from __future__ import annotations

import torch
from torch import nn

from ..config import DataConfig, ModelConfig
from ..types import BatchTensors
from .common import (
    CrossAttentionBlock,
    FeedForwardBlock,
    SelfAttentionBlock,
    TransformerEncoderBlock,
    UnifiedFeatureEncoder,
    build_mlp_head,
    build_position_encoding,
    masked_mean,
)


class InterFormerLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float, ffn_multiplier: int) -> None:
        super().__init__()
        self.seq_self = SelfAttentionBlock(hidden_dim, num_heads, dropout)
        self.static_self = SelfAttentionBlock(hidden_dim, num_heads, dropout)
        self.seq_cross = CrossAttentionBlock(hidden_dim, num_heads, dropout)
        self.static_cross = CrossAttentionBlock(hidden_dim, num_heads, dropout)
        self.seq_ff = FeedForwardBlock(hidden_dim, ffn_multiplier, dropout)
        self.static_ff = FeedForwardBlock(hidden_dim, ffn_multiplier, dropout)

    def forward(
        self,
        seq_tokens: torch.Tensor,
        seq_mask: torch.Tensor,
        static_tokens: torch.Tensor,
        static_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_tokens = self.seq_self(seq_tokens, token_mask=seq_mask, attn_mask=None)
        static_tokens = self.static_self(static_tokens, token_mask=static_mask, attn_mask=None)
        seq_tokens = self.seq_cross(seq_tokens, static_tokens, query_mask=seq_mask, key_mask=static_mask)
        static_tokens = self.static_cross(static_tokens, seq_tokens, query_mask=static_mask, key_mask=seq_mask)
        seq_tokens = self.seq_ff(seq_tokens, token_mask=seq_mask)
        static_tokens = self.static_ff(static_tokens, token_mask=static_mask)
        return seq_tokens, static_tokens


class InterFormer(nn.Module):
    def __init__(self, config: ModelConfig, data_config: DataConfig, dense_dim: int) -> None:
        super().__init__()
        self.head_hidden_dim = config.head_hidden_dim or config.hidden_dim * 2
        self.encoder = UnifiedFeatureEncoder(config, data_config, dense_dim)
        self.seq_pre_encoder = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                    ffn_multiplier=config.ffn_multiplier,
                )
                for _ in range(max(config.sequence_layers, 1))
            ]
        )
        self.layers = nn.ModuleList(
            [
                InterFormerLayer(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                    ffn_multiplier=config.ffn_multiplier,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.head = build_mlp_head(config.hidden_dim * 2, self.head_hidden_dim, config.dropout)

    def forward(self, batch: BatchTensors) -> torch.Tensor:
        views = self.encoder.build_views(batch)
        seq_tokens = views.flat_history_tokens
        seq_mask = batch.history_mask
        static_tokens = views.static_tokens
        static_mask = views.static_mask

        for block in self.seq_pre_encoder:
            seq_tokens = block(seq_tokens, token_mask=seq_mask, attn_mask=None)
        for layer in self.layers:
            seq_tokens, static_tokens = layer(seq_tokens, seq_mask, static_tokens, static_mask)

        seq_pool = masked_mean(seq_tokens, seq_mask)
        static_pool = masked_mean(static_tokens, static_mask)
        return self.head(torch.cat([seq_pool, static_pool], dim=-1)).squeeze(-1)


class OneTrans(nn.Module):
    def __init__(self, config: ModelConfig, data_config: DataConfig, dense_dim: int) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.head_hidden_dim = config.head_hidden_dim or config.hidden_dim * 2
        self.encoder = UnifiedFeatureEncoder(config, data_config, dense_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_dim) * 0.02)
        self.segment_embedding = nn.Embedding(max(config.segment_count, 4), config.hidden_dim, padding_idx=0)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                    ffn_multiplier=config.ffn_multiplier,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        self.head = build_mlp_head(config.hidden_dim, self.head_hidden_dim, config.dropout)

    def forward(self, batch: BatchTensors) -> torch.Tensor:
        views = self.encoder.build_views(batch)
        batch_size = batch.batch_size
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, views.static_tokens, views.flat_history_tokens], dim=1)
        tokens = tokens + build_position_encoding(tokens.size(1), tokens.size(-1), tokens.device, tokens.dtype)

        static_len = views.static_tokens.size(1)
        history_len = views.flat_history_tokens.size(1)
        segment_ids = torch.cat(
            [
                torch.zeros((batch_size, 1), dtype=torch.long, device=tokens.device),
                torch.ones((batch_size, static_len), dtype=torch.long, device=tokens.device),
                torch.full((batch_size, history_len), 2, dtype=torch.long, device=tokens.device),
            ],
            dim=1,
        )
        tokens = tokens + self.segment_embedding(segment_ids)

        mask = torch.cat(
            [
                torch.ones((batch_size, 1), dtype=torch.bool, device=tokens.device),
                views.static_mask,
                batch.history_mask,
            ],
            dim=1,
        )
        for layer in self.layers:
            tokens = layer(tokens, token_mask=mask, attn_mask=None)
        tokens = self.final_norm(tokens)
        return self.head(tokens[:, 0, :]).squeeze(-1)


class QueryDecoder(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, num_queries: int, num_layers: int, dropout: float, ffn_multiplier: int) -> None:
        super().__init__()
        self.query_seed = nn.Parameter(torch.randn(1, num_queries, hidden_dim) * 0.02)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "cross": CrossAttentionBlock(hidden_dim, num_heads, dropout),
                        "self": SelfAttentionBlock(hidden_dim, num_heads, dropout),
                        "ff": FeedForwardBlock(hidden_dim, ffn_multiplier, dropout),
                    }
                )
            )

    def forward(self, seq_tokens: torch.Tensor, seq_mask: torch.Tensor) -> torch.Tensor:
        batch_size = seq_tokens.size(0)
        queries = self.query_seed.expand(batch_size, -1, -1)
        query_mask = torch.ones(queries.shape[:2], dtype=torch.bool, device=queries.device)
        for layer in self.layers:
            queries = layer["cross"](queries, seq_tokens, query_mask=query_mask, key_mask=seq_mask)
            queries = layer["self"](queries, token_mask=query_mask, attn_mask=None)
            queries = layer["ff"](queries, token_mask=query_mask)
        return queries


class QueryBooster(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float, ffn_multiplier: int) -> None:
        super().__init__()
        self.cross = CrossAttentionBlock(hidden_dim, num_heads, dropout)
        self.ff = FeedForwardBlock(hidden_dim, ffn_multiplier, dropout)

    def forward(
        self,
        static_tokens: torch.Tensor,
        static_mask: torch.Tensor,
        queries: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> torch.Tensor:
        boosted = self.cross(static_tokens, queries, query_mask=static_mask, key_mask=query_mask)
        return self.ff(boosted, token_mask=static_mask)


class HyFormer(nn.Module):
    def __init__(self, config: ModelConfig, data_config: DataConfig, dense_dim: int) -> None:
        super().__init__()
        self.head_hidden_dim = config.head_hidden_dim or config.hidden_dim * 2
        self.encoder = UnifiedFeatureEncoder(config, data_config, dense_dim)
        self.static_encoder = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                    ffn_multiplier=config.ffn_multiplier,
                )
                for _ in range(max(config.static_layers, 1))
            ]
        )
        self.seq_encoder = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                    ffn_multiplier=config.ffn_multiplier,
                )
                for _ in range(max(config.sequence_layers, 1))
            ]
        )
        self.query_decoders = nn.ModuleList(
            [
                QueryDecoder(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    num_queries=max(config.num_queries, 1),
                    num_layers=max(config.query_decoder_layers, 1),
                    dropout=config.dropout,
                    ffn_multiplier=config.ffn_multiplier,
                )
                for _ in data_config.sequence_names
            ]
        )
        self.booster = QueryBooster(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            ffn_multiplier=config.ffn_multiplier,
        )
        self.fusion_layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                    ffn_multiplier=config.ffn_multiplier,
                )
                for _ in range(max(config.fusion_layers, 1))
            ]
        )
        self.head = build_mlp_head(config.hidden_dim * 2, self.head_hidden_dim, config.dropout)

    def forward(self, batch: BatchTensors) -> torch.Tensor:
        views = self.encoder.build_views(batch)
        static_tokens = views.static_tokens
        static_mask = views.static_mask
        for block in self.static_encoder:
            static_tokens = block(static_tokens, token_mask=static_mask, attn_mask=None)

        all_queries: list[torch.Tensor] = []
        for group_index, decoder in enumerate(self.query_decoders):
            sequence_tokens = views.sequence_tokens[:, group_index, :, :]
            sequence_mask = batch.sequence_mask[:, group_index, :]
            encoded_sequence = sequence_tokens
            for block in self.seq_encoder:
                encoded_sequence = block(encoded_sequence, token_mask=sequence_mask, attn_mask=None)
            all_queries.append(decoder(encoded_sequence, sequence_mask))

        queries = torch.cat(all_queries, dim=1)
        query_mask = torch.ones(queries.shape[:2], dtype=torch.bool, device=queries.device)
        boosted_static = self.booster(static_tokens, static_mask, queries, query_mask)

        fused = torch.cat([queries, boosted_static], dim=1)
        fused_mask = torch.cat([query_mask, static_mask], dim=1)
        for block in self.fusion_layers:
            fused = block(fused, token_mask=fused_mask, attn_mask=None)

        query_count = queries.size(1)
        query_pool = fused[:, :query_count, :].mean(dim=1)
        boosted_pool = masked_mean(fused[:, query_count:, :], static_mask)
        return self.head(torch.cat([query_pool, boosted_pool], dim=-1)).squeeze(-1)


__all__ = ["HyFormer", "InterFormer", "OneTrans"]