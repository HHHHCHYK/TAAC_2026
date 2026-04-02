from __future__ import annotations

import torch
from torch import nn

from ..config import DataConfig, ModelConfig
from ..types import BatchTensors
from .common import (
    CrossAttentionBlock,
    DINActivationUnit,
    TransformerEncoderBlock,
    UnifiedFeatureEncoder,
    build_mlp_head,
    build_pooled_memory,
    build_position_encoding,
    make_recsys_attn_mask,
    masked_mean,
)


class DeepContextNet(nn.Module):
    def __init__(self, config: ModelConfig, data_config: DataConfig, dense_dim: int) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.head_hidden_dim = config.head_hidden_dim or config.hidden_dim * 2
        self.encoder = UnifiedFeatureEncoder(config, data_config, dense_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        self.blocks = nn.ModuleList(
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
        self.output = build_mlp_head(config.hidden_dim * 5, self.head_hidden_dim, config.dropout)

    def forward(self, batch: BatchTensors) -> torch.Tensor:
        batch_size = batch.batch_size
        views = self.encoder.build_views(batch)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        candidate_token = views.candidate_token.unsqueeze(1)
        tokens = torch.cat([cls_token, views.dense_token, views.context_tokens, candidate_token, views.flat_history_tokens], dim=1)
        tokens = tokens + build_position_encoding(tokens.size(1), tokens.size(-1), tokens.device, tokens.dtype)

        candidate_index = 2 + views.context_tokens.size(1)
        mask = torch.cat(
            [
                torch.ones((batch_size, 2), dtype=torch.bool, device=tokens.device),
                batch.context_mask,
                torch.ones((batch_size, 1), dtype=torch.bool, device=tokens.device),
                batch.history_mask,
            ],
            dim=1,
        )
        for block in self.blocks:
            tokens = block(tokens, token_mask=mask, attn_mask=None)

        cls_output = tokens[:, 0, :]
        candidate_output = tokens[:, candidate_index, :]
        history_output = tokens[:, candidate_index + 1 :, :]
        history_summary = masked_mean(history_output, batch.history_mask)
        interaction = cls_output * candidate_output
        difference = torch.abs(candidate_output - history_summary)
        fused = torch.cat([cls_output, candidate_output, history_summary, interaction, difference], dim=-1)
        return self.output(fused).squeeze(-1)


class UniRecBackboneBase(nn.Module):
    def __init__(self, config: ModelConfig, data_config: DataConfig, dense_dim: int) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.head_hidden_dim = config.head_hidden_dim or config.hidden_dim * 2
        self.encoder = UnifiedFeatureEncoder(config, data_config, dense_dim)
        self.feature_cross_layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                    ffn_multiplier=config.ffn_multiplier,
                )
                for _ in range(max(config.feature_cross_layers, 1))
            ]
        )
        self.interest_attention = DINActivationUnit(config.hidden_dim)
        self.blocks = nn.ModuleList(
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

    def _encode_backbone(self, batch: BatchTensors) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        batch_size = batch.batch_size
        views = self.encoder.build_views(batch)
        context_tokens = views.context_tokens
        for layer in self.feature_cross_layers:
            context_tokens = layer(context_tokens, token_mask=batch.context_mask, attn_mask=None)

        interest_token = self.interest_attention(views.candidate_token, views.flat_history_tokens, batch.history_mask).unsqueeze(1)
        prefix_tokens = torch.cat([views.dense_token, context_tokens, interest_token, views.flat_history_tokens], dim=1)
        static_prefix_len = 1 + context_tokens.size(1) + 1
        candidate_start_offset = prefix_tokens.size(1)
        tokens = torch.cat([prefix_tokens, views.candidate_token.unsqueeze(1)], dim=1)
        tokens = tokens + build_position_encoding(tokens.size(1), tokens.size(-1), tokens.device, tokens.dtype)

        prefix_mask = torch.cat(
            [
                torch.ones((batch_size, 1), dtype=torch.bool, device=tokens.device),
                batch.context_mask,
                torch.ones((batch_size, 1), dtype=torch.bool, device=tokens.device),
                batch.history_mask,
            ],
            dim=1,
        )
        sequence_mask = torch.cat(
            [
                prefix_mask,
                torch.ones((batch_size, 1), dtype=torch.bool, device=tokens.device),
            ],
            dim=1,
        )
        attn_mask = make_recsys_attn_mask(
            seq_len=tokens.size(1),
            static_prefix_len=static_prefix_len,
            candidate_start_offset=candidate_start_offset,
            device=tokens.device,
        )
        for block in self.blocks:
            tokens = block(tokens, token_mask=sequence_mask, attn_mask=attn_mask)

        return tokens, views.dense_summary, static_prefix_len, candidate_start_offset


class UniRecModel(UniRecBackboneBase):
    def __init__(self, config: ModelConfig, data_config: DataConfig, dense_dim: int) -> None:
        super().__init__(config=config, data_config=data_config, dense_dim=dense_dim)
        self.output = build_mlp_head(config.hidden_dim * 6, self.head_hidden_dim, config.dropout)

    def forward(self, batch: BatchTensors) -> torch.Tensor:
        tokens, dense_summary, static_prefix_len, candidate_start_offset = self._encode_backbone(batch)
        candidate_output = tokens[:, candidate_start_offset, :]
        interest_output = tokens[:, static_prefix_len - 1, :]
        history_summary = masked_mean(tokens[:, static_prefix_len:candidate_start_offset, :], batch.history_mask)
        route = candidate_output * interest_output
        difference = torch.abs(candidate_output - history_summary)
        fused = torch.cat([candidate_output, interest_output, history_summary, dense_summary, route, difference], dim=-1)
        return self.output(fused).squeeze(-1)


class UniRecDINReadoutModel(UniRecBackboneBase):
    def __init__(self, config: ModelConfig, data_config: DataConfig, dense_dim: int) -> None:
        super().__init__(config=config, data_config=data_config, dense_dim=dense_dim)
        self.readout_attention = DINActivationUnit(config.hidden_dim)
        self.output = build_mlp_head(config.hidden_dim * 8, self.head_hidden_dim, config.dropout)

    def forward(self, batch: BatchTensors) -> torch.Tensor:
        tokens, dense_summary, static_prefix_len, candidate_start_offset = self._encode_backbone(batch)
        candidate_output = tokens[:, candidate_start_offset, :]
        interest_output = tokens[:, static_prefix_len - 1, :]
        history_outputs = tokens[:, static_prefix_len:candidate_start_offset, :]
        history_summary = masked_mean(history_outputs, batch.history_mask)
        readout_summary = self.readout_attention(candidate_output, history_outputs, batch.history_mask)
        route = candidate_output * interest_output
        readout_route = candidate_output * readout_summary
        difference = torch.abs(candidate_output - readout_summary)
        fused = torch.cat(
            [
                candidate_output,
                interest_output,
                history_summary,
                readout_summary,
                dense_summary,
                route,
                readout_route,
                difference,
            ],
            dim=-1,
        )
        return self.output(fused).squeeze(-1)


class UniScaleFormer(nn.Module):
    def __init__(self, config: ModelConfig, data_config: DataConfig, dense_dim: int) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.head_hidden_dim = config.head_hidden_dim or config.hidden_dim * 2
        self.recent_seq_len = max(config.recent_seq_len, 1)
        self.memory_slots = max(config.memory_slots, 1)
        self.encoder = UnifiedFeatureEncoder(config, data_config, dense_dim)
        self.cross_attention = CrossAttentionBlock(config.hidden_dim, config.num_heads, config.dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                    ffn_multiplier=config.ffn_multiplier,
                )
                for _ in range(max(config.num_layers - 1, 1))
            ]
        )
        self.output = build_mlp_head(config.hidden_dim * 6, self.head_hidden_dim, config.dropout)

    def forward(self, batch: BatchTensors) -> torch.Tensor:
        batch_size = batch.batch_size
        views = self.encoder.build_views(batch)
        recent_len = min(self.recent_seq_len, views.flat_history_tokens.size(1))
        local_history = views.flat_history_tokens[:, :recent_len]
        local_mask = batch.history_mask[:, :recent_len]
        memory_tokens, memory_mask = build_pooled_memory(
            history_embeddings=views.flat_history_tokens,
            history_mask=batch.history_mask,
            recent_seq_len=recent_len,
            memory_slots=self.memory_slots,
        )
        memory_context = torch.cat([local_history, memory_tokens], dim=1)
        memory_context_mask = torch.cat([local_mask, memory_mask], dim=1)

        candidate_query = views.candidate_token.unsqueeze(1)
        if memory_context.size(1) > 0:
            candidate_token = self.cross_attention(
                query_tokens=candidate_query,
                key_value_tokens=memory_context,
                query_mask=torch.ones((batch_size, 1), dtype=torch.bool, device=candidate_query.device),
                key_mask=memory_context_mask,
            )
        else:
            candidate_token = candidate_query

        prefix_tokens = torch.cat([views.dense_token, views.context_tokens, memory_context], dim=1)
        candidate_start_offset = prefix_tokens.size(1)
        tokens = torch.cat([prefix_tokens, candidate_token], dim=1)
        tokens = tokens + build_position_encoding(tokens.size(1), tokens.size(-1), tokens.device, tokens.dtype)

        sequence_mask = torch.cat(
            [
                torch.ones((batch_size, 1), dtype=torch.bool, device=tokens.device),
                batch.context_mask,
                memory_context_mask,
                torch.ones((batch_size, 1), dtype=torch.bool, device=tokens.device),
            ],
            dim=1,
        )
        for block in self.blocks:
            tokens = block(tokens, token_mask=sequence_mask, attn_mask=None)

        candidate_output = tokens[:, candidate_start_offset, :]
        context_summary = masked_mean(tokens[:, 1 : 1 + views.context_tokens.size(1), :], batch.context_mask)
        memory_summary = masked_mean(tokens[:, 1 + views.context_tokens.size(1) : candidate_start_offset, :], memory_context_mask)
        interaction = candidate_output * memory_summary
        difference = torch.abs(candidate_output - views.candidate_token)
        fused = torch.cat([candidate_output, views.candidate_token, context_summary, memory_summary, interaction, difference], dim=-1)
        return self.output(fused).squeeze(-1)


__all__ = ["DeepContextNet", "UniRecDINReadoutModel", "UniRecModel", "UniScaleFormer"]