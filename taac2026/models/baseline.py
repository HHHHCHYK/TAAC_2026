from __future__ import annotations

import torch
from torch import nn

from ..config import DataConfig, ModelConfig
from ..types import BatchTensors
from .common import (
    DINActivationUnit,
    TransformerEncoderBlock,
    UnifiedFeatureEncoder,
    build_mlp_head,
    build_position_encoding,
    make_recsys_attn_mask,
    masked_mean,
)


class GrokBackboneBase(nn.Module):
    def __init__(self, config: ModelConfig, data_config: DataConfig, dense_dim: int) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.head_hidden_dim = config.head_hidden_dim or config.hidden_dim * 2
        self.encoder = UnifiedFeatureEncoder(config, data_config, dense_dim)
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

    def _encode_backbone(
        self,
        batch: BatchTensors,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int, torch.Tensor]:
        batch_size = batch.batch_size
        views = self.encoder.build_views(batch)

        prefix_tokens = torch.cat([views.dense_token, views.context_tokens, views.flat_history_tokens], dim=1)
        static_prefix_len = 1 + views.context_tokens.size(1)
        candidate_start_offset = prefix_tokens.size(1)
        tokens = torch.cat([prefix_tokens, views.candidate_token.unsqueeze(1)], dim=1)
        tokens = tokens + build_position_encoding(tokens.size(1), tokens.size(-1), tokens.device, tokens.dtype)

        prefix_mask = torch.cat(
            [
                torch.ones((batch_size, 1), dtype=torch.bool, device=tokens.device),
                batch.context_mask,
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

        return tokens, prefix_mask, static_prefix_len, candidate_start_offset, views.candidate_token


class GrokUnifiedBaseline(GrokBackboneBase):
    def __init__(self, config: ModelConfig, data_config: DataConfig, dense_dim: int) -> None:
        super().__init__(config=config, data_config=data_config, dense_dim=dense_dim)
        self.output = build_mlp_head(self.hidden_dim * 4, self.head_hidden_dim, config.dropout)

    def forward(self, batch: BatchTensors) -> torch.Tensor:
        tokens, prefix_mask, _, candidate_start_offset, candidate_seed = self._encode_backbone(batch)
        candidate_output = tokens[:, candidate_start_offset, :]
        prefix_summary = masked_mean(tokens[:, :candidate_start_offset, :], prefix_mask)
        interaction = candidate_output * prefix_summary
        difference = torch.abs(candidate_output - candidate_seed)
        fused = torch.cat([candidate_output, prefix_summary, interaction, difference], dim=-1)
        return self.output(fused).squeeze(-1)


class GrokDINReadoutBaseline(GrokBackboneBase):
    def __init__(self, config: ModelConfig, data_config: DataConfig, dense_dim: int) -> None:
        super().__init__(config=config, data_config=data_config, dense_dim=dense_dim)
        self.interest_attention = DINActivationUnit(config.hidden_dim)
        self.output = build_mlp_head(self.hidden_dim * 7, self.head_hidden_dim, config.dropout)

    def forward(self, batch: BatchTensors) -> torch.Tensor:
        tokens, prefix_mask, static_prefix_len, candidate_start_offset, candidate_seed = self._encode_backbone(batch)
        candidate_output = tokens[:, candidate_start_offset, :]
        prefix_summary = masked_mean(tokens[:, :candidate_start_offset, :], prefix_mask)
        history_outputs = tokens[:, static_prefix_len:candidate_start_offset, :]
        history_summary = masked_mean(history_outputs, batch.history_mask)
        interest_summary = self.interest_attention(candidate_output, history_outputs, batch.history_mask)
        interest_interaction = candidate_output * interest_summary
        prefix_interaction = candidate_output * prefix_summary
        candidate_gap = torch.abs(candidate_output - candidate_seed)
        fused = torch.cat(
            [
                candidate_output,
                prefix_summary,
                history_summary,
                interest_summary,
                interest_interaction,
                prefix_interaction,
                candidate_gap,
            ],
            dim=-1,
        )
        return self.output(fused).squeeze(-1)


__all__ = ["GrokDINReadoutBaseline", "GrokUnifiedBaseline"]