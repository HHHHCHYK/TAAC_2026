from __future__ import annotations

import torch
from torch import nn

from ..config import DataConfig, ModelConfig
from ..types import BatchTensors
from .common import DINActivationUnit, UnifiedFeatureEncoder, build_mlp_head, masked_mean


class NativeDINBase(nn.Module):
    def __init__(self, config: ModelConfig, data_config: DataConfig, dense_dim: int) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.head_hidden_dim = config.head_hidden_dim or config.hidden_dim * 2
        self.encoder = UnifiedFeatureEncoder(config, data_config, dense_dim)
        self.context_projection = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
        )


class CreatorwyxDINAdapter(NativeDINBase):
    def __init__(self, config: ModelConfig, data_config: DataConfig, dense_dim: int) -> None:
        super().__init__(config=config, data_config=data_config, dense_dim=dense_dim)
        self.din_attention = DINActivationUnit(config.hidden_dim)
        self.output = build_mlp_head(config.hidden_dim * 7, self.head_hidden_dim, config.dropout)

    def forward(self, batch: BatchTensors) -> torch.Tensor:
        views = self.encoder.build_views(batch)
        candidate_summary = views.candidate_token
        context_summary = masked_mean(views.context_tokens, batch.context_mask)
        history_summary = self.din_attention(candidate_summary, views.flat_history_tokens, batch.history_mask)
        context_enhanced = self.context_projection(torch.cat([context_summary, history_summary], dim=-1))
        interaction = candidate_summary * history_summary
        difference = torch.abs(candidate_summary - history_summary)
        context_interaction = candidate_summary * context_enhanced
        fused = torch.cat(
            [
                candidate_summary,
                history_summary,
                context_summary,
                context_enhanced,
                interaction,
                difference,
                context_interaction + views.dense_summary,
            ],
            dim=-1,
        )
        return self.output(fused).squeeze(-1)


class CreatorwyxGroupedDINAdapter(NativeDINBase):
    def __init__(self, config: ModelConfig, data_config: DataConfig, dense_dim: int) -> None:
        super().__init__(config=config, data_config=data_config, dense_dim=dense_dim)
        self.action_attention = DINActivationUnit(config.hidden_dim)
        self.content_attention = DINActivationUnit(config.hidden_dim)
        self.item_attention = DINActivationUnit(config.hidden_dim)
        self.route_gate = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 3),
        )
        self.output = build_mlp_head(config.hidden_dim * 11, self.head_hidden_dim, config.dropout)

    def forward(self, batch: BatchTensors) -> torch.Tensor:
        views = self.encoder.build_views(batch)
        candidate_summary = views.candidate_token
        context_summary = masked_mean(views.context_tokens, batch.context_mask)

        action_tokens = views.sequence_tokens[:, 0]
        content_tokens = views.sequence_tokens[:, 1]
        item_tokens = views.sequence_tokens[:, 2]
        action_mask = batch.sequence_mask[:, 0]
        content_mask = batch.sequence_mask[:, 1]
        item_mask = batch.sequence_mask[:, 2]

        route_presence = torch.stack(
            [action_mask.any(dim=1), content_mask.any(dim=1), item_mask.any(dim=1)],
            dim=1,
        )
        route_mean_stack = torch.stack(
            [
                masked_mean(action_tokens, action_mask),
                masked_mean(content_tokens, content_mask),
                masked_mean(item_tokens, item_mask),
            ],
            dim=1,
        )
        global_summary = (
            route_mean_stack * route_presence.unsqueeze(-1).float()
        ).sum(dim=1) / route_presence.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        context_enhanced = self.context_projection(torch.cat([context_summary, global_summary], dim=-1))

        action_summary = self.action_attention(candidate_summary, action_tokens, action_mask)
        content_summary = self.content_attention(candidate_summary, content_tokens, content_mask)
        item_summary = self.item_attention(candidate_summary, item_tokens, item_mask)

        route_logits = self.route_gate(torch.cat([candidate_summary, context_enhanced, views.dense_summary], dim=-1))
        route_logits = route_logits.masked_fill(~route_presence, -1e4)
        route_weights = torch.softmax(route_logits, dim=-1)
        route_weights = route_weights * route_presence.float()
        route_weights = route_weights / route_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        route_stack = torch.stack([action_summary, content_summary, item_summary], dim=1)
        grouped_summary = (route_stack * route_weights.unsqueeze(-1)).sum(dim=1)
        route_spread = route_stack.std(dim=1, correction=0)
        grouped_interaction = candidate_summary * grouped_summary
        grouped_gap = torch.abs(candidate_summary - grouped_summary)

        fused = torch.cat(
            [
                candidate_summary,
                context_summary,
                context_enhanced,
                action_summary,
                content_summary,
                item_summary,
                grouped_summary,
                route_spread,
                grouped_interaction,
                grouped_gap,
                views.dense_summary,
            ],
            dim=-1,
        )
        return self.output(fused).squeeze(-1)


__all__ = ["CreatorwyxDINAdapter", "CreatorwyxGroupedDINAdapter"]