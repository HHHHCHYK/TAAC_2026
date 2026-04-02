from __future__ import annotations

import torch
from torch import nn

from ..config import DataConfig, ModelConfig
from ..types import BatchTensors
from .common import (
    DINActivationUnit,
    TransformerEncoderBlock,
    UnifiedFeatureEncoder,
    build_causal_attention_mask,
    build_mlp_head,
    build_pooled_memory,
    masked_attention_pool,
    masked_mean,
)


class SequenceModelBase(nn.Module):
    def __init__(self, config: ModelConfig, data_config: DataConfig, dense_dim: int) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.head_hidden_dim = config.head_hidden_dim or config.hidden_dim * 2
        self.recent_seq_len = max(config.recent_seq_len, 1)
        self.memory_slots = max(config.memory_slots, 1)
        self.encoder = UnifiedFeatureEncoder(config, data_config, dense_dim)
        self.context_projection = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
        )
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

    def encode_inputs(
        self,
        batch: BatchTensors,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        views = self.encoder.build_views(batch)
        candidate_summary = views.candidate_token
        context_summary = masked_mean(views.context_tokens, batch.context_mask)
        history_prior = masked_mean(views.flat_history_tokens, batch.history_mask)
        dense_summary = views.dense_summary
        context_enhanced = self.context_projection(torch.cat([context_summary, history_prior], dim=-1))
        return candidate_summary, context_summary, views.flat_history_tokens, dense_summary, context_enhanced

    def encode_history(self, history_embeddings: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        if history_embeddings.size(1) == 0:
            return history_embeddings

        encoded = history_embeddings
        attn_mask = build_causal_attention_mask(history_embeddings.size(1), history_embeddings.device)
        for block in self.blocks:
            encoded = block(encoded, token_mask=history_mask, attn_mask=attn_mask)
        return encoded


class TencentSASRecAdapter(SequenceModelBase):
    def __init__(self, config: ModelConfig, data_config: DataConfig, dense_dim: int) -> None:
        super().__init__(config=config, data_config=data_config, dense_dim=dense_dim)
        self.output = build_mlp_head(config.hidden_dim * 8, self.head_hidden_dim, config.dropout)

    def forward(self, batch: BatchTensors) -> torch.Tensor:
        candidate_summary, context_summary, history_embeddings, dense_summary, context_enhanced = self.encode_inputs(batch)
        encoded_history = self.encode_history(history_embeddings, batch.history_mask)
        history_summary = masked_attention_pool(encoded_history, batch.history_mask, candidate_summary)
        recent_history = encoded_history[:, : self.recent_seq_len]
        recent_mask = batch.history_mask[:, : self.recent_seq_len]
        recent_summary = (
            masked_mean(recent_history, recent_mask)
            if recent_history.size(1) > 0
            else torch.zeros_like(candidate_summary)
        )
        interaction = candidate_summary * history_summary
        difference = torch.abs(candidate_summary - history_summary)
        fused = torch.cat(
            [
                candidate_summary,
                context_summary,
                history_summary,
                recent_summary,
                context_enhanced,
                interaction,
                difference,
                dense_summary,
            ],
            dim=-1,
        )
        return self.output(fused).squeeze(-1)


class RetrievalStyleAdapter(SequenceModelBase):
    def __init__(self, config: ModelConfig, data_config: DataConfig, dense_dim: int, variant: str) -> None:
        super().__init__(config=config, data_config=data_config, dense_dim=dense_dim)
        self.variant = variant
        self.global_attention = DINActivationUnit(config.hidden_dim)
        self.action_attention = DINActivationUnit(config.hidden_dim)
        self.content_attention = DINActivationUnit(config.hidden_dim)
        self.item_attention = DINActivationUnit(config.hidden_dim)
        self.route_gate = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 3),
        )
        hidden_dim = self.head_hidden_dim * (2 if variant == "omnigenrec_adapter" else 1)
        self.output = build_mlp_head(config.hidden_dim * 12, hidden_dim, config.dropout)

    def forward(self, batch: BatchTensors) -> torch.Tensor:
        candidate_summary, context_summary, history_embeddings, dense_summary, context_enhanced = self.encode_inputs(batch)
        encoded_history = self.encode_history(history_embeddings, batch.history_mask)

        global_summary = self.global_attention(candidate_summary, encoded_history, batch.history_mask)
        local_history = encoded_history[:, : self.recent_seq_len]
        local_mask = batch.history_mask[:, : self.recent_seq_len]
        local_summary = self.global_attention(candidate_summary, local_history, local_mask)
        memory_tokens, memory_mask = build_pooled_memory(
            history_embeddings=encoded_history,
            history_mask=batch.history_mask,
            recent_seq_len=self.recent_seq_len,
            memory_slots=self.memory_slots,
        )
        memory_summary = masked_attention_pool(memory_tokens, memory_mask, candidate_summary)

        action_mask = batch.history_mask & (batch.history_group_ids == 1)
        content_mask = batch.history_mask & (batch.history_group_ids == 2)
        item_mask = batch.history_mask & (batch.history_group_ids == 3)
        action_summary = self.action_attention(candidate_summary, encoded_history, action_mask)
        content_summary = self.content_attention(candidate_summary, encoded_history, content_mask)
        item_summary = self.item_attention(candidate_summary, encoded_history, item_mask)

        route_logits = self.route_gate(torch.cat([candidate_summary, context_enhanced, dense_summary], dim=-1))
        route_weights = torch.softmax(route_logits, dim=-1)
        route_stack = torch.stack([action_summary, content_summary, item_summary], dim=1)
        grouped_summary = (route_stack * route_weights.unsqueeze(-1)).sum(dim=1)
        route_spread = route_stack.std(dim=1, correction=0)

        if self.variant == "omnigenrec_adapter":
            grouped_summary = grouped_summary + 0.5 * local_summary

        interaction_global = candidate_summary * global_summary
        interaction_grouped = candidate_summary * grouped_summary
        local_memory_gap = torch.abs(local_summary - memory_summary)
        fused = torch.cat(
            [
                candidate_summary,
                context_summary,
                context_enhanced,
                global_summary,
                local_summary,
                memory_summary,
                grouped_summary,
                route_spread,
                interaction_global,
                interaction_grouped,
                local_memory_gap,
                dense_summary,
            ],
            dim=-1,
        )
        return self.output(fused).squeeze(-1)


__all__ = ["RetrievalStyleAdapter", "TencentSASRecAdapter"]