from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..config import DataConfig, ModelConfig
from ..types import BatchTensors


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).float()
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def masked_attention_pool(values: torch.Tensor, mask: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    if values.size(1) == 0:
        return torch.zeros_like(query)

    scores = (values * query.unsqueeze(1)).sum(dim=-1) / math.sqrt(max(values.size(-1), 1))
    scores = scores.masked_fill(~mask, -1e4)
    weights = torch.softmax(scores, dim=-1)
    weights = weights * mask.float()
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return torch.bmm(weights.unsqueeze(1), values).squeeze(1)


def build_position_encoding(length: int, hidden_dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if length == 0:
        return torch.zeros((1, 0, hidden_dim), device=device, dtype=dtype)

    positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, hidden_dim, 2, device=device, dtype=torch.float32) * (-math.log(10_000.0) / max(hidden_dim, 1))
    )
    encoding = torch.zeros((length, hidden_dim), device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * div_term)
    encoding[:, 1::2] = torch.cos(positions * div_term)
    return encoding.unsqueeze(0).to(dtype=dtype)


def build_causal_attention_mask(length: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones((length, length), dtype=torch.bool, device=device), diagonal=1)


def make_recsys_attn_mask(
    seq_len: int,
    static_prefix_len: int,
    candidate_start_offset: int,
    device: torch.device,
) -> torch.Tensor:
    allowed = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)
    allowed[:static_prefix_len, :static_prefix_len] = True

    history_len = candidate_start_offset - static_prefix_len
    if history_len > 0:
        allowed[static_prefix_len:candidate_start_offset, :static_prefix_len] = True
        allowed[static_prefix_len:candidate_start_offset, static_prefix_len:candidate_start_offset] = torch.tril(
            torch.ones((history_len, history_len), dtype=torch.bool, device=device)
        )

    candidate_len = seq_len - candidate_start_offset
    if candidate_len > 0:
        allowed[candidate_start_offset:, :candidate_start_offset] = True
        allowed[candidate_start_offset:, candidate_start_offset:] = torch.eye(
            candidate_len,
            dtype=torch.bool,
            device=device,
        )
    return ~allowed


def build_pooled_memory(
    history_embeddings: torch.Tensor,
    history_mask: torch.Tensor,
    recent_seq_len: int,
    memory_slots: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    far_history = history_embeddings[:, recent_seq_len:]
    far_mask = history_mask[:, recent_seq_len:]
    if far_history.size(1) == 0 or memory_slots <= 0:
        return history_embeddings[:, :0], history_mask[:, :0]

    masked_far_history = far_history * far_mask.unsqueeze(-1).float()
    pooled_values = F.adaptive_avg_pool1d(masked_far_history.transpose(1, 2), memory_slots).transpose(1, 2)
    pooled_mask = F.adaptive_avg_pool1d(far_mask.float().unsqueeze(1), memory_slots).squeeze(1)
    memory_embeddings = pooled_values / pooled_mask.unsqueeze(-1).clamp_min(1e-6)
    memory_mask = pooled_mask > 0
    return memory_embeddings, memory_mask


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, query_len, _ = query.shape
        key_len = key.size(1)

        q = self.query_projection(query).view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key_projection(key).view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value_projection(value).view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        valid_mask = torch.ones_like(scores, dtype=torch.bool)
        if attn_mask is not None:
            if attn_mask.ndim == 2:
                blocked = attn_mask.view(1, 1, query_len, key_len)
            elif attn_mask.ndim == 3:
                blocked = attn_mask.unsqueeze(1)
            else:
                blocked = attn_mask
            valid_mask = valid_mask & ~blocked
            scores = scores.masked_fill(blocked, -1e4)

        if key_mask is not None:
            expanded_key_mask = key_mask[:, None, None, :].bool()
            valid_mask = valid_mask & expanded_key_mask
            scores = scores.masked_fill(~expanded_key_mask, -1e4)

        weights = torch.softmax(scores, dim=-1)
        weights = weights * valid_mask.float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        weights = self.dropout(weights)

        attended = (weights @ v).transpose(1, 2).contiguous().view(batch_size, query_len, self.hidden_dim)
        return self.output_projection(attended)


class FeedForwardBlock(nn.Module):
    def __init__(self, hidden_dim: int, ffn_multiplier: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ffn_multiplier, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        output = tokens + self.dropout(self.ffn(self.norm(tokens)))
        if token_mask is not None:
            output = output * token_mask.unsqueeze(-1).float()
        return output


class SelfAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.attention = MultiHeadAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attended = self.attention(
            query=self.norm(tokens),
            key=self.norm(tokens),
            value=self.norm(tokens),
            key_mask=token_mask,
            attn_mask=attn_mask,
        )
        output = tokens + self.dropout(attended)
        if token_mask is not None:
            output = output * token_mask.unsqueeze(-1).float()
        return output


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.key_value_norm = nn.LayerNorm(hidden_dim)
        self.attention = MultiHeadAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query_tokens: torch.Tensor,
        key_value_tokens: torch.Tensor,
        query_mask: torch.Tensor | None = None,
        key_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attended = self.attention(
            query=self.query_norm(query_tokens),
            key=self.key_value_norm(key_value_tokens),
            value=self.key_value_norm(key_value_tokens),
            key_mask=key_mask,
            attn_mask=None,
        )
        output = query_tokens + self.dropout(attended)
        if query_mask is not None:
            output = output * query_mask.unsqueeze(-1).float()
        return output


class TransformerEncoderBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float, ffn_multiplier: int) -> None:
        super().__init__()
        self.attention = SelfAttentionBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        self.feed_forward = FeedForwardBlock(hidden_dim=hidden_dim, ffn_multiplier=ffn_multiplier, dropout=dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens = self.attention(tokens, token_mask=token_mask, attn_mask=attn_mask)
        return self.feed_forward(tokens, token_mask=token_mask)


class DINActivationUnit(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.dnn = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, query: torch.Tensor, keys: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if keys.size(1) == 0:
            return torch.zeros_like(query)

        query_expanded = query.unsqueeze(1).expand(-1, keys.size(1), -1)
        attention_input = torch.cat(
            [query_expanded, keys, query_expanded - keys, query_expanded * keys],
            dim=-1,
        )
        attention_scores = self.dnn(attention_input).squeeze(-1)
        attention_scores = attention_scores.masked_fill(~mask, -1e4)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = attention_weights * mask.float()
        attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return torch.bmm(attention_weights.unsqueeze(1), keys).squeeze(1)


@dataclass(slots=True)
class FeatureViews:
    dense_summary: torch.Tensor
    dense_token: torch.Tensor
    context_tokens: torch.Tensor
    candidate_token: torch.Tensor
    flat_history_tokens: torch.Tensor
    sequence_tokens: torch.Tensor
    static_tokens: torch.Tensor
    static_mask: torch.Tensor


class UnifiedFeatureEncoder(nn.Module):
    def __init__(self, model_config: ModelConfig, data_config: DataConfig, dense_dim: int) -> None:
        super().__init__()
        self.embedding_dim = model_config.embedding_dim
        self.hidden_dim = model_config.hidden_dim
        self.sequence_count = len(data_config.sequence_names)

        self.token_embedding = nn.Embedding(model_config.vocab_size, model_config.embedding_dim, padding_idx=0)
        self.source_embedding = nn.Embedding(4, model_config.hidden_dim)
        self.sequence_group_embedding = nn.Embedding(self.sequence_count + 1, model_config.hidden_dim, padding_idx=0)
        self.context_projection = nn.Sequential(
            nn.Linear(model_config.embedding_dim, model_config.hidden_dim),
            nn.LayerNorm(model_config.hidden_dim),
            nn.SiLU(),
        )
        self.component_projection = nn.Sequential(
            nn.Linear(model_config.embedding_dim * 2, model_config.hidden_dim),
            nn.LayerNorm(model_config.hidden_dim),
            nn.SiLU(),
        )
        self.time_projection = nn.Sequential(
            nn.Linear(1, model_config.hidden_dim),
            nn.SiLU(),
            nn.Linear(model_config.hidden_dim, model_config.hidden_dim),
        )
        self.dense_projection = nn.Sequential(
            nn.Linear(dense_dim, model_config.hidden_dim),
            nn.LayerNorm(model_config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(model_config.dropout),
        )
        self.candidate_projection = nn.Sequential(
            nn.Linear(model_config.embedding_dim + model_config.hidden_dim, model_config.hidden_dim),
            nn.LayerNorm(model_config.hidden_dim),
            nn.SiLU(),
        )
        self.embedding_dropout = nn.Dropout(model_config.dropout)

    def _component_summary(self, component_tokens: torch.Tensor, component_mask: torch.Tensor) -> torch.Tensor:
        component_embeddings = self.token_embedding(component_tokens)
        component_weights = component_mask.unsqueeze(-1).float()
        summarized = (component_embeddings * component_weights).sum(dim=-2)
        return summarized / component_weights.sum(dim=-2).clamp_min(1.0)

    def encode_flat_history(self, batch: BatchTensors) -> torch.Tensor:
        history_embeddings = self.token_embedding(batch.history_tokens)
        component_summary = self._component_summary(batch.history_component_tokens, batch.history_component_mask)
        history_tokens = self.component_projection(torch.cat([history_embeddings, component_summary], dim=-1))
        history_tokens = history_tokens + self.sequence_group_embedding(batch.history_group_ids)
        history_tokens = history_tokens + self.time_projection(batch.history_time_gaps.unsqueeze(-1))
        return history_tokens + self.source_embedding.weight[2].view(1, 1, -1)

    def encode_grouped_sequences(self, batch: BatchTensors) -> torch.Tensor:
        batch_size, group_count, group_len = batch.sequence_tokens.shape
        sequence_embeddings = self.token_embedding(batch.sequence_tokens)
        component_summary = self._component_summary(batch.sequence_component_tokens, batch.sequence_component_mask)
        sequence_tokens = self.component_projection(torch.cat([sequence_embeddings, component_summary], dim=-1))

        group_ids = (
            torch.arange(1, group_count + 1, device=batch.sequence_tokens.device, dtype=torch.long)
            .view(1, group_count, 1)
            .expand(batch_size, group_count, group_len)
        )
        sequence_tokens = sequence_tokens + self.sequence_group_embedding(group_ids)
        sequence_tokens = sequence_tokens + self.time_projection(batch.sequence_time_gaps.unsqueeze(-1))
        return sequence_tokens + self.source_embedding.weight[2].view(1, 1, 1, -1)

    def build_views(self, batch: BatchTensors) -> FeatureViews:
        batch_size = batch.batch_size
        dense_summary = self.dense_projection(batch.dense_features)
        dense_token = dense_summary.unsqueeze(1) + self.source_embedding.weight[0].view(1, 1, -1)

        context_embeddings = self.token_embedding(batch.context_tokens)
        context_tokens = self.context_projection(context_embeddings)
        context_tokens = context_tokens + self.source_embedding.weight[1].view(1, 1, -1)

        candidate_embeddings = self.token_embedding(batch.candidate_tokens)
        candidate_summary = masked_mean(candidate_embeddings, batch.candidate_mask)
        candidate_token = self.candidate_projection(torch.cat([candidate_summary, dense_summary], dim=-1))
        candidate_token = candidate_token + self.source_embedding.weight[3].view(1, -1)

        flat_history_tokens = self.encode_flat_history(batch)
        sequence_tokens = self.encode_grouped_sequences(batch)

        static_tokens = torch.cat([dense_token, context_tokens, candidate_token.unsqueeze(1)], dim=1)
        static_mask = torch.cat(
            [
                torch.ones((batch_size, 1), dtype=torch.bool, device=batch.context_tokens.device),
                batch.context_mask,
                torch.ones((batch_size, 1), dtype=torch.bool, device=batch.context_tokens.device),
            ],
            dim=1,
        )

        return FeatureViews(
            dense_summary=dense_summary,
            dense_token=self.embedding_dropout(dense_token),
            context_tokens=self.embedding_dropout(context_tokens),
            candidate_token=self.embedding_dropout(candidate_token),
            flat_history_tokens=self.embedding_dropout(flat_history_tokens),
            sequence_tokens=self.embedding_dropout(sequence_tokens),
            static_tokens=self.embedding_dropout(static_tokens),
            static_mask=static_mask,
        )


def build_mlp_head(input_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    )


__all__ = [
    "CrossAttentionBlock",
    "DINActivationUnit",
    "FeedForwardBlock",
    "MultiHeadAttention",
    "FeatureViews",
    "SelfAttentionBlock",
    "TransformerEncoderBlock",
    "UnifiedFeatureEncoder",
    "build_causal_attention_mask",
    "build_mlp_head",
    "build_pooled_memory",
    "build_position_encoding",
    "make_recsys_attn_mask",
    "masked_attention_pool",
    "masked_mean",
]