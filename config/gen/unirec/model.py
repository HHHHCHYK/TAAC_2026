from __future__ import annotations

import math
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn

from taac2026.domain.config import DataConfig, ModelConfig
from taac2026.domain.types import BatchTensors

from .data import TIME_GAP_BUCKET_COUNT
from .utils import masked_mean


def masked_last(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
	positions = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0).expand_as(mask)
	last_indices = (positions * mask.long()).max(dim=1).values
	batch_indices = torch.arange(tokens.shape[0], device=tokens.device)
	return tokens[batch_indices, last_indices]


@lru_cache(maxsize=8)
def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
	row_idx = torch.arange(seq_len, device=device).unsqueeze(1)
	col_idx = torch.arange(seq_len, device=device).unsqueeze(0)
	return (col_idx <= row_idx).unsqueeze(0).unsqueeze(0)


@lru_cache(maxsize=8)
def build_unified_attention_mask(
	seq_len: int,
	n_feature_tokens: int,
	n_special_tokens: int,
	global_window: int,
	local_window: int,
	device: torch.device,
) -> torch.Tensor:
	row_idx = torch.arange(seq_len, device=device).unsqueeze(1)
	col_idx = torch.arange(seq_len, device=device).unsqueeze(0)
	causal = col_idx <= row_idx
	global_m = col_idx < global_window
	local_m = (row_idx - col_idx) < max(1, local_window)
	mask = causal & (global_m | local_m)

	if n_feature_tokens > 0:
		mask[:n_feature_tokens, :n_feature_tokens] = True
		mask[n_feature_tokens:, :n_feature_tokens] = True

	if n_special_tokens > 0:
		mask[-n_special_tokens:, :] = True

	return mask.unsqueeze(0).unsqueeze(0)


class SiLUAttention(nn.Module):
	def __init__(self, dim: int, num_heads: int, dropout: float = 0.1) -> None:
		super().__init__()
		if dim % num_heads != 0:
			raise ValueError("hidden_dim must be divisible by num_heads")
		self.dim = dim
		self.num_heads = num_heads
		self.head_dim = dim // num_heads
		self.qkuv_proj = nn.Linear(dim, 4 * dim, bias=False)
		self.out_proj = nn.Linear(dim, dim, bias=False)
		self.norm_attn = nn.LayerNorm(self.head_dim)
		self.dropout = nn.Dropout(dropout)

	def forward(self, hidden_states: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
		batch_size, token_count, hidden_dim = hidden_states.shape
		qkuv = F.silu(self.qkuv_proj(hidden_states))
		query, key, value, gate = qkuv.chunk(4, dim=-1)

		query = query.view(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)
		key = key.view(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)
		value = value.view(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)
		gate = gate.view(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)

		scores = torch.matmul(query, key.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
		weights = F.silu(scores)
		if attn_mask is not None:
			weights = weights * attn_mask.to(dtype=weights.dtype)
		weights = self.dropout(weights)

		attended = torch.matmul(weights, value)
		attended = self.norm_attn(attended)
		attended = attended * gate
		attended = attended.transpose(1, 2).reshape(batch_size, token_count, hidden_dim)
		return self.out_proj(attended)


class PointwiseFeedForward(nn.Module):
	def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
		super().__init__()
		self.w1 = nn.Linear(dim, hidden_dim, bias=False)
		self.w2 = nn.Linear(hidden_dim, dim, bias=False)
		self.w3 = nn.Linear(dim, hidden_dim, bias=False)
		self.dropout = nn.Dropout(dropout)

	def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
		return self.dropout(self.w2(F.silu(self.w1(hidden_states)) * self.w3(hidden_states)))


class UnifiedTransducerBlock(nn.Module):
	def __init__(self, dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
		super().__init__()
		self.norm1 = nn.LayerNorm(dim)
		self.norm2 = nn.LayerNorm(dim)
		self.attn = SiLUAttention(dim, num_heads, dropout)
		self.ffn = PointwiseFeedForward(dim, ffn_dim, dropout)
		self.dropout = nn.Dropout(dropout)

	def forward(self, hidden_states: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
		hidden_states = hidden_states + self.dropout(self.attn(self.norm1(hidden_states), attn_mask))
		hidden_states = hidden_states + self.dropout(self.ffn(self.norm2(hidden_states)))
		return hidden_states


class BlockAttnRes(nn.Module):
	"""Block Attention Residuals (Kimi, arXiv:2603.15031).

	Per-layer pseudo-queries (initialized to zero) attend over block
	summaries (last-token pooled).  Replaces fixed residual accumulation
	with learned softmax attention over depth.

	Benefits:
	  - Mitigates PreNorm dilution with bounded output magnitudes
	  - Equivalent to ~1.25x compute advantage in scaling law
	  - <2% inference overhead (last-token summary is O(N·D))
	"""

	def __init__(self, dim: int, total_layers: int) -> None:
		super().__init__()
		self.dim = dim
		self.pseudo_queries = nn.ParameterList(
			[nn.Parameter(torch.zeros(dim)) for _ in range(total_layers)]
		)
		NormClass = nn.RMSNorm if hasattr(nn, "RMSNorm") else nn.LayerNorm
		self.res_norm = NormClass(dim)

	def forward(
		self,
		layer_idx: int,
		layer_output: torch.Tensor,
		block_summaries: list[torch.Tensor],
		partial_sum: torch.Tensor,
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""Returns (h_l, updated_partial_sum, current_last_token)."""
		current_last = layer_output[:, -1, :]  # (B, D)
		partial_sum = partial_sum + current_last

		if len(block_summaries) == 0:
			return layer_output, partial_sum, current_last

		w_l = self.pseudo_queries[layer_idx]  # (D,)
		keys = torch.stack(block_summaries, dim=1)  # (B, N, D)
		keys = self.res_norm(keys)

		scores = torch.matmul(keys, w_l) * (1.0 / math.sqrt(self.dim))  # (B, N)
		weights = F.softmax(scores, dim=-1)  # (B, N)

		inter_block = torch.matmul(weights.unsqueeze(1), keys).squeeze(1)  # (B, D)
		combined = layer_output + inter_block.unsqueeze(1)  # (B, L, D)

		return combined, partial_sum, current_last


class BranchTransducer(nn.Module):
	def __init__(self, dim: int, num_heads: int, num_layers: int, dropout: float) -> None:
		super().__init__()
		self.layers = nn.ModuleList(
			[UnifiedTransducerBlock(dim=dim, num_heads=num_heads, ffn_dim=dim * 4, dropout=dropout) for _ in range(num_layers)]
		)
		self.norm = nn.LayerNorm(dim)

	def forward(self, hidden_states: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
		attn_mask = build_causal_mask(hidden_states.shape[1], hidden_states.device)
		attn_mask = attn_mask & token_mask.unsqueeze(1).unsqueeze(-1) & token_mask.unsqueeze(1).unsqueeze(2)
		for layer in self.layers:
			hidden_states = layer(hidden_states, attn_mask)
			hidden_states = hidden_states * token_mask.unsqueeze(-1).float()
		hidden_states = self.norm(hidden_states)
		return masked_last(hidden_states, token_mask)


class GatedFusion(nn.Module):
	def __init__(self, dim: int, num_branches: int) -> None:
		super().__init__()
		self.gate_proj = nn.Linear(dim, num_branches, bias=False)
		self.value_projs = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(num_branches)])

	def forward(self, branch_outputs: list[torch.Tensor]) -> torch.Tensor:
		stacked = torch.stack(branch_outputs, dim=1)
		gate_input = stacked.mean(dim=1)
		gates = torch.softmax(self.gate_proj(gate_input), dim=-1).unsqueeze(-1)
		values = torch.stack([proj(branch_output) for proj, branch_output in zip(self.value_projs, branch_outputs, strict=True)], dim=1)
		return (gates * values).sum(dim=1)


class MixtureOfTransducers(nn.Module):
	def __init__(self, dim: int, num_heads: int, num_layers: int, dropout: float, num_branches: int) -> None:
		super().__init__()
		self.branches = nn.ModuleList([BranchTransducer(dim, num_heads, num_layers, dropout) for _ in range(num_branches)])
		self.fusion = GatedFusion(dim, num_branches)
		self._cuda_streams: list[torch.cuda.Stream] | None = None
		self._streams_device: torch.device | None = None

	def _get_cuda_streams(self, device: torch.device) -> list[torch.cuda.Stream]:
		"""Lazily create and cache CUDA streams for branch parallelism."""
		if self._cuda_streams is None or self._streams_device != device:
			self._cuda_streams = [torch.cuda.Stream(device=device) for _ in range(len(self.branches) - 1)]
			self._streams_device = device
		return self._cuda_streams

	def forward(
		self,
		branch_tokens_list: list[torch.Tensor],
		branch_mask_list: list[torch.Tensor],
	) -> torch.Tensor:
		if branch_tokens_list[0].is_cuda and len(self.branches) >= 3:
			# CUDA stream parallelism for independent branches
			streams = self._get_cuda_streams(branch_tokens_list[0].device)
			outputs: list[torch.Tensor | None] = [None] * len(self.branches)
			for i, stream in enumerate(streams):
				with torch.cuda.stream(stream):
					outputs[i] = self.branches[i](branch_tokens_list[i], branch_mask_list[i])
			outputs[-1] = self.branches[-1](branch_tokens_list[-1], branch_mask_list[-1])
			for stream in streams:
				torch.cuda.current_stream().wait_stream(stream)
			return self.fusion(outputs)  # type: ignore[arg-type]
		outputs = [branch(tokens, mask) for branch, tokens, mask in zip(self.branches, branch_tokens_list, branch_mask_list, strict=True)]
		return self.fusion(outputs)


class UniRecModel(nn.Module):
	def __init__(self, data_config: DataConfig, model_config: ModelConfig, dense_dim: int) -> None:
		super().__init__()
		self.data_config = data_config
		self.model_config = model_config
		self.hidden_dim = model_config.hidden_dim
		self.sequence_count = len(data_config.sequence_names)
		self.feature_token_names = (
			"user",
			"context",
			"dense",
			"candidate",
			"candidate_post",
			"candidate_author",
		)
		self.n_feature_tokens = len(self.feature_token_names)
		self.global_window = max(128, self.n_feature_tokens + 8)
		self.local_window = 128
		self.truncated_seq_len = max(0, model_config.recent_seq_len)
		self.attn_res_block_size = max(1, model_config.memory_slots)

		self.token_embedding = nn.Embedding(
			num_embeddings=model_config.vocab_size,
			embedding_dim=model_config.embedding_dim,
			padding_idx=0,
		)
		self.token_projection = (
			nn.Identity()
			if model_config.embedding_dim == model_config.hidden_dim
			else nn.Linear(model_config.embedding_dim, model_config.hidden_dim)
		)
		self.dense_projection = nn.Sequential(
			nn.Linear(dense_dim, model_config.hidden_dim),
			nn.LayerNorm(model_config.hidden_dim),
			nn.SiLU(),
		)
		self.feature_projections = nn.ModuleDict(
			{
				name: nn.Sequential(
					nn.Linear(model_config.hidden_dim, model_config.hidden_dim),
					nn.LayerNorm(model_config.hidden_dim),
					nn.SiLU(),
				)
				for name in self.feature_token_names
			}
		)
		self.field_embedding = nn.Embedding(self.n_feature_tokens + 8, model_config.hidden_dim)
		self.segment_embedding = nn.Embedding(self.sequence_count + 2, model_config.hidden_dim)
		self.history_group_embedding = nn.Embedding(self.sequence_count + 1, model_config.hidden_dim, padding_idx=0)
		self.sequence_id_embedding = nn.Embedding(self.sequence_count + 1, model_config.hidden_dim, padding_idx=0)
		self.sequence_position_embedding = nn.Embedding(data_config.max_seq_len, model_config.hidden_dim)
		self.time_gap_embedding = nn.Embedding(TIME_GAP_BUCKET_COUNT + 1, model_config.hidden_dim, padding_idx=0)
		self.event_projection = nn.Sequential(
			nn.LayerNorm(model_config.hidden_dim * 6),
			nn.Linear(model_config.hidden_dim * 6, model_config.hidden_dim * 4),
			nn.SiLU(),
			nn.Dropout(model_config.dropout),
			nn.Linear(model_config.hidden_dim * 4, model_config.hidden_dim),
		)
		self.empty_sequence_tokens = nn.Parameter(torch.randn(self.sequence_count, model_config.hidden_dim) * 0.02)
		feature_cross_depth = max(1, model_config.feature_cross_layers)
		self.feature_cross_layers = nn.ModuleList(
			[
				nn.TransformerEncoderLayer(
					d_model=model_config.hidden_dim,
					nhead=model_config.num_heads,
					dim_feedforward=int(model_config.hidden_dim * model_config.ffn_multiplier),
					dropout=model_config.dropout,
					activation="gelu",
					batch_first=True,
				)
				for _ in range(feature_cross_depth)
			]
		)
		self.interest_query = nn.Linear(model_config.hidden_dim, model_config.hidden_dim, bias=False)
		self.interest_key = nn.Linear(model_config.hidden_dim, model_config.hidden_dim, bias=False)
		self.interest_value = nn.Linear(model_config.hidden_dim, model_config.hidden_dim, bias=False)
		self.interest_out = nn.Sequential(
			nn.Linear(model_config.hidden_dim, model_config.hidden_dim),
			nn.SiLU(),
			nn.Linear(model_config.hidden_dim, model_config.hidden_dim),
		)
		self.target_fusion = nn.Sequential(
			nn.Linear(model_config.hidden_dim * 3, model_config.hidden_dim),
			nn.SiLU(),
			nn.Linear(model_config.hidden_dim, model_config.hidden_dim),
		)

		mot_layers = max(0, model_config.sequence_layers)
		self.mot = None
		self.mot_projection = None
		if mot_layers > 0:
			self.mot = MixtureOfTransducers(model_config.hidden_dim, model_config.num_heads, mot_layers, model_config.dropout, self.sequence_count)
			self.mot_projection = nn.Linear(model_config.hidden_dim, model_config.hidden_dim)

		full_layers = model_config.static_layers if model_config.static_layers > 0 else max(1, model_config.num_layers)
		truncated_layers = model_config.fusion_layers if model_config.fusion_layers > 0 else max(0, model_config.num_layers - full_layers)
		self.full_blocks = nn.ModuleList(
			[
				UnifiedTransducerBlock(
					dim=model_config.hidden_dim,
					num_heads=model_config.num_heads,
					ffn_dim=int(model_config.hidden_dim * model_config.ffn_multiplier),
					dropout=model_config.dropout,
				)
				for _ in range(full_layers)
			]
		)
		self.truncated_blocks = nn.ModuleList(
			[
				UnifiedTransducerBlock(
					dim=model_config.hidden_dim,
					num_heads=model_config.num_heads,
					ffn_dim=int(model_config.hidden_dim * model_config.ffn_multiplier),
					dropout=model_config.dropout,
				)
				for _ in range(truncated_layers)
			]
		)
		self.block_attn_res = BlockAttnRes(model_config.hidden_dim, len(self.full_blocks) + len(self.truncated_blocks))
		self.final_norm = nn.LayerNorm(model_config.hidden_dim)
		head_hidden_dim = model_config.head_hidden_dim or model_config.hidden_dim * 2
		self.head = nn.Sequential(
			nn.Linear(model_config.hidden_dim, head_hidden_dim),
			nn.SiLU(),
			nn.Dropout(model_config.dropout),
			nn.Linear(head_hidden_dim, head_hidden_dim // 2),
			nn.SiLU(),
			nn.Dropout(model_config.dropout),
			nn.Linear(head_hidden_dim // 2, 1),
		)
		self._init_weights()

	def _init_weights(self) -> None:
		for module in self.modules():
			if isinstance(module, nn.Linear):
				nn.init.xavier_uniform_(module.weight)
				if module.bias is not None:
					nn.init.zeros_(module.bias)
			elif isinstance(module, nn.Embedding):
				nn.init.normal_(module.weight, std=0.02)
				if module.padding_idx is not None:
					nn.init.zeros_(module.weight[module.padding_idx])

	def _require(self, tensor: torch.Tensor | None, name: str) -> torch.Tensor:
		if tensor is None:
			raise RuntimeError(f"Batch is missing required tensor: {name}")
		return tensor

	def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
		return self.token_projection(self.token_embedding(tokens))

	def _build_feature_tokens(self, batch: BatchTensors) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		user_tokens = self._require(batch.user_tokens, "user_tokens")
		user_mask = self._require(batch.user_mask, "user_mask")
		candidate_post_tokens = self._require(batch.candidate_post_tokens, "candidate_post_tokens")
		candidate_post_mask = self._require(batch.candidate_post_mask, "candidate_post_mask")
		candidate_author_tokens = self._require(batch.candidate_author_tokens, "candidate_author_tokens")
		candidate_author_mask = self._require(batch.candidate_author_mask, "candidate_author_mask")

		summaries = {
			"user": masked_mean(self._embed_tokens(user_tokens), user_mask),
			"context": masked_mean(self._embed_tokens(batch.context_tokens), batch.context_mask),
			"dense": self.dense_projection(batch.dense_features),
			"candidate": masked_mean(self._embed_tokens(batch.candidate_tokens), batch.candidate_mask),
			"candidate_post": masked_mean(self._embed_tokens(candidate_post_tokens), candidate_post_mask),
			"candidate_author": masked_mean(self._embed_tokens(candidate_author_tokens), candidate_author_mask),
		}
		feature_tokens = torch.stack(
			[self.feature_projections[name](summaries[name]) for name in self.feature_token_names],
			dim=1,
		)
		field_ids = torch.arange(self.n_feature_tokens, device=feature_tokens.device).unsqueeze(0).expand(batch.batch_size, -1)
		feature_tokens = feature_tokens + self.field_embedding(field_ids)
		for layer in self.feature_cross_layers:
			feature_tokens = layer(feature_tokens)

		user_pool = feature_tokens[:, :3, :].mean(dim=1)
		item_pool = feature_tokens[:, 3:, :].mean(dim=1)
		return feature_tokens, user_pool, item_pool

	def _build_event_tokens(self, batch: BatchTensors) -> torch.Tensor:
		history_post_tokens = self._require(batch.history_post_tokens, "history_post_tokens")
		history_author_tokens = self._require(batch.history_author_tokens, "history_author_tokens")
		history_action_tokens = self._require(batch.history_action_tokens, "history_action_tokens")
		history_time_gap = self._require(batch.history_time_gap, "history_time_gap")
		history_group_ids = self._require(batch.history_group_ids, "history_group_ids")

		history_hidden = self._embed_tokens(batch.history_tokens)
		post_hidden = self._embed_tokens(history_post_tokens)
		author_hidden = self._embed_tokens(history_author_tokens)
		action_hidden = self._embed_tokens(history_action_tokens)
		gap_hidden = self.time_gap_embedding(history_time_gap.clamp(min=0, max=TIME_GAP_BUCKET_COUNT))
		group_hidden = self.history_group_embedding(history_group_ids)
		event_inputs = torch.cat(
			[
				history_hidden,
				post_hidden,
				author_hidden,
				action_hidden,
				gap_hidden,
				group_hidden,
			],
			dim=-1,
		)
		return self.event_projection(event_inputs) * batch.history_mask.unsqueeze(-1).float()

	def _split_history_by_group(self, batch: BatchTensors, event_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		history_time_gap = self._require(batch.history_time_gap, "history_time_gap")
		history_group_ids = self._require(batch.history_group_ids, "history_group_ids")
		batch_size, _, hidden_dim = event_tokens.shape
		max_seq_len = self.data_config.max_seq_len
		device = event_tokens.device
		branch_tokens = event_tokens.new_zeros(batch_size, self.sequence_count, max_seq_len, hidden_dim)
		branch_mask = torch.zeros(batch_size, self.sequence_count, max_seq_len, dtype=torch.bool, device=device)
		branch_times = event_tokens.new_zeros(batch_size, self.sequence_count, max_seq_len)

		for batch_index in range(batch_size):
			valid_positions = torch.nonzero(batch.history_mask[batch_index], as_tuple=False).squeeze(-1)
			for sequence_index in range(self.sequence_count):
				if valid_positions.numel() > 0:
					selected_positions = valid_positions[
						history_group_ids[batch_index, valid_positions] == (sequence_index + 1)
					]
				else:
					selected_positions = valid_positions
				selected_positions = selected_positions[-max_seq_len:]
				selected_count = int(selected_positions.numel())
				if selected_count == 0:
					branch_tokens[batch_index, sequence_index, -1] = self.empty_sequence_tokens[sequence_index]
					branch_mask[batch_index, sequence_index, -1] = True
					continue

				start = max_seq_len - selected_count
				branch_tokens[batch_index, sequence_index, start:] = event_tokens[batch_index, selected_positions]
				branch_mask[batch_index, sequence_index, start:] = True
				gaps = history_time_gap[batch_index, selected_positions].float()
				recency = float(TIME_GAP_BUCKET_COUNT + 1) - gaps.clamp(min=0.0, max=float(TIME_GAP_BUCKET_COUNT + 1))
				base = torch.arange(selected_count, device=device, dtype=torch.float32) * float(TIME_GAP_BUCKET_COUNT + 1)
				branch_times[batch_index, sequence_index, start:] = base + recency

		positions = torch.arange(max_seq_len, device=device)
		branch_tokens = branch_tokens + self.sequence_position_embedding(positions).view(1, 1, max_seq_len, hidden_dim)
		sequence_ids = torch.arange(1, self.sequence_count + 1, device=device)
		branch_tokens = branch_tokens + self.sequence_id_embedding(sequence_ids).view(1, self.sequence_count, 1, hidden_dim)
		branch_tokens = branch_tokens * branch_mask.unsqueeze(-1).float()
		return branch_tokens, branch_mask, branch_times

	def _build_unified_sequence(self, batch: BatchTensors) -> tuple[torch.Tensor, torch.Tensor, int, int]:
		feature_tokens, user_pool, item_pool = self._build_feature_tokens(batch)
		event_tokens = self._build_event_tokens(batch)
		branch_tokens, branch_mask, _ = self._split_history_by_group(batch, event_tokens)

		per_branch_tokens = [branch_tokens[:, i] for i in range(self.sequence_count)]
		per_branch_mask = [branch_mask[:, i] for i in range(self.sequence_count)]

		all_sequence_tokens = torch.cat(per_branch_tokens, dim=1)
		all_sequence_mask = torch.cat(per_branch_mask, dim=1)
		query = self.interest_query(item_pool).unsqueeze(1)
		key = self.interest_key(all_sequence_tokens)
		value = self.interest_value(all_sequence_tokens)
		attention_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(query.shape[-1])
		attention_scores = attention_scores.masked_fill(~all_sequence_mask.unsqueeze(1), -1.0e4)
		attention_weights = torch.softmax(attention_scores, dim=-1)
		attention_weights = attention_weights * all_sequence_mask.unsqueeze(1).float()
		attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
		interest = torch.matmul(attention_weights, value).squeeze(1)
		interest_token = self.interest_out(interest).unsqueeze(1)

		target_token = self.target_fusion(torch.cat([user_pool, item_pool, user_pool * item_pool], dim=-1)).unsqueeze(1)

		sequence_parts = [feature_tokens] + per_branch_tokens
		special_segment_id = self.sequence_count + 1
		segment_ids = [
			torch.zeros(batch.batch_size, self.n_feature_tokens, dtype=torch.long, device=feature_tokens.device),
		] + [
			torch.full((batch.batch_size, per_branch_tokens[i].shape[1]), i + 1, dtype=torch.long, device=feature_tokens.device)
			for i in range(self.sequence_count)
		]
		padding_parts = [
			torch.ones(batch.batch_size, self.n_feature_tokens, dtype=torch.bool, device=feature_tokens.device),
		] + list(per_branch_mask)

		if self.mot is not None and self.mot_projection is not None:
			mot_token = self.mot_projection(
				self.mot(
					per_branch_tokens,
					per_branch_mask,
				)
			).unsqueeze(1)
			sequence_parts.append(mot_token)
			segment_ids.append(torch.full((batch.batch_size, 1), special_segment_id, dtype=torch.long, device=feature_tokens.device))
			padding_parts.append(torch.ones(batch.batch_size, 1, dtype=torch.bool, device=feature_tokens.device))

		sequence_parts.append(interest_token)
		segment_ids.append(torch.full((batch.batch_size, 1), special_segment_id, dtype=torch.long, device=feature_tokens.device))
		padding_parts.append(torch.ones(batch.batch_size, 1, dtype=torch.bool, device=feature_tokens.device))

		sequence_parts.append(target_token)
		segment_ids.append(torch.full((batch.batch_size, 1), special_segment_id, dtype=torch.long, device=feature_tokens.device))
		padding_parts.append(torch.ones(batch.batch_size, 1, dtype=torch.bool, device=feature_tokens.device))

		unified_tokens = torch.cat(sequence_parts, dim=1)
		segments = torch.cat(segment_ids, dim=1)
		padding_mask = torch.cat(padding_parts, dim=1)
		unified_tokens = unified_tokens + self.segment_embedding(segments)
		n_special_tokens = 2 + (1 if self.mot is not None and self.mot_projection is not None else 0)
		return unified_tokens, padding_mask, self.n_feature_tokens, n_special_tokens

	def _compose_attention_mask(
		self,
		padding_mask: torch.Tensor,
		n_feature_tokens: int,
		n_special_tokens: int,
	) -> torch.Tensor:
		attn_mask = build_unified_attention_mask(
			seq_len=padding_mask.shape[1],
			n_feature_tokens=n_feature_tokens,
			n_special_tokens=n_special_tokens,
			global_window=self.global_window,
			local_window=self.local_window,
			device=padding_mask.device,
		)
		return attn_mask & padding_mask.unsqueeze(1).unsqueeze(-1) & padding_mask.unsqueeze(1).unsqueeze(2)

	def forward(self, batch: BatchTensors) -> torch.Tensor:
		hidden_states, padding_mask, n_feature_tokens, n_special_tokens = self._build_unified_sequence(batch)
		attention_mask = self._compose_attention_mask(padding_mask, n_feature_tokens, n_special_tokens)

		batch_size = hidden_states.shape[0]
		dim = hidden_states.shape[2]
		block_summaries: list[torch.Tensor] = []
		partial_sum = hidden_states.new_zeros((batch_size, dim))
		layer_idx = 0

		for block in self.full_blocks:
			hidden_states = block(hidden_states, attention_mask)
			hidden_states = hidden_states * padding_mask.unsqueeze(-1).float()
			hidden_states, partial_sum, _ = self.block_attn_res(
				layer_idx, hidden_states, block_summaries, partial_sum,
			)
			hidden_states = hidden_states * padding_mask.unsqueeze(-1).float()
			layer_idx += 1
			if layer_idx % self.attn_res_block_size == 0:
				block_summaries.append(partial_sum)
				partial_sum = hidden_states.new_zeros((batch_size, dim))

		# Save remaining partial_sum if there is an incomplete residual-attention block.
		if layer_idx % self.attn_res_block_size != 0:
			block_summaries.append(partial_sum)
			partial_sum = hidden_states.new_zeros((batch_size, dim))

		if self.truncated_blocks and self.truncated_seq_len > 0:
			special_start = hidden_states.shape[1] - n_special_tokens
			sequence_region = hidden_states[:, n_feature_tokens:special_start]
			sequence_mask = padding_mask[:, n_feature_tokens:special_start]
			recent_len = min(self.truncated_seq_len, sequence_region.shape[1])
			if recent_len > 0:
				recent_tokens = sequence_region[:, -recent_len:]
				recent_mask = sequence_mask[:, -recent_len:]
				truncated_tokens = torch.cat(
					[
						hidden_states[:, :n_feature_tokens],
						recent_tokens,
						hidden_states[:, special_start:],
					],
					dim=1,
				)
				truncated_padding = torch.cat(
					[
						padding_mask[:, :n_feature_tokens],
						recent_mask,
						padding_mask[:, special_start:],
					],
					dim=1,
				)
				truncated_mask = self._compose_attention_mask(truncated_padding, n_feature_tokens, n_special_tokens)
				for block in self.truncated_blocks:
					truncated_tokens = block(truncated_tokens, truncated_mask)
					truncated_tokens = truncated_tokens * truncated_padding.unsqueeze(-1).float()
					truncated_tokens, partial_sum, _ = self.block_attn_res(
						layer_idx, truncated_tokens, block_summaries, partial_sum,
					)
					layer_idx += 1
					if layer_idx % self.attn_res_block_size == 0:
						block_summaries.append(partial_sum)
						partial_sum = hidden_states.new_zeros((batch_size, dim))

				updated_recent = truncated_tokens[:, n_feature_tokens:n_feature_tokens + recent_len]
				updated_special = truncated_tokens[:, -n_special_tokens:]
				frozen_prefix = sequence_region[:, :-recent_len]
				frozen_prefix_mask = sequence_mask[:, :-recent_len]
				hidden_states = torch.cat(
					[
						hidden_states[:, :n_feature_tokens],
						frozen_prefix,
						updated_recent,
						updated_special,
					],
					dim=1,
				)
				padding_mask = torch.cat(
					[
						padding_mask[:, :n_feature_tokens],
						frozen_prefix_mask,
						recent_mask,
						padding_mask[:, special_start:],
					],
					dim=1,
				)

		hidden_states = self.final_norm(hidden_states)
		return self.head(hidden_states[:, -1, :]).squeeze(-1)


def build_model_component(data_config: DataConfig, model_config: ModelConfig, dense_dim: int) -> UniRecModel:
	return UniRecModel(data_config=data_config, model_config=model_config, dense_dim=dense_dim)