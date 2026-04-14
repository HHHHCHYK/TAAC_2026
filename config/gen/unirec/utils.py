from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer


def masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
	weights = mask.unsqueeze(-1).float()
	summed = (tokens * weights).sum(dim=1)
	counts = weights.sum(dim=1).clamp_min(1.0)
	return summed / counts


class DisabledAuxiliaryLoss:
	enabled = False
	requires_aux = False


class PairwiseAUCLoss(nn.Module):
	def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
		positive_mask = labels > 0.5
		negative_mask = ~positive_mask
		if positive_mask.sum() == 0 or negative_mask.sum() == 0:
			return logits.new_tensor(0.0)
		positive_scores = logits[positive_mask]
		negative_scores = logits[negative_mask]
		margins = positive_scores.unsqueeze(1) - negative_scores.unsqueeze(0)
		return F.softplus(-margins).mean()


class CombinedRankingLoss(nn.Module):
	def __init__(self, pos_weight: torch.Tensor, pairwise_weight: float) -> None:
		super().__init__()
		self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
		self.pairwise = PairwiseAUCLoss()
		self.pairwise_weight = min(max(pairwise_weight, 0.0), 1.0)

	def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
		bce_loss = self.bce(logits, labels)
		pairwise_loss = self.pairwise(logits, labels)
		return (1.0 - self.pairwise_weight) * bce_loss + self.pairwise_weight * pairwise_loss


class Muon(Optimizer):
	def __init__(
		self,
		params,
		lr: float = 1.0e-3,
		momentum: float = 0.95,
		ns_steps: int = 5,
		weight_decay: float = 0.0,
	) -> None:
		defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps, weight_decay=weight_decay)
		super().__init__(params, defaults)

	@staticmethod
	def _newton_schulz(gradient: torch.Tensor, steps: int = 5) -> torch.Tensor:
		a, b, c = (3.4445, -4.7750, 2.0315)
		matrix = gradient.float()
		transpose = False
		if matrix.shape[0] < matrix.shape[1]:
			matrix = matrix.transpose(0, 1)
			transpose = True
		matrix = matrix / (matrix.norm() + 1.0e-7)
		for _ in range(steps):
			gram = matrix @ matrix.transpose(0, 1)
			poly = b * gram + c * gram @ gram
			matrix = a * matrix + poly @ matrix
		if transpose:
			matrix = matrix.transpose(0, 1)
		return matrix.to(dtype=gradient.dtype)

	@torch.no_grad()
	def step(self, closure=None):
		loss = None
		if closure is not None:
			with torch.enable_grad():
				loss = closure()

		for group in self.param_groups:
			lr = group["lr"]
			momentum = group["momentum"]
			ns_steps = group["ns_steps"]
			weight_decay = group["weight_decay"]
			for parameter in group["params"]:
				if parameter.grad is None:
					continue
				gradient = parameter.grad
				state = self.state[parameter]
				if weight_decay > 0:
					parameter.mul_(1.0 - lr * weight_decay)
				if "momentum_buffer" not in state:
					state["momentum_buffer"] = torch.zeros_like(gradient)
				buffer = state["momentum_buffer"]
				buffer.mul_(momentum).add_(gradient)
				if parameter.ndim >= 2:
					update = self._newton_schulz(buffer.reshape(buffer.shape[0], -1), ns_steps).view_as(parameter)
				else:
					update = buffer
				parameter.add_(update, alpha=-lr)
		return loss


class CombinedOptimizer:
	def __init__(self, optimizers: list[Optimizer]) -> None:
		self.optimizers = optimizers
		self.param_groups = []
		for optimizer in optimizers:
			self.param_groups.extend(optimizer.param_groups)

	def zero_grad(self, set_to_none: bool = True) -> None:
		for optimizer in self.optimizers:
			optimizer.zero_grad(set_to_none=set_to_none)

	def step(self, closure=None) -> None:
		for optimizer in self.optimizers:
			optimizer.step(closure)

	def state_dict(self):
		return [optimizer.state_dict() for optimizer in self.optimizers]

	def load_state_dict(self, state_dicts) -> None:
		for optimizer, state_dict in zip(self.optimizers, state_dicts, strict=False):
			optimizer.load_state_dict(state_dict)


def build_loss_stack(data_config, model_config, train_config, data_stats, device):
	del data_config
	del model_config
	pos_weight = torch.tensor([data_stats.pos_weight], dtype=torch.float32, device=device)
	return CombinedRankingLoss(pos_weight=pos_weight, pairwise_weight=train_config.pairwise_weight), DisabledAuxiliaryLoss()


def build_optimizer_component(model, train_config):
	no_decay_terms = {"bias", "LayerNorm.weight", "LayerNorm.bias", "norm.weight", "norm.bias"}
	muon_params = []
	adamw_decay = []
	adamw_no_decay = []

	for name, parameter in model.named_parameters():
		if not parameter.requires_grad:
			continue
		is_embedding = "embed" in name.lower() or "embedding" in name.lower()
		uses_no_decay = any(term in name for term in no_decay_terms)
		if parameter.ndim == 2 and not is_embedding and not uses_no_decay:
			muon_params.append(parameter)
		elif uses_no_decay:
			adamw_no_decay.append(parameter)
		else:
			adamw_decay.append(parameter)

	optimizers: list[Optimizer] = []
	if muon_params:
		optimizers.append(
			Muon(
				muon_params,
				lr=0.02,
				momentum=0.95,
				ns_steps=5,
				weight_decay=train_config.weight_decay,
			)
		)
	adamw_groups = []
	if adamw_decay:
		adamw_groups.append({"params": adamw_decay, "weight_decay": train_config.weight_decay})
	if adamw_no_decay:
		adamw_groups.append({"params": adamw_no_decay, "weight_decay": 0.0})
	if adamw_groups:
		optimizers.append(torch.optim.AdamW(adamw_groups, lr=train_config.learning_rate))
	if len(optimizers) == 1:
		return optimizers[0]
	return CombinedOptimizer(optimizers)