from __future__ import annotations

from typing import Callable

from torch import nn

from ..config import DataConfig, ModelConfig
from .baseline import GrokDINReadoutBaseline, GrokUnifiedBaseline
from .din import CreatorwyxDINAdapter, CreatorwyxGroupedDINAdapter
from .novel import HyFormer, InterFormer, OneTrans
from .sequence import RetrievalStyleAdapter, TencentSASRecAdapter
from .unified import DeepContextNet, UniRecDINReadoutModel, UniRecModel, UniScaleFormer


ModelFactory = Callable[[ModelConfig, DataConfig, int], nn.Module]


def _build_retrieval_variant(variant: str) -> ModelFactory:
    def factory(config: ModelConfig, data_config: DataConfig, dense_dim: int) -> nn.Module:
        return RetrievalStyleAdapter(config=config, data_config=data_config, dense_dim=dense_dim, variant=variant)

    return factory


MODEL_REGISTRY: dict[str, ModelFactory] = {
    "baseline": GrokUnifiedBaseline,
    "grok_baseline": GrokUnifiedBaseline,
    "grok_din_readout": GrokDINReadoutBaseline,
    "creatorwyx_din_adapter": CreatorwyxDINAdapter,
    "creatorwyx_grouped_din_adapter": CreatorwyxGroupedDINAdapter,
    "tencent_sasrec_adapter": TencentSASRecAdapter,
    "zcyeee_retrieval_adapter": _build_retrieval_variant("zcyeee_retrieval_adapter"),
    "o_o_retrieval_adapter": _build_retrieval_variant("o_o_retrieval_adapter"),
    "oo_retrieval_adapter": _build_retrieval_variant("o_o_retrieval_adapter"),
    "omnigenrec_adapter": _build_retrieval_variant("omnigenrec_adapter"),
    "deep_context_net": DeepContextNet,
    "unirec": UniRecModel,
    "unirec_din_readout": UniRecDINReadoutModel,
    "uniscaleformer": UniScaleFormer,
    "interformer": InterFormer,
    "onetrans": OneTrans,
    "hyformer": HyFormer,
}


def build_model(config: ModelConfig, data_config: DataConfig, dense_dim: int) -> nn.Module:
    name = config.name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model name: {config.name}")
    return MODEL_REGISTRY[name](config, data_config, dense_dim)


__all__ = [
    "CreatorwyxDINAdapter",
    "CreatorwyxGroupedDINAdapter",
    "DeepContextNet",
    "GrokDINReadoutBaseline",
    "GrokUnifiedBaseline",
    "HyFormer",
    "InterFormer",
    "OneTrans",
    "TencentSASRecAdapter",
    "UniRecDINReadoutModel",
    "UniRecModel",
    "UniScaleFormer",
    "build_model",
]