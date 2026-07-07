"""将棋AI モデル定義."""

from __future__ import annotations

from models.dataset import (
    ShogiRankingPairDataset,
    ShogiValueDataset,
    collate_fn,
    ranking_collate_fn,
)
from models.features import compute_all_features, compute_attack_map, compute_king_safety
from models.sfen_parser import ParsedPosition, parse_sfen
from models.value_transformer import (
    ValueTransformer,
    cp_to_wdl,
    denormalize_cp,
    normalize_cp,
    wdl_to_cp,
)

__all__ = [
    "ValueTransformer",
    "ShogiValueDataset",
    "ShogiRankingPairDataset",
    "collate_fn",
    "ranking_collate_fn",
    "ParsedPosition",
    "parse_sfen",
    "normalize_cp",
    "denormalize_cp",
    "cp_to_wdl",
    "wdl_to_cp",
    "compute_all_features",
    "compute_attack_map",
    "compute_king_safety",
]
