"""データセットの事前テンソル化キャッシュに関する等価性・メモ化テスト.

キャッシュ有効時（cache_tensors=True）と無効時（False）で全サンプル・
全フィールドがビット単位で一致することを保証する。学習の意味論を
変えないことがこの変更の最重要要件のため、既存の学習ロジック
（cp_noise/normalize_turn/augment_flip/target_mode等）を網羅的に
比較する。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from models.dataset import (
    ShogiDeltaPairDataset,
    ShogiRankingPairDataset,
    ShogiValueDataset,
    parse_positions_incremental,
)
from models.sfen_parser import parse_sfen as real_parse_sfen

BOARD_SFEN = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL"


def _assert_items_equal(
    item_a: dict[str, torch.Tensor], item_b: dict[str, torch.Tensor]
) -> None:
    """2つのdataset[idx]の結果が全キーでビット単位一致することを確認."""
    assert set(item_a.keys()) == set(item_b.keys())
    for key in item_a:
        assert torch.equal(item_a[key], item_b[key]), f"key={key}"


def _assert_datasets_equal(ds_a, ds_b) -> None:  # noqa: ANN001
    """2つのdatasetの全アイテムが一致することを確認."""
    assert len(ds_a) == len(ds_b)
    for i in range(len(ds_a)):
        _assert_items_equal(ds_a[i], ds_b[i])


class TestParsePositionsIncremental:
    """parse_positions_incrementalの等価性テスト."""

    def test_matches_parse_sfen_individually(self) -> None:
        sfens = [
            # 孤児: リスト先頭で親（3手少ない局面）が未登場
            "startpos moves 7g7f 3c3d 8h2b+",
            "startpos",
            "startpos moves 7g7f",
            "startpos moves 7g7f 3c3d",
            # 上と同じキーの重複（既にcacheにある状態での再パース）
            "startpos moves 7g7f 3c3d 8h2b+",
            # 別ブランチ（startposはcache済みなので差分適用される）
            "startpos moves 2g2f",
            # "sfen <board> <turn> <hand> <count>" 形式（movesなし）
            f"sfen {BOARD_SFEN} b - 1",
            # 同形式 + moves 1手（差分適用の対象になるはず）
            f"sfen {BOARD_SFEN} b - 1 moves 7g7f",
        ]

        results = parse_positions_incremental(sfens)

        assert len(results) == len(sfens)
        for sfen, result in zip(sfens, results):
            expected = real_parse_sfen(sfen)
            assert torch.equal(result.board, expected.board), sfen
            assert torch.equal(result.hand, expected.hand), sfen
            assert torch.equal(result.turn, expected.turn), sfen


@pytest.fixture
def value_data(tmp_path: Path) -> Path:
    """2対局×各4局面 + source付き分岐レコード1件を持つJSONLを生成."""
    records = [
        # game 0: 7g7f, 3c3d, 8h2b+（角を取って成る捕獲付き手順）
        {
            "sfen": "startpos", "score_cp": 50, "ply": 0,
            "game_id": 0, "result": "black_win",
        },
        {
            "sfen": "startpos moves 7g7f", "score_cp": -30, "ply": 1,
            "game_id": 0, "result": "black_win",
        },
        {
            "sfen": "startpos moves 7g7f 3c3d", "score_cp": 60, "ply": 2,
            "game_id": 0, "result": "black_win",
        },
        {
            "sfen": "startpos moves 7g7f 3c3d 8h2b+", "score_cp": -20,
            "ply": 3, "game_id": 0, "result": "black_win",
        },
        # 分岐レコード: game0 ply2局面 + 本譜とは異なる手（親+1手）
        {
            "sfen": "startpos moves 7g7f 3c3d 2g2f", "score_cp": 15,
            "ply": 3, "game_id": 0, "result": "black_win",
            "source": "multipv",
            "candidates": [
                {"move": "8h2b+", "score_cp": -20, "rank": 1},
                {"move": "2g2f", "score_cp": 15, "rank": 2},
            ],
        },
        # game 1: 2g2f, 8c8d, 2f2e（捕獲なし）
        {
            "sfen": "startpos", "score_cp": 10, "ply": 0,
            "game_id": 1, "result": "white_win",
        },
        {
            "sfen": "startpos moves 2g2f", "score_cp": -15, "ply": 1,
            "game_id": 1, "result": "white_win",
        },
        {
            "sfen": "startpos moves 2g2f 8c8d", "score_cp": 25, "ply": 2,
            "game_id": 1, "result": "white_win",
        },
        {
            "sfen": "startpos moves 2g2f 8c8d 2f2e", "score_cp": -40,
            "ply": 3, "game_id": 1, "result": "white_win",
        },
    ]
    path = tmp_path / "value_data.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))
    return path


class TestShogiValueDatasetCache:
    """ShogiValueDatasetのキャッシュ等価性テスト."""

    @pytest.mark.parametrize(
        "config",
        [
            {},
            {"use_features": True},
            {"normalize_turn": True},
            {"augment_flip": True},
            {"target_mode": "wdl"},
            {
                "use_features": True,
                "normalize_turn": True,
                "augment_flip": True,
            },
        ],
        ids=[
            "default", "use_features", "normalize_turn",
            "augment_flip", "wdl", "combined",
        ],
    )
    def test_value_dataset_equivalence(
        self, value_data: Path, config: dict
    ) -> None:
        ds_cache = ShogiValueDataset(value_data, cache_tensors=True, **config)
        ds_nocache = ShogiValueDataset(
            value_data, cache_tensors=False, **config
        )
        _assert_datasets_equal(ds_cache, ds_nocache)

    def test_cp_noise_not_frozen(self, value_data: Path) -> None:
        dataset = ShogiValueDataset(
            value_data, cache_tensors=True, cp_noise=50.0
        )
        v1 = dataset[0]["value"].item()
        v2 = dataset[0]["value"].item()
        assert v1 != v2


@pytest.fixture
def ranking_data(tmp_path: Path) -> Path:
    """candidatesフィールド付きの小規模データを生成（test_ranking.py準拠）."""
    records = [
        {
            "sfen": "startpos", "score_cp": 50, "ply": 0,
            "game_id": 0, "result": "black_win",
            "candidates": [
                {"move": "2g2f", "score_cp": 50, "rank": 1},
                {"move": "7g7f", "score_cp": 30, "rank": 2},
                {"move": "9g9f", "score_cp": -80, "rank": 3},
            ],
        },
        {
            "sfen": "startpos moves 2g2f", "score_cp": -40, "ply": 1,
            "game_id": 0, "result": "black_win",
            "candidates": [
                {"move": "8c8d", "score_cp": -40, "rank": 1},
                {"move": "3c3d", "score_cp": -90, "rank": 2},
            ],
        },
        {
            "sfen": "startpos", "score_cp": 10, "ply": 0,
            "game_id": 1, "result": "white_win",
            "candidates": [
                {"move": "7g7f", "score_cp": 10, "rank": 1},
                {"move": "1g1f", "score_cp": -200, "rank": 2},
            ],
        },
    ]
    path = tmp_path / "ranking.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))
    return path


class TestShogiRankingPairDatasetCache:
    """ShogiRankingPairDatasetのキャッシュ等価性・メモ化テスト."""

    @pytest.mark.parametrize(
        "config",
        [
            {},
            {"use_features": True, "normalize_turn": True, "augment_flip": True},
        ],
        ids=["default", "combined"],
    )
    def test_ranking_pair_equivalence(
        self, ranking_data: Path, config: dict
    ) -> None:
        ds_cache = ShogiRankingPairDataset(
            ranking_data, min_gap_cp=30.0, cache_tensors=True, **config
        )
        ds_nocache = ShogiRankingPairDataset(
            ranking_data, min_gap_cp=30.0, cache_tensors=False, **config
        )
        _assert_datasets_equal(ds_cache, ds_nocache)

    def test_pair_memoization_hits(
        self, ranking_data: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dataset = ShogiRankingPairDataset(
            ranking_data, min_gap_cp=30.0, cache_tensors=True
        )
        calls = {"n": 0}

        def spy(sfen: str):  # noqa: ANN202
            calls["n"] += 1
            return real_parse_sfen(sfen)

        monkeypatch.setattr("models.dataset.parse_sfen", spy)

        _ = dataset[0]
        count_after_first = calls["n"]
        assert count_after_first > 0

        _ = dataset[0]
        assert calls["n"] == count_after_first


@pytest.fixture
def delta_pair_data(tmp_path: Path) -> Path:
    """gen_perturb_pairs.py出力形式の小規模ペアデータを生成（test_ranking.py準拠）."""
    records = [
        {
            "sfen_a": "startpos moves 2g2f", "sfen_b": "startpos moves 7g7f",
            "score_cp_a": -30, "score_cp_b": -60,
            "pair_type": "rewind_branch", "game_id": 0, "material_diff": 0,
        },
        {
            "sfen_a": "startpos moves 2g2f 8c8d",
            "sfen_b": "startpos moves 2g2f 3c3d",
            "score_cp_a": 40, "score_cp_b": 10,
            "pair_type": "move_dest", "game_id": 1, "material_diff": 90,
        },
    ]
    path = tmp_path / "delta_pairs.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))
    return path


class TestShogiDeltaPairDatasetCache:
    """ShogiDeltaPairDatasetのキャッシュ等価性テスト."""

    @pytest.mark.parametrize(
        "config",
        [
            {},
            {"use_features": True, "normalize_turn": True, "augment_flip": True},
        ],
        ids=["default", "combined"],
    )
    def test_delta_pair_equivalence(
        self, delta_pair_data: Path, config: dict
    ) -> None:
        ds_cache = ShogiDeltaPairDataset(
            delta_pair_data, cache_tensors=True, **config
        )
        ds_nocache = ShogiDeltaPairDataset(
            delta_pair_data, cache_tensors=False, **config
        )
        _assert_datasets_equal(ds_cache, ds_nocache)
