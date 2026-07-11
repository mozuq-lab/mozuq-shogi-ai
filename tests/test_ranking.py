"""ranking学習・オフライン一致率関連のテスト."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from models import ShogiRankingPairDataset, ranking_collate_fn
from models.dataset import child_sfen
from scripts.move_agreement import load_candidate_samples, measure_agreement_offline
from train.train import compute_ranking_loss, select_val_games


@pytest.fixture
def ranking_data(tmp_path: Path) -> Path:
    """candidatesフィールド付きの小規模データを生成."""
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
        # candidatesなしのレコードはペア構築の対象外
        {
            "sfen": "startpos moves 7g7f", "score_cp": 0, "ply": 1,
            "game_id": 1, "result": "white_win",
        },
    ]
    path = tmp_path / "ranking.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))
    return path


class TestChildSfen:
    """child_sfenのテスト."""

    def test_startpos(self) -> None:
        assert child_sfen("startpos", "7g7f") == "startpos moves 7g7f"

    def test_startpos_with_moves(self) -> None:
        assert (
            child_sfen("startpos moves 7g7f", "3c3d")
            == "startpos moves 7g7f 3c3d"
        )

    def test_sfen_form(self) -> None:
        sfen = "sfen lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
        assert child_sfen(sfen, "7g7f") == sfen + " moves 7g7f"


class TestShogiRankingPairDataset:
    """ShogiRankingPairDatasetのテスト."""

    def test_pair_count_with_gap_filter(self, ranking_data: Path) -> None:
        # game0-rec1: (2g2f,9g9f), (7g7f,9g9f)（2g2f vs 7g7fはgap20で除外）
        # game0-rec2: (8c8d,3c3d)
        # game1-rec3: (7g7f,1g1f)
        dataset = ShogiRankingPairDataset(ranking_data, min_gap_cp=30.0)
        assert len(dataset) == 4

    def test_better_move_comes_first(self, ranking_data: Path) -> None:
        dataset = ShogiRankingPairDataset(ranking_data, min_gap_cp=30.0)
        triples = {p[:3] for p in dataset.pairs}
        assert ("startpos", "2g2f", "9g9f") in triples
        assert ("startpos moves 2g2f", "8c8d", "3c3d") in triples

    def test_min_gap_zero_includes_all_pairs(self, ranking_data: Path) -> None:
        dataset = ShogiRankingPairDataset(ranking_data, min_gap_cp=0.0)
        # rec1は3候補で3ペア、rec2/rec3は各1ペア
        assert len(dataset) == 5

    def test_game_id_filters(self, ranking_data: Path) -> None:
        train_ds = ShogiRankingPairDataset(
            ranking_data, min_gap_cp=30.0, exclude_game_ids={0}
        )
        val_ds = ShogiRankingPairDataset(
            ranking_data, min_gap_cp=30.0, include_game_ids={0}
        )
        assert len(train_ds) == 1
        assert len(val_ds) == 3

    def test_getitem_shapes_and_turn(self, ranking_data: Path) -> None:
        dataset = ShogiRankingPairDataset(ranking_data, min_gap_cp=30.0)
        item = dataset[0]
        assert item["board_a"].shape == (81,)
        assert item["board_b"].shape == (81,)
        assert item["hand_a"].shape == (14,)
        # 子局面同士は同じ手番（親の相手側）
        assert item["turn_a"].item() == item["turn_b"].item()
        # 親がstartpos（先手番）なので子局面は後手番
        assert item["turn_a"].item() == 1

    def test_normalize_turn_forces_black_view(self, ranking_data: Path) -> None:
        dataset = ShogiRankingPairDataset(
            ranking_data, min_gap_cp=30.0, normalize_turn=True
        )
        for i in range(len(dataset)):
            item = dataset[i]
            assert item["turn_a"].item() == 0
            assert item["turn_b"].item() == 0

    def test_augment_flip_doubles(self, ranking_data: Path) -> None:
        base = ShogiRankingPairDataset(ranking_data, min_gap_cp=30.0)
        flipped = ShogiRankingPairDataset(
            ranking_data, min_gap_cp=30.0, augment_flip=True
        )
        assert len(flipped) == len(base) * 2
        # 反転版は盤面が異なるがテンソル形状は同じ
        item = flipped[len(base)]
        assert item["board_a"].shape == (81,)

    def test_use_features(self, ranking_data: Path) -> None:
        dataset = ShogiRankingPairDataset(
            ranking_data, min_gap_cp=30.0, use_features=True
        )
        item = dataset[0]
        assert item["features_a"].shape == (81, 10)
        assert item["features_b"].shape == (81, 10)

    def test_collate(self, ranking_data: Path) -> None:
        dataset = ShogiRankingPairDataset(ranking_data, min_gap_cp=30.0)
        batch = ranking_collate_fn([dataset[0], dataset[1]])
        assert batch["board_a"].shape == (2, 81)
        assert batch["board_b"].shape == (2, 81)
        assert batch["turn_a"].shape == (2,)


class TestDeltaTarget:
    """delta_target（ΔV回帰ターゲット）のテスト."""

    def _pair_index(
        self, dataset: ShogiRankingPairDataset, triple: tuple[str, str, str]
    ) -> int:
        return next(
            i for i, p in enumerate(dataset.pairs) if p[:3] == triple
        )

    def test_cp_mode_value(self, ranking_data: Path) -> None:
        # (2g2f: +50, 9g9f: -80) → n(−50) − n(+80)、n=tanh(cp/1200)
        dataset = ShogiRankingPairDataset(
            ranking_data, min_gap_cp=30.0, cp_scale=1200.0
        )
        idx = self._pair_index(dataset, ("startpos", "2g2f", "9g9f"))
        expected = math.tanh(-50 / 1200.0) - math.tanh(80 / 1200.0)
        assert dataset[idx]["delta_target"].item() == pytest.approx(
            expected, abs=1e-6
        )

    def test_delta_negative_for_all_pairs(self, ranking_data: Path) -> None:
        # 良い手側の子局面ターゲットは常に小さい（相手番視点）
        dataset = ShogiRankingPairDataset(ranking_data, min_gap_cp=30.0)
        for i in range(len(dataset)):
            assert dataset[i]["delta_target"].item() < 0

    def test_wdl_mode_value(self, ranking_data: Path) -> None:
        dataset = ShogiRankingPairDataset(
            ranking_data, min_gap_cp=30.0, target_mode="wdl", wdl_scale=600.0
        )
        idx = self._pair_index(dataset, ("startpos", "2g2f", "9g9f"))

        def n(cp: float) -> float:
            return 2.0 / (1.0 + math.exp(-cp / 600.0)) - 1.0

        expected = n(-50.0) - n(80.0)
        assert dataset[idx]["delta_target"].item() == pytest.approx(
            expected, abs=1e-6
        )

    def test_collate_includes_delta_target(self, ranking_data: Path) -> None:
        dataset = ShogiRankingPairDataset(ranking_data, min_gap_cp=30.0)
        batch = ranking_collate_fn([dataset[0], dataset[1]])
        assert batch["delta_target"].shape == (2,)


class TestComputeRankingLoss:
    """compute_ranking_lossのテスト."""

    def test_correct_order_low_loss(self) -> None:
        # 良い手の子局面value（相手番視点）が低い = 正しい順序
        value_better = torch.tensor([-0.5])
        value_worse = torch.tensor([0.5])
        loss = compute_ranking_loss(value_better, value_worse)
        assert loss.item() == pytest.approx(math.log(1 + math.exp(-1.0)), abs=1e-5)

    def test_wrong_order_high_loss(self) -> None:
        value_better = torch.tensor([0.5])
        value_worse = torch.tensor([-0.5])
        loss = compute_ranking_loss(value_better, value_worse)
        assert loss.item() == pytest.approx(math.log(1 + math.exp(1.0)), abs=1e-5)

    def test_correct_less_than_wrong(self) -> None:
        correct = compute_ranking_loss(torch.tensor([-0.3]), torch.tensor([0.3]))
        wrong = compute_ranking_loss(torch.tensor([0.3]), torch.tensor([-0.3]))
        assert correct.item() < wrong.item()

    def test_tie_gives_log2(self) -> None:
        loss = compute_ranking_loss(torch.tensor([0.0]), torch.tensor([0.0]))
        assert loss.item() == pytest.approx(math.log(2.0), abs=1e-5)


class TestSelectValGames:
    """select_val_gamesのテスト."""

    def test_deterministic(self) -> None:
        game_ids = list(range(10)) * 5
        assert select_val_games(game_ids, 0.2) == select_val_games(game_ids, 0.2)

    def test_single_game_returns_none(self) -> None:
        assert select_val_games([0, 0, 0], 0.2) is None

    def test_at_least_one_val_game(self) -> None:
        val_games = select_val_games([0, 1], 0.1)
        assert val_games is not None
        assert len(val_games) == 1


class _FakeEvaluator:
    """find_best_moveが固定の手を返すテスト用評価器."""

    def __init__(self, move: str) -> None:
        self.move = move

    def find_best_move(self, board) -> tuple[str, int]:  # noqa: ANN001
        return self.move, 0


class TestOfflineAgreement:
    """オフライン指し手一致率のテスト."""

    def test_load_candidate_samples(self, ranking_data: Path) -> None:
        samples = load_candidate_samples(ranking_data)
        # candidates付きは3レコード（重複sfenの"startpos"は1つに統合）
        assert len(samples) == 2
        sfens = {s["sfen"] for s in samples}
        assert sfens == {"startpos", "startpos moves 2g2f"}

    def test_agreement_and_hit_rate(self, ranking_data: Path) -> None:
        # "2g2f"はstartposのrank1（一致）、"startpos moves 2g2f"では
        # candidatesに含まれない（不一致・hitなし）
        evaluator = _FakeEvaluator("2g2f")
        summary = measure_agreement_offline(evaluator, ranking_data)
        assert summary["total"] == 2
        assert summary["matched"] == 1
        assert summary["agreement"] == pytest.approx(0.5)
        assert summary["multipv_hit"] == 1
        assert summary["multipv_hit_rate"] == pytest.approx(0.5)

    def test_no_candidates_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "no_candidates.jsonl"
        path.write_text(
            '{"sfen": "startpos", "score_cp": 0, "ply": 0, "game_id": 0}'
        )
        with pytest.raises(ValueError):
            measure_agreement_offline(_FakeEvaluator("7g7f"), path)

    def test_regret_zero_for_best_move(self, ranking_data: Path) -> None:
        # "2g2f"はstartposのrank1（regret 0）。
        # "startpos moves 2g2f"では候補外なのでcensored
        evaluator = _FakeEvaluator("2g2f")
        summary = measure_agreement_offline(evaluator, ranking_data)
        assert summary["regret_mean_cp"] == pytest.approx(0.0)
        assert summary["regret_samples"] == 1
        assert summary["regret_censored"] == 1

    def test_regret_nonzero_for_worse_move(self, ranking_data: Path) -> None:
        # "9g9f"はstartposのrank3: regret = 50 − (−80) = 130cp
        evaluator = _FakeEvaluator("9g9f")
        summary = measure_agreement_offline(evaluator, ranking_data)
        assert summary["regret_mean_cp"] == pytest.approx(130.0)
        assert summary["regret_samples"] == 1
