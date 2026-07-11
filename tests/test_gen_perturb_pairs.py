"""摂動ペア生成（局面感度蒸留）のテスト."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import shogi

from tools.gen_perturb_pairs import (
    board_perturb_pairs,
    build_pairs,
    is_stable,
    label_pairs,
    load_mainline_records,
    material_balance,
    move_destination_pairs,
    promotion_pairs,
    rewind_branch_pairs,
)


class TestMaterialBalance:
    """material_balanceのテスト."""

    def test_startpos_is_zero(self) -> None:
        assert material_balance(shogi.Board()) == 0

    def test_hand_piece_counted(self) -> None:
        # 先手が歩を1枚持っている（盤上は玉のみ対称）
        board = shogi.Board("4k4/9/9/9/9/9/9/9/4K4 b P 1")
        assert material_balance(board) == 90

    def test_side_to_move_view(self) -> None:
        # 同一局面でも手番側視点なので符号が反転する
        board_black = shogi.Board("4k4/9/9/9/9/9/9/9/4K4 b P 1")
        board_white = shogi.Board("4k4/9/9/9/9/9/9/9/4K4 w P 1")
        assert material_balance(board_black) == -material_balance(board_white)


class TestMoveDestinationPairs:
    """move_destination_pairsのテスト."""

    def test_same_from_square(self) -> None:
        pairs = move_destination_pairs(shogi.Board(), max_pairs_per_group=2)
        assert len(pairs) > 0
        for usi_a, usi_b in pairs:
            # 同一グループ（同じ駒）のペアなので移動元表記が一致する
            assert usi_a[:2] == usi_b[:2]
            assert usi_a != usi_b

    def test_max_pairs_per_group(self) -> None:
        few = move_destination_pairs(shogi.Board(), max_pairs_per_group=1)
        many = move_destination_pairs(shogi.Board(), max_pairs_per_group=10)
        assert len(few) <= len(many)


class TestPromotionPairs:
    """promotion_pairsのテスト."""

    def test_startpos_has_none(self) -> None:
        assert promotion_pairs(shogi.Board()) == []

    def test_pawn_promotion_pair(self) -> None:
        # 5d歩が5cへ進む手は成/不成の両方が合法
        board = shogi.Board("4k4/9/9/4P4/9/9/9/9/4K4 b - 1")
        pairs = promotion_pairs(board)
        assert ("5d5c+", "5d5c") in pairs


class TestRewindBranchPairs:
    """rewind_branch_pairsのテスト."""

    def test_pairs_from_candidates(self) -> None:
        records = [
            {
                "sfen": "startpos", "ply": 0, "game_id": 3,
                "candidates": [
                    {"move": "2g2f", "score_cp": 50, "rank": 1},
                    {"move": "7g7f", "score_cp": 30, "rank": 2},
                ],
            },
            {"sfen": "startpos moves 2g2f", "ply": 1, "game_id": 3},
        ]
        pairs = rewind_branch_pairs(records)
        # 本譜(2g2f)と異なる候補7g7fの分岐のみペアになる
        assert len(pairs) == 1
        assert pairs[0]["sfen_a"] == "startpos moves 2g2f"
        assert pairs[0]["sfen_b"] == "startpos moves 7g7f"
        assert pairs[0]["pair_type"] == "rewind_branch"
        assert pairs[0]["game_id"] == 3

    def test_no_next_record_no_pair(self) -> None:
        records = [
            {
                "sfen": "startpos", "ply": 0, "game_id": 0,
                "candidates": [{"move": "2g2f", "score_cp": 50, "rank": 1}],
            },
        ]
        assert rewind_branch_pairs(records) == []


class TestBoardPerturbPairs:
    """board_perturb_pairsのテスト."""

    def test_generates_move_dest_pairs(self) -> None:
        record = {"sfen": "startpos", "game_id": 7}
        pairs = board_perturb_pairs(record, max_pairs_per_group=1)
        assert len(pairs) > 0
        for pair in pairs:
            assert pair["game_id"] == 7
            assert pair["pair_type"] in ("move_dest", "promotion")
            assert pair["sfen_a"].startswith("startpos moves ")


class TestIsStable:
    """is_stable（安定性フィルタ）のテスト."""

    def test_same_sign_stable(self) -> None:
        assert is_stable(100, 50)
        assert is_stable(-100, -20)

    def test_sign_flip_unstable(self) -> None:
        assert not is_stable(100, -50)

    def test_zero_is_stable(self) -> None:
        assert is_stable(0, 100)
        assert is_stable(100, 0)


class TestLabelPairs:
    """label_pairs（エンジンラベル付け+安定性フィルタ）のテスト."""

    @staticmethod
    def _make_evaluate(scores: dict[str, dict[int, int]]):
        def evaluate(sfen: str, nodes: int) -> int | None:
            return scores.get(sfen, {}).get(nodes)
        return evaluate

    def test_labels_and_material_diff(self) -> None:
        pairs = [{
            "sfen_a": "startpos moves 2g2f",
            "sfen_b": "startpos moves 7g7f",
            "pair_type": "rewind_branch",
            "game_id": 0,
        }]
        evaluate = self._make_evaluate({
            "startpos moves 2g2f": {200: -30, 50: -20},
            "startpos moves 7g7f": {200: -60, 50: -40},
        })
        labeled, dropped = label_pairs(
            pairs, evaluate, label_nodes=200, stability_nodes=50
        )
        assert dropped == 0
        assert len(labeled) == 1
        assert labeled[0]["score_cp_a"] == -30
        assert labeled[0]["score_cp_b"] == -60
        # 序盤の歩の差し替えなので素材は一致
        assert labeled[0]["material_diff"] == 0

    def test_unstable_pair_dropped(self) -> None:
        pairs = [{
            "sfen_a": "startpos moves 2g2f",
            "sfen_b": "startpos moves 7g7f",
            "pair_type": "rewind_branch",
            "game_id": 0,
        }]
        # ラベル探索と安定性探索で差分の符号が反転 → 破棄
        evaluate = self._make_evaluate({
            "startpos moves 2g2f": {200: -30, 50: -80},
            "startpos moves 7g7f": {200: -60, 50: -40},
        })
        labeled, dropped = label_pairs(
            pairs, evaluate, label_nodes=200, stability_nodes=50
        )
        assert labeled == []
        assert dropped == 1

    def test_turn_mismatch_raises(self) -> None:
        pairs = [{
            "sfen_a": "startpos",
            "sfen_b": "startpos moves 7g7f",
            "pair_type": "move_dest",
            "game_id": 0,
        }]
        with pytest.raises(ValueError, match="手番"):
            label_pairs(pairs, lambda s, n: 0, 200, 50)

    def test_none_score_dropped(self) -> None:
        pairs = [{
            "sfen_a": "startpos moves 2g2f",
            "sfen_b": "startpos moves 7g7f",
            "pair_type": "rewind_branch",
            "game_id": 0,
        }]
        labeled, dropped = label_pairs(pairs, lambda s, n: None, 200, 50)
        assert labeled == []
        assert dropped == 1

    def test_shared_sfen_a_is_memoized(self) -> None:
        """同一(sfen, nodes)は複数ペアにまたがってもevaluateが1回だけ呼ばれる.

        rewind_branchペアでは本譜の次局面（sfen_a）が候補手の数だけ
        繰り返し登場する。最もコストの高いlabel_nodesの探索が
        重複実行されないことを、呼び出し回数を数えるfake evaluateで検証する。
        """
        calls: list[tuple[str, int]] = []
        scores = {
            "startpos moves 2g2f": {200: -30, 50: -20},
            "startpos moves 7g7f": {200: -60, 50: -40},
            "startpos moves 6g6f": {200: -55, 50: -35},
        }

        def evaluate(sfen: str, nodes: int) -> int | None:
            calls.append((sfen, nodes))
            return scores.get(sfen, {}).get(nodes)

        # 2ペアがsfen_a（本譜の次局面）を共有する
        pairs = [
            {
                "sfen_a": "startpos moves 2g2f",
                "sfen_b": "startpos moves 7g7f",
                "pair_type": "rewind_branch",
                "game_id": 0,
            },
            {
                "sfen_a": "startpos moves 2g2f",
                "sfen_b": "startpos moves 6g6f",
                "pair_type": "rewind_branch",
                "game_id": 0,
            },
        ]
        labeled, dropped = label_pairs(
            pairs, evaluate, label_nodes=200, stability_nodes=50
        )
        assert dropped == 0
        assert len(labeled) == 2

        # ユニークな(sfen, nodes)の数だけしかevaluateが呼ばれない
        unique_keys = {(sfen, nodes) for sfen, nodes in calls}
        assert len(unique_keys) == 6  # {2g2f,7g7f,6g6f} x {200,50}
        assert len(calls) == len(unique_keys)

    def test_none_stability_score_dropped(self) -> None:
        pairs = [{
            "sfen_a": "startpos moves 2g2f",
            "sfen_b": "startpos moves 7g7f",
            "pair_type": "rewind_branch",
            "game_id": 0,
        }]
        # ラベル探索（nodes=200）は有効値、安定性探索（nodes=50）はNone → 破棄
        evaluate = self._make_evaluate({
            "startpos moves 2g2f": {200: -30},
            "startpos moves 7g7f": {200: -60},
        })
        labeled, dropped = label_pairs(
            pairs, evaluate, label_nodes=200, stability_nodes=50
        )
        assert labeled == []
        assert dropped == 1


class TestLoadAndBuild:
    """load_mainline_records / build_pairsのテスト."""

    @pytest.fixture
    def data_path(self, tmp_path: Path) -> Path:
        records = [
            {
                "sfen": "startpos", "score_cp": 50, "ply": 0, "game_id": 0,
                "candidates": [
                    {"move": "2g2f", "score_cp": 50, "rank": 1},
                    {"move": "7g7f", "score_cp": 30, "rank": 2},
                ],
            },
            {"sfen": "startpos moves 2g2f", "score_cp": -40, "ply": 1,
             "game_id": 0},
            # 分岐レコード（source付き）は本譜として扱わない
            {"sfen": "startpos moves 2g2f 3c3d", "score_cp": 20, "ply": 2,
             "game_id": 0, "source": "multipv"},
            {"sfen": "startpos", "score_cp": 10, "ply": 0, "game_id": 1},
        ]
        path = tmp_path / "data.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in records))
        return path

    def test_load_mainline_records(self, data_path: Path) -> None:
        games = load_mainline_records(data_path)
        assert set(games.keys()) == {0, 1}
        assert len(games[0]) == 2  # source付きは除外
        assert games[0][0]["ply"] == 0

    def test_build_pairs_caps_per_game(self, data_path: Path) -> None:
        games = load_mainline_records(data_path)
        pairs = build_pairs(games, max_pairs_per_game=3, seed=42)
        per_game: dict[int, int] = {}
        for pair in pairs:
            per_game[pair["game_id"]] = per_game.get(pair["game_id"], 0) + 1
        assert all(count <= 3 for count in per_game.values())
        assert len(pairs) > 0
