"""摂動ペア生成（局面感度蒸留）のテスト."""

from __future__ import annotations

import shogi

from tools.gen_perturb_pairs import (
    board_perturb_pairs,
    is_stable,
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
