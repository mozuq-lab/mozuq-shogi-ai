"""計測スクリプト関連のテスト."""

from __future__ import annotations

import random

import shogi

from scripts.move_agreement import (
    board_from_sfen_line,
    engine_position_args,
    summarize,
)
from scripts.selfplay_match import elo_diff, play_game, random_opening


class TestBoardFromSfenLine:
    """board_from_sfen_lineのテスト."""

    def test_startpos(self) -> None:
        board = board_from_sfen_line("startpos")
        assert board.sfen() == shogi.Board().sfen()

    def test_startpos_with_moves(self) -> None:
        board = board_from_sfen_line("startpos moves 7g7f 3c3d")
        expected = shogi.Board()
        expected.push(shogi.Move.from_usi("7g7f"))
        expected.push(shogi.Move.from_usi("3c3d"))
        assert board.sfen() == expected.sfen()

    def test_sfen_form(self) -> None:
        sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
        board = board_from_sfen_line(f"sfen {sfen}")
        assert board.sfen() == sfen


class TestEnginePositionArgs:
    """engine_position_argsのテスト."""

    def test_startpos(self) -> None:
        sfen_arg, moves = engine_position_args("startpos")
        assert sfen_arg is None
        assert moves == []

    def test_startpos_with_moves(self) -> None:
        sfen_arg, moves = engine_position_args("startpos moves 7g7f 3c3d")
        assert sfen_arg is None
        assert moves == ["7g7f", "3c3d"]

    def test_sfen_form(self) -> None:
        sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
        sfen_arg, moves = engine_position_args(f"sfen {sfen} moves 7g7f")
        assert sfen_arg == sfen
        assert moves == ["7g7f"]


class TestSummarize:
    """summarizeのテスト."""

    def test_empty(self) -> None:
        summary = summarize([])
        assert summary["total"] == 0
        assert summary["agreement"] == 0.0

    def test_overall_agreement(self) -> None:
        results = [(0, True), (10, False), (20, True), (40, True)]
        summary = summarize(results)
        assert summary["total"] == 4
        assert summary["matched"] == 3
        assert summary["agreement"] == 0.75

    def test_phase_split(self) -> None:
        results = [
            (0, True),     # opening
            (29, False),   # opening
            (30, True),    # middlegame
            (79, True),    # middlegame
            (80, False),   # endgame
            (150, True),   # endgame
        ]
        summary = summarize(results)
        assert summary["opening"]["total"] == 2
        assert summary["opening"]["matched"] == 1
        assert summary["middlegame"]["total"] == 2
        assert summary["middlegame"]["matched"] == 2
        assert summary["endgame"]["total"] == 2
        assert summary["endgame"]["agreement"] == 0.5


def _random_move_fn(rng: random.Random):
    """ランダムに合法手を選ぶMoveFn（テスト用）."""

    def fn(board: shogi.Board) -> tuple[str, int]:
        legal = list(board.legal_moves)
        return rng.choice(legal).usi(), 0

    return fn


class TestPlayGame:
    """play_gameのテスト."""

    def test_max_moves_draw(self) -> None:
        """最大手数に達したら引き分けになる."""
        rng = random.Random(0)
        result = play_game(
            shogi.Board(), _random_move_fn(rng), _random_move_fn(rng), max_moves=10
        )
        assert result == "draw"

    def test_random_game_terminates(self) -> None:
        """ランダム同士の対局が正常な結果で終了する."""
        rng = random.Random(1)
        result = play_game(
            shogi.Board(), _random_move_fn(rng), _random_move_fn(rng), max_moves=512
        )
        assert result in ("black_win", "white_win", "draw")

    def test_resign_black(self) -> None:
        """先手が投了したら後手勝ち."""

        def resign_fn(board: shogi.Board) -> tuple[str, int]:
            return "resign", -30000

        rng = random.Random(2)
        result = play_game(
            shogi.Board(), resign_fn, _random_move_fn(rng), max_moves=10
        )
        assert result == "white_win"


class TestRandomOpening:
    """random_openingのテスト."""

    def test_moves_applied(self) -> None:
        rng = random.Random(42)
        board = random_opening(rng, 8)
        assert board.move_number == 9  # 8手進んだ局面

    def test_deterministic(self) -> None:
        board1 = random_opening(random.Random(42), 8)
        board2 = random_opening(random.Random(42), 8)
        assert board1.sfen() == board2.sfen()


class TestEloDiff:
    """elo_diffのテスト."""

    def test_even(self) -> None:
        assert elo_diff(0.5) == 0.0

    def test_positive(self) -> None:
        assert 180 < elo_diff(0.75) < 200  # 勝率75% ≈ +191

    def test_symmetric(self) -> None:
        assert abs(elo_diff(0.6) + elo_diff(0.4)) < 1e-9
