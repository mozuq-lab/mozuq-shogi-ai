"""計測スクリプト関連のテスト."""

from __future__ import annotations

import shogi

from scripts.move_agreement import (
    board_from_sfen_line,
    engine_position_args,
    summarize,
)


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
