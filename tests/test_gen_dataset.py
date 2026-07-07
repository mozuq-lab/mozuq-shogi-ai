"""データ生成関連のテスト."""

from __future__ import annotations

from pathlib import Path

from shogi_utils.usi_engine import MultiPVEntry, USIEngine
from tools.gen_dataset import (
    PositionRecord,
    build_candidates,
    build_multipv_records,
    build_pv_leaf_record,
    get_legal_moves_usi,
    mate_to_cp,
    record_to_dict,
)


def _make_engine() -> USIEngine:
    """パース処理テスト用のUSIEngine（プロセスは起動しない）."""
    return USIEngine(Path("dummy"))


class TestParseSearchResult:
    """USIエンジン出力パースのテスト."""

    def test_single_pv(self) -> None:
        lines = [
            "info depth 5 score cp 30 nodes 1000 pv 7g7f 3c3d 2g2f",
            "info depth 10 score cp 55 nodes 50000 pv 2g2f 8c8d",
            "bestmove 2g2f",
        ]
        result = _make_engine()._parse_search_result(lines)
        assert result.bestmove == "2g2f"
        assert result.score_cp == 55
        assert result.pv == ["2g2f", "8c8d"]
        assert result.multipv == []

    def test_multipv_entries(self) -> None:
        lines = [
            "info depth 8 multipv 1 score cp 50 pv 2g2f 8c8d",
            "info depth 8 multipv 2 score cp 20 pv 7g7f 3c3d",
            "info depth 8 multipv 3 score cp -100 pv 9g9f 8c8d",
            "info depth 10 multipv 1 score cp 60 pv 2g2f 3c3d",
            "info depth 10 multipv 2 score cp 10 pv 7g7f 8c8d",
            "info depth 10 multipv 3 score mate -5 pv 9g9f 8c8d",
            "bestmove 2g2f",
        ]
        result = _make_engine()._parse_search_result(lines)

        # bestmoveの評価はmultipv 1の最終行から取る
        assert result.score_cp == 60
        assert result.pv == ["2g2f", "3c3d"]

        # 各順位の最終行（最深）が使われる
        assert len(result.multipv) == 3
        assert result.multipv[0].rank == 1
        assert result.multipv[0].move == "2g2f"
        assert result.multipv[0].score_cp == 60
        assert result.multipv[1].rank == 2
        assert result.multipv[1].move == "7g7f"
        assert result.multipv[1].score_cp == 10
        assert result.multipv[2].rank == 3
        assert result.multipv[2].score_mate == -5

    def test_multipv_last_line_not_best(self) -> None:
        """MultiPV時、最後のinfo行（最下位）をbestmove評価に使わない."""
        lines = [
            "info depth 10 multipv 1 score cp 100 pv 2g2f",
            "info depth 10 multipv 2 score cp -500 pv 9g9f",
            "bestmove 2g2f",
        ]
        result = _make_engine()._parse_search_result(lines)
        assert result.score_cp == 100  # multipv 2の-500ではない


class TestMateToCp:
    """mate_to_cpのテスト."""

    def test_cp_passthrough(self) -> None:
        assert mate_to_cp(150, None) == 150

    def test_mate_positive(self) -> None:
        assert mate_to_cp(None, 5) == 30000

    def test_mate_negative(self) -> None:
        assert mate_to_cp(None, -3) == -30000

    def test_both_none(self) -> None:
        assert mate_to_cp(None, None) is None


class TestBuildPvLeafRecord:
    """build_pv_leaf_recordのテスト."""

    def test_even_pv_keeps_sign(self) -> None:
        """偶数手のPVでは末端局面も同じ手番なので符号維持."""
        record = build_pv_leaf_record(
            moves=[], ply=0, game_id=7, score_cp=80, pv=["7g7f", "3c3d"]
        )
        assert record is not None
        assert record.sfen == "startpos moves 7g7f 3c3d"
        assert record.score_cp == 80
        assert record.ply == 2
        assert record.game_id == 7
        assert record.source == "pv_leaf"

    def test_odd_pv_flips_sign(self) -> None:
        """奇数手のPVでは末端局面は相手番なので符号反転."""
        record = build_pv_leaf_record(
            moves=[], ply=0, game_id=0, score_cp=80, pv=["7g7f"]
        )
        assert record is not None
        assert record.score_cp == -80
        assert record.ply == 1

    def test_with_prior_moves(self) -> None:
        record = build_pv_leaf_record(
            moves=["7g7f", "3c3d"], ply=2, game_id=0, score_cp=-50,
            pv=["2g2f", "8c8d"],
        )
        assert record is not None
        assert record.sfen == "startpos moves 7g7f 3c3d 2g2f 8c8d"
        assert record.ply == 4

    def test_empty_pv_returns_none(self) -> None:
        assert build_pv_leaf_record([], 0, 0, 50, []) is None

    def test_illegal_pv_returns_none(self) -> None:
        """非合法手を含むPVはNone（初期局面で1a1bは指せない）."""
        assert build_pv_leaf_record([], 0, 0, 50, ["1a1b"]) is None

    def test_malformed_pv_returns_none(self) -> None:
        assert build_pv_leaf_record([], 0, 0, 50, ["xxxx"]) is None


class TestBuildMultipvRecords:
    """build_multipv_recordsのテスト."""

    def _entries(self):
        from shogi_utils.usi_engine import MultiPVEntry

        return [
            MultiPVEntry(rank=1, move="2g2f", score_cp=60, pv=["2g2f"]),
            MultiPVEntry(rank=2, move="7g7f", score_cp=20, pv=["7g7f"]),
            MultiPVEntry(rank=3, move="9g9f", score_mate=-4, pv=["9g9f"]),
        ]

    def test_rank1_excluded_and_sign_flipped(self) -> None:
        records = build_multipv_records([], 0, 3, self._entries())

        # rank 1は除外（次plyの通常探索で記録されるため）
        assert len(records) == 2

        # rank 2: 手番側+20 → 子局面（相手番）視点で-20
        assert records[0].sfen == "startpos moves 7g7f"
        assert records[0].score_cp == -20
        assert records[0].ply == 1
        assert records[0].source == "multipv"

        # rank 3: mate -4 → -30000 → 子局面視点で+30000
        assert records[1].score_cp == 30000

    def test_with_prior_moves(self) -> None:
        from shogi_utils.usi_engine import MultiPVEntry

        entries = [
            MultiPVEntry(rank=1, move="8c8d", score_cp=10, pv=["8c8d"]),
            MultiPVEntry(rank=2, move="3c3d", score_cp=-30, pv=["3c3d"]),
        ]
        records = build_multipv_records(["7g7f"], 1, 0, entries)
        assert len(records) == 1
        assert records[0].sfen == "startpos moves 7g7f 3c3d"
        assert records[0].score_cp == 30
        assert records[0].ply == 2

    def test_illegal_move_filtered(self) -> None:
        """legal_moves指定時、非合法な候補手の子局面レコードは作られない."""
        records = build_multipv_records(
            [], 0, 0, self._entries(), legal_moves={"2g2f"}
        )
        # rank 2の7g7f・rank 3の9g9fが非合法扱いで除外され、rank 1は元々除外
        assert records == []


class TestGetLegalMovesUsi:
    """get_legal_moves_usiのテスト."""

    def test_startpos(self) -> None:
        legal = get_legal_moves_usi([])
        assert "7g7f" in legal
        assert "2g2f" in legal
        assert len(legal) == 30  # 初期局面の合法手数

    def test_after_moves(self) -> None:
        legal = get_legal_moves_usi(["7g7f"])
        assert "3c3d" in legal  # 後手の手番
        assert "7g7f" not in legal


class TestRecordToDict:
    """record_to_dictのテスト."""

    def test_source_none_omitted(self) -> None:
        record = PositionRecord(
            sfen="startpos", score_cp=0, ply=0, game_id=0, result="draw"
        )
        d = record_to_dict(record)
        assert "source" not in d
        assert d["sfen"] == "startpos"

    def test_source_kept(self) -> None:
        record = PositionRecord(
            sfen="startpos", score_cp=0, ply=0, game_id=0, source="pv_leaf"
        )
        d = record_to_dict(record)
        assert d["source"] == "pv_leaf"

    def test_dict_input(self) -> None:
        d = record_to_dict({"sfen": "startpos", "score_cp": 1, "source": None})
        assert "source" not in d

    def test_candidates_none_omitted(self) -> None:
        record = PositionRecord(
            sfen="startpos", score_cp=0, ply=0, game_id=0, result="draw"
        )
        d = record_to_dict(record)
        assert "candidates" not in d

    def test_candidates_kept(self) -> None:
        candidates = [{"move": "2g2f", "score_cp": 50, "rank": 1}]
        record = PositionRecord(
            sfen="startpos", score_cp=50, ply=0, game_id=0,
            candidates=candidates,
        )
        d = record_to_dict(record)
        assert d["candidates"] == candidates


class TestBuildCandidates:
    """build_candidatesのテスト."""

    def test_basic(self) -> None:
        entries = [
            MultiPVEntry(rank=2, move="7g7f", score_cp=30),
            MultiPVEntry(rank=1, move="2g2f", score_cp=55),
        ]
        candidates = build_candidates(entries)
        assert candidates == [
            {"move": "2g2f", "score_cp": 55, "rank": 1},
            {"move": "7g7f", "score_cp": 30, "rank": 2},
        ]

    def test_mate_converted(self) -> None:
        entries = [
            MultiPVEntry(rank=1, move="2g2f", score_mate=3),
            MultiPVEntry(rank=2, move="7g7f", score_mate=-5),
        ]
        candidates = build_candidates(entries)
        assert candidates[0]["score_cp"] == 30000
        assert candidates[1]["score_cp"] == -30000

    def test_no_score_entry_skipped(self) -> None:
        entries = [
            MultiPVEntry(rank=1, move="2g2f", score_cp=10),
            MultiPVEntry(rank=2, move="7g7f"),  # スコアなし
        ]
        candidates = build_candidates(entries)
        assert len(candidates) == 1
        assert candidates[0]["move"] == "2g2f"

    def test_empty_returns_none(self) -> None:
        assert build_candidates([]) is None
        assert build_candidates([MultiPVEntry(rank=1, move="2g2f")]) is None

    def test_illegal_move_filtered(self) -> None:
        """legal_moves指定時、非合法な候補手（壊れたMultiPV行）は除外される."""
        entries = [
            MultiPVEntry(rank=1, move="2g2f", score_cp=10),
            MultiPVEntry(rank=2, move="2g2g", score_cp=5),  # 非合法
        ]
        candidates = build_candidates(entries, legal_moves={"2g2f", "7g7f"})
        assert len(candidates) == 1
        assert candidates[0]["move"] == "2g2f"

    def test_no_legal_moves_keeps_all(self) -> None:
        """legal_moves未指定なら従来どおりフィルタしない."""
        entries = [
            MultiPVEntry(rank=1, move="2g2f", score_cp=10),
            MultiPVEntry(rank=2, move="2g2g", score_cp=5),
        ]
        candidates = build_candidates(entries)
        assert len(candidates) == 2
