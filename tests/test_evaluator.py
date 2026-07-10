"""Evaluatorのテスト."""

from __future__ import annotations

from pathlib import Path

import pytest
import shogi
import torch

from engine.evaluator import Evaluator
from models import ValueTransformer


@pytest.fixture
def tiny_checkpoint(tmp_path: Path) -> Path:
    """テスト用の小さなモデルcheckpointを作成."""
    config = {
        "d_model": 32,
        "n_heads": 2,
        "n_layers": 1,
        "ffn_dim": 64,
        "use_features": False,
        "use_attention_pooling": True,
        "normalize_turn": False,
        "cp_scale": 1200.0,
        "target_mode": "cp",
    }
    model = ValueTransformer(
        d_model=32, n_heads=2, n_layers=1, ffn_dim=64, dropout=0.0
    )
    path = tmp_path / "tiny.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": config}, path)
    return path


class TestFindBestMoveTerminal:
    """find_best_moveの終端局面処理のテスト."""

    def test_mate_in_one_returns_mate_score(self, tiny_checkpoint: Path) -> None:
        """1手詰めはNN評価に依らず必ず選ばれ、+30000を返す.

        後手玉5a、先手歩5c、先手持ち駒に金。G*5bが唯一の詰み
        （金は歩に支えられ、玉の逃げ場は全て金の利き）。
        """
        evaluator = Evaluator(tiny_checkpoint, device="cpu")
        board = shogi.Board("4k4/9/4P4/9/9/9/9/9/9 b G 1")

        move, score = evaluator.find_best_move(board)

        assert move == "G*5b"
        assert score == 30000

    def test_mate_check_does_not_mutate_board(self, tiny_checkpoint: Path) -> None:
        """詰みチェック後も盤面が元の状態に戻っていること."""
        evaluator = Evaluator(tiny_checkpoint, device="cpu")
        board = shogi.Board("4k4/9/4P4/9/9/9/9/9/9 b G 1")
        sfen_before = board.sfen()

        evaluator.find_best_move(board)

        assert board.sfen() == sfen_before

    def test_no_legal_moves_returns_resign(self, tiny_checkpoint: Path) -> None:
        """合法手がない（詰まされている）場合はresignと-30000を返す."""
        evaluator = Evaluator(tiny_checkpoint, device="cpu")
        # G*5bの詰み局面を後手番から見た状態
        board = shogi.Board("4k4/4G4/4P4/9/9/9/9/9/9 w - 1")

        move, score = evaluator.find_best_move(board)

        assert move == "resign"
        assert score == -30000

    def test_normal_position_returns_legal_move(self, tiny_checkpoint: Path) -> None:
        """詰みのない通常局面では合法手を返す（従来動作の確認）."""
        evaluator = Evaluator(tiny_checkpoint, device="cpu")
        board = shogi.Board()  # 初期局面

        move, score = evaluator.find_best_move(board)

        legal_usi = {m.usi() for m in board.legal_moves}
        assert move in legal_usi
        assert -30000 < score < 30000
