"""USIプロトコルサーバー.

標準入出力を通じてUSIプロトコルで通信するエンジン。

使用例:
    PYTHONPATH=. python engine/usi_server.py --model checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# プロジェクトルートをパスに追加（PYTHONPATH指定なしでも実行可能に）
sys.path.insert(0, str(Path(__file__).parent.parent))

import shogi

from engine.evaluator import Evaluator

# ログ設定（stderrに出力）
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

ENGINE_NAME = "Shogi-AI"
ENGINE_AUTHOR = "Shogi-AI Project"


class USIServer:
    """USIプロトコルサーバー."""

    def __init__(self, model_path: Path, device: str = "auto") -> None:
        """初期化.

        Args:
            model_path: モデルのチェックポイントパス
            device: 推論デバイス
        """
        self.model_path = model_path
        self.device = device
        self.evaluator: Evaluator | None = None
        self.board: shogi.Board = shogi.Board()
        self.running = True

    def run(self) -> None:
        """メインループ."""
        logger.info("USI server started")

        while self.running:
            try:
                line = input()
                logger.debug(f"Received: {line}")
                self._handle_command(line.strip())
            except EOFError:
                break
            except KeyboardInterrupt:
                break

        logger.info("USI server stopped")

    def _handle_command(self, command: str) -> None:
        """コマンドを処理."""
        if not command:
            return

        parts = command.split()
        cmd = parts[0]

        if cmd == "usi":
            self._cmd_usi()
        elif cmd == "isready":
            self._cmd_isready()
        elif cmd == "setoption":
            self._cmd_setoption(parts)
        elif cmd == "usinewgame":
            self._cmd_usinewgame()
        elif cmd == "position":
            self._cmd_position(parts)
        elif cmd == "go":
            self._cmd_go(parts)
        elif cmd == "stop":
            self._cmd_stop()
        elif cmd == "quit":
            self._cmd_quit()
        elif cmd == "gameover":
            self._cmd_gameover(parts)
        else:
            logger.warning(f"Unknown command: {command}")

    def _send(self, message: str) -> None:
        """メッセージを送信."""
        print(message, flush=True)
        logger.debug(f"Sent: {message}")

    def _cmd_usi(self) -> None:
        """usiコマンド."""
        self._send(f"id name {ENGINE_NAME}")
        self._send(f"id author {ENGINE_AUTHOR}")
        # オプションは今のところなし
        self._send("usiok")

    def _cmd_isready(self) -> None:
        """isreadyコマンド."""
        # 初回のみモデルを読み込む
        if self.evaluator is None:
            logger.info(f"Loading model: {self.model_path}")
            self.evaluator = Evaluator(self.model_path, self.device)
            logger.info("Model loaded")

        self._send("readyok")

    def _cmd_setoption(self, parts: list[str]) -> None:
        """setoptionコマンド."""
        # 今のところオプションは無視
        pass

    def _cmd_usinewgame(self) -> None:
        """usinewgameコマンド."""
        self.board = shogi.Board()

    def _cmd_position(self, parts: list[str]) -> None:
        """positionコマンド."""
        if len(parts) < 2:
            return

        if parts[1] == "startpos":
            self.board = shogi.Board()
            moves_idx = 2
        elif parts[1] == "sfen":
            # sfen形式: position sfen <board> <turn> <hand> <move_count> [moves ...]
            if len(parts) < 6:
                return
            sfen_str = " ".join(parts[2:6])
            self.board = shogi.Board(sfen_str)
            moves_idx = 6
        else:
            return

        # movesがあれば適用
        if len(parts) > moves_idx and parts[moves_idx] == "moves":
            for move_usi in parts[moves_idx + 1 :]:
                move = shogi.Move.from_usi(move_usi)
                self.board.push(move)

    def _cmd_go(self, parts: list[str]) -> None:
        """goコマンド."""
        if self.evaluator is None:
            logger.error("Model not loaded")
            self._send("bestmove resign")
            return

        # 探索を実行
        best_move, score = self.evaluator.find_best_move(self.board)

        # info出力
        self._send(f"info score cp {score} pv {best_move}")
        self._send(f"bestmove {best_move}")

    def _cmd_stop(self) -> None:
        """stopコマンド."""
        # 1手読みなので即座に返せる（何もしない）
        pass

    def _cmd_quit(self) -> None:
        """quitコマンド."""
        self.running = False

    def _cmd_gameover(self, parts: list[str]) -> None:
        """gameoverコマンド."""
        # 何もしない
        pass


def main() -> None:
    """エントリーポイント."""
    parser = argparse.ArgumentParser(description="Shogi-AI USI Engine")
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/best.pt",
        help="モデルのチェックポイントパス",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="推論デバイス（auto/cuda/mps/cpu）",
    )

    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)

    server = USIServer(model_path, args.device)
    server.run()


if __name__ == "__main__":
    main()
