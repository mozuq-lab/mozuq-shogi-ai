"""USIエンジンラッパー

水匠5などのUSIプロトコル対応エンジンを制御するクラス。
"""

from __future__ import annotations

import subprocess
import re
import threading
import queue
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class MultiPVEntry:
    """MultiPVの候補手（1局面の上位N手のうちの1つ）"""
    rank: int  # multipv順位（1が最善）
    move: str  # 候補手（PVの初手）
    score_cp: Optional[int] = None  # centipawn（現局面の手番側視点）
    score_mate: Optional[int] = None  # 詰み手数
    pv: list[str] = None  # 読み筋

    def __post_init__(self):
        if self.pv is None:
            self.pv = []


@dataclass
class SearchResult:
    """探索結果"""
    bestmove: str
    score_cp: Optional[int] = None  # centipawn
    score_mate: Optional[int] = None  # 詰み手数（正:勝ち、負:負け）
    pv: list[str] = None  # 最善手順
    depth: int = 0
    nodes: int = 0
    multipv: list[MultiPVEntry] = None  # MultiPV候補（MultiPV>1設定時のみ）

    def __post_init__(self):
        if self.pv is None:
            self.pv = []
        if self.multipv is None:
            self.multipv = []


class USIEngine:
    """USIプロトコル対応エンジンのラッパー"""

    def __init__(self, engine_path: Path, working_dir: Optional[Path] = None):
        """
        Args:
            engine_path: エンジンの実行ファイルパス
            working_dir: 作業ディレクトリ（Noneの場合はエンジンのディレクトリ）
        """
        self.engine_path = Path(engine_path)
        self.working_dir = working_dir or self.engine_path.parent
        self.process: Optional[subprocess.Popen] = None
        self._output_queue: queue.Queue = queue.Queue()
        self._reader_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """エンジンを起動"""
        if self.process is not None:
            raise RuntimeError("Engine already started")

        self.process = subprocess.Popen(
            [str(self.engine_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(self.working_dir),
            bufsize=1,
        )

        # 出力読み取りスレッドを開始
        self._reader_thread = threading.Thread(target=self._read_output, daemon=True)
        self._reader_thread.start()

    def _read_output(self) -> None:
        """エンジンの出力を非同期で読み取る"""
        while self.process and self.process.poll() is None:
            try:
                line = self.process.stdout.readline()
                if line:
                    self._output_queue.put(line.strip())
            except Exception:
                break

    def _send_command(self, command: str) -> None:
        """コマンドを送信"""
        if self.process is None:
            raise RuntimeError("Engine not started")
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def _wait_for_response(self, expected: str, timeout: float = 30.0) -> list[str]:
        """特定のレスポンスを待つ"""
        lines = []
        import time
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                line = self._output_queue.get(timeout=0.1)
                lines.append(line)
                if line.startswith(expected):
                    return lines
            except queue.Empty:
                continue

        raise TimeoutError(f"Timeout waiting for '{expected}'")

    def init_usi(self) -> None:
        """USI初期化"""
        self._send_command("usi")
        self._wait_for_response("usiok")

    def set_option(self, name: str, value: str | int | bool) -> None:
        """オプションを設定"""
        if isinstance(value, bool):
            value = "true" if value else "false"
        self._send_command(f"setoption name {name} value {value}")

    def is_ready(self) -> None:
        """準備完了を待つ"""
        self._send_command("isready")
        self._wait_for_response("readyok", timeout=60.0)

    def new_game(self) -> None:
        """新規対局"""
        self._send_command("usinewgame")

    def set_position(self, sfen: Optional[str] = None, moves: Optional[list[str]] = None) -> None:
        """局面を設定

        Args:
            sfen: SFEN文字列（Noneの場合は初期局面）
            moves: 指し手リスト
        """
        if sfen is None:
            cmd = "position startpos"
        else:
            cmd = f"position sfen {sfen}"

        if moves:
            cmd += " moves " + " ".join(moves)

        self._send_command(cmd)

    def go(
        self,
        movetime: Optional[int] = None,
        depth: Optional[int] = None,
        nodes: Optional[int] = None,
    ) -> SearchResult:
        """探索を開始

        Args:
            movetime: 思考時間（ミリ秒）
            depth: 探索深さ
            nodes: 探索ノード数

        Returns:
            SearchResult: 探索結果
        """
        cmd = "go"
        if movetime is not None:
            cmd += f" movetime {movetime}"
        elif depth is not None:
            cmd += f" depth {depth}"
        elif nodes is not None:
            cmd += f" nodes {nodes}"

        self._send_command(cmd)

        # bestmoveを待つ
        lines = self._wait_for_response("bestmove", timeout=120.0)

        return self._parse_search_result(lines)

    def go_random(self) -> SearchResult:
        """ランダムな合法手を選択

        Returns:
            SearchResult: ランダムに選ばれた手（score_cpは0）
        """
        self._send_command("go random")

        # bestmoveを待つ
        lines = self._wait_for_response("bestmove", timeout=30.0)

        result = SearchResult(bestmove="resign", score_cp=0)
        for line in lines:
            if line.startswith("bestmove"):
                parts = line.split()
                if len(parts) >= 2:
                    result.bestmove = parts[1]
                break

        return result

    @staticmethod
    def _parse_score_pv(line: str) -> tuple[Optional[int], Optional[int], list[str]]:
        """info行からscore cp/mateとpvを抽出

        Args:
            line: エンジンのinfo行

        Returns:
            (score_cp, score_mate, pv)のタプル
        """
        score_cp = None
        score_mate = None

        cp_match = re.search(r"score cp (-?\d+)", line)
        if cp_match:
            score_cp = int(cp_match.group(1))

        mate_match = re.search(r"score mate (-?\d+)", line)
        if mate_match:
            score_mate = int(mate_match.group(1))

        pv_match = re.search(r" pv (.+)$", line)
        pv = pv_match.group(1).split() if pv_match else []

        return score_cp, score_mate, pv

    def _parse_search_result(self, lines: list[str]) -> SearchResult:
        """探索結果をパース"""
        result = SearchResult(bestmove="resign")

        # 最後のinfo行を探す
        # MultiPV有効時は各順位の最終行を記録し、bestmoveの評価にはmultipv 1を使う
        last_info = None
        multipv_lines: dict[int, str] = {}
        for line in lines:
            if line.startswith("info") and "score" in line:
                mpv_match = re.search(r"\bmultipv (\d+)\b", line)
                if mpv_match:
                    rank = int(mpv_match.group(1))
                    multipv_lines[rank] = line
                    if rank == 1:
                        last_info = line
                else:
                    last_info = line
            elif line.startswith("bestmove"):
                parts = line.split()
                if len(parts) >= 2:
                    result.bestmove = parts[1]

        if last_info:
            result.score_cp, result.score_mate, result.pv = self._parse_score_pv(
                last_info
            )

            # depth
            depth_match = re.search(r"depth (\d+)", last_info)
            if depth_match:
                result.depth = int(depth_match.group(1))

            # nodes
            nodes_match = re.search(r"nodes (\d+)", last_info)
            if nodes_match:
                result.nodes = int(nodes_match.group(1))

        # MultiPVエントリを構築（順位2以上が存在する場合のみ）
        if len(multipv_lines) >= 2:
            for rank in sorted(multipv_lines):
                score_cp, score_mate, pv = self._parse_score_pv(multipv_lines[rank])
                if not pv:
                    continue
                result.multipv.append(MultiPVEntry(
                    rank=rank,
                    move=pv[0],
                    score_cp=score_cp,
                    score_mate=score_mate,
                    pv=pv,
                ))

        return result

    def quit(self) -> None:
        """エンジンを終了"""
        if self.process:
            try:
                self._send_command("quit")
                self.process.wait(timeout=5.0)
            except Exception:
                self.process.kill()
            finally:
                self.process = None

    def __enter__(self) -> "USIEngine":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.quit()


def get_engine_path(engine: str = "suisho5") -> Path:
    """エンジンのパスを取得

    Args:
        engine: エンジン名 ("suisho5" または "hao")

    Returns:
        エンジンの実行ファイルパス

    OSに応じて適切なバイナリを返す:
    - Mac: YaneuraOu-mac (suisho5のみ)
    - Windows: YaneuraOu_NNUE_halfKP256-V830Git_AVX2.exe
    """
    import platform

    base_dir = Path(__file__).parent.parent

    if engine == "hao":
        engine_dir = base_dir / "external" / "shogi-cli" / "hao"
        if platform.system() == "Windows":
            return engine_dir / "YaneuraOu_NNUE_halfKP256-V830Git_AVX2.exe"
        else:
            return engine_dir / "YaneuraOu-mac"
    else:
        # suisho5 (デフォルト)
        engine_dir = base_dir / "external" / "shogi-cli" / "suisho5"
        if platform.system() == "Windows":
            return engine_dir / "YaneuraOu_NNUE_halfKP256-V830Git_AVX2.exe"
        else:
            return engine_dir / "YaneuraOu-mac"


def get_default_engine_path() -> Path:
    """デフォルトのエンジンパスを取得（後方互換性のため）"""
    return get_engine_path("suisho5")
