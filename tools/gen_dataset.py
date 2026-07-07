#!/usr/bin/env python3
"""教師データ生成スクリプト

水匠5同士の自己対局を行い、各局面の評価値を収集する。
弱いAIとの対局データ生成にも対応。
"""

from __future__ import annotations

import json
import argparse
import random
import sys
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import shogi

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from shogi_utils.usi_engine import USIEngine, MultiPVEntry, get_engine_path, get_default_engine_path


@dataclass
class PositionRecord:
    """局面レコード"""
    sfen: str           # 局面（SFEN形式）
    score_cp: int       # 評価値（centipawn、手番側から見た値）
    ply: int            # 手数
    game_id: int        # 対局ID
    result: Optional[str] = None  # 対局結果（"black_win", "white_win", "draw"）
    source: Optional[str] = None  # レコード由来（None=対局局面, "pv_leaf", "multipv"）
    # MultiPV候補手のリスト（[{"move", "score_cp", "rank"}, ...]、手番側視点）
    candidates: Optional[list[dict]] = None


def record_to_dict(record: PositionRecord | dict) -> dict:
    """レコードをJSONL書き込み用の辞書に変換（source/candidatesがNoneなら省略）.

    Args:
        record: PositionRecordまたは辞書

    Returns:
        書き込み用の辞書
    """
    d = asdict(record) if isinstance(record, PositionRecord) else dict(record)
    if d.get("source") is None:
        d.pop("source", None)
    if d.get("candidates") is None:
        d.pop("candidates", None)
    return d


def mate_to_cp(score_cp: Optional[int], score_mate: Optional[int]) -> Optional[int]:
    """cp/mateスコアをcpに統一（mateは±30000に変換）.

    Args:
        score_cp: centipawnスコア
        score_mate: 詰み手数

    Returns:
        centipawnスコア。どちらも無ければNone
    """
    if score_cp is not None:
        return score_cp
    if score_mate is not None:
        return 30000 if score_mate > 0 else -30000
    return None


def build_pv_leaf_record(
    moves: list[str],
    ply: int,
    game_id: int,
    score_cp: int,
    pv: list[str],
) -> Optional[PositionRecord]:
    """PV末端局面のレコードを構築.

    探索評価値は読み筋（PV）末端の静的評価に対応するため、
    PVを適用した「静止した局面」にラベルを付けることで、
    1手読みで使う評価関数の学習に適したデータになる。

    Args:
        moves: 現局面までの指し手リスト
        ply: 現局面の手数
        game_id: 対局ID
        score_cp: 現局面の探索評価値（手番側視点）
        pv: 読み筋（USI形式）

    Returns:
        PV末端局面のレコード。PVが空または非合法手を含む場合None
    """
    if not pv:
        return None

    # PVの合法性を検証（エンジンのPVはhash再構築で稀に壊れることがある）
    board = shogi.Board()
    for move_usi in moves:
        board.push_usi(move_usi)
    for move_usi in pv:
        try:
            move = shogi.Move.from_usi(move_usi)
        except ValueError:
            return None
        if move not in board.legal_moves:
            return None
        board.push(move)

    # score_cpは現局面の手番側視点。PVが奇数手なら末端局面は相手番なので符号反転
    leaf_score = score_cp if len(pv) % 2 == 0 else -score_cp
    sfen = "startpos moves " + " ".join(list(moves) + list(pv))

    return PositionRecord(
        sfen=sfen,
        score_cp=leaf_score,
        ply=ply + len(pv),
        game_id=game_id,
        source="pv_leaf",
    )


def get_legal_moves_usi(moves: list[str]) -> set[str]:
    """現局面の合法手集合（USI形式）を取得.

    Args:
        moves: 現局面までの指し手リスト

    Returns:
        合法手のUSI表記の集合
    """
    board = shogi.Board()
    for move_usi in moves:
        board.push_usi(move_usi)
    return {m.usi() for m in board.legal_moves}


def build_candidates(
    entries: list[MultiPVEntry],
    legal_moves: Optional[set[str]] = None,
) -> Optional[list[dict]]:
    """MultiPV候補手リストを親局面レコード用に構築.

    同一探索・同一深さの評価値なので候補手間の比較に使える。
    ranking lossの学習ペアや、オフライン指し手一致率の教師として利用する。

    Args:
        entries: MultiPV候補のリスト
        legal_moves: 現局面の合法手集合。指定時、非合法な候補手を除外する
            （エンジンのMultiPV行はhash再構築で稀に壊れることがあるため）

    Returns:
        [{"move", "score_cp", "rank"}, ...]（rank昇順）。有効な候補が無ければNone
    """
    candidates: list[dict] = []
    for entry in entries:
        score = mate_to_cp(entry.score_cp, entry.score_mate)
        if score is None:
            continue
        if legal_moves is not None and entry.move not in legal_moves:
            continue
        candidates.append({
            "move": entry.move,
            "score_cp": score,
            "rank": entry.rank,
        })

    if not candidates:
        return None
    candidates.sort(key=lambda c: c["rank"])
    return candidates


def build_multipv_records(
    moves: list[str],
    ply: int,
    game_id: int,
    entries: list[MultiPVEntry],
    legal_moves: Optional[set[str]] = None,
) -> list[PositionRecord]:
    """MultiPV候補手の子局面レコードを構築.

    順位2以下の候補手について、その手を指した後の局面に評価値を付ける。
    1回の探索から「良い手と悪い手の評価差」を含むデータが得られる。
    順位1（最善手）の子局面は次の手番の通常探索で記録されるため除外する。

    Args:
        moves: 現局面までの指し手リスト
        ply: 現局面の手数
        game_id: 対局ID
        entries: MultiPV候補のリスト
        legal_moves: 現局面の合法手集合。指定時、非合法な候補手を除外する

    Returns:
        子局面レコードのリスト
    """
    records: list[PositionRecord] = []
    for entry in entries:
        if entry.rank <= 1:
            continue

        score = mate_to_cp(entry.score_cp, entry.score_mate)
        if score is None:
            continue

        if legal_moves is not None and entry.move not in legal_moves:
            continue

        # entryのscoreは現局面の手番側視点。子局面は相手番なので符号反転
        sfen = "startpos moves " + " ".join(list(moves) + [entry.move])
        records.append(PositionRecord(
            sfen=sfen,
            score_cp=-score,
            ply=ply + 1,
            game_id=game_id,
            source="multipv",
        ))

    return records


class SelfPlayGenerator:
    """自己対局によるデータ生成"""

    def __init__(
        self,
        engine_path: Path,
        depth: Optional[int] = None,
        movetime: Optional[int] = None,
        max_ply: int = 256,
        use_book: bool = False,
        random_opening_ply: int = 0,
        random_type: str = "full",
        weak_side: Optional[str] = None,
        weak_prob: float = 0.5,
        record_pv_leaf: bool = False,
        multipv: int = 1,
    ):
        """
        Args:
            engine_path: エンジンパス
            depth: 探索深さ（depthとmovetimeのどちらか一方を指定）
            movetime: 1手あたりの思考時間（ミリ秒）
            max_ply: 最大手数
            use_book: 定跡使用
            random_opening_ply: 序盤のランダム手数（0の場合はランダム化しない）
            random_type: ランダム手の生成方式 ("engine"=go random, "full"=完全ランダム)
            weak_side: 弱い側 ("black", "white", "alternate", None)
            weak_prob: 弱い側がランダム手を指す確率 (0.0〜1.0)
            record_pv_leaf: PV末端局面のレコードも記録する
            multipv: MultiPV数（2以上で候補手の子局面レコードも記録）
        """
        self.engine_path = engine_path
        self.depth = depth
        self.movetime = movetime
        self.max_ply = max_ply
        self.use_book = use_book
        self.random_opening_ply = random_opening_ply
        self.random_type = random_type
        self.weak_side = weak_side
        self.weak_prob = weak_prob
        self.record_pv_leaf = record_pv_leaf
        self.multipv = multipv

        # depthとmovetimeのどちらも指定されていない場合はdepth=10をデフォルトに
        if self.depth is None and self.movetime is None:
            self.depth = 10

    def _is_weak_turn(self, ply: int, game_id: int) -> bool:
        """このターンが弱い側の手番かどうかを判定

        Args:
            ply: 現在の手数（0始まり、偶数=先手、奇数=後手）
            game_id: 対局ID（alternateモードで使用）

        Returns:
            弱い側の手番ならTrue
        """
        if self.weak_side is None:
            return False
        elif self.weak_side == "black":
            return ply % 2 == 0  # 先手（偶数手）
        elif self.weak_side == "white":
            return ply % 2 == 1  # 後手（奇数手）
        elif self.weak_side == "alternate":
            # 対局ごとに弱い側を交互に変える
            return (ply + game_id) % 2 == 0
        elif self.weak_side == "both":
            return True  # 両側とも弱い
        return False

    def _get_random_move(self, moves: list[str]) -> str | None:
        """python-shogiを使ってランダムな合法手を取得

        Args:
            moves: これまでの指し手リスト（USI形式）

        Returns:
            ランダムに選ばれた合法手（USI形式）、合法手がなければNone
        """
        board = shogi.Board()
        for move_usi in moves:
            board.push_usi(move_usi)

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        random_move = random.choice(legal_moves)
        return random_move.usi()

    def play_game(self, game_id: int) -> list[PositionRecord]:
        """1対局を行い、局面データを収集

        Args:
            game_id: 対局ID

        Returns:
            局面レコードのリスト
        """
        records: list[PositionRecord] = []
        moves: list[str] = []
        result: Optional[str] = None

        with USIEngine(self.engine_path) as engine:
            engine.init_usi()
            engine.set_option("USI_OwnBook", self.use_book)
            engine.set_option("Threads", 1)  # 再現性のため1スレッド
            if self.multipv > 1:
                engine.set_option("MultiPV", self.multipv)
            engine.is_ready()
            engine.new_game()

            for ply in range(self.max_ply):
                # 局面を設定
                engine.set_position(moves=moves if moves else None)

                is_weak_turn = self._is_weak_turn(ply, game_id)

                # 手の決定ロジック:
                # - random_opening期間中:
                #   - random_type=="engine": go_random()使用（評価値なし）
                #   - random_type=="full": 探索後に完全ランダム手で置換（評価値あり）
                # - random_opening後 + weak_side設定時:
                #   - 弱い側: weak_probに従ってランダム手
                #   - 強い側: 通常探索
                # - random_opening後 + weak_side未設定:
                #   - 両側: 通常探索

                use_random_opening = ply < self.random_opening_ply

                if use_random_opening and self.random_type == "engine":
                    # エンジンのgo random使用（評価値なし）
                    search_result = engine.go_random()
                else:
                    # 通常探索で評価値を取得
                    search_result = engine.go(depth=self.depth, movetime=self.movetime)

                    if use_random_opening and self.random_type == "full":
                        # 完全ランダム手で置換（評価値は維持）
                        random_move = self._get_random_move(moves)
                        if random_move is not None:
                            search_result.bestmove = random_move
                    elif self.weak_side is not None and is_weak_turn:
                        # 弱い側: 確率に従ってランダム手で置換
                        if random.random() < self.weak_prob:
                            random_move = self._get_random_move(moves)
                            if random_move is not None:
                                search_result.bestmove = random_move

                # 投了チェック
                if search_result.bestmove in ("resign", "win"):
                    if search_result.bestmove == "resign":
                        # 手番側が投了 = 相手の勝ち
                        result = "white_win" if ply % 2 == 0 else "black_win"
                    else:
                        # 入玉宣言勝ち
                        result = "black_win" if ply % 2 == 0 else "white_win"
                    break

                # 評価値を取得（手番側から見た値）
                score_cp = mate_to_cp(
                    search_result.score_cp, search_result.score_mate
                )
                if score_cp is None:
                    score_cp = 0  # フォールバック

                # SFEN取得のため一時的に局面を設定して'd'コマンドを使う代わりに、
                # movesから構築したSFENを保存（簡易版）
                # 注: 正確なSFENを取得するにはcshogiを使用するのがベスト
                sfen = self._moves_to_sfen_approx(moves, ply)

                # MultiPV有効時は候補手リストを親レコードに付与
                # （壊れたMultiPV行の混入を防ぐため合法手のみ採用）
                candidates: Optional[list[dict]] = None
                multipv_legal: Optional[set[str]] = None
                if self.multipv > 1 and search_result.multipv:
                    multipv_legal = get_legal_moves_usi(moves)
                    candidates = build_candidates(
                        search_result.multipv, multipv_legal
                    )

                records.append(PositionRecord(
                    sfen=sfen,
                    score_cp=score_cp,
                    ply=ply,
                    game_id=game_id,
                    candidates=candidates,
                ))

                # PV末端局面のレコードを追加
                if self.record_pv_leaf and search_result.pv:
                    leaf = build_pv_leaf_record(
                        moves, ply, game_id, score_cp, search_result.pv
                    )
                    if leaf is not None:
                        records.append(leaf)

                # MultiPV候補手の子局面レコードを追加
                if self.multipv > 1 and search_result.multipv:
                    records.extend(build_multipv_records(
                        moves, ply, game_id, search_result.multipv,
                        multipv_legal,
                    ))

                # 指し手を追加
                moves.append(search_result.bestmove)

                # 詰みの場合は終了
                if search_result.score_mate is not None:
                    if search_result.score_mate > 0:
                        result = "black_win" if ply % 2 == 0 else "white_win"
                    else:
                        result = "white_win" if ply % 2 == 0 else "black_win"
                    break

            # 最大手数到達
            if result is None:
                result = "draw"

        # 結果を各レコードに設定
        for record in records:
            record.result = result

        return records

    def _moves_to_sfen_approx(self, moves: list[str], ply: int) -> str:
        """指し手リストから近似SFENを生成（簡易版）

        正確なSFENを得るにはcshogiを使用すべき。
        ここでは「startpos moves ...」形式で保存。
        """
        if not moves:
            return "startpos"
        return "startpos moves " + " ".join(moves)


def _play_game_worker(args: tuple) -> list[dict]:
    """ワーカープロセスで1対局を実行

    Args:
        args: (game_id, engine_path, depth, movetime, max_ply, use_book,
               random_opening_ply, random_type, weak_side, weak_prob,
               record_pv_leaf, multipv)

    Returns:
        局面レコードのリスト（辞書形式）
    """
    (game_id, engine_path, depth, movetime, max_ply, use_book,
     random_opening_ply, random_type, weak_side, weak_prob,
     record_pv_leaf, multipv) = args

    generator = SelfPlayGenerator(
        engine_path=Path(engine_path),
        depth=depth,
        movetime=movetime,
        max_ply=max_ply,
        use_book=use_book,
        random_opening_ply=random_opening_ply,
        random_type=random_type,
        weak_side=weak_side,
        weak_prob=weak_prob,
        record_pv_leaf=record_pv_leaf,
        multipv=multipv,
    )

    try:
        records = generator.play_game(game_id)
        return [record_to_dict(r) for r in records]
    except Exception as e:
        print(f"Game {game_id} error: {e}", flush=True)
        return []


def generate_dataset_parallel(
    num_games: int,
    output_path: Path,
    engine_path: Path,
    depth: Optional[int] = None,
    movetime: Optional[int] = None,
    max_ply: int = 256,
    use_book: bool = False,
    random_opening_ply: int = 0,
    random_type: str = "full",
    num_workers: int = 4,
    weak_side: Optional[str] = None,
    weak_prob: float = 0.5,
    record_pv_leaf: bool = False,
    multipv: int = 1,
) -> int:
    """データセットを並列生成

    Args:
        num_games: 対局数
        output_path: 出力ファイルパス
        engine_path: エンジンパス
        depth: 探索深さ
        movetime: 思考時間（ミリ秒）
        max_ply: 最大手数
        use_book: 定跡使用
        random_opening_ply: 序盤のランダム手数
        random_type: ランダム手の生成方式 ("engine" or "full")
        num_workers: ワーカー数
        weak_side: 弱い側 ("black", "white", "alternate", None)
        weak_prob: 弱い側がランダム手を指す確率
        record_pv_leaf: PV末端局面のレコードも記録する
        multipv: MultiPV数（2以上で候補手の子局面レコードも記録）

    Returns:
        生成した局面数
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ワーカーへの引数を準備
    worker_args = [
        (game_id, str(engine_path), depth, movetime, max_ply, use_book,
         random_opening_ply, random_type, weak_side, weak_prob,
         record_pv_leaf, multipv)
        for game_id in range(num_games)
    ]

    total_positions = 0
    completed_games = 0

    with open(output_path, "w", encoding="utf-8") as f:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # 全てのタスクをサブミット
            futures = {executor.submit(_play_game_worker, args): args[0] for args in worker_args}

            for future in as_completed(futures):
                game_id = futures[future]
                completed_games += 1

                try:
                    records = future.result()
                    for record in records:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")

                    total_positions += len(records)
                    result = records[0]["result"] if records else "N/A"
                    print(f"[{completed_games}/{num_games}] Game {game_id}: {len(records)} positions, result: {result}")

                except Exception as e:
                    print(f"[{completed_games}/{num_games}] Game {game_id} failed: {e}")

    return total_positions


def generate_dataset(
    num_games: int,
    output_path: Path,
    engine_path: Path,
    depth: Optional[int] = None,
    movetime: Optional[int] = None,
    max_ply: int = 256,
    use_book: bool = False,
    random_opening_ply: int = 0,
    random_type: str = "full",
    weak_side: Optional[str] = None,
    weak_prob: float = 0.5,
    record_pv_leaf: bool = False,
    multipv: int = 1,
) -> int:
    """データセットを生成

    Args:
        num_games: 対局数
        output_path: 出力ファイルパス
        engine_path: エンジンパス
        depth: 探索深さ
        movetime: 思考時間（ミリ秒）
        max_ply: 最大手数
        use_book: 定跡使用
        random_opening_ply: 序盤のランダム手数
        random_type: ランダム手の生成方式 ("engine" or "full")
        weak_side: 弱い側 ("black", "white", "alternate", None)
        weak_prob: 弱い側がランダム手を指す確率
        record_pv_leaf: PV末端局面のレコードも記録する
        multipv: MultiPV数（2以上で候補手の子局面レコードも記録）

    Returns:
        生成した局面数
    """
    generator = SelfPlayGenerator(
        engine_path=engine_path,
        depth=depth,
        movetime=movetime,
        max_ply=max_ply,
        use_book=use_book,
        random_opening_ply=random_opening_ply,
        random_type=random_type,
        weak_side=weak_side,
        weak_prob=weak_prob,
        record_pv_leaf=record_pv_leaf,
        multipv=multipv,
    )

    total_positions = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for game_id in range(num_games):
            print(f"Game {game_id + 1}/{num_games}...", end=" ", flush=True)

            try:
                records = generator.play_game(game_id)
                for record in records:
                    f.write(json.dumps(record_to_dict(record), ensure_ascii=False) + "\n")

                total_positions += len(records)
                print(f"{len(records)} positions, result: {records[0].result if records else 'N/A'}")

            except Exception as e:
                print(f"Error: {e}")
                continue

    return total_positions


def main():
    parser = argparse.ArgumentParser(description="教師データ生成")
    parser.add_argument("--num-games", "-n", type=int, default=1, help="対局数")
    parser.add_argument("--output", "-o", type=str, default=None, help="出力ファイル")
    parser.add_argument("--depth", "-d", type=int, default=10, help="探索深さ（デフォルト: 10）")
    parser.add_argument("--movetime", "-t", type=int, default=None, help="思考時間(ms)（指定するとdepthより優先）")
    parser.add_argument("--max-ply", type=int, default=256, help="最大手数")
    parser.add_argument("--use-book", action="store_true", help="定跡使用")
    parser.add_argument("--random-opening", "-r", type=int, default=32, help="序盤のランダム手数（デフォルト: 32）")
    parser.add_argument("--random-type", type=str, default="full", choices=["engine", "full"],
                        help="ランダム手の生成方式 (engine=go random, full=完全ランダム, デフォルト: full)")
    parser.add_argument("--engine", type=str, default=None, help="エンジンパス（直接指定）")
    parser.add_argument("--engine-type", type=str, default="suisho5", choices=["suisho5", "hao"], help="エンジン種類（デフォルト: suisho5）")
    parser.add_argument("--workers", "-w", type=int, default=1, help="並列ワーカー数（デフォルト: 1）")
    parser.add_argument("--weak-side", type=str, default=None, choices=["black", "white", "alternate", "both"],
                        help="弱い側 (black=先手, white=後手, alternate=交互, both=両方)")
    parser.add_argument("--weak-prob", type=float, default=0.5,
                        help="弱い側がランダム手を指す確率 (0.0-1.0, デフォルト: 0.5)")
    parser.add_argument("--record-pv-leaf", action="store_true",
                        help="PV末端局面（静止局面）のレコードも記録")
    parser.add_argument("--multipv", type=int, default=1,
                        help="MultiPV数（2以上で候補手の子局面レコードも記録、デフォルト: 1）")

    args = parser.parse_args()

    # デフォルト出力ファイル名
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"data/raw/positions_{timestamp}.jsonl"

    # エンジンパス
    if args.engine:
        engine_path = Path(args.engine)
    else:
        engine_path = get_engine_path(args.engine_type)

    if not engine_path.exists():
        print(f"Error: Engine not found: {engine_path}")
        return 1

    output_path = Path(args.output)

    # movetimeが指定された場合はdepthを無効化
    depth = None if args.movetime else args.depth
    movetime = args.movetime

    print(f"Settings:")
    print(f"  Engine: {engine_path}")
    print(f"  Games: {args.num_games}")
    if movetime:
        print(f"  Movetime: {movetime}ms")
    else:
        print(f"  Depth: {depth}")
    print(f"  Max ply: {args.max_ply}")
    print(f"  Use book: {args.use_book}")
    print(f"  Random opening: {args.random_opening} ply (type={args.random_type})")
    print(f"  Workers: {args.workers}")
    if args.weak_side:
        print(f"  Weak side: {args.weak_side} (prob={args.weak_prob})")
    if args.record_pv_leaf:
        print(f"  Record PV leaf: enabled")
    if args.multipv > 1:
        print(f"  MultiPV: {args.multipv}")
    print(f"  Output: {output_path}")
    print()

    if args.workers > 1:
        total = generate_dataset_parallel(
            num_games=args.num_games,
            output_path=output_path,
            engine_path=engine_path,
            depth=depth,
            movetime=movetime,
            max_ply=args.max_ply,
            use_book=args.use_book,
            random_opening_ply=args.random_opening,
            random_type=args.random_type,
            num_workers=args.workers,
            weak_side=args.weak_side,
            weak_prob=args.weak_prob,
            record_pv_leaf=args.record_pv_leaf,
            multipv=args.multipv,
        )
    else:
        total = generate_dataset(
            num_games=args.num_games,
            output_path=output_path,
            engine_path=engine_path,
            depth=depth,
            movetime=movetime,
            max_ply=args.max_ply,
            use_book=args.use_book,
            random_opening_ply=args.random_opening,
            random_type=args.random_type,
            weak_side=args.weak_side,
            weak_prob=args.weak_prob,
            record_pv_leaf=args.record_pv_leaf,
            multipv=args.multipv,
        )

    print(f"\nDone! Total positions: {total}")
    print(f"Output: {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
