"""局面感度蒸留用の摂動ペア生成スクリプト.

学習JSONLの本譜局面から、到達可能性の安全リスト内の摂動ペアを作り、
エンジンで両局面をラベル付けしてΔV学習用のペアJSONLを出力する。

ペア種別:
    rewind_branch: 本譜の次局面 vs 同一親局面から別候補手で分岐した局面
    move_dest:     同一駒を異なるマスへ動かす合法手ペアの子局面
    promotion:     同じ移動で成/不成だけが異なる合法手ペアの子局面（素材一致）

使用例:
    python tools/gen_perturb_pairs.py \\
        --data data/raw/dataset_mpv.jsonl \\
        -o data/raw/perturb_pairs.jsonl \\
        --label-nodes 200000 --stability-nodes 50000
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import shogi

from models.dataset import child_sfen
from scripts.move_agreement import board_from_sfen_line, engine_position_args
from shogi_utils import USIEngine, get_engine_path
from tools.gen_dataset import mate_to_cp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 駒価値（駒割の一致判定に使う。厳密なチューニング値である必要はない）
PIECE_VALUES: dict[int, int] = {
    shogi.PAWN: 90,
    shogi.LANCE: 315,
    shogi.KNIGHT: 405,
    shogi.SILVER: 495,
    shogi.GOLD: 540,
    shogi.BISHOP: 855,
    shogi.ROOK: 990,
    shogi.KING: 0,
    shogi.PROM_PAWN: 540,
    shogi.PROM_LANCE: 540,
    shogi.PROM_KNIGHT: 540,
    shogi.PROM_SILVER: 540,
    shogi.PROM_BISHOP: 945,
    shogi.PROM_ROOK: 1080,
}


def material_balance(board: shogi.Board) -> int:
    """手番側から見た駒割（盤上+持ち駒、玉除く）を計算.

    Args:
        board: 対象局面

    Returns:
        手番側の駒価値合計 − 相手側の駒価値合計（cp相当）
    """
    balance = 0
    for square in shogi.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        value = PIECE_VALUES[piece.piece_type]
        balance += value if piece.color == board.turn else -value
    for color in shogi.COLORS:
        sign = 1 if color == board.turn else -1
        for piece_type, count in board.pieces_in_hand[color].items():
            balance += sign * PIECE_VALUES[piece_type] * count
    return balance


def move_destination_pairs(
    board: shogi.Board,
    max_pairs_per_group: int = 1,
    rng: random.Random | None = None,
) -> list[tuple[str, str]]:
    """同一駒を異なるマスへ動かす合法手ペアを列挙.

    打つ手は同じ駒種の打ち先違いを同一グループとして扱う。
    成/不成の違いはpromotion_pairsの担当なので、同じ成り状態同士のみ
    ペアにする。

    Args:
        board: 親局面
        max_pairs_per_group: 1グループ（同一駒）あたりの最大ペア数
        rng: サンプリング用乱数（Noneで先頭から選択）

    Returns:
        (USI手a, USI手b)のリスト
    """
    groups: dict[tuple, list[shogi.Move]] = defaultdict(list)
    for move in board.legal_moves:
        if move.from_square is None:
            key = ("drop", move.drop_piece_type)
        else:
            key = ("move", move.from_square, bool(move.promotion))
        groups[key].append(move)

    pairs: list[tuple[str, str]] = []
    for moves in groups.values():
        if len(moves) < 2:
            continue
        combos = [
            (a, b) for i, a in enumerate(moves) for b in moves[i + 1:]
        ]
        if rng is not None:
            rng.shuffle(combos)
        for move_a, move_b in combos[:max_pairs_per_group]:
            pairs.append((move_a.usi(), move_b.usi()))
    return pairs


def promotion_pairs(board: shogi.Board) -> list[tuple[str, str]]:
    """同じ移動で成/不成だけが異なる合法手ペアを列挙（素材一致）.

    Args:
        board: 親局面

    Returns:
        (成る手のUSI, 成らない手のUSI)のリスト
    """
    by_fromto: dict[tuple[int, int], dict[bool, shogi.Move]] = defaultdict(dict)
    for move in board.legal_moves:
        if move.from_square is None:
            continue
        by_fromto[(move.from_square, move.to_square)][
            bool(move.promotion)
        ] = move
    return [
        (variants[True].usi(), variants[False].usi())
        for variants in by_fromto.values()
        if True in variants and False in variants
    ]


def rewind_branch_pairs(records: list[dict]) -> list[dict]:
    """本譜レコード列から（本譜の次局面, 分岐局面）ペアを構築.

    candidates付きレコードr(t)と本譜の次レコードr(t+1)について、
    candidatesの本譜と異なる手で分岐した局面とr(t+1)のペアを作る。
    どのplyでも分岐させるため、設計書の「k手巻き戻し」はこの
    per-record適用で網羅される。両局面は同一ply・同一手番。

    Args:
        records: 同一対局の本譜レコード（sourceフィールド無し）のリスト

    Returns:
        sfen_a（本譜側）/sfen_b（分岐側）/pair_type/game_idを持つ
        dictのリスト（評価値ラベルは未付与）
    """
    pairs: list[dict] = []
    by_ply = {r["ply"]: r for r in records}
    for ply, record in by_ply.items():
        candidates = record.get("candidates")
        next_record = by_ply.get(ply + 1)
        if not candidates or next_record is None:
            continue
        next_sfen_parts = next_record["sfen"].split()
        if len(next_sfen_parts) < 2:
            continue
        played = next_sfen_parts[-1]
        for cand in candidates:
            if cand["move"] == played:
                continue
            pairs.append({
                "sfen_a": next_record["sfen"],
                "sfen_b": child_sfen(record["sfen"], cand["move"]),
                "pair_type": "rewind_branch",
                "game_id": record.get("game_id", 0),
            })
    return pairs


def board_perturb_pairs(
    record: dict,
    max_pairs_per_group: int = 1,
    rng: random.Random | None = None,
) -> list[dict]:
    """1つの本譜レコードからmove_dest/promotionペアを構築.

    Args:
        record: 本譜レコード（sfenとgame_idを使用）
        max_pairs_per_group: move_destの1駒あたり最大ペア数
        rng: サンプリング用乱数

    Returns:
        sfen_a/sfen_b/pair_type/game_idを持つdictのリスト（ラベル未付与）
    """
    board = board_from_sfen_line(record["sfen"])
    game_id = record.get("game_id", 0)
    pairs: list[dict] = []
    for move_a, move_b in move_destination_pairs(
        board, max_pairs_per_group, rng
    ):
        pairs.append({
            "sfen_a": child_sfen(record["sfen"], move_a),
            "sfen_b": child_sfen(record["sfen"], move_b),
            "pair_type": "move_dest",
            "game_id": game_id,
        })
    for move_a, move_b in promotion_pairs(board):
        pairs.append({
            "sfen_a": child_sfen(record["sfen"], move_a),
            "sfen_b": child_sfen(record["sfen"], move_b),
            "pair_type": "promotion",
            "game_id": game_id,
        })
    return pairs


def is_stable(delta_label: int, delta_stability: int) -> bool:
    """2回の評価で差分の符号が反転していないか判定.

    どちらかの差分が0の場合は「反転」とは言えないため安定扱いとする。

    Args:
        delta_label: ラベル探索での score_a − score_b
        delta_stability: 安定性確認探索での score_a − score_b

    Returns:
        安定ならTrue
    """
    if delta_label == 0 or delta_stability == 0:
        return True
    return (delta_label > 0) == (delta_stability > 0)
