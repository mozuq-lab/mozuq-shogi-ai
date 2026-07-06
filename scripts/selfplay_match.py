"""モデル同士の自己対局スクリプト.

2つのcheckpointを1手読み同士で対局させ、勝敗を集計する。
開始局面はランダムな序盤数手で多様化し、同一開始局面から
先後を入れ替えて2局ずつ対局する（ペア対局）。

使用例:
    # 新旧モデルを20ペア（40局）対局
    PYTHONPATH=. python scripts/selfplay_match.py \
        --model-a checkpoints/new.pt \
        --model-b checkpoints/old.pt \
        --games 20

    # 結果をJSONに保存
    PYTHONPATH=. python scripts/selfplay_match.py \
        --model-a checkpoints/new.pt --model-b checkpoints/old.pt \
        --games 20 --output reports/match.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).parent.parent))

import shogi

from engine.evaluator import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 手を選ぶ関数の型: Boardを受け取り(USI表記, 評価値)を返す
MoveFn = Callable[[shogi.Board], tuple[str, int]]


def play_game(
    board: shogi.Board,
    black_fn: MoveFn,
    white_fn: MoveFn,
    max_moves: int = 256,
) -> str:
    """1局を対局.

    Args:
        board: 開始局面（破壊的に使用される）
        black_fn: 先手の指し手選択関数
        white_fn: 後手の指し手選択関数
        max_moves: 最大手数（超えたら引き分け）

    Returns:
        対局結果 ("black_win", "white_win", "draw")
    """
    for _ in range(max_moves):
        if not any(board.legal_moves):
            # 合法手なし = 手番側の負け（詰み or ステイルメイト）
            return "white_win" if board.turn == shogi.BLACK else "black_win"

        if board.is_fourfold_repetition():
            return "draw"

        move_fn = black_fn if board.turn == shogi.BLACK else white_fn
        move_usi, _ = move_fn(board)

        if move_usi == "resign":
            return "white_win" if board.turn == shogi.BLACK else "black_win"

        board.push(shogi.Move.from_usi(move_usi))

    return "draw"


def random_opening(rng: random.Random, num_moves: int, max_retries: int = 10) -> shogi.Board:
    """ランダムな序盤局面を生成.

    Args:
        rng: 乱数生成器
        num_moves: ランダムに進める手数
        max_retries: 途中で終局した場合のやり直し回数

    Returns:
        生成された開始局面
    """
    for _ in range(max_retries):
        board = shogi.Board()
        ok = True
        for _ in range(num_moves):
            legal = list(board.legal_moves)
            if not legal:
                ok = False
                break
            board.push(rng.choice(legal))
        if ok:
            return board
    # やり直し上限に達したら初期局面を返す
    return shogi.Board()


def elo_diff(score_rate: float) -> float:
    """勝率からレート差を推定.

    Args:
        score_rate: Aのスコア率（勝=1、引分=0.5として集計）

    Returns:
        推定レート差（Aが上ならプラス）
    """
    clamped = min(max(score_rate, 0.001), 0.999)
    return 400.0 * math.log10(clamped / (1.0 - clamped))


def run_match(
    model_a: str,
    model_b: str,
    games: int,
    opening_moves: int,
    max_moves: int,
    device: str,
    seed: int,
) -> dict:
    """ペア対局でマッチを実行.

    各ペアは同一のランダム開始局面から、A先手とB先手の2局を対局する。

    Args:
        model_a: モデルAのcheckpointパス
        model_b: モデルBのcheckpointパス
        games: ペア数（実対局数は2倍）
        opening_moves: 開始局面のランダム手数
        max_moves: 1局の最大手数
        device: 推論デバイス
        seed: 乱数シード

    Returns:
        集計結果の辞書
    """
    ev_a = Evaluator(model_a, device=device)
    ev_b = Evaluator(model_b, device=device)
    fn_a: MoveFn = ev_a.find_best_move
    fn_b: MoveFn = ev_b.find_best_move

    rng = random.Random(seed)
    a_wins = 0
    b_wins = 0
    draws = 0

    for pair in range(games):
        opening = random_opening(rng, opening_moves)
        opening_sfen = opening.sfen()

        # 1局目: Aが先手
        result = play_game(shogi.Board(opening_sfen), fn_a, fn_b, max_moves)
        if result == "black_win":
            a_wins += 1
        elif result == "white_win":
            b_wins += 1
        else:
            draws += 1

        # 2局目: Bが先手（同一開始局面）
        result = play_game(shogi.Board(opening_sfen), fn_b, fn_a, max_moves)
        if result == "black_win":
            b_wins += 1
        elif result == "white_win":
            a_wins += 1
        else:
            draws += 1

        total = a_wins + b_wins + draws
        logger.info(
            f"Pair {pair + 1}/{games}: A={a_wins} B={b_wins} draw={draws} "
            f"(score_rate={(a_wins + 0.5 * draws) / total:.3f})"
        )

    total = a_wins + b_wins + draws
    score_rate = (a_wins + 0.5 * draws) / total if total > 0 else 0.5
    return {
        "model_a": str(model_a),
        "model_b": str(model_b),
        "games": total,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "draws": draws,
        "score_rate_a": score_rate,
        "elo_diff_a": elo_diff(score_rate),
    }


def main() -> None:
    """エントリーポイント."""
    parser = argparse.ArgumentParser(description="モデル同士の自己対局")
    parser.add_argument("--model-a", type=str, required=True, help="モデルA checkpoint")
    parser.add_argument("--model-b", type=str, required=True, help="モデルB checkpoint")
    parser.add_argument("--games", type=int, default=20, help="ペア数（実対局数は2倍）")
    parser.add_argument("--opening-moves", type=int, default=8, help="開始局面のランダム手数")
    parser.add_argument("--max-moves", type=int, default=256, help="1局の最大手数")
    parser.add_argument("--device", type=str, default="auto", help="推論デバイス")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード")
    parser.add_argument("--output", type=str, default=None, help="結果JSONの出力先")
    args = parser.parse_args()

    summary = run_match(
        model_a=args.model_a,
        model_b=args.model_b,
        games=args.games,
        opening_moves=args.opening_moves,
        max_moves=args.max_moves,
        device=args.device,
        seed=args.seed,
    )

    print(f"\n=== 対局結果 ({summary['games']}局) ===")
    print(f"A ({summary['model_a']}): {summary['a_wins']}勝")
    print(f"B ({summary['model_b']}): {summary['b_wins']}勝")
    print(f"引き分け: {summary['draws']}")
    print(f"Aのスコア率: {summary['score_rate_a']:.3f}")
    print(f"推定レート差: {summary['elo_diff_a']:+.0f}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
