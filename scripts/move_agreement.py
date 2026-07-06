"""指し手一致率測定スクリプト.

JSONLデータの各局面で、モデルの1手読みが選ぶ手と
教師エンジンの最善手との一致率を測定する。
評価関数の質を対局なしで直接測る指標として使用する。

使用例:
    # 200局面をランダムサンプリングして深さ10の教師と比較
    PYTHONPATH=. python scripts/move_agreement.py \
        --model checkpoints/best.pt \
        --data data/raw/dataset.jsonl \
        --depth 10 --limit 200

    # 結果をJSONに保存
    PYTHONPATH=. python scripts/move_agreement.py \
        --model checkpoints/best.pt --data data/raw/dataset.jsonl \
        --depth 10 --limit 200 --output reports/agreement.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import shogi

from engine.evaluator import Evaluator
from shogi_utils import USIEngine, get_engine_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ply帯の定義（序盤/中盤/終盤）
PHASE_BOUNDS = {
    "opening": (0, 29),
    "middlegame": (30, 79),
    "endgame": (80, 10000),
}


def board_from_sfen_line(sfen: str) -> shogi.Board:
    """データセットのsfenフィールドからshogi.Boardを構築.

    Args:
        sfen: "startpos [moves ...]" または "sfen <board> <turn> <hand> <count> [moves ...]"

    Returns:
        構築されたBoard

    Raises:
        ValueError: 不正な形式の場合
    """
    parts = sfen.strip().split()
    if not parts:
        raise ValueError("Empty sfen")

    if parts[0] == "startpos":
        board = shogi.Board()
        moves_idx = 1
    elif parts[0] == "sfen" and len(parts) >= 5:
        board = shogi.Board(" ".join(parts[1:5]))
        moves_idx = 5
    else:
        raise ValueError(f"Cannot parse sfen: {sfen}")

    if len(parts) > moves_idx and parts[moves_idx] == "moves":
        for move_usi in parts[moves_idx + 1 :]:
            board.push(shogi.Move.from_usi(move_usi))

    return board


def engine_position_args(sfen: str) -> tuple[str | None, list[str]]:
    """データセットのsfenフィールドをUSIEngine.set_positionの引数に変換.

    Args:
        sfen: "startpos [moves ...]" または "sfen ... [moves ...]"

    Returns:
        (sfen引数, moves引数)のタプル。startposの場合sfen引数はNone。
    """
    parts = sfen.strip().split()

    if parts[0] == "startpos":
        moves = parts[2:] if len(parts) > 1 and parts[1] == "moves" else []
        return None, moves

    if parts[0] == "sfen" and len(parts) >= 5:
        board_sfen = " ".join(parts[1:5])
        moves: list[str] = []
        if len(parts) > 5 and parts[5] == "moves":
            moves = parts[6:]
        return board_sfen, moves

    raise ValueError(f"Cannot parse sfen: {sfen}")


def summarize(results: list[tuple[int, bool]]) -> dict:
    """一致判定結果を集計.

    Args:
        results: (ply, 一致したか)のリスト

    Returns:
        全体およびply帯別の一致率を含む辞書
    """
    summary: dict = {
        "total": len(results),
        "matched": sum(1 for _, m in results if m),
    }
    summary["agreement"] = (
        summary["matched"] / summary["total"] if summary["total"] > 0 else 0.0
    )

    for phase, (lo, hi) in PHASE_BOUNDS.items():
        phase_results = [m for ply, m in results if lo <= ply <= hi]
        summary[phase] = {
            "total": len(phase_results),
            "matched": sum(phase_results),
            "agreement": (
                sum(phase_results) / len(phase_results) if phase_results else 0.0
            ),
        }

    return summary


def measure_agreement(
    model_path: str,
    data_path: str,
    depth: int,
    limit: int | None,
    engine_type: str,
    device: str,
    seed: int,
) -> dict:
    """指し手一致率を測定.

    Args:
        model_path: モデルcheckpointのパス
        data_path: JSONLデータのパス
        depth: 教師エンジンの探索深さ
        limit: 測定局面数の上限（Noneで全局面）
        engine_type: 教師エンジン種別（suisho5/hao）
        device: 推論デバイス
        seed: サンプリング用シード

    Returns:
        集計結果の辞書
    """
    # 局面の読み込み（重複sfenは除外）
    samples: list[dict] = []
    seen: set[str] = set()
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            if sample["sfen"] not in seen:
                seen.add(sample["sfen"])
                samples.append(sample)

    if limit is not None and len(samples) > limit:
        rng = random.Random(seed)
        samples = rng.sample(samples, limit)

    logger.info(f"Positions to evaluate: {len(samples)}")

    evaluator = Evaluator(model_path, device=device)
    results: list[tuple[int, bool]] = []
    skipped = 0

    with USIEngine(get_engine_path(engine_type)) as engine:
        engine.init_usi()
        engine.set_option("USI_OwnBook", False)
        engine.is_ready()

        for i, sample in enumerate(samples):
            sfen = sample["sfen"]
            ply = sample.get("ply", 0)

            board = board_from_sfen_line(sfen)
            legal_moves = list(board.legal_moves)
            if len(legal_moves) <= 1:
                # 選択の余地がない局面は測定対象外
                skipped += 1
                continue

            # モデルの1手読み
            model_move, _ = evaluator.find_best_move(board)

            # 教師エンジンの最善手
            engine_sfen, engine_moves = engine_position_args(sfen)
            engine.set_position(sfen=engine_sfen, moves=engine_moves)
            search = engine.go(depth=depth)

            if search.bestmove in ("resign", "win"):
                skipped += 1
                continue

            results.append((ply, model_move == search.bestmove))

            if (i + 1) % 50 == 0:
                current = summarize(results)
                logger.info(
                    f"{i + 1}/{len(samples)}: agreement={current['agreement']:.3f}"
                )

    summary = summarize(results)
    summary["skipped"] = skipped
    summary["depth"] = depth
    summary["model"] = str(model_path)
    summary["data"] = str(data_path)
    return summary


def main() -> None:
    """エントリーポイント."""
    parser = argparse.ArgumentParser(description="指し手一致率測定")
    parser.add_argument("--model", type=str, required=True, help="モデルcheckpoint")
    parser.add_argument("--data", type=str, required=True, help="JSONLデータ")
    parser.add_argument("--depth", type=int, default=10, help="教師エンジンの探索深さ")
    parser.add_argument("--limit", type=int, default=200, help="測定局面数の上限（0で無制限）")
    parser.add_argument("--engine-type", type=str, default="suisho5",
                        choices=["suisho5", "hao"], help="教師エンジン")
    parser.add_argument("--device", type=str, default="auto", help="推論デバイス")
    parser.add_argument("--seed", type=int, default=42, help="サンプリング用シード")
    parser.add_argument("--output", type=str, default=None, help="結果JSONの出力先")
    args = parser.parse_args()

    summary = measure_agreement(
        model_path=args.model,
        data_path=args.data,
        depth=args.depth,
        limit=args.limit if args.limit > 0 else None,
        engine_type=args.engine_type,
        device=args.device,
        seed=args.seed,
    )

    print(f"\n=== 指し手一致率 (depth {summary['depth']}) ===")
    print(f"全体:     {summary['agreement']:.3f} ({summary['matched']}/{summary['total']})")
    for phase, label in [("opening", "序盤"), ("middlegame", "中盤"), ("endgame", "終盤")]:
        p = summary[phase]
        print(f"{label} :   {p['agreement']:.3f} ({p['matched']}/{p['total']})")
    print(f"スキップ: {summary['skipped']}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
