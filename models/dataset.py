"""将棋局面データセット."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset

from models.sfen_parser import parse_sfen
from models.value_transformer import cp_to_wdl, normalize_cp
from models.features import compute_all_features

if TYPE_CHECKING:
    pass


def flip_board_horizontal(board: torch.Tensor) -> torch.Tensor:
    """盤面を左右反転.

    Args:
        board: 盤面テンソル (81,) 駒種ID

    Returns:
        左右反転した盤面テンソル (81,)
    """
    # 9x9に変形して左右反転し、1Dに戻す
    board_2d = board.view(9, 9)
    flipped = torch.flip(board_2d, dims=[1])
    return flipped.view(81)


def flip_hand(hand: torch.Tensor) -> torch.Tensor:
    """持ち駒を先手/後手入れ替え.

    持ち駒インデックス: 先手0-6, 後手7-13

    Args:
        hand: 持ち駒テンソル (14,)

    Returns:
        入れ替えた持ち駒テンソル (14,)
    """
    flipped = torch.zeros_like(hand)
    flipped[:7] = hand[7:]   # 後手の持ち駒 → 先手に
    flipped[7:] = hand[:7]   # 先手の持ち駒 → 後手に
    return flipped


def flip_board_turn(board: torch.Tensor) -> torch.Tensor:
    """盤面の駒を先手/後手入れ替え.

    駒ID: 0=空, 1-14=先手, 15-28=後手

    Args:
        board: 盤面テンソル (81,) 駒種ID

    Returns:
        先後入れ替えた盤面テンソル (81,)
    """
    flipped = board.clone()

    # 先手の駒 (1-14) → 後手の駒 (15-28)
    black_mask = (board >= 1) & (board <= 14)
    flipped[black_mask] = board[black_mask] + 14

    # 後手の駒 (15-28) → 先手の駒 (1-14)
    white_mask = (board >= 15) & (board <= 28)
    flipped[white_mask] = board[white_mask] - 14

    return flipped


def flip_board_vertical(board: torch.Tensor) -> torch.Tensor:
    """盤面を上下反転（手番反転時に使用）.

    Args:
        board: 盤面テンソル (81,) 駒種ID

    Returns:
        上下反転した盤面テンソル (81,)
    """
    board_2d = board.view(9, 9)
    flipped = torch.flip(board_2d, dims=[0])
    return flipped.view(81)


def normalize_to_black_view(
    board: torch.Tensor,
    hand: torch.Tensor,
    turn: torch.Tensor,
    value: float,
    outcome: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    """後手番の局面を先手視点に正規化.

    後手番の場合:
    - 盤面を180度回転（上下左右反転）
    - 駒の先後を入れ替え
    - 持ち駒の先後を入れ替え
    - 評価値・勝敗は「手番側視点」のラベルなので変更しない
      （先後入替後は新しい先手＝元の手番側であり、視点が一致するため）

    Args:
        board: 盤面テンソル (81,)
        hand: 持ち駒テンソル (14,)
        turn: 手番テンソル ()
        value: 正規化評価値（手番側視点）
        outcome: 勝敗ラベル（手番側視点）

    Returns:
        正規化された (board, hand, turn, value, outcome)
    """
    if turn.item() == 0:
        # 先手番: そのまま
        return board, hand, turn, value, outcome

    # 後手番: 先手視点に変換
    # 180度回転 = 上下反転 + 左右反転
    flipped_board = flip_board_vertical(flip_board_horizontal(board))
    # 駒の先後入れ替え
    flipped_board = flip_board_turn(flipped_board)
    # 持ち駒の先後入れ替え
    flipped_hand = flip_hand(hand)
    # 手番を先手に
    new_turn = torch.tensor(0, dtype=torch.long)

    return flipped_board, flipped_hand, new_turn, value, outcome


def augment_horizontal_flip(
    board: torch.Tensor,
    hand: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """左右反転によるデータ拡張.

    将棋は左右対称なので、盤面を左右反転してもルール上有効。

    Args:
        board: 盤面テンソル (81,)
        hand: 持ち駒テンソル (14,)

    Returns:
        左右反転した (board, hand)
    """
    return flip_board_horizontal(board), hand  # 持ち駒は変わらない


class ShogiValueDataset(Dataset):
    """将棋局面評価値データセット.

    JONLファイルから局面と評価値を読み込み、モデル入力形式に変換する。

    Args:
        data_path: JONLデータファイルのパス
        cp_scale: centipawn正規化のスケール（デフォルト: 1200）
        use_features: 拡張特徴量を使用するかどうか（デフォルト: False）
        cp_noise: 評価値に加えるノイズの標準偏差（デフォルト: 0、無効）
        cp_filter_threshold: 評価値フィルタの閾値（デフォルト: None、無効）
        normalize_turn: 後手番を先手視点に正規化（デフォルト: False）
        augment_flip: 左右反転でデータ拡張（デフォルト: False）
        drop_zero_cp: score_cp==0の局面を除外（デフォルト: False）。
            旧方式（--random-type engine）で生成したデータはランダム手の局面に
            ダミーの評価値0が記録されているため、その除去に使用する。
        cp_clamp: 評価値を±この値に丸める（デフォルト: None、無効）。
            cp_filter_thresholdの「除外」と異なり、大差局面を学習に残せる。
        target_mode: 評価値ターゲットの空間（デフォルト: "cp"）。
            "cp": tanh(score_cp / cp_scale)
            "wdl": 勝率 sigmoid(score_cp / wdl_scale) と実際の勝敗を
                   elmo式にブレンドし、[-1, 1]にマップ
        wdl_scale: cp→勝率変換のシグモイドスケール（デフォルト: 600）
        wdl_lambda: elmoブレンドの教師評価値の重み（デフォルト: 0.5）。
            target = wdl_lambda * 評価値勝率 + (1 - wdl_lambda) * 勝敗
    """

    def __init__(
        self,
        data_path: str | Path,
        cp_scale: float = 1200.0,
        use_features: bool = False,
        cp_noise: float = 0.0,
        cp_filter_threshold: float | None = None,
        normalize_turn: bool = False,
        augment_flip: bool = False,
        drop_zero_cp: bool = False,
        cp_clamp: float | None = None,
        target_mode: str = "cp",
        wdl_scale: float = 600.0,
        wdl_lambda: float = 0.5,
    ) -> None:
        if target_mode not in ("cp", "wdl"):
            raise ValueError(f"Unknown target_mode: {target_mode}")

        self.data_path = Path(data_path)
        self.cp_scale = cp_scale
        self.use_features = use_features
        self.cp_noise = cp_noise
        self.cp_filter_threshold = cp_filter_threshold
        self.normalize_turn = normalize_turn
        self.augment_flip = augment_flip
        self.drop_zero_cp = drop_zero_cp
        self.cp_clamp = cp_clamp
        self.target_mode = target_mode
        self.wdl_scale = wdl_scale
        self.wdl_lambda = wdl_lambda
        self.samples: list[dict] = []

        self._load_data()

    def _load_data(self) -> None:
        """データファイルを読み込む."""
        with open(self.data_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)

                # 評価値フィルタ: 極端な評価値を除外
                if self.cp_filter_threshold is not None:
                    score_cp = sample.get("score_cp", 0)
                    if abs(score_cp) > self.cp_filter_threshold:
                        continue

                # ダミーラベル除去: 評価値0の局面を除外
                if self.drop_zero_cp and sample.get("score_cp", 0) == 0:
                    continue

                self.samples.append(sample)

    def __len__(self) -> int:
        base_len = len(self.samples)
        if self.augment_flip:
            return base_len * 2  # 元データ + 左右反転
        return base_len

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """指定インデックスのサンプルを取得.

        Args:
            idx: サンプルインデックス

        Returns:
            dict with keys:
                - board: 盤面テンソル (81,)
                - hand: 持ち駒テンソル (14,)
                - turn: 手番テンソル ()
                - value: 正規化評価値テンソル ()
                - outcome: 勝敗ラベル (1.0: 手番側勝ち, 0.0: 手番側負け, 0.5: 引き分け)
                - features: 拡張特徴量テンソル (81, 10) [use_features=True時のみ]
        """
        # 左右反転拡張: 後半のインデックスは反転版
        apply_flip = False
        if self.augment_flip:
            base_len = len(self.samples)
            if idx >= base_len:
                idx = idx - base_len
                apply_flip = True

        sample = self.samples[idx]
        sfen = sample["sfen"]
        score_cp = sample["score_cp"]

        # 評価値ノイズ付与（学習時の過学習抑制）
        if self.cp_noise > 0:
            score_cp = score_cp + random.gauss(0, self.cp_noise)

        # 評価値クランプ（大差局面を除外せず丸めて学習に残す）
        if self.cp_clamp is not None:
            score_cp = max(-self.cp_clamp, min(self.cp_clamp, score_cp))

        # SFENをパース
        parsed = parse_sfen(sfen)

        # 勝敗ラベルを手番視点に変換
        result_str = sample.get("result", "draw")
        ply = sample.get("ply", 0)
        is_black_turn = (ply % 2 == 0)  # 偶数手目は先手番

        if result_str == "black_win":
            outcome = 1.0 if is_black_turn else 0.0
        elif result_str == "white_win":
            outcome = 0.0 if is_black_turn else 1.0
        else:  # draw
            outcome = 0.5

        # 評価値ターゲットを計算
        if self.target_mode == "wdl":
            # elmo式: 教師評価値の勝率と実際の勝敗をブレンドし、[-1, 1]にマップ
            eval_wr = cp_to_wdl(score_cp, self.wdl_scale)
            blended = self.wdl_lambda * eval_wr + (1.0 - self.wdl_lambda) * outcome
            value = 2.0 * blended - 1.0
        else:
            value = normalize_cp(score_cp, self.cp_scale)

        board = parsed.board
        hand = parsed.hand
        turn = parsed.turn

        # 手番正規化: 後手番を先手視点に変換
        if self.normalize_turn:
            board, hand, turn, value, outcome = normalize_to_black_view(
                board, hand, turn, value, outcome
            )

        # 左右反転拡張
        if apply_flip:
            board, hand = augment_horizontal_flip(board, hand)

        result = {
            "board": board,
            "hand": hand,
            "turn": turn,
            "value": torch.tensor(value, dtype=torch.float32),
            "outcome": torch.tensor(outcome, dtype=torch.float32),
        }

        # 拡張特徴量を追加（変換後の盤面から計算）
        if self.use_features:
            features = compute_all_features(board)
            # (81, 10) のテンソルにまとめる
            # [attack(2), king_dist(2), piece_value(1), control(1), king_safety(4)]
            result["features"] = torch.cat([
                features["attack_map"],        # (81, 2)
                features["king_distance"],     # (81, 2)
                features["piece_value"].unsqueeze(1),  # (81, 1)
                features["control"].unsqueeze(1),      # (81, 1)
                features["king_safety"],       # (81, 4)
            ], dim=1)

        return result


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """バッチをまとめる関数.

    Args:
        batch: サンプルのリスト

    Returns:
        バッチ化されたテンソルの辞書
    """
    result = {
        "board": torch.stack([s["board"] for s in batch]),
        "hand": torch.stack([s["hand"] for s in batch]),
        "turn": torch.stack([s["turn"] for s in batch]),
        "value": torch.stack([s["value"] for s in batch]),
        "outcome": torch.stack([s["outcome"] for s in batch]),
    }

    # 拡張特徴量がある場合
    if "features" in batch[0]:
        result["features"] = torch.stack([s["features"] for s in batch])

    return result
