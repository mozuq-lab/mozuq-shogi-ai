"""Value Network学習スクリプト.

使用例:
    # Mac環境での動作確認（小規模）
    python train/train.py --data data/raw/hao_trial_500_depth10.jsonl --epochs 5 --batch-size 64

    # Windows環境での本格学習
    python train/train.py --data data/raw/large_dataset.jsonl --epochs 100 --batch-size 512 --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

# プロジェクトルートをパスに追加（PYTHONPATH指定なしでも実行可能に）
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from models import (
    ShogiDeltaPairDataset,
    ShogiRankingPairDataset,
    ShogiValueDataset,
    ValueTransformer,
    collate_fn,
    ranking_collate_fn,
)

if TYPE_CHECKING:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """学習設定."""

    # データ
    data_path: str = "data/raw/hao_trial_500_depth10.jsonl"
    val_split: float = 0.1

    # モデル
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    ffn_dim: int = 512
    dropout: float = 0.1

    # 学習
    epochs: int = 100
    batch_size: int = 512
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 5

    # 評価値正規化
    cp_scale: float = 1200.0

    # デバイス
    device: str = "auto"

    # 保存
    output_dir: str = "checkpoints"
    save_every: int = 10
    log_every: int = 100

    # 再開
    resume: str | None = None

    # 拡張特徴量
    use_features: bool = False

    # Pooling方式
    use_attention_pooling: bool = True

    # モデル構造オプション
    use_king_relative: bool = False
    use_2d_pos: bool = False
    use_discrete_hand: bool = False

    # 勝敗補助損失
    aux_loss_weight: float = 0.1

    # 学習安定化
    grad_clip_norm: float = 1.0
    label_smoothing: float = 0.05

    # データ前処理
    cp_noise: float = 0.0
    cp_filter_threshold: float | None = None
    drop_zero_cp: bool = False
    cp_clamp: float | None = None

    # 評価値ターゲット（cp: tanh正規化, wdl: 勝率空間でelmoブレンド）
    target_mode: str = "cp"
    wdl_scale: float = 600.0
    wdl_lambda: float = 0.5

    # データ拡張
    normalize_turn: bool = False
    augment_flip: bool = False

    # データローダー
    num_workers: int = 0

    # 分散正則化（デフォルト無効）
    # 過去にnormalize_turnのラベル符号バグで出力が定数に崩壊する問題があり、
    # その対症療法として導入された。バグ修正済みのため通常は不要。
    variance_reg_weight: float = 0.0

    # EMA (Exponential Moving Average)
    # 0で無効。有効時（推奨: 0.999）はvalidateとbest.pt保存にEMA重みを使用
    ema_decay: float = 0.0

    # 評価値損失の種類（mse: 二乗誤差, huber: 教師の探索ノイズにロバスト）
    value_loss: str = "mse"
    huber_delta: float = 0.5

    # Pairwise ranking損失（0=無効）
    # candidatesフィールド付きデータ（gen_dataset.py --multipv 2以上）が必要。
    # 同一親局面の候補手ペアについて「良い手の子局面の評価が悪い手より
    # 高くなる（手番側視点では低くなる）」関係を学習する
    ranking_weight: float = 0.0
    ranking_min_gap: float = 30.0

    # ΔV差分回帰損失（0=無効）
    # candidatesペア（および--delta-dataの摂動ペア）の評価値差分を
    # Huber回帰する。rankingが順序のみ学ぶのに対し差分量まで合わせる
    delta_weight: float = 0.0

    # 摂動ペアデータ（gen_perturb_pairs.py出力、Noneで無効）
    # delta_weight > 0 のとき、candidatesペアに加えて摂動ペアのΔVも学習する
    delta_data: str | None = None
    # material_diff==0（素材一致）のペアのみ使用
    delta_same_material_only: bool = False

    # オフライン指し手一致率の計測（Noneで無効）
    # candidatesフィールド付きJSONLを指定すると、定期checkpoint保存時と
    # 学習終了時にrank1手との一致率を計測してログに記録する
    agreement_data: str | None = None
    agreement_limit: int = 200


@dataclass
class TrainState:
    """学習状態."""

    epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float("inf")
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    # ranking検証メトリクス（epochごと: {"epoch", "loss", "accuracy"}）
    ranking_val: list[dict] = field(default_factory=list)
    # 指し手一致率の履歴（{"epoch", "agreement", ...}）
    agreement: list[dict] = field(default_factory=list)
    # 摂動ペア検証メトリクス（epochごと: {"epoch", "mae"}）
    delta_val: list[dict] = field(default_factory=list)


def compute_value_loss(
    value: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "mse",
    huber_delta: float = 0.5,
) -> torch.Tensor:
    """評価値損失を計算.

    Args:
        value: モデル出力 (batch, 1)
        target: ターゲット (batch, 1)
        loss_type: 損失の種類（"mse" または "huber"）
        huber_delta: Huber lossの遷移点

    Returns:
        損失テンソル（スカラー）
    """
    if loss_type == "huber":
        return nn.functional.huber_loss(value, target, delta=huber_delta)
    if loss_type == "mse":
        return nn.functional.mse_loss(value, target)
    raise ValueError(f"Unknown value_loss: {loss_type}")


def compute_outcome_loss(
    outcome: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """重み付き勝敗BCE損失を計算.

    pv_leaf/multipvなどの分岐局面には本譜の勝敗が当てはまらないため、
    outcome_weight=0として損失から除外する。

    Args:
        outcome: 勝率予測 (batch, 1)
        target: 勝敗ターゲット (batch, 1)
        weight: サンプル重み (batch, 1)。0のサンプルは損失に寄与しない

    Returns:
        重み付き平均BCE損失（有効サンプルが無ければ0）
    """
    bce = nn.functional.binary_cross_entropy(outcome, target, reduction="none")
    total = weight.sum()
    if total.item() == 0.0:
        return bce.sum() * 0.0
    return (bce * weight).sum() / total


def compute_ranking_loss(
    value_better: torch.Tensor,
    value_worse: torch.Tensor,
) -> torch.Tensor:
    """Pairwise ranking損失（RankNet形式）を計算.

    ペアの各要素は「親局面で候補手を1手指した後の子局面」の評価値で、
    子局面の手番側（=親の相手側）視点。親にとって良い手ほど子局面の
    評価値は低くなるべきなので、value_better < value_worse を目指す。

    loss = -log sigmoid(value_worse - value_better)
         = softplus(value_better - value_worse)

    Args:
        value_better: 良い手の子局面の評価値 (batch,)
        value_worse: 悪い手の子局面の評価値 (batch,)

    Returns:
        損失テンソル（スカラー）
    """
    return nn.functional.softplus(value_better - value_worse).mean()


def compute_delta_loss(
    value_a: torch.Tensor,
    value_b: torch.Tensor,
    delta_target: torch.Tensor,
    huber_delta: float = 0.5,
) -> torch.Tensor:
    """ΔV（ペア間の評価値差分）のHuber損失を計算.

    絶対値の一致ではなく「似た局面間の評価差」を教師差分に合わせる。
    1手読みの手選びはargmaxで決まるため、差分方向の誤差が着手を
    直接左右する。ラベルを完全に学習できれば差分も合うが、有限容量では
    この損失が差分方向の誤差を優先的に抑える再重み付けとして働く。

    Args:
        value_a: ペアA側の評価値 (batch,)
        value_b: ペアB側の評価値 (batch,)
        delta_target: value_a − value_b の教師ターゲット (batch,)
        huber_delta: Huber lossの遷移点

    Returns:
        損失テンソル（スカラー）
    """
    return nn.functional.huber_loss(
        value_a - value_b, delta_target, delta=huber_delta
    )


def _ranking_forward(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """rankingペアバッチを1回のforwardで評価し、(良い手側, 悪い手側)の値を返す."""
    n = batch["board_a"].size(0)
    board = torch.cat([batch["board_a"], batch["board_b"]]).to(device)
    hand = torch.cat([batch["hand_a"], batch["hand_b"]]).to(device)
    turn = torch.cat([batch["turn_a"], batch["turn_b"]]).to(device)
    features = None
    if "features_a" in batch:
        features = torch.cat([batch["features_a"], batch["features_b"]]).to(device)

    value, _ = model(board, hand, turn, features)
    value = value.view(-1)
    return value[:n], value[n:]


def split_by_game(
    dataset: ShogiValueDataset,
    val_split: float,
    seed: int = 42,
) -> tuple[Subset, Subset]:
    """対局（game_id）単位で訓練/検証データを分割.

    同一対局内の局面は強く相関するため、ランダム分割では
    train/valに近傍局面が跨りval_lossが楽観的になる。
    game_id単位で分割することでリークを防ぐ。

    augment_flip有効時は、反転版サンプル（インデックス後半）も
    元サンプルと同じ側に割り当てる。

    Args:
        dataset: 分割対象のデータセット
        val_split: 検証データの割合
        seed: シャッフル用シード

    Returns:
        (訓練Subset, 検証Subset)
    """
    game_ids = [s.get("game_id", 0) for s in dataset.samples]
    val_games = select_val_games(game_ids, val_split, seed)

    if val_games is None:
        # game_idが無い/1対局のみの場合はランダム分割にフォールバック
        logger.warning(
            "game_idが不足しているためランダム分割にフォールバックします"
        )
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_subset, val_subset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )
        return train_subset, val_subset

    base_len = len(dataset.samples)
    train_indices: list[int] = []
    val_indices: list[int] = []
    for idx, gid in enumerate(game_ids):
        if gid in val_games:
            val_indices.append(idx)
        else:
            train_indices.append(idx)

    # augment_flip有効時: 反転版（idx + base_len）も同じ側に割り当てる
    if dataset.augment_flip:
        train_indices += [idx + base_len for idx in train_indices]
        val_indices += [idx + base_len for idx in val_indices]

    n_games = len(set(game_ids))
    logger.info(
        f"Game-level split: {n_games - len(val_games)} games (train) / "
        f"{len(val_games)} games (val)"
    )
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def select_val_games(
    game_ids: list[int],
    val_split: float,
    seed: int = 42,
) -> set[int] | None:
    """検証に割り当てる対局IDの集合を決定する.

    split_by_gameと同じseedで呼べば同じ分割が得られるため、
    rankingペアデータセット等で同一の対局分割を共有できる。

    Args:
        game_ids: 各サンプルのgame_idのリスト
        val_split: 検証データの割合
        seed: 分割用シード

    Returns:
        検証用game_idの集合。対局数が2未満の場合None（ランダム分割にフォールバック）
    """
    unique_games = sorted(set(game_ids))
    if len(unique_games) < 2:
        return None

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(unique_games), generator=generator).tolist()
    n_val_games = max(1, int(len(unique_games) * val_split))
    return {unique_games[i] for i in perm[:n_val_games]}


def get_device(device_str: str) -> torch.device:
    """デバイスを取得."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: TrainConfig,
    state: TrainState,
    ema_model: AveragedModel | None = None,
) -> None:
    """チェックポイントを保存.

    EMA有効時はmodel_state_dictにEMA重み（推論用）を、
    raw_model_state_dictに生の重み（学習再開用）を保存する。
    """
    checkpoint = {
        "model_state_dict": (
            ema_model.module.state_dict() if ema_model is not None
            else model.state_dict()
        ),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": vars(config),
        "state": vars(state),
    }
    if ema_model is not None:
        checkpoint["raw_model_state_dict"] = model.state_dict()
    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved: {path}")


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ema_model: AveragedModel | None = None,
) -> tuple[TrainConfig, TrainState]:
    """チェックポイントを読み込み.

    EMA付きcheckpointの場合、modelには生の重み（raw_model_state_dict）、
    ema_modelにはEMA重み（model_state_dict）を復元する。
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    raw_state_dict = checkpoint.get(
        "raw_model_state_dict", checkpoint["model_state_dict"]
    )
    model.load_state_dict(raw_state_dict)

    if ema_model is not None:
        ema_model.module.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    config = TrainConfig(**checkpoint["config"])
    state = TrainState(**checkpoint["state"])

    logger.info(f"Checkpoint loaded: {path} (epoch {state.epoch})")
    return config, state


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    state: TrainState,
    log_every: int,
    use_features: bool = False,
    aux_loss_weight: float = 0.1,
    grad_clip_norm: float = 1.0,
    label_smoothing: float = 0.05,
    variance_reg_weight: float = 0.0,
    ema_model: AveragedModel | None = None,
    value_loss_type: str = "mse",
    huber_delta: float = 0.5,
    ranking_loader: DataLoader | None = None,
    ranking_weight: float = 0.0,
    delta_weight: float = 0.0,
    delta_loader: DataLoader | None = None,
) -> float:
    """1エポック学習."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    # rankingペアはメインバッチ1回につき1バッチ消費（尽きたら周回）
    ranking_iter = iter(ranking_loader) if ranking_loader is not None else None
    # 摂動ペア（--delta-data）も同様にメインバッチ1回につき1バッチ消費
    delta_iter = iter(delta_loader) if delta_loader is not None else None

    for batch in loader:
        board = batch["board"].to(device)
        hand = batch["hand"].to(device)
        turn = batch["turn"].to(device)
        target_value = batch["value"].to(device).unsqueeze(1)
        target_outcome = batch["outcome"].to(device).unsqueeze(1)
        outcome_weight = batch["outcome_weight"].to(device).unsqueeze(1)
        features = batch.get("features")
        if features is not None:
            features = features.to(device)

        # Label Smoothing: [0, 1] → [smoothing, 1 - smoothing]
        smoothed_outcome = target_outcome * (1 - 2 * label_smoothing) + label_smoothing

        optimizer.zero_grad()
        value, outcome = model(board, hand, turn, features)

        # デバッグ: 最初の数ステップと定期的に形状と値を確認
        is_debug_step = state.global_step < 5 or state.global_step % 1000 == 0
        if is_debug_step:
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(f"--- Debug Step {state.global_step} (lr={current_lr:.2e}) ---")
            logger.info(f"Value: mean={value.mean().item():.4f}, std={value.std().item():.4f}, min={value.min().item():.4f}, max={value.max().item():.4f}")
            logger.info(f"Target: mean={target_value.mean().item():.4f}, std={target_value.std().item():.4f}")

        # 評価値損失（主タスク）
        # 念のため形状を再確認して強制的に合わせる
        value = value.view(-1, 1)
        target_value = target_value.view(-1, 1)
        value_loss = compute_value_loss(
            value, target_value, value_loss_type, huber_delta
        )

        # 勝敗損失（補助タスク、Label Smoothing適用、分岐局面は重み0で除外）
        outcome = outcome.view(-1, 1)
        smoothed_outcome = smoothed_outcome.view(-1, 1)
        outcome_loss = compute_outcome_loss(
            outcome, smoothed_outcome, outcome_weight
        )

        # 分散維持の正則化（定数出力への崩壊を防ぐ）
        # 出力の標準偏差が小さいとペナルティ（負の項なので最大化される）
        variance_reg = value.std()

        # Pairwise ranking損失 + ΔV差分回帰損失（同じforwardを共有）
        ranking_loss: torch.Tensor | None = None
        delta_loss: torch.Tensor | None = None
        if ranking_iter is not None:
            try:
                ranking_batch = next(ranking_iter)
            except StopIteration:
                ranking_iter = iter(ranking_loader)
                ranking_batch = next(ranking_iter)
            value_better, value_worse = _ranking_forward(
                model, ranking_batch, device
            )
            if ranking_weight > 0:
                ranking_loss = compute_ranking_loss(value_better, value_worse)
            if delta_weight > 0 and "delta_target" in ranking_batch:
                delta_loss = compute_delta_loss(
                    value_better, value_worse,
                    ranking_batch["delta_target"].to(device),
                    huber_delta,
                )

        # 摂動ペアのΔV差分回帰損失
        perturb_loss: torch.Tensor | None = None
        if delta_iter is not None:
            try:
                delta_batch = next(delta_iter)
            except StopIteration:
                delta_iter = iter(delta_loader)
                delta_batch = next(delta_iter)
            value_a, value_b = _ranking_forward(model, delta_batch, device)
            perturb_loss = compute_delta_loss(
                value_a, value_b,
                delta_batch["delta_target"].to(device),
                huber_delta,
            )

        # 合計損失
        # variance_regを引くことで、stdを大きく維持するインセンティブを与える
        loss = value_loss + aux_loss_weight * outcome_loss - variance_reg_weight * variance_reg
        if ranking_loss is not None:
            loss = loss + ranking_weight * ranking_loss
        if delta_loss is not None:
            loss = loss + delta_weight * delta_loss
        if perturb_loss is not None:
            loss = loss + delta_weight * perturb_loss
        loss.backward()

        # デバッグ: 勾配の確認
        if is_debug_step:
            grad_norm = sum(p.grad.norm() for p in model.parameters() if p.grad is not None)
            logger.info(f"Total grad norm: {grad_norm:.4f}")

        # Gradient Clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

        optimizer.step()

        # EMA更新
        if ema_model is not None:
            ema_model.update_parameters(model)

        total_loss += loss.item()
        num_batches += 1
        state.global_step += 1

        if state.global_step % log_every == 0:
            ranking_str = (
                f", ranking={ranking_loss.item():.6f}"
                if ranking_loss is not None else ""
            )
            delta_str = (
                f", delta={delta_loss.item():.6f}"
                if delta_loss is not None else ""
            )
            perturb_str = (
                f", perturb={perturb_loss.item():.6f}"
                if perturb_loss is not None else ""
            )
            logger.info(
                f"Step {state.global_step}: loss={loss.item():.6f} "
                f"(value={value_loss.item():.6f}, outcome={outcome_loss.item():.6f}, "
                f"var_reg={variance_reg.item():.4f}{ranking_str}{delta_str}{perturb_str})"
            )

    return total_loss / num_batches


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_features: bool = False,
    aux_loss_weight: float = 0.1,
    value_loss_type: str = "mse",
    huber_delta: float = 0.5,
) -> float:
    """バリデーション."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        board = batch["board"].to(device)
        hand = batch["hand"].to(device)
        turn = batch["turn"].to(device)
        target_value = batch["value"].to(device).unsqueeze(1)
        target_outcome = batch["outcome"].to(device).unsqueeze(1)
        outcome_weight = batch["outcome_weight"].to(device).unsqueeze(1)
        features = batch.get("features")
        if features is not None:
            features = features.to(device)

        value, outcome = model(board, hand, turn, features)

        # 評価値損失（主タスク）
        value_loss = compute_value_loss(
            value, target_value, value_loss_type, huber_delta
        )

        # 勝敗損失（補助タスク、分岐局面は重み0で除外）
        outcome_loss = compute_outcome_loss(
            outcome, target_outcome, outcome_weight
        )

        # 合計損失
        loss = value_loss + aux_loss_weight * outcome_loss

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


@torch.no_grad()
def validate_ranking(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float]:
    """rankingペアの検証（損失・ペア正答率・ΔV平均絶対誤差）.

    ペア正答率は「良い手の子局面の評価値が悪い手より低い」割合で、
    1手読みの手選びの正しさを直接反映する指標。delta_maeは
    差分量の予測誤差（ΔV回帰の検証指標）。

    Args:
        model: 評価するモデル
        loader: rankingペアの検証ローダー
        device: デバイス

    Returns:
        (ranking損失, ペア正答率, ΔV平均絶対誤差)
    """
    model.eval()
    total_loss = 0.0
    total_delta_err = 0.0
    correct = 0
    total = 0

    for batch in loader:
        value_better, value_worse = _ranking_forward(model, batch, device)
        loss = compute_ranking_loss(value_better, value_worse)
        total_loss += loss.item() * value_better.size(0)
        correct += (value_better < value_worse).sum().item()
        total += value_better.size(0)
        if "delta_target" in batch:
            target = batch["delta_target"].to(device)
            total_delta_err += (
                (value_better - value_worse - target).abs().sum().item()
            )

    if total == 0:
        return 0.0, 0.0, 0.0
    return total_loss / total, correct / total, total_delta_err / total


@torch.no_grad()
def validate_delta(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """摂動ペアのΔV平均絶対誤差を計算.

    Args:
        model: 評価するモデル
        loader: 摂動ペアの検証ローダー
        device: デバイス

    Returns:
        ΔV平均絶対誤差
    """
    model.eval()
    total_err = 0.0
    total = 0
    for batch in loader:
        value_a, value_b = _ranking_forward(model, batch, device)
        target = batch["delta_target"].to(device)
        total_err += (value_a - value_b - target).abs().sum().item()
        total += value_a.size(0)
    return total_err / total if total else 0.0


def agreement_data_overlaps(data_path: str, agreement_data: str | None) -> bool:
    """一致率計測データが学習データと同一ファイルかどうか判定.

    学習データでの計測は生成条件レベルの過適合を検出できないため、
    独立したholdoutデータの使用を促す警告に使う。

    Args:
        data_path: 学習データのパス
        agreement_data: 一致率計測データのパス（未指定はNone）

    Returns:
        同一ファイルならTrue
    """
    if not agreement_data:
        return False
    try:
        return Path(agreement_data).resolve() == Path(data_path).resolve()
    except OSError:
        return False


def run_offline_agreement(
    checkpoint_path: Path,
    data_path: str,
    limit: int,
    device: str,
) -> dict:
    """保存済みcheckpointでオフライン指し手一致率を計測.

    Evaluator（推論と同じ経路）を通すことで、学習時と推論時の
    前処理の食い違いも検出できる。

    Args:
        checkpoint_path: 計測対象のcheckpointパス
        data_path: candidatesフィールド付きJSONLデータのパス
        limit: 測定局面数の上限
        device: 推論デバイス

    Returns:
        一致率の集計結果
    """
    # エンジン関連の依存を学習時のみ読み込む
    from engine.evaluator import Evaluator
    from scripts.move_agreement import measure_agreement_offline

    evaluator = Evaluator(checkpoint_path, device=device)
    return measure_agreement_offline(evaluator, data_path, limit=limit)


def write_log(log_path: Path, config: TrainConfig, state: TrainState) -> None:
    """学習ログをJSONファイルに書き出す.

    Args:
        log_path: 出力先のJSONファイルパス
        config: 学習設定
        state: 学習状態
    """
    log_data = {
        "config": vars(config),
        "train_losses": state.train_losses,
        "val_losses": state.val_losses,
        "best_val_loss": state.best_val_loss,
        "ranking_val": state.ranking_val,
        "agreement": state.agreement,
        "delta_val": state.delta_val,
    }
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)


def _make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    collate: Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]],
    config: TrainConfig,
    device: torch.device,
) -> DataLoader:
    """共通設定でDataLoaderを構築する.

    num_workers/pin_memory/persistent_workersの組み合わせを一箇所に集約する。

    Args:
        dataset: ラップするデータセット
        batch_size: バッチサイズ
        shuffle: サンプル順をシャッフルするかどうか
        collate: バッチ構築に使うcollate関数
        config: 学習設定（num_workersを参照）
        device: 実行デバイス（pin_memoryの判定に使用）

    Returns:
        構築されたDataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
    )


def _build_main_loaders(
    config: TrainConfig, device: torch.device
) -> tuple[ShogiValueDataset, DataLoader, DataLoader]:
    """メインデータセットとtrain/val用DataLoaderを構築する.

    対局（game_id）単位でtrain/valを分割し、同一対局の局面が
    train/valに跨るリークを防ぐ。

    Args:
        config: 学習設定
        device: 実行デバイス

    Returns:
        (データセット全体, 訓練用DataLoader, 検証用DataLoader)
    """
    logger.info(f"Loading dataset: {config.data_path}")
    logger.info(f"Use features: {config.use_features}")
    dataset = ShogiValueDataset(
        config.data_path,
        cp_scale=config.cp_scale,
        use_features=config.use_features,
        cp_noise=config.cp_noise,
        cp_filter_threshold=config.cp_filter_threshold,
        normalize_turn=config.normalize_turn,
        augment_flip=config.augment_flip,
        drop_zero_cp=config.drop_zero_cp,
        cp_clamp=config.cp_clamp,
        target_mode=config.target_mode,
        wdl_scale=config.wdl_scale,
        wdl_lambda=config.wdl_lambda,
    )
    logger.info(f"Dataset size: {len(dataset)}")

    # 訓練/検証分割（対局単位でリークを防ぐ）
    train_dataset, val_dataset = split_by_game(dataset, config.val_split)
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = _make_loader(
        train_dataset, config.batch_size, True, collate_fn, config, device
    )
    val_loader = _make_loader(
        val_dataset, config.batch_size, False, collate_fn, config, device
    )

    return dataset, train_loader, val_loader


def _build_ranking_loaders(
    config: TrainConfig,
    dataset: ShogiValueDataset,
    device: torch.device,
) -> tuple[DataLoader | None, DataLoader | None]:
    """rankingペアのtrain/val用DataLoaderを構築する.

    config.ranking_weight <= 0 かつ config.delta_weight <= 0の場合は
    (None, None)を返す。メインデータセットと同じ対局分割
    （select_val_games）を共有することで、rankingペアだけが検証局面を
    訓練に使うリークを防ぐ。

    Args:
        config: 学習設定
        dataset: メインのShogiValueDataset（game_id分割の共有に使用）
        device: 実行デバイス

    Returns:
        (rankingペア訓練用DataLoader, rankingペア検証用DataLoader)。
        ranking_weight <= 0 かつ delta_weight <= 0の場合は(None, None)。

    Raises:
        ValueError: game_idが2対局以上ないため対局単位の分割を共有できない場合、
            またはrankingペアが1つも構築できない場合。
    """
    if config.ranking_weight <= 0 and config.delta_weight <= 0:
        return None, None

    game_ids = [s.get("game_id", 0) for s in dataset.samples]
    val_games = select_val_games(game_ids, config.val_split)
    if val_games is None:
        # 対局単位の分割を共有できないと、rankingペアだけが
        # 検証局面を訓練に使うリークが起きるため明示エラーにする
        raise ValueError(
            "--ranking-weight/--delta-weightには"
            "game_idが2対局以上あるデータが必要です"
        )

    ranking_train_dataset = ShogiRankingPairDataset(
        config.data_path,
        use_features=config.use_features,
        normalize_turn=config.normalize_turn,
        augment_flip=config.augment_flip,
        min_gap_cp=config.ranking_min_gap,
        exclude_game_ids=val_games,
        cp_scale=config.cp_scale,
        target_mode=config.target_mode,
        wdl_scale=config.wdl_scale,
    )
    if len(ranking_train_dataset) == 0:
        raise ValueError(
            "rankingペアが構築できません。--ranking-weight/--delta-weightには"
            "candidatesフィールド付きデータ（gen_dataset.py --multipv 2以上）"
            "が必要です"
        )

    # ペアは1サンプル2局面なので、バッチサイズを半分にして
    # 1ステップあたりのforward局面数をメインバッチと揃える
    ranking_batch_size = max(1, config.batch_size // 2)
    ranking_train_loader = _make_loader(
        ranking_train_dataset,
        ranking_batch_size,
        True,
        ranking_collate_fn,
        config,
        device,
    )

    ranking_val_loader: DataLoader | None = None
    n_val_pairs = 0
    if val_games is not None:
        ranking_val_dataset = ShogiRankingPairDataset(
            config.data_path,
            use_features=config.use_features,
            normalize_turn=config.normalize_turn,
            augment_flip=False,
            min_gap_cp=config.ranking_min_gap,
            include_game_ids=val_games,
            cp_scale=config.cp_scale,
            target_mode=config.target_mode,
            wdl_scale=config.wdl_scale,
        )
        n_val_pairs = len(ranking_val_dataset)
        if n_val_pairs > 0:
            ranking_val_loader = _make_loader(
                ranking_val_dataset,
                ranking_batch_size,
                False,
                ranking_collate_fn,
                config,
                device,
            )
    logger.info(
        f"Ranking pairs: train={len(ranking_train_dataset)}, "
        f"val={n_val_pairs} (min_gap={config.ranking_min_gap}cp)"
    )

    return ranking_train_loader, ranking_val_loader


def _build_delta_loaders(
    config: TrainConfig,
    dataset: ShogiValueDataset,
    device: torch.device,
) -> tuple[DataLoader | None, DataLoader | None]:
    """摂動ペア（--delta-data）のtrain/val用DataLoaderを構築する.

    メインデータセットと同じ対局分割を共有する。gen_perturb_pairs.pyは
    入力JSONLのgame_idをペアへ引き継ぐため、学習データと同じJSONLから
    生成したペアであれば検証対局由来のペアが訓練に混ざらない。

    Args:
        config: 学習設定
        dataset: メインのShogiValueDataset（game_id分割の共有に使用）
        device: 実行デバイス

    Returns:
        (訓練用DataLoader, 検証用DataLoader)。delta_data未指定または
        delta_weight <= 0の場合は(None, None)。

    Raises:
        ValueError: 対局単位の分割を共有できない場合、
            またはペアが1つも構築できない場合。
    """
    if not config.delta_data or config.delta_weight <= 0:
        return None, None

    game_ids = [s.get("game_id", 0) for s in dataset.samples]
    val_games = select_val_games(game_ids, config.val_split)
    if val_games is None:
        raise ValueError(
            "--delta-dataにはgame_idが2対局以上あるデータが必要です"
        )

    dataset_kwargs = dict(
        use_features=config.use_features,
        normalize_turn=config.normalize_turn,
        cp_scale=config.cp_scale,
        target_mode=config.target_mode,
        wdl_scale=config.wdl_scale,
        same_material_only=config.delta_same_material_only,
    )
    delta_train_dataset = ShogiDeltaPairDataset(
        config.delta_data,
        augment_flip=config.augment_flip,
        exclude_game_ids=val_games,
        **dataset_kwargs,
    )
    if len(delta_train_dataset) == 0:
        raise ValueError(
            f"--delta-dataからペアを構築できません: {config.delta_data}"
        )

    pair_batch_size = max(1, config.batch_size // 2)
    delta_train_loader = _make_loader(
        delta_train_dataset, pair_batch_size, True, ranking_collate_fn,
        config, device,
    )

    delta_val_dataset = ShogiDeltaPairDataset(
        config.delta_data,
        augment_flip=False,
        include_game_ids=val_games,
        **dataset_kwargs,
    )
    delta_val_loader: DataLoader | None = None
    if len(delta_val_dataset) > 0:
        delta_val_loader = _make_loader(
            delta_val_dataset, pair_batch_size, False, ranking_collate_fn,
            config, device,
        )
    logger.info(
        f"Delta pairs: train={len(delta_train_dataset)}, "
        f"val={len(delta_val_dataset)}"
    )
    return delta_train_loader, delta_val_loader


def _build_model_and_optimizer(
    config: TrainConfig, device: torch.device
) -> tuple[
    nn.Module,
    AveragedModel | None,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
]:
    """モデル・EMAモデル・optimizer・schedulerを構築する.

    Args:
        config: 学習設定
        device: 実行デバイス

    Returns:
        (モデル, EMAモデル（ema_decay<=0の場合はNone）, optimizer, scheduler)
    """
    model = ValueTransformer(
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        ffn_dim=config.ffn_dim,
        dropout=config.dropout,
        use_features=config.use_features,
        use_attention_pooling=config.use_attention_pooling,
        use_king_relative=config.use_king_relative,
        use_2d_pos=config.use_2d_pos,
        use_discrete_hand=config.use_discrete_hand,
    ).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # EMAモデル
    ema_model: AveragedModel | None = None
    if config.ema_decay > 0:
        ema_model = AveragedModel(
            model, multi_avg_fn=get_ema_multi_avg_fn(config.ema_decay)
        )
        logger.info(f"EMA enabled: decay={config.ema_decay}")

    # オプティマイザ・スケジューラ
    optimizer = AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.epochs - config.warmup_epochs,
        eta_min=config.lr * 0.01,
    )

    return model, ema_model, optimizer, scheduler


def train(config: TrainConfig) -> None:
    """学習メイン処理."""
    # デバイス
    device = get_device(config.device)
    logger.info(f"Using device: {device}")

    if agreement_data_overlaps(config.data_path, config.agreement_data):
        logger.warning(
            "agreement_dataが学習データと同一ファイルです。"
            "計測は独立したholdoutデータで行うことを推奨します"
        )

    # データセットとメインのtrain/valローダー
    dataset, train_loader, val_loader = _build_main_loaders(config, device)

    # rankingペアデータセット（メインと同じ対局分割を共有してリークを防ぐ）
    ranking_train_loader, ranking_val_loader = _build_ranking_loaders(
        config, dataset, device
    )

    # 摂動ペアローダー（--delta-data、対局分割をメインと共有）
    delta_train_loader, delta_val_loader = _build_delta_loaders(
        config, dataset, device
    )

    # モデル・EMAモデル・オプティマイザ・スケジューラ
    model, ema_model, optimizer, scheduler = _build_model_and_optimizer(
        config, device
    )

    # 状態
    state = TrainState()

    # 再開
    if config.resume:
        resume_path = Path(config.resume)
        if resume_path.exists():
            _, state = load_checkpoint(
                resume_path, model, optimizer, scheduler, ema_model
            )

    # 出力ディレクトリ
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 学習ログ
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"log_{run_name}.json"

    # 学習ループ
    logger.info("Starting training...")
    start_time = time.time()

    for epoch in range(state.epoch, config.epochs):
        state.epoch = epoch
        epoch_start = time.time()

        # Warmup
        if epoch < config.warmup_epochs:
            warmup_lr = config.lr * (epoch + 1) / config.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group["lr"] = warmup_lr

        # 学習
        train_loss = train_epoch(
            model, train_loader, optimizer, device, state, config.log_every,
            use_features=config.use_features,
            aux_loss_weight=config.aux_loss_weight,
            grad_clip_norm=config.grad_clip_norm,
            label_smoothing=config.label_smoothing,
            variance_reg_weight=config.variance_reg_weight,
            ema_model=ema_model,
            value_loss_type=config.value_loss,
            huber_delta=config.huber_delta,
            ranking_loader=ranking_train_loader,
            ranking_weight=config.ranking_weight,
            delta_weight=config.delta_weight,
            delta_loader=delta_train_loader,
        )
        state.train_losses.append(train_loss)

        # バリデーション（EMA有効時はEMA重みで評価）
        eval_model = ema_model.module if ema_model is not None else model
        val_loss = validate(
            eval_model, val_loader, device,
            use_features=config.use_features,
            aux_loss_weight=config.aux_loss_weight,
            value_loss_type=config.value_loss,
            huber_delta=config.huber_delta,
        )
        state.val_losses.append(val_loss)

        # ranking検証（ペア正答率は1手読みの手選び精度の直接的な指標）
        if ranking_val_loader is not None:
            rank_loss, rank_acc, delta_mae = validate_ranking(
                eval_model, ranking_val_loader, device
            )
            state.ranking_val.append(
                {"epoch": epoch + 1, "loss": rank_loss,
                 "accuracy": rank_acc, "delta_mae": delta_mae}
            )
            logger.info(
                f"Ranking val: loss={rank_loss:.6f}, "
                f"pair_accuracy={rank_acc:.3f}, delta_mae={delta_mae:.4f}"
            )

        # 摂動ペア検証（ΔV差分の予測誤差）
        if delta_val_loader is not None:
            delta_mae = validate_delta(eval_model, delta_val_loader, device)
            state.delta_val.append({"epoch": epoch + 1, "mae": delta_mae})
            logger.info(f"Delta val: mae={delta_mae:.4f}")

        # スケジューラ更新（warmup後）
        if epoch >= config.warmup_epochs:
            scheduler.step()

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch + 1}/{config.epochs}: "
            f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
            f"lr={current_lr:.2e}, time={epoch_time:.1f}s"
        )

        # ベストモデル保存
        best_updated = False
        if val_loss < state.best_val_loss:
            state.best_val_loss = val_loss
            best_updated = True
            save_checkpoint(
                output_dir / "best.pt",
                model, optimizer, scheduler, config, state,
                ema_model=ema_model,
            )

        # 定期保存
        if (epoch + 1) % config.save_every == 0:
            checkpoint_path = output_dir / f"epoch_{epoch + 1:04d}.pt"
            save_checkpoint(
                checkpoint_path,
                model, optimizer, scheduler, config, state,
                ema_model=ema_model,
            )

            # 保存したcheckpointでオフライン指し手一致率を計測
            if config.agreement_data:
                result = run_offline_agreement(
                    checkpoint_path, config.agreement_data,
                    config.agreement_limit, config.device,
                )
                state.agreement.append({"epoch": epoch + 1, **result})
                logger.info(
                    f"Agreement (epoch {epoch + 1}): "
                    f"{result['agreement']:.3f} "
                    f"(multipv_hit={result['multipv_hit_rate']:.3f}, "
                    f"regret_mean={result['regret_mean_cp']:.1f}cp)"
                )
                # 計測結果を含めて保存し直す（checkpoint単体で一致率を参照可能に）
                save_checkpoint(
                    checkpoint_path,
                    model, optimizer, scheduler, config, state,
                    ema_model=ema_model,
                )
                # 同一epochでbest.ptも更新されていれば計測結果を反映
                # （この時点では重みが同一なので再保存で問題ない）
                if best_updated:
                    save_checkpoint(
                        output_dir / "best.pt",
                        model, optimizer, scheduler, config, state,
                        ema_model=ema_model,
                    )

        # ログ保存
        write_log(log_path, config, state)

    # 最終保存
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time / 60:.1f} minutes")
    final_path = output_dir / "final.pt"
    save_checkpoint(
        final_path,
        model, optimizer, scheduler, config, state,
        ema_model=ema_model,
    )

    # 最終モデルでオフライン指し手一致率を計測
    if config.agreement_data:
        result = run_offline_agreement(
            final_path, config.agreement_data,
            config.agreement_limit, config.device,
        )
        state.agreement.append({"epoch": config.epochs, "final": True, **result})
        logger.info(
            f"Agreement (final): {result['agreement']:.3f} "
            f"(序盤={result['opening']['agreement']:.3f}, "
            f"中盤={result['middlegame']['agreement']:.3f}, "
            f"終盤={result['endgame']['agreement']:.3f}, "
            f"multipv_hit={result['multipv_hit_rate']:.3f}, "
            f"regret_mean={result['regret_mean_cp']:.1f}cp)"
        )
        # 計測結果を含めて保存し直す（checkpoint単体で一致率を参照可能に）
        save_checkpoint(
            final_path,
            model, optimizer, scheduler, config, state,
            ema_model=ema_model,
        )

        # best.pt（既定の推論・比較対象）も自身の重みで計測し、結果を書き戻す。
        # 学習終了時のmodelはbestとは別の重みなのでsave_checkpointは使えず、
        # checkpoint辞書のstateだけを更新して保存し直す
        best_path = output_dir / "best.pt"
        if best_path.exists():
            best_result = run_offline_agreement(
                best_path, config.agreement_data,
                config.agreement_limit, config.device,
            )
            best_ckpt = torch.load(
                best_path, map_location="cpu", weights_only=False
            )
            best_epoch = best_ckpt["state"].get("epoch", 0) + 1
            entry = {"epoch": best_epoch, "best": True, **best_result}
            state.agreement.append(entry)
            best_agreement = list(best_ckpt["state"].get("agreement", []))
            best_agreement.append(entry)
            best_ckpt["state"]["agreement"] = best_agreement
            torch.save(best_ckpt, best_path)
            logger.info(
                f"Agreement (best, epoch {best_epoch}): "
                f"{best_result['agreement']:.3f} "
                f"(multipv_hit={best_result['multipv_hit_rate']:.3f}, "
                f"regret_mean={best_result['regret_mean_cp']:.1f}cp)"
            )

        write_log(log_path, config, state)


def main() -> None:
    """エントリーポイント."""
    parser = argparse.ArgumentParser(description="Value Network学習")
    parser.add_argument("--data", type=str, required=True, help="データファイルパス")
    parser.add_argument("--epochs", type=int, default=100, help="エポック数")
    parser.add_argument("--batch-size", type=int, default=512, help="バッチサイズ")
    parser.add_argument("--lr", type=float, default=3e-4, help="学習率")
    parser.add_argument("--device", type=str, default="auto", help="デバイス")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="出力ディレクトリ")
    parser.add_argument("--resume", type=str, default=None, help="再開するチェックポイント")
    parser.add_argument("--val-split", type=float, default=0.1, help="検証データ割合")
    parser.add_argument("--save-every", type=int, default=10, help="保存間隔（エポック）")
    parser.add_argument("--log-every", type=int, default=100, help="ログ間隔（ステップ）")

    # モデルパラメータ
    parser.add_argument("--d-model", type=int, default=256, help="埋め込み次元")
    parser.add_argument("--n-heads", type=int, default=4, help="アテンションヘッド数")
    parser.add_argument("--n-layers", type=int, default=4, help="レイヤー数")
    parser.add_argument("--ffn-dim", type=int, default=512, help="FFN次元")
    parser.add_argument("--dropout", type=float, default=0.1, help="ドロップアウト率")

    # 拡張特徴量
    parser.add_argument("--use-features", action="store_true", help="拡張特徴量を使用")

    # Pooling方式
    parser.add_argument("--no-attention-pooling", action="store_true", help="Mean Poolingを使用（デフォルトはAttention Pooling）")

    # モデル構造オプション
    parser.add_argument("--use-king-relative", action="store_true", help="玉相対位置埋め込みを使用")
    parser.add_argument("--use-2d-pos", action="store_true", help="2D位置埋め込み（段・筋）を使用")
    parser.add_argument("--use-discrete-hand", action="store_true", help="持ち駒枚数を離散埋め込みで扱う")

    # 補助損失
    parser.add_argument("--aux-loss-weight", type=float, default=0.1, help="勝敗補助損失の重み")

    # 学習安定化
    parser.add_argument("--warmup-epochs", type=int, default=5, help="ウォームアップエポック数")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0, help="勾配クリッピングのmax_norm")
    parser.add_argument("--label-smoothing", type=float, default=0.05, help="Label Smoothingの強度")

    # データ前処理
    parser.add_argument("--cp-scale", type=float, default=1200.0, help="評価値正規化のスケール（デフォルト: 1200）")
    parser.add_argument("--cp-noise", type=float, default=0.0, help="評価値ノイズの標準偏差（cp）")
    parser.add_argument("--cp-filter-threshold", type=float, default=None, help="評価値フィルタの閾値（cp）")
    parser.add_argument("--drop-zero-cp", action="store_true", help="score_cp==0の局面を除外（旧方式データのダミーラベル対策）")
    parser.add_argument("--cp-clamp", type=float, default=None, help="評価値を±この値に丸める（filterと異なり大差局面を学習に残す）")

    # 評価値ターゲット
    parser.add_argument("--target-mode", type=str, default="cp", choices=["cp", "wdl"], help="ターゲット空間（cp: tanh正規化, wdl: 勝率でelmoブレンド）")
    parser.add_argument("--wdl-scale", type=float, default=600.0, help="cp→勝率変換のシグモイドスケール")
    parser.add_argument("--wdl-lambda", type=float, default=0.5, help="elmoブレンドの教師評価値の重み（0〜1）")

    # データ拡張
    parser.add_argument("--normalize-turn", action="store_true", help="後手番を先手視点に正規化")
    parser.add_argument("--augment-flip", action="store_true", help="左右反転でデータ拡張（2倍）")

    # データローダー
    parser.add_argument("--num-workers", type=int, default=0, help="データローダーのワーカー数")

    # 分散正則化（通常は不要。過去のラベル符号バグの対症療法として存在）
    parser.add_argument("--variance-reg-weight", type=float, default=0.0, help="分散正則化の重み（デフォルト: 0=無効）")

    # EMA
    parser.add_argument("--ema-decay", type=float, default=0.0, help="EMA減衰率（0=無効、推奨: 0.999）")

    # 評価値損失
    parser.add_argument("--value-loss", type=str, default="mse", choices=["mse", "huber"], help="評価値損失の種類")
    parser.add_argument("--huber-delta", type=float, default=0.5, help="Huber lossの遷移点")

    # Pairwise ranking損失
    parser.add_argument("--ranking-weight", type=float, default=0.0,
                        help="ranking損失の重み（0=無効。candidatesフィールド付きデータが必要）")
    parser.add_argument("--ranking-min-gap", type=float, default=30.0,
                        help="ペアとして採用する最小評価値差（cp）")
    parser.add_argument("--delta-weight", type=float, default=0.0,
                        help="ΔV差分回帰損失の重み（0=無効。candidatesペアの評価値差分を回帰）")
    parser.add_argument("--delta-data", type=str, default=None,
                        help="摂動ペアJSONL（gen_perturb_pairs.py出力。--delta-weightと併用）")
    parser.add_argument("--delta-same-material-only", action="store_true",
                        help="素材一致（material_diff==0）の摂動ペアのみ使用")

    # オフライン指し手一致率
    parser.add_argument("--agreement-data", type=str, default=None,
                        help="一致率計測用のcandidatesフィールド付きJSONL（指定時、定期保存と学習終了時に計測）")
    parser.add_argument("--agreement-limit", type=int, default=200,
                        help="一致率計測の局面数上限")

    args = parser.parse_args()

    # argparseの結果をTrainConfigへ自動転記する。
    # フィールド名とargのdest（ハイフン→アンダースコア）は基本的に一致するが、
    # 以下の2つだけ名前が異なる/論理が反転しているため個別に扱う:
    #   --data                  -> data_path
    #   --no-attention-pooling  -> use_attention_pooling（store_trueを論理反転）
    # weight_decayのように対応するCLIオプションが無いフィールドは、
    # TrainConfigのデフォルト値がそのまま使われる（従来の挙動と同じ）。
    config_kwargs: dict[str, object] = {}
    for f in fields(TrainConfig):
        if f.name == "data_path":
            config_kwargs[f.name] = args.data
        elif f.name == "use_attention_pooling":
            config_kwargs[f.name] = not args.no_attention_pooling
        elif hasattr(args, f.name):
            config_kwargs[f.name] = getattr(args, f.name)

    config = TrainConfig(**config_kwargs)

    train(config)


if __name__ == "__main__":
    main()
