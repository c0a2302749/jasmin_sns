"""run/metrics.py — F-12: ペアの評価時刻 T* を決定し、指標を計算する。

設計決定参照:
- F-12: 両アームの評価時刻の統一と、因果効果・分極化指標の計算
- A5: E_rate (主指標)、E_maj、flip_rate (個人反転率) 等の定義
- A10: T* = max(T_conv_ctrl, T_conv_treat) (ともに収束した遅い方のターン、または T_max)
- D-分極化: 人数重み付き母分散 (within) と 群平均差 (between)
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from config import Config
from run.arm import ArmRecord
from world.build import World


def t_star(ctrl: ArmRecord, treat: ArmRecord, config: Config) -> int:
    """両アームとも収束条件を満たしたターン T* を決定する (A10)。

    両アームがともに収束した時点のうち遅い方（max）を返す。
    片方でも T_max まで収束しなかった場合は T_max を返す。

    Args:
        ctrl (ArmRecord): 介入なしアームの実行記録。
        treat (ArmRecord): 介入ありアームの実行記録。
        config (Config): 実験設定（上限に model.T_max を用いる）。

    Returns:
        int: 共通評価時刻 T*。
    """
    t_max = config.model.T_max
    if ctrl.t_conv is None or treat.t_conv is None:
        return t_max
    return max(ctrl.t_conv, treat.t_conv)


def _polarization(O_t: np.ndarray, theta: float = 0.5) -> dict[str, float]:
    """θ で2群（賛成 / 反対）に分け、分極化指標 (within, between) を計算する。

    within は人数重み付き母分散、between は群平均差。
    片群が空の場合、群間距離 (between) は math.nan とする。

    Args:
        O_t (np.ndarray): あるターンの全参加者の意見配列。
        theta (float): 二値化の閾値。

    Returns:
        dict[str, float]: {"pol_within": within, "pol_between": between}。
    """
    high = O_t[O_t >= theta]
    low = O_t[O_t < theta]
    
    within = 0.0
    n_total = O_t.size
    
    if high.size > 0:
        within += float(np.var(high, ddof=0)) * (high.size / n_total)
    if low.size > 0:
        within += float(np.var(low, ddof=0)) * (low.size / n_total)
        
    if high.size > 0 and low.size > 0:
        between = float(abs(high.mean() - low.mean()))
    else:
        between = math.nan
        
    return {"pol_within": within, "pol_between": between}


def calculate_metrics(
    ctrl: ArmRecord,
    treat: ArmRecord,
    world: World,
    config: Config,
    W_level: float,
    ablation: str,
) -> dict[str, Any]:
    """対アーム (ctrl, treat) から共通評価時刻 T* の評価指標を計算する (A5, F-12)。

    行を識別する W と ablation 条件はアームの実行記録ではなく実験条件なので、
    run_checks と同じく引数で受け取る（ArmRecord はシミュレーション結果のみを保持）。

    Args:
        ctrl (ArmRecord): 介入なしアームの実行記録。
        treat (ArmRecord): 介入ありアームの実行記録。
        world (World): シード確定済みの世界（seed と n_clipped_O0 の取得に用いる）。
        config (Config): 実験設定（T* の上限に model.T_max、二値化に metrics.theta を用いる）。
        W_level (float): 介入アームの介入強度 W。
        ablation (str): 適用したAblation条件（"none" | "ABL-SYS" | "ABL-SNS" | "ABL-REACT"）。

    Returns:
        dict[str, Any]: io/schema.py の SUMMARY_SCHEMA に合致する1行分の指標辞書。
    """
    # 評価時刻 T* の決定
    t = t_star(ctrl, treat, config)
    theta = config.metrics.theta  # 二値化の閾値 (A5, metrics.theta)

    O_ctrl = ctrl.O[t]
    O_treat = treat.O[t]

    # 投票の二値化と割合の計算
    vote_ctrl = O_ctrl >= theta
    vote_treat = O_treat >= theta

    rate_ctrl = float(np.mean(vote_ctrl))
    rate_treat = float(np.mean(vote_treat))

    # 指標の計算
    e_rate = rate_treat - rate_ctrl
    e_maj = int(rate_treat >= 0.5) - int(rate_ctrl >= 0.5)
    flip_rate = float(np.mean(vote_treat != vote_ctrl))
    dO_mean = float(np.mean(O_treat - O_ctrl))

    pol_ctrl = _polarization(O_ctrl, theta)
    pol_treat = _polarization(O_treat, theta)

    # io/schema.py の SUMMARY_SCHEMA に合致する辞書を返す
    return {
        "seed": world.seed,
        "W": W_level,
        "ablation_condition": ablation,
        "E_rate": e_rate,
        "E_maj": e_maj,
        "flip_rate": flip_rate,
        "dO_mean": dO_mean,
        "pol_within_treat": pol_treat["pol_within"],
        "pol_within_ctrl": pol_ctrl["pol_within"],
        "pol_between_treat": pol_treat["pol_between"],
        "pol_between_ctrl": pol_ctrl["pol_between"],
        "T_star": t,
        "t_conv_treat": treat.t_conv if treat.t_conv is not None else -1,
        "t_conv_ctrl": ctrl.t_conv if ctrl.t_conv is not None else -1,
        "n_clipped_O0": world.n_clipped_O0,
        "n_clipped_R_treat": treat.n_clipped_R,
        "n_clipped_R_ctrl": ctrl.n_clipped_R,
        "n_reactance_active": treat.n_reactance_active,
    }