"""layers/governance.py — F-08: 実効介入強度 W_eff の計算。

設計決定参照:
- FR-P4, A13: 区分関数を同等な閉形式 max(0, W - (1+k)*max(0, W - θ_react)) で実装
- FR-G2: 時間変動介入への拡張点として w_of_t のインタフェースを保持
- FR-G3: ガバナンス層は乱数を消費しない
- A12: W, theta_react, W_effは同一スケール
- NFR-3: 状態を持たない純粋関数モジュールとし、sns.py とは互いにimportしない
"""
from __future__ import annotations

import numpy as np


def w_of_t(t: int, w_const: float) -> float:
    """介入強度の時刻依存インタフェース (FR-G2)。
    
    将来的に介入強度 W を時間変化（漸増・パルス状など）させる場合の拡張点。
    第1反復では定数介入のみを扱うため、そのまま定数を返す。
    """
    return w_const


def calculate_w_eff(
    W: float,
    theta_react: np.ndarray,
    k_react: float,
) -> tuple[np.ndarray, int]:
    """介入強度 W と参加者の反発限界値 theta_react から W_eff 配列を計算する。

    Args:
        W (float): 運営からの設定介入強度 (0.0 ～ 1.0)。
        theta_react (np.ndarray): 全参加者の介入反発限界値配列 θ_react (1D ndarray)。
        k_react (float): リアクタンス減衰係数 k。

    Returns:
        tuple[np.ndarray, int]: (W_eff, n_reactance_active) のタプル。W_eff は
            全参加者の実効知覚強度配列（不変 ndarray）、n_reactance_active は
            リアクタンスが発動した人数 (W > θ_react)。
    """
    # 閉形式: max(0, W - (1 + k) * max(0, W - theta_react)) (A13)
    # 乱数を一切消費せずに numpy の配列演算のみで完結させる (FR-G3)
    excess = np.maximum(0.0, W - theta_react)
    w_eff_arr = np.maximum(0.0, W - (1.0 + k_react) * excess)

    # 呼び出し側（arm.py）で意図せず値を書き換えてしまうのを防ぐ
    w_eff_arr.flags.writeable = False
    
    # リアクタンスが発動した人数をカウント (W > θ_react)
    n_active = int(np.sum(W > theta_react))

    return w_eff_arr, n_active