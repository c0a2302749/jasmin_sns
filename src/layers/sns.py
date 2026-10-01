"""layers/sns.py — F-09: 近傍集約 S_O, S_R を計算する。

設計決定参照:
- FR-S2, FR-S3, FR-S4: 近傍集約の計算式（読むだけ、配列を書き換えない）
- A14: |N_i|=0 の防御

このファイルはnumpyのみに依存する。governance.py とは互いにimportしない
（NFR-3, tests/test_imports.pyで機械確認）。
"""
from __future__ import annotations

from collections.abc import Sequence
import numpy as np


def social_signals(
    i: int, 
    neighbors: Sequence[np.ndarray], 
    O: np.ndarray, 
    R: np.ndarray
) -> tuple[float, float]:
    """参加者iの近傍から S_O（差分）と S_R（絶対量）を計算する。

    S_O,i(t) = mean_{j in N_i} O_j(t) - O_i(t)   （ズレが大きいほど追従が強い）
    S_R,i(t) = mean_{j in N_i} R_j(t)             （平均そのもの）

    配列の現在値を読むだけで、O・Rを書き換えない（FR-S4）。

    Args:
        i (int): 対象の参加者インデックス。
        neighbors (Sequence[np.ndarray]): 全参加者の隣接インデックス配列のシーケンス。
        O (np.ndarray): 全参加者の現在の意見配列 (1D ndarray)。
        R (np.ndarray): 全参加者の現在のリスク認知配列 (1D ndarray)。

    Returns:
        tuple[float, float]: (S_O, S_R) のタプル。S_O は意見のズレ
            （近傍平均 O - 自分の O）、S_R はリスク認知の絶対量（近傍平均 R）。
    """
    neigh = neighbors[i]
    
    # 防御的ハンドリング (A14): 孤立ノードの場合は社会的影響なし
    if neigh.size == 0:
        return 0.0, 0.0

    # 1. S_O: 近傍の意見平均 - 自分の現在の意見 O_i (FR-S2)
    s_o = float(np.mean(O[neigh]) - O[i])

    # 2. S_R: 近傍のリスク認知平均 (FR-S3)
    s_r = float(np.mean(R[neigh]))

    return s_o, s_r