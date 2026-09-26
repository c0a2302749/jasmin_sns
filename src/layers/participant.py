"""layers/participant.py — F-10: 1人分の R, O を更新する。

設計決定参照:
- FR-P2, FR-P3: O および R の更新式実装
- A1: 同一参加者内で R(t) を先に計算し O(t+1) に用いる順序
- A4: γ(O - O0) の O は O(t) を参照する
- NFR-4: 項ごとの値を debug モードで返せる（追跡可能性）

R, O とも状態配列や World を直接受け取らず、スカラー値のみを受け取る純粋関数とする。
呼び出し側（run/arm.py）が配列の読み書きを担う。
"""
from __future__ import annotations

import numpy as np


def update_R(
    w_sys: float,
    w_eff: float,
    w_sns: float,
    S_R: float,
    b: float,
) -> tuple[float, float]:
    """参加者1人分のリスク認知 R(t) を計算する (FR-P3)。

    R(t) = Clip(w_sys * W_eff + w_sns * S_R + b, 0, 1)
    A1: 同一ターン内で O の更新より先に呼び出される。

    Args:
        w_sys (float): 制度的介入係数。
        w_eff (float): 実効知覚強度。
        w_sns (float): 社会的影響係数。
        S_R (float): 近傍のリスク認知平均。
        b (float): ベースラインリスク認知。

    Returns:
        tuple[float, float]: (new_r, r_raw) のタプル。new_r は [0, 1] に
            クリップされた新しい R、r_raw はクリップ前の生の値
            （天井効果診断・デバッグ用）。
    """
    r_raw = float(w_sys * w_eff + w_sns * S_R + b)
    new_r = float(np.clip(r_raw, 0.0, 1.0))
    return new_r, r_raw


def update_O(
    O_i: float,
    O0_i: float,
    alpha: float,
    beta: float,
    gamma: float,
    S_O: float,
    R_i: float,
    S: float,
    debug: bool = False,
) -> tuple[float, dict[str, float] | None]:
    """参加者1人分の意見 O(t+1) を計算する (FR-P2, A4)。

    O(t+1) = (tanh((O(t) + α*S_O - β*R_i - γ*(O(t) - O0_i) - 0.5) * S) + 1) / 2
    R_i は同一ターンで既に更新済みの新しい R を使用する (A1)。

    Args:
        O_i (float): 現在の意見 O(t)。
        O0_i (float): 初期意見 O0。
        alpha (float): 同調圧力係数。
        beta (float): リスク認知係数。
        gamma (float): 初期意見固執係数。
        S_O (float): 近傍との意見ズレ。
        R_i (float): 同一ターンで更新済みのリスク認知 R(t)。
        S (float): スケーリング係数。
        debug (bool): True の場合に項ごとの計算結果も返す。

    Returns:
        tuple[float, dict[str, float] | None]: (new_o, debug_terms) のタプル。
            new_o は新しい意見 O (0.0 ～ 1.0)、debug_terms は debug=True の
            場合のみ各項の計算結果辞書、それ以外は None。
    """
    # 項ごとの分解計算
    term_alpha = alpha * S_O
    term_beta = beta * R_i
    term_gamma = gamma * (O_i - O0_i)

    # 論文の更新式本体
    pre_tanh = (O_i + term_alpha - term_beta - term_gamma - 0.5) * S
    new_o = float((np.tanh(pre_tanh) + 1.0) / 2.0)

    debug_terms = None
    if debug:
        debug_terms = {
            "term_alpha_SO": float(term_alpha),
            "term_beta_R": float(term_beta),
            "term_gamma_diff": float(term_gamma),
            "pre_tanh": float(pre_tanh),
        }

    return new_o, debug_terms