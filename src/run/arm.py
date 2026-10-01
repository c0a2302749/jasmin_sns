"""run/arm.py — F-11: 1アームを T_max まで走らせ、記録と収束ターンを返す。

設計決定参照:
- FR-E2: T_max までの実行と停止判定
- A1: 同一参加者内で R(t) を先に更新し、その新しい R(t) を O(t+1) に用いる
- A2: Gauss-Seidel 型（即時インプレース更新）を既定とし、Jacobi 型を切替可能に保持
- A10: 対比較の時刻整合のため、途中で収束判定を満たしても必ず T_max まで完走する
- NFR-4: debug モード時に項ごとの分解値を記録する

ArmRecord（値を持ち回るクラス）もここに置く。
layers/governance と layers/sns の両方を呼ぶのはこのモジュールと
layers/participant.py だけであり、sns ⇄ governance は互いを知らない (NFR-3)。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from config import Config
from layers import governance, participant, sns
from world.ablation import Attrs
from world.build import World


@dataclass(frozen=True)
class ArmRecord:
    """1アーム分の全シミュレーション結果。振る舞いを持たないデータの入れ物。

    Attributes:
        O (np.ndarray): 全参加者の意見履歴。shape (T_max+1, n)。
        R (np.ndarray): 全参加者のリスク認知履歴。shape (T_max+1, n)。
        W_eff (np.ndarray): 全参加者の実効知覚強度。shape (n,)。
        S_O (np.ndarray): 意見ズレの履歴。shape (T_max, n)。
        S_R (np.ndarray): リスク認知集約の履歴。shape (T_max, n)。
        max_delta (np.ndarray): ターンごとの最大変化量。shape (T_max,)。
        t_conv (int | None): 収束ターン。未収束時は None。
        n_clipped_R (int): R の生値が [0, 1] 外となりクリップされた延べ回数。
        n_reactance_active (int): リアクタンスが発動した人数 (W_eff < W)。
        terms (dict[str, np.ndarray] | None): debug 時のみ項ごとの分解値。
    """

    O: np.ndarray                  # shape (T_max+1, n)
    R: np.ndarray                  # shape (T_max+1, n)
    W_eff: np.ndarray              # shape (n,) 
    S_O: np.ndarray                # shape (T_max, n)
    S_R: np.ndarray                # shape (T_max, n)
    max_delta: np.ndarray          # shape (T_max,)
    t_conv: int | None             # 収束ターン。未収束時は None
    n_clipped_R: int               # R の生値が [0, 1] 外となりクリップされた延べ回数
    n_reactance_active: int        # リアクタンスが発動した人数 (W_eff < W)
    terms: dict[str, np.ndarray] | None = field(default=None)  # debug_terms 時のみ


def run_arm(
    world: World, 
    attrs: Attrs, 
    W: float, 
    config: Config, 
    debug: bool = False
) -> ArmRecord:
    """1アームを T_max まで走らせる。停止条件を満たしても走行は止めない（A10）。

    Gauss-Seidel型 (既定, A2): 配列をインプレース更新し、処理時点で
    配列にある値（更新済み近傍は新値、未更新近傍は旧値）を参照する。
    Jacobi型 (update_mode="jacobi", 検証専用): ターン開始時点の値のみを参照し、
    参加者の処理順序を入れ替えても結果がbit一致する。

    Args:
        world (World): シード確定済みの不変な世界。
        attrs (Attrs): Ablation適用済みの属性配列。
        W (float): 介入強度。
        config (Config): 実験設定。
        debug (bool): True の場合に項ごとの分解値も記録する。

    Raises:
        ValueError: 未知の update_mode の場合。

    Returns:
        ArmRecord: 1アーム分の全シミュレーション結果。
    """
    n = config.network.n
    T_max = config.model.T_max
    T_min = config.model.T_min
    eps = config.model.eps
    update_mode = config.stopping.update_mode
    S = config.model.S

    O = np.empty((T_max + 1, n), dtype=np.float64)
    R = np.empty((T_max + 1, n), dtype=np.float64)
    S_O_hist = np.zeros((T_max, n), dtype=np.float64)
    S_R_hist = np.zeros((T_max, n), dtype=np.float64)
    max_delta = np.zeros(T_max, dtype=np.float64)

    O[0] = world.O0
    R[0] = world.R0  # ターン0: 介入・社会的影響は作用しない（A0）

    # 第1反復では W(t) は定数を返す（FR-G2の拡張点）。
    w_const = governance.w_of_t(0, W)
    w_eff_arr, n_reactance_active = governance.calculate_w_eff(
        W=w_const, 
        theta_react=attrs.theta_react, 
        k_react=attrs.k_react
    )

    n_clipped_R = 0
    t_conv: int | None = None

    debug_terms: dict[str, np.ndarray] | None = None
    if debug:
        debug_terms = {
            "term_alpha_SO": np.zeros((T_max, n), dtype=np.float64),
            "term_beta_R": np.zeros((T_max, n), dtype=np.float64),
            "term_gamma_diff": np.zeros((T_max, n), dtype=np.float64),
            "pre_tanh": np.zeros((T_max, n), dtype=np.float64),
        }

    for t in range(T_max):
        order = world.update_order[t]
        O_curr = O[t].copy()
        R_curr = R[t].copy()

        if update_mode == "jacobi":
            O_read, R_read = O[t], R[t]        # ターン開始時点の値のみ参照
        elif update_mode == "gauss_seidel":
            O_read, R_read = O_curr, R_curr    # インプレース更新される最新値を参照
        else:
            raise ValueError(f"未知のupdate_mode: {update_mode}")

        for _i in order:
            i = int(_i)
            # 1. 近傍からの集約
            s_o, s_r = sns.social_signals(i, world.neighbors, O_read, R_read)
            S_O_hist[t, i] = s_o
            S_R_hist[t, i] = s_r

            # 2. R の更新（同一ターンで先に計算: A1）
            r_new, r_raw = participant.update_R(
                w_sys=float(attrs.w_sys[i]),
                w_eff=float(w_eff_arr[i]),
                w_sns=float(attrs.w_sns[i]),
                S_R=float(s_r),
                b=float(attrs.b[i])
            )
            if r_raw < 0.0 or r_raw > 1.0:
                n_clipped_R += 1

            # 3. O の更新（新しい R を即時利用）
            o_new, terms = participant.update_O(
                O_i=float(O_read[i]),
                O0_i=float(world.O0[i]),
                alpha=float(attrs.alpha[i]),
                beta=float(attrs.beta[i]),
                gamma=float(attrs.gamma[i]),
                S_O=float(s_o),
                R_i=float(r_new),
                S=float(S),
                debug=debug
            )

            R_curr[i] = r_new
            O_curr[i] = o_new
            
            if debug and terms is not None and debug_terms is not None:
                for key, value in terms.items():
                    debug_terms[key][t, i] = value

        O[t + 1] = O_curr
        R[t + 1] = R_curr

        # 収束判定 (最大値ノルムによる個人単位の判定: A10)
        delta = float(np.max(np.abs(O[t + 1] - O[t])))
        max_delta[t] = delta
        if t_conv is None and (t + 1) >= T_min and delta < eps:
            t_conv = t + 1

    # 返す前に全配列を不変化 (writeable=False) して事故を防ぐ
    for arr in (O, R, w_eff_arr, S_O_hist, S_R_hist, max_delta):
        arr.flags.writeable = False
        
    if debug_terms is not None:
        for v in debug_terms.values():
            v.flags.writeable = False

    return ArmRecord(
        O=O, 
        R=R, 
        W_eff=w_eff_arr, 
        S_O=S_O_hist, 
        S_R=S_R_hist,
        max_delta=max_delta, 
        t_conv=t_conv,
        n_clipped_R=n_clipped_R, 
        n_reactance_active=n_reactance_active,
        terms=debug_terms,
    )