"""analysis/stats.py — F-16: ブートストラップ信頼区間の計算 (FR-E6)。

設計決定参照:
- FR-E6: 異なる乱数シードで複数回試行し、Eの平均と信頼区間を報告する
- D-統計: percentile法。B, seed は configs の metrics.bootstrap_B / bootstrap_seed

RNG境界 (NFR-1/FR-G3, verify_invariants.py:RNG_BOUNDARY_STATIC):
src/ 内で `np.random.*` を呼べるのは rng.py のみのため、ここでは
標準ライブラリ `random.Random(seed)` で復元抽出する。シミュレーションの
Generator系列には一切触れない。
"""
from __future__ import annotations

import random
import statistics


def bootstrap_ci(
    values: list[float],
    B: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, float]:
    """平均値のpercentileブートストラップ信頼区間を返す。

    Args:
        values: シードごとの観測値 (例: あるWの E_rate 30件)。
        B: リサンプリング回数 (config.metrics.bootstrap_B)。
        seed: 整数シード (config.metrics.bootstrap_seed)。
        alpha: 有意水準。既定0.05で95%区間。

    Returns:
        {"mean": 標本平均, "lo": 下側分位点, "hi": 上側分位点, "B": B}。
        空入力では全て float("nan") を返す。
    """
    n = len(values)
    if n == 0 or B < 1:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "B": float(B)}
    mean = float(statistics.fmean(values))
    if n == 1:
        return {"mean": mean, "lo": mean, "hi": mean, "B": float(B)}
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(B):
        s = 0.0
        for _ in range(n):
            s += values[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo_idx = min(max(int((alpha / 2) * B), 0), B - 1)
    hi_idx = min(max(int((1 - alpha / 2) * B), 0), B - 1)
    return {"mean": mean, "lo": float(means[lo_idx]), "hi": float(means[hi_idx]), "B": float(B)}


def bootstrap_paired_diff(
    a_vals: list[float],
    b_vals: list[float],
    B: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, float]:
    """同一シードで対にした差のブートストラップ検定 (percentile法)。

    非単調性 (例: E_rate(W=0.5) vs E_rate(W=1.0)) を同一シードのペア差
    d_s = a_s - b_s として検定する。シード間の相関を保持するため、
    差のベクトル d に対して通常の percentile ブートストラップを行う。
    H0: mean(d) = 0。CI が 0 を含まなければ 5% 水準で有意。

    Args:
        a_vals: 条件Aのシードごとの観測値 (例: W=0.5 の E_rate 30件)。
        b_vals: 条件Bのシードごとの観測値 (同順・同シード)。
        B: リサンプリング回数。
        seed: 整数シード (復元抽出用。シミュレーション系列には触れない)。
        alpha: 有意水準。既定0.05で95%区間。

    Returns:
        {"mean": 差の平均, "lo": 下側分位点, "hi": 上側分位点,
         "B": B, "n": 対数, "significant": 0を含まないか (1.0/0.0)}。
        長さ不一致・空入力では全て nan (significant は nan)。
    """
    if len(a_vals) != len(b_vals) or len(a_vals) == 0 or B < 1:
        return {
            "mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
            "B": float(B), "n": float(len(a_vals)),
            "significant": float("nan"),
        }
    diffs = [float(a) - float(b) for a, b in zip(a_vals, b_vals)]
    res = bootstrap_ci(diffs, B=B, seed=seed, alpha=alpha)
    lo, hi = res["lo"], res["hi"]
    sig = 1.0 if (hi < 0.0 or lo > 0.0) else 0.0
    return {"mean": res["mean"], "lo": lo, "hi": hi,
            "B": float(B), "n": float(len(diffs)), "significant": sig}


def summarize_emaj(values: list[int]) -> dict[str, float]:
    """E_maj (-1/0/+1) の度数分布と平均を返す (決定的集計、RNG不使用)。

    Args:
        values: シードごとの E_maj 観測値。

    Returns:
        {"n": 件数, "mean": 平均, "p_minus1": -1の割合,
         "p_0": 0の割合, "p_plus1": +1の割合}。
    """
    n = len(values)
    if n == 0:
        return {"n": 0.0, "mean": float("nan"),
                "p_minus1": float("nan"), "p_0": float("nan"), "p_plus1": float("nan")}
    mean = float(statistics.fmean(values))
    return {
        "n": float(n),
        "mean": mean,
        "p_minus1": float(sum(1 for v in values if v == -1) / n),
        "p_0": float(sum(1 for v in values if v == 0) / n),
        "p_plus1": float(sum(1 for v in values if v == 1) / n),
    }
