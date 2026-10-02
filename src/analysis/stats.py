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
