"""scripts/analyze_ci_emaj.py — E_rate 95%CI・E_maj・対ペア差検定・型別ΔO_mean。

実行: uv run python scripts/analyze_ci_emaj.py --n-seeds 30 --step 0.05
  (重い: 30seed x 20W。試しは --n-seeds 5 で)
  第2シナリオ (O0=0.9): --mu 0.9 を付ける (stdは継承)。

出力:
  T1. W ごとの E_rate 平均[95%CI] (percentile bootstrap, B=2000) と E_maj 分布
  T2. 非単調性の対ペア検定: ピークW vs W=1.0 の差 d_s を同一シードで対にし bootstrap。
      ピークが同値で複数ある場合は最小のWを用いた旨を明示する。
  T3. 型別 ΔO_mean (T* での mean(O_treat-O_ctrl) を persona 別に分解)
  T4. 評価時点 T* の分布 (T_min=10のため10〜12に収まるかの確認)。
      収束が速い設定では「収束速度差の混入防止」の設計が実質的に働かない。
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from analysis.stats import bootstrap_ci, bootstrap_paired_diff, summarize_emaj
from config import load
from run.arm import run_arm
from run.metrics import calculate_metrics, persona_dO_means
from runlog import log_header
from world.ablation import Ablation, apply_ablation
from world.build import build_world


def main() -> None:
    ap = argparse.ArgumentParser(description="E_rate CI / E_maj / paired test / persona dO")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n-seeds", type=int, default=30)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--mu", type=float, default=None,
                    help="init_opinion.mean を上書き (第2シナリオ用。例: 0.9)")
    args = ap.parse_args()

    config = load(args.config)
    if args.mu is not None:
        config = replace(config, init_opinion=replace(config.init_opinion, mean=args.mu))
    B = args.B
    boot_seed = config.metrics.bootstrap_seed
    theta = config.metrics.theta
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    log_header(config, f"seeds={seeds[0]}..{seeds[-1]} ({len(seeds)}) "
                       f"step={args.step} B={B} mu={config.init_opinion.mean}")
    W_grid = [round(x, 4) for x in np.arange(0.0, 1.0 + args.step / 2, args.step)]
    W_grid = [w for w in W_grid if 0.0 < w <= 1.0]

    per_W_E: dict[float, list[float]] = {w: [] for w in W_grid}
    per_W_M: dict[float, list[int]] = {w: [] for w in W_grid}
    per_W_dO: dict[float, list[float]] = {w: [] for w in W_grid}
    per_W_T: dict[float, list[int]] = {w: [] for w in W_grid}
    per_W_persona: dict[float, dict[str, list[float]]] = {w: {} for w in W_grid}
    n_clipped_O0 = 0

    for seed in seeds:
        world = build_world(config, seed)
        n_clipped_O0 += world.n_clipped_O0
        attrs = apply_ablation(world.params, Ablation.NONE, config.model.k_react)
        ctrl = run_arm(world, attrs, W=0.0, config=config)
        for W in W_grid:
            treat = run_arm(world, attrs, W=W, config=config)
            m = calculate_metrics(ctrl, treat, world, config, W_level=W, ablation="none")
            per_W_E[W].append(m["E_rate"])
            per_W_M[W].append(m["E_maj"])
            per_W_dO[W].append(m["dO_mean"])
            per_W_T[W].append(m["T_star"])
            pd = persona_dO_means(ctrl, treat, world, config)
            for k, v in pd.items():
                per_W_persona[W].setdefault(k, []).append(v)

    n = config.network.n
    print(f"O0 clip: n_clipped_O0={n_clipped_O0}/{len(seeds) * n} "
          f"= {n_clipped_O0 / (len(seeds) * n):.4f}")
    print("=" * 100)
    print("T1. E_rate mean [95% CI] と E_maj 分布 (ablation=none)")
    print(f"seeds={len(seeds)} B={B} boot_seed={boot_seed} theta={theta}")
    print("=" * 100)
    print("  W      E_rate_mean  lo        hi        dO_mean    E_maj: n  mean   p(-1)  p(0)  p(+1)")
    for W in W_grid:
        ci = bootstrap_ci(per_W_E[W], B=B, seed=boot_seed)
        em = summarize_emaj(per_W_M[W])
        dO = float(np.mean(per_W_dO[W]))
        print(f"  {W:.2f}   {ci['mean']:+.5f}  {ci['lo']:+.5f}  {ci['hi']:+.5f}  "
              f"{dO:+.5f}   {int(em['n']):3d}  {em['mean']:+.3f}  "
              f"{em['p_minus1']:.3f}  {em['p_0']:.3f}  {em['p_plus1']:.3f}")

    # T2. 非単調性の対ペア検定: |E|最大のW vs W=1.0。同値は最小のWを用いる。
    means = {W: float(np.mean(per_W_E[W])) for W in W_grid}
    peak_abs = max(abs(v) for v in means.values())
    tied = sorted(w for w in W_grid if abs(means[w]) >= peak_abs - 1e-12)
    peak = min(tied)
    tail = max(W_grid)
    pd = bootstrap_paired_diff(per_W_E[peak], per_W_E[tail], B=B, seed=boot_seed)
    # 符号を揃えるため絶対値ではなく生の差で検定 (両方とも負の想定)。
    # peak が負・tail が負 (0に近い) なら差は負になる。
    print()
    print("=" * 100)
    print(f"T2. 非単調性の対ペア bootstrap 検定: W={peak:.2f} vs W={tail:.2f} (同一シードで対)")
    print("=" * 100)
    if len(tied) > 1:
        print(f"  ※同値{len(tied)}点 ({', '.join(f'{w:.2f}' for w in tied)}) のうち最小のWを用いた。")
    print(f"  mean(E_peak)={means[peak]:+.5f} mean(E_tail)={means[tail]:+.5f}")
    print(f"  d_s = E_peak,s - E_tail,s の平均 {pd['mean']:+.5f} "
          f"95%CI [{pd['lo']:+.5f}, {pd['hi']:+.5f}] "
          f"significant(0を含まない)={bool(pd['significant'])}")
    # 参考: 独立2標本として差を取った場合との比較用に相関も表示
    c = float(np.corrcoef(per_W_E[peak], per_W_E[tail])[0, 1]) if len(seeds) > 2 else float("nan")
    print(f"  参考: 2条件のシード間相関 r={c:.3f} (共通乱数で正の相関→対検定が高検出力)")
    print(f"  ※seeds 0-29で選んだ対比較は seeds 30-59 の再検定 (retest_peak.py) で確認する。")

    print()
    print("=" * 100)
    print("T3. 型別 ΔO_mean (T* での mean(O_treat-O_ctrl), シード平均)")
    print("=" * 100)
    names = sorted({k for w in W_grid for k in per_W_persona[w]})
    print("  W      " + "  ".join(f"{nm:>12s}" for nm in names))
    for W in W_grid:
        cells = []
        for nm in names:
            vals = per_W_persona[W].get(nm, [])
            cells.append(f"{float(np.mean(vals)):+.5f}" if vals else "      n/a")
        print(f"  {W:.2f}   " + "  ".join(f"{c:>12s}" for c in cells))

    print()
    print("=" * 100)
    print("T4. 評価時点 T* の分布 (T_min=10)")
    print("=" * 100)
    for W in (0.5, 1.0):
        if W not in per_W_T:
            continue
        cnt = Counter(per_W_T[W])
        print(f"  W={W:.2f} min={min(per_W_T[W])} max={max(per_W_T[W])} "
              + " ".join(f"T={t}:{cnt[t]}" for t in sorted(cnt)))


if __name__ == "__main__":
    main()
