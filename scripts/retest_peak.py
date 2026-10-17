"""scripts/retest_peak.py — B2: ピーク再検定 (seeds 30-59, W=0.45 vs 1.00のみ)。

実行: uv run python scripts/retest_peak.py --seed-start 30 --n-seeds 30
  30シード x (ctrl1 + treat2) = 90アームで軽い。
旧結果 (seeds 0-29): d=-0.00933 95%CI [-0.01333,-0.00567] 有意。
本実行を確認の検定として前面に出す (ピーク選択に使っていないシード)。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from analysis.stats import bootstrap_paired_diff
from config import load
from run.arm import run_arm
from run.metrics import calculate_metrics
from runlog import log_header
from world.ablation import Ablation, apply_ablation
from world.build import build_world


def main() -> None:
    ap = argparse.ArgumentParser(description="ピーク再検定 W=0.45 vs 1.00")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seed-start", type=int, default=30)
    ap.add_argument("--n-seeds", type=int, default=30)
    ap.add_argument("--B", type=int, default=10000)
    args = ap.parse_args()

    config = load(args.config)
    boot_seed = config.metrics.bootstrap_seed
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    log_header(config, f"seeds={seeds[0]}..{seeds[-1]} ({len(seeds)}) B={args.B}")
    e_peak: list[float] = []
    e_tail: list[float] = []
    for seed in seeds:
        world = build_world(config, seed)
        attrs = apply_ablation(world.params, Ablation.NONE, config.model.k_react)
        ctrl = run_arm(world, attrs, W=0.0, config=config)
        t_peak = run_arm(world, attrs, W=0.45, config=config)
        t_tail = run_arm(world, attrs, W=1.0, config=config)
        e_peak.append(calculate_metrics(
            ctrl, t_peak, world, config, W_level=0.45, ablation="none")["E_rate"])
        e_tail.append(calculate_metrics(
            ctrl, t_tail, world, config, W_level=1.0, ablation="none")["E_rate"])
    pd = bootstrap_paired_diff(e_peak, e_tail, B=args.B, seed=boot_seed)
    # 共通ルール: 絶対値の差 d_abs = |E_peak|-|E_tail| も併記 (両者負なら -d_signed)。
    abs_peak = [abs(v) for v in e_peak]
    abs_tail = [abs(v) for v in e_tail]
    pd_abs = bootstrap_paired_diff(abs_peak, abs_tail, B=args.B, seed=boot_seed)
    print("=" * 96)
    print(f"B2. 再検定 seeds={seeds[0]}..{seeds[-1]} ({len(seeds)}) W=0.45 vs 1.00 B={args.B}")
    print("=" * 96)
    print(f"  mean(E_0.45)={float(np.mean(e_peak)):+.5f} mean(E_1.0)={float(np.mean(e_tail)):+.5f}")
    print(f"  d_s平均(signed E_peak-E_tail) {pd['mean']:+.5f} 95%CI [{pd['lo']:+.5f}, {pd['hi']:+.5f}] "
          f"significant={bool(pd['significant'])} (旧B=2000: -0.00933 [-0.01333,-0.00567] True)")
    print(f"  d_abs平均(|E_peak|-|E_tail|) {pd_abs['mean']:+.5f} 95%CI [{pd_abs['lo']:+.5f}, {pd_abs['hi']:+.5f}] "
          f"significant={bool(pd_abs['significant'])}")


if __name__ == "__main__":
    main()
