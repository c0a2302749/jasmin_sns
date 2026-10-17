"""scripts/diagnose_beta0.py — β=0 診断実行 (C1 が通った原因の特定)。

実行: wsl bash -lc "cd ~/jasmin_sns && uv run python scripts/diagnose_beta0.py --n-seeds 30"

C1 (ベースライン再現: W=0 で群間距離 |mean_H-mean_L| が初期より拡大) は
κ=2 (中立が非拡大の境界)・同調で合意に寄るはずのモデルで 29/30 と通った。
原因候補:
  H1: -βR 項 (R0=b>0 が全員を一様に押し下げ、分布を非対称に歪める) が主因。
  H2: α による同調・ネットワーク効果が主因。
β=0 (全員の beta を 0 に置換、他は同一世界) で C1 が崩れれば H1 を支持、
残れば H2 (α側) の寄与が大きい。対照として α=0 も併記する。
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from config import load
from run.arm import run_arm
from runlog import log_header
from world.ablation import Ablation, apply_ablation
from world.build import build_world


def pol_between(O_t: np.ndarray, theta: float) -> float:
    high = O_t[O_t >= theta]
    low = O_t[O_t < theta]
    if high.size == 0 or low.size == 0:
        return 0.0
    return float(abs(high.mean() - low.mean()))


def run_condition(config, seeds, zero_field: str | None):
    """zero_field: None=baseline, "beta"=β0, "alpha"=α0。同一世界で比較。"""
    theta = config.metrics.theta
    c1_hits = 0
    d_pol: list[float] = []
    d_mean: list[float] = []
    for seed in seeds:
        world = build_world(config, seed)
        if zero_field is None:
            attrs = apply_ablation(world.params, Ablation.NONE, config.model.k_react)
        else:
            params = [replace(p, **{zero_field: 0.0}) for p in world.params]
            attrs = apply_ablation(params, Ablation.NONE, config.model.k_react)
        rec = run_arm(world, attrs, W=0.0, config=config)
        t_end = rec.t_conv if rec.t_conv is not None else rec.O.shape[0] - 1
        p0, pt = pol_between(rec.O[0], theta), pol_between(rec.O[t_end], theta)
        d_pol.append(pt - p0)
        d_mean.append(float(np.mean(rec.O[t_end]) - np.mean(rec.O[0])))
        if pt > p0:
            c1_hits += 1
    return c1_hits / len(seeds), float(np.mean(d_pol)), float(np.mean(d_mean)), d_pol


def main() -> None:
    ap = argparse.ArgumentParser(description="beta=0 診断")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n-seeds", type=int, default=30)
    ap.add_argument("--seed-start", type=int, default=0)
    args = ap.parse_args()

    config = load(args.config)
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    log_header(config, f"seeds={seeds[0]}..{seeds[-1]} ({len(seeds)})")
    print("=" * 96)
    print("β=0 診断: 同一世界・W=0 で baseline / β=0 / α=0(対照) の C1 を比較")
    print("=" * 96)
    rows = {}
    for label, field in (("baseline", None), ("beta=0", "beta"), ("alpha=0(対照)", "alpha")):
        rate, mean_dpol, mean_dmean, _ = run_condition(config, seeds, field)
        rows[label] = (rate, mean_dpol, mean_dmean)
        print(f"  {label:14s} C1率={rate:.3f} ({int(rate*len(seeds))}/{len(seeds)})  "
              f"Δpol_between平均={mean_dpol:+.5f}  Δmean(O)平均={mean_dmean:+.5f}")
    base, b0, a0 = rows["baseline"][0], rows["beta=0"][0], rows["alpha=0(対照)"][0]
    print()
    if b0 < base - 1e-12 and b0 < 0.80:
        print("  → 判定: β=0 で C1 が崩れた (-βR がベースライン再現の主因, H1 を支持)。")
    elif b0 >= 0.80:
        print("  → 判定: β=0 でも C1 が残った (-βR だけでは説明できない。α・初期分布側 H2 の寄与大)。")
    else:
        print("  → 判定: 部分的に低下。両経路の複合と解釈する。")
    print(f"  → 対照 α=0 の C1率={a0:.3f} (同調を消したときの残差が -βR+初期固執の寄与の目安)。")


if __name__ == "__main__":
    main()
