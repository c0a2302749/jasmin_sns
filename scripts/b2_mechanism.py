"""scripts/b2_mechanism.py — B2: 非単調性の機構切り分け (主実験 mu=0.80)。

variant:
  baseline  : 現行 (R クリップあり, theta_react は分布)
  no_clip   : R の [0,1] クリップを外す。非単調が消えれば R の可動域制約が原因。
  theta05   : 全員 theta_react=0.5。ピーク位置が theta 分布で決まるかの確認。

各 W で E_rate と平均 W_eff (全員) を出し、E_rate を平均 W_eff の関数として見る。
主指標: E_rate。非単調は |E_rate| のピーク後に減少するかで判定。

実行: uv run python scripts/b2_mechanism.py --o0mean 0.8 --variant baseline
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
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
    ap = argparse.ArgumentParser(description="B2 非単調性の機構切り分け")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--o0mean", type=float, default=0.8)
    ap.add_argument("--variant", default="baseline",
                    choices=["baseline", "no_clip", "theta05"])
    ap.add_argument("--n-seeds", type=int, default=30)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--raw-dir", default=None)
    args = ap.parse_args()

    base = load(args.config)
    cfg = replace(base, init_opinion=replace(base.init_opinion, mean=args.o0mean))
    B = cfg.metrics.bootstrap_B
    bs = cfg.metrics.bootstrap_seed
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    W_grid = [round(x, 4) for x in np.arange(args.step, 1.0 + args.step / 2, args.step)]
    clip_R = args.variant != "no_clip"
    theta_override = 0.5 if args.variant == "theta05" else None
    log_header(cfg, f"B2 variant={args.variant} mu={args.o0mean} seeds={seeds[0]}..{seeds[-1]} clipR={clip_R}")

    raw_path = None
    if args.raw_dir is not None:
        rd = Path(args.raw_dir)
        rd.mkdir(parents=True, exist_ok=True)
        raw_path = rd / f"b2_{args.variant}_mu{args.o0mean}.jsonl"

    e: dict[float, list[float]] = {w: [] for w in W_grid}
    weff: dict[float, list[float]] = {w: [] for w in W_grid}
    rows_raw: list[dict] = []
    for seed in seeds:
        world = build_world(cfg, seed)
        params = world.params
        if theta_override is not None:
            params = [replace(p, theta_react=theta_override) for p in world.params]
        attrs = apply_ablation(params, Ablation.NONE, cfg.model.k_react)
        ctrl = run_arm(world, attrs, W=0.0, config=cfg, clip_R=clip_R)
        for W in W_grid:
            treat = run_arm(world, attrs, W=W, config=cfg, clip_R=clip_R)
            m = calculate_metrics(ctrl, treat, world, cfg, W_level=W, ablation="none")
            e[W].append(m["E_rate"])
            weff[W].append(float(np.mean(treat.W_eff)))
            rows_raw.append({"variant": args.variant, "seed": seed, "W": W,
                             "E_rate": m["E_rate"], "mean_Weff": float(np.mean(treat.W_eff))})
    if raw_path is not None:
        with raw_path.open("w", encoding="utf-8") as f:
            for r in rows_raw:
                f.write(json.dumps(r) + "\n")

    means = {W: float(np.mean(e[W])) for W in W_grid}
    mweff = {W: float(np.mean(weff[W])) for W in W_grid}
    peak = max(W_grid, key=lambda w: abs(means[w]))
    tail = max(W_grid)
    # ピーク後の減少開始
    dec_start = None
    for a, b in zip(W_grid, W_grid[1:]):
        if b > peak and abs(means[b]) < abs(means[a]) - 1e-12:
            dec_start = b
            break
    pd = bootstrap_paired_diff(e[peak], e[tail], B=B, seed=bs)

    print("=" * 88)
    print(f"B2 機構切り分け: variant={args.variant} mu={args.o0mean} "
          f"seeds={seeds[0]}..{seeds[-1]} ({len(seeds)}) clip_R={clip_R}")
    print("=" * 88)
    print("  W      E_rate      平均W_eff   |E_rate|")
    for W in W_grid:
        print(f"  {W:.2f}   {means[W]:+.5f}   {mweff[W]:.4f}     {abs(means[W]):.5f}")
    print()
    print(f"  ピーク W={peak:.2f} (|E_rate|={abs(means[peak]):.5f})  減少開始 W={dec_start}")
    print(f"  d_s(E_peak-E_tail)={pd['mean']:+.5f} 95%CI [{pd['lo']:+.5f},{pd['hi']:+.5f}] "
          f"significant={bool(pd['significant'])}")
    mono = all(abs(means[a]) <= abs(means[b]) + 1e-12 for a, b in zip(W_grid, W_grid[1:]))
    print(f"  全域で単調増加か: {mono}  → {'非単調なし(山が消えた)' if mono else '非単調が残る'}")
    print()
    print("  E_rate を平均 W_eff の関数として (W: meanWeff -> E_rate):")
    print("   " + "  ".join(f"{W:.2f}:{mweff[W]:.3f}->{means[W]:+.4f}" for W in W_grid))


if __name__ == "__main__":
    main()
