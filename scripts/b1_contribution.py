"""scripts/b1_contribution.py — B1: 観測変化に占める介入の寄与 (主実験 mu=0.80)。

各シードで初期 / ctrl(W=0) 最終 / treat(W*) 最終 の賛成率と O_mean を取り、
観測変化 (treat最終 - 初期) のうち E_rate (treat最終 - ctrl最終) が占める割合を出す。
残りは参加者内部と社会的影響による変化。

実行: uv run python scripts/b1_contribution.py --o0mean 0.8 --w 0.6
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from analysis.stats import bootstrap_ci
from config import load
from run.arm import run_arm
from run.metrics import calculate_metrics
from runlog import log_header
from world.ablation import Ablation, apply_ablation
from world.build import build_world


def main() -> None:
    ap = argparse.ArgumentParser(description="B1 観測変化への介入寄与")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--o0mean", type=float, default=0.8)
    ap.add_argument("--w", type=float, default=0.6)
    ap.add_argument("--n-seeds", type=int, default=30)
    ap.add_argument("--seed-start", type=int, default=0)
    args = ap.parse_args()

    cfg = replace(load(args.config), init_opinion=replace(
        load(args.config).init_opinion, mean=args.o0mean))
    B = cfg.metrics.bootstrap_B
    bs = cfg.metrics.bootstrap_seed
    theta = cfg.metrics.theta
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    log_header(cfg, f"B1 mu={args.o0mean} W*={args.w} seeds={seeds[0]}..{seeds[-1]} B={B}")

    rows = []
    for seed in seeds:
        world = build_world(cfg, seed)
        attrs = apply_ablation(world.params, Ablation.NONE, cfg.model.k_react)
        ctrl = run_arm(world, attrs, W=0.0, config=cfg)
        treat = run_arm(world, attrs, W=args.w, config=cfg)
        m = calculate_metrics(ctrl, treat, world, cfg, W_level=args.w, ablation="none")
        t = m["T_star"]
        init_rate = float(np.mean(world.O0 >= theta))
        rc = float(np.mean(ctrl.O[t] >= theta))
        rt = float(np.mean(treat.O[t] >= theta))
        om_init = float(np.mean(world.O0))
        om_ctrl = float(np.mean(ctrl.O[t]))
        om_treat = float(np.mean(treat.O[t]))
        rows.append((m["E_rate"], init_rate, rc, rt, om_init, om_ctrl, om_treat))

    a = np.array(rows)
    E, init_r, rc_r, rt_r, om_i, om_c, om_t = [a[:, i] for i in range(7)]
    obs_rate = rt_r - init_r
    share = np.where(np.abs(obs_rate) > 1e-12, E / obs_rate, np.nan)
    obs_om = om_t - om_i
    share_om = np.where(np.abs(obs_om) > 1e-12, (om_t - om_c) / obs_om, np.nan)

    print("=" * 96)
    print(f"B1. 観測変化への介入寄与  設定: O0平均={args.o0mean}, W*={args.w}, "
          f"seeds={seeds[0]}..{seeds[-1]} ({len(seeds)})")
    print("=" * 96)
    print(f"  賛成率: 初期={init_r.mean():.4f}  ctrl={rc_r.mean():.4f}  treat={rt_r.mean():.4f}")
    print(f"  O_mean: 初期={om_i.mean():.4f}  ctrl={om_c.mean():.4f}  treat={om_t.mean():.4f}")
    print(f"  E_rate(平均)={E.mean():+.5f}")
    print(f"  観測変化(treat最終-初期)={obs_rate.mean():+.4f}"
          f"  [95%CI {bootstrap_ci(list(obs_rate), B=B, seed=bs)['lo']:+.4f},"
          f"{bootstrap_ci(list(obs_rate), B=B, seed=bs)['hi']:+.4f}]")
    print(f"  寄与割合 E_rate/観測変化 = {np.nanmean(share):.4f} "
          f"(シード毎平均。95%CI {bootstrap_ci([s for s in share if s==s], B=B, seed=bs)['lo']:.4f},"
          f"{bootstrap_ci([s for s in share if s==s], B=B, seed=bs)['hi']:.4f})")
    print(f"  平均ベースの寄与 = {E.mean()/obs_rate.mean():.4f}")
    print(f"  参考 O_mean: (treat-ctrl)/(treat-初期) = {(om_t-om_c).mean()/(om_t-om_i).mean():.4f}")
    print()
    print(f"  残り {1 - E.mean()/obs_rate.mean():.4f} は参加者内部と社会的影響による変化。")


if __name__ == "__main__":
    main()
