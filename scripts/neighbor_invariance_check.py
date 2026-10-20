#!/usr/bin/env python
"""scripts/neighbor_invariance_check.py — 5章 line 620 の検査。

ABL-SNS+ALPHA（w_sns=0 かつ alpha=0）では参加者は互いに独立になり、
更新式の近傍項（alpha*S_O, w_sns*S_R）が両方とも消える。したがって
近傍トポロジーを変えても SNS_ALPHA アームの軌跡・結果は変化しないはず。

検査内容（シードごと）:
1. [INVARIANCE] 同一 seed で build_world → 別の連結WSグラフに近傍のみ差し替え
   (O0/R0/params/update_order は固定)、SNS_ALPHA アームを両方で走らせ、
   O 軌跡・t_conv・評価時意見が一致すること (max|ΔO|≈0)。
2. [DISCRIMINATING] 対照として none アームでは近傍差し替えで結果が変わること
   (max|ΔO|>0)。これがないと INVARIANCE が自明に成立してしまう。

実行: uv run python scripts/neighbor_invariance_check.py --config configs/default.yaml --n-seeds 10
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
from world.network import build_neighbors


def alt_neighbors(config, seed: int) -> tuple[np.ndarray, ...]:
    """build_world とは独立した乱数シードで別の連結WSグラフを凍結して返す。"""
    net_cfg = config.network
    rng = np.random.default_rng(seed)  # rng.py 経由でない明示シード (検査専用)
    return build_neighbors(
        net_cfg.n, net_cfg.k, net_cfg.p, rng, max_tries=net_cfg.max_tries
    )


def eval_state(rec, theta: float) -> np.ndarray:
    """評価時点 (t_conv 相当; 未収束なら最終) の意見を返す。"""
    t = rec.t_conv if rec.t_conv is not None else rec.O.shape[0] - 1
    return rec.O[t]


def main() -> None:
    ap = argparse.ArgumentParser(description="ABL-SNS+ALPHA 近傍不変性の検査")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--W", type=float, default=0.5)
    args = ap.parse_args()

    config = load(args.config)
    log_header(config)
    theta = config.metrics.theta
    print("================================================================================")
    print("近傍不変性の検査: ABL-SNS+ALPHA は近傍非依存 / none は近傍に依存 (discriminating)")
    print(f"config={args.config} seeds={args.seed_start}..{args.seed_start + args.n_seeds - 1} W={args.W}")
    print("================================================================================")

    max_o_inv = 0.0        # SNS_ALPHA: O 軌跡の最大差 (0 であるべき)
    max_eval_inv = 0.0     # SNS_ALPHA: 評価時意見の最大差 (0 であるべき)
    rate_inv_diffs = []    # SNS_ALPHA: 評価時賛成率の差 (0 であるべき)
    tconv_mismatch = 0     # SNS_ALPHA: t_conv 不一致数
    max_o_none = 0.0       # none: O 軌跡の最大差 (>0 であるべき)
    rate_none_diffs = []   # none: 評価時賛成率の差 (>0 であるべき)

    for seed in range(args.seed_start, args.seed_start + args.n_seeds):
        world = build_world(config, seed)
        nbrs2 = alt_neighbors(config, seed + 1_000_003)
        world2 = replace(world, neighbors=nbrs2)

        # SNS_ALPHA アーム (w_sns=0, alpha=0)
        attrs_sa = apply_ablation(world.params, Ablation.SNS_ALPHA, config.model.k_react)
        rec_sa1 = run_arm(world, attrs_sa, W=args.W, config=config)
        rec_sa2 = run_arm(world2, attrs_sa, W=args.W, config=config)
        max_o_inv = max(max_o_inv, float(np.max(np.abs(rec_sa1.O - rec_sa2.O))))
        ev1, ev2 = eval_state(rec_sa1, theta), eval_state(rec_sa2, theta)
        max_eval_inv = max(max_eval_inv, float(np.max(np.abs(ev1 - ev2))))
        rate_inv_diffs.append(abs(float(np.mean(ev1 >= theta)) - float(np.mean(ev2 >= theta))))
        if rec_sa1.t_conv != rec_sa2.t_conv:
            tconv_mismatch += 1

        # none アーム (対照: 近傍差し替えで変化するはず)
        attrs_none = apply_ablation(world.params, Ablation.NONE, config.model.k_react)
        rec_n1 = run_arm(world, attrs_none, W=args.W, config=config)
        rec_n2 = run_arm(world2, attrs_none, W=args.W, config=config)
        max_o_none = max(max_o_none, float(np.max(np.abs(rec_n1.O - rec_n2.O))))
        ne1, ne2 = eval_state(rec_n1, theta), eval_state(rec_n2, theta)
        rate_none_diffs.append(abs(float(np.mean(ne1 >= theta)) - float(np.mean(ne2 >= theta))))

    print("\n[INVARIANCE] ABL-SNS+ALPHA: 近傍を変えても SNS_ALPHA 軌跡・結果が不変")
    print(f"  max|ΔO| (全軌跡)     = {max_o_inv:.3e}  (0 であるべき)")
    print(f"  max|ΔO| (評価時点)   = {max_eval_inv:.3e}  (0 であるべき)")
    print(f"  max|Δ賛成率|          = {max(rate_inv_diffs):.3e}  (0 であるべき)")
    print(f"  t_conv 不一致シード数 = {tconv_mismatch} / {args.n_seeds}")
    inv_pass = (max_o_inv < 1e-12 and max_eval_inv < 1e-12
                and max(rate_inv_diffs) < 1e-12 and tconv_mismatch == 0)
    print(f"  {'[PASS]' if inv_pass else '[FAIL]'} NEIGHBOR_INVARIANT_IN_BOTH_ZERO: "
          f"近傍トポロジーの変化が SNS_ALPHA 結果に伝播しない")

    print("\n[DISCRIMINATING] none: 対照として近傍差し替えで結果が変わること")
    print(f"  max|ΔO| (全軌跡)  = {max_o_none:.3e}  (>0 であるべき)")
    print(f"  max|Δ賛成率|       = {max(rate_none_diffs):.3e}  (>0 であるべき)")
    disc_pass = max_o_none > 1e-6 and max(rate_none_diffs) > 1e-9
    print(f"  {'[PASS]' if disc_pass else '[FAIL]'} NEIGHBOR_SENSITIVE_IN_NONE: "
          f"検査が自明でなく近傍の影響を捉えている")

    print(f"\n総合: {'PASS' if (inv_pass and disc_pass) else 'FAIL'}")


if __name__ == "__main__":
    main()
