"""scripts/check_emaj_rates.py — B1: E_maj=0の理由特定。

- B1a: 既定設定で介入なし賛成率 rate_ctrl の水準を確認する
  (ctrlのみ30アームで足りる。rate_ctrlはWに依存しない)。
  シフトは標本の初期平均を基準にする (diagnose_beta0 と同一定義)。
  評価時点は単腕のt_conv（対比較のT*ではない。収束後の差は微小）。
- B1b: O0平均0.7/0.9の変異設定で W={0.45,0.5,1.0} の E_rate/E_maj を測り、
  E_majが識別力を持つ適用範囲を示す (std=0.1継承)。

実行: uv run python scripts/check_emaj_rates.py --n-seeds 30
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from analysis.stats import summarize_emaj
from config import load
from run.arm import run_arm
from run.metrics import calculate_metrics, t_star
from runlog import log_header
from world.ablation import Ablation, apply_ablation
from world.build import build_world


def main() -> None:
    ap = argparse.ArgumentParser(description="E_maj=0の理由: rate_ctrl水準とO0変異でのE_maj")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n-seeds", type=int, default=30)
    ap.add_argument("--seed-start", type=int, default=0)
    args = ap.parse_args()

    config = load(args.config)
    theta = config.metrics.theta
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    log_header(config, f"seeds={seeds[0]}..{seeds[-1]} ({len(seeds)}) theta={theta}")

    print("=" * 96)
    print("B1a. 介入なし賛成率 rate_ctrl の水準 (既定設定、単腕t_conv評価)")
    print("=" * 96)
    rate_ctrls: list[float] = []
    mean_os: list[float] = []
    mean_o0s: list[float] = []
    nclip_o0: list[int] = []
    for seed in seeds:
        world = build_world(config, seed)
        attrs = apply_ablation(world.params, Ablation.NONE, config.model.k_react)
        ctrl = run_arm(world, attrs, W=0.0, config=config)
        t = ctrl.t_conv if ctrl.t_conv is not None else ctrl.O.shape[0] - 1
        rate_ctrls.append(float(np.mean(ctrl.O[t] >= theta)))
        mean_os.append(float(np.mean(ctrl.O[t])))
        mean_o0s.append(float(np.mean(world.O0)))
        nclip_o0.append(world.n_clipped_O0)
    rc = np.array(rate_ctrls)
    n = config.network.n
    print(f"  mean(rate_ctrl)={float(np.mean(rc)):.5f}  "
          f"min={float(np.min(rc)):.3f} max={float(np.max(rc)):.3f}")
    print(f"  mean(O0標本)={float(np.mean(mean_o0s)):.5f} "
          f"mean(O_ctrl)={float(np.mean(mean_os)):.5f} "
          f"シフト(標本基準)={float(np.mean(mean_os)) - float(np.mean(mean_o0s)):+.5f}")
    print(f"  n_clipped_O0={sum(nclip_o0)}/{len(seeds) * n} "
          f"= {sum(nclip_o0) / (len(seeds) * n):.5f} (5σ離れ・影響ほぼなしの確認)")
    print(f"  rate_ctrl>=0.5 のシード数: {int(np.sum(rc >= 0.5))}/{len(seeds)}")
    print(f"  rate_ctrl>=0.5 が0件なら、両アームは常に0.5の同側→E_maj=0が必然。")
    hist, edges = np.histogram(rc, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.01])
    print("  分布: " + ", ".join(f"[{lo:.1f},{hi:.1f}):{c}" for (lo, hi), c in zip(zip(edges, edges[1:]), hist)))

    print()
    print("=" * 96)
    print("B1b. O0平均0.7/0.9での E_rate / E_maj (W=0.45,0.5,1.0, std=0.1継承)")
    print("=" * 96)
    for mu in (0.7, 0.9):
        cfg7 = replace(config, init_opinion=replace(config.init_opinion, mean=mu))
        clip = 0
        for W in (0.45, 0.5, 1.0):
            es: list[float] = []
            ms: list[int] = []
            rcs: list[float] = []
            rts: list[float] = []
            for seed in seeds:
                world = build_world(cfg7, seed)
                if W == 0.45:
                    clip += world.n_clipped_O0
                attrs = apply_ablation(world.params, Ablation.NONE, cfg7.model.k_react)
                ctrl = run_arm(world, attrs, W=0.0, config=cfg7)
                treat = run_arm(world, attrs, W=W, config=cfg7)
                m = calculate_metrics(ctrl, treat, world, cfg7, W_level=W, ablation="none")
                es.append(m["E_rate"])
                ms.append(m["E_maj"])
                t = t_star(ctrl, treat, cfg7)
                rcs.append(float(np.mean(ctrl.O[t] >= theta)))
                rts.append(float(np.mean(treat.O[t] >= theta)))
            em = summarize_emaj(ms)
            print(f"  mu={mu:.1f} W={W:.2f} E_rate={float(np.mean(es)):+.5f} "
                  f"mean(rate_ctrl)={float(np.mean(rcs)):.3f} mean(rate_treat)={float(np.mean(rts)):.3f} "
                  f"E_maj: mean={em['mean']:+.3f} p(-1)={em['p_minus1']:.3f} "
                  f"p(0)={em['p_0']:.3f} p(+1)={em['p_plus1']:.3f}")
        print(f"  mu={mu:.1f} n_clipped_O0={clip}/{len(seeds) * n} "
              f"= {clip / (len(seeds) * n):.4f}")


if __name__ == "__main__":
    main()
