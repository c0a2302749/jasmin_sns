"""scripts/negative_controls.py — 陰性対照 C2・C4 と persona_B 感度。

実行: uv run python scripts/negative_controls.py --n-seeds 30
  部分実行: --only c2 | c4 | personab

C2 (天井): 全員の b=0.9 に置換 → W=0でRが上限に張り付く設定。
  予想: Wを動かしてもEがほぼ変わらない (天井がEの読み取りを潰す)。
  対照: 既定bでの用量反応 (EがWで変化)。
C4 (短T_max): T_max=10/12 vs 500 (W=0.5)。
  収束率は介入なしアームのみで数える (C4の定義どおり)。
  予想: T_max=10で収束率が下がり、基準が不合格を検出する。
persona_B: 別ペルソナ表で結論の形状 (非単調＋ABL-SYS消失) が保たれるか。
  (選定の物語ではなく感度分析として扱う)
C3の陰性対照は作らない (k=0で非単調が既に消えており重なるため)。
C1の陰性対照も作らない (b=0はβ=0と同義、O0平均は感度分析済み)。
"""
from __future__ import annotations

import argparse
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


def run_pair(config, seed, params, W):
    world = build_world(config, seed)
    use_params = params if params is not None else world.params
    attrs = apply_ablation(use_params, Ablation.NONE, config.model.k_react)
    ctrl = run_arm(world, attrs, W=0.0, config=config)
    treat = run_arm(world, attrs, W=W, config=config)
    m = calculate_metrics(ctrl, treat, world, config, W_level=W, ablation="none")
    return m, ctrl, treat, world, attrs


def main() -> None:
    ap = argparse.ArgumentParser(description="陰性対照 C2/C4 + persona_B感度")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n-seeds", type=int, default=30)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--only", default="all",
                    choices=["all", "c2", "c4", "personab"])
    args = ap.parse_args()

    config = load(args.config)
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    log_header(config, f"seeds={seeds[0]}..{seeds[-1]} ({len(seeds)}) only={args.only}")
    n = config.network.n
    do = lambda name: args.only in ("all", name)

    if do("c2"):
        print("=" * 96)
        print("C2. 天井の陰性対照: b=0.9 (W=0でRが上限に張り付く)")
        print("=" * 96)
        W_grid = (0.25, 0.5, 0.75, 1.0)
        for label, b_override in (("既定b(対照)", None), ("b=0.9(天井)", 0.9)):
            es: dict[float, list[float]] = {w: [] for w in W_grid}
            clip0: list[float] = []
            for seed in seeds:
                world = build_world(config, seed)
                params = None
                if b_override is not None:
                    params = [replace(p, b=b_override) for p in world.params]
                # W=0のclip率はctrl側から直接取る
                use_params = params if params is not None else world.params
                attrs = apply_ablation(use_params, Ablation.NONE, config.model.k_react)
                c = run_arm(world, attrs, W=0.0, config=config)
                denom = config.model.T_max * n
                clip0.append(c.n_clipped_R / denom)
                for W in W_grid:
                    t = run_arm(world, attrs, W=W, config=config)
                    m = calculate_metrics(c, t, world, config, W_level=W, ablation="none")
                    es[W].append(m["E_rate"])
            means = {w: float(np.mean(es[w])) for w in W_grid}
            spread = max(means.values()) - min(means.values())
            print(f"  {label}: W=0 clip率={float(np.mean(clip0)):.4f} "
                  + " ".join(f"E({w})={means[w]:+.5f}" for w in W_grid)
                  + f" 振れ幅={spread:.5f}")

    print()
    if do("c4"):
        print("=" * 96)
        print("C4. 短T_maxの陰性対照: T_max=10/12 vs 500 (W=0.5、収束率は介入なしのみ)")
        print("=" * 96)
        for tmax in (10, 12, 500):
            cfg = replace(config, model=replace(config.model, T_max=tmax))
            conv = 0
            es: list[float] = []
            for seed in seeds:
                m, ctrl, treat, _, _ = run_pair(cfg, seed, None, 0.5)
                es.append(m["E_rate"])
                if ctrl.t_conv is not None:
                    conv += 1
            print(f"  T_max={tmax}: 収束率(ctrlのみ)={conv}/{len(seeds)}={conv/len(seeds):.3f} "
                  f"E(0.5)={float(np.mean(es)):+.5f}")

    print()
    if do("personab"):
        print("=" * 96)
        print("persona_B感度: 非単調 (|E0.5|>|E1.0|) と ABL-SYS消失")
        print("=" * 96)
        config_dir = Path(config.source_path).parent
        new_path = str((config_dir / "persona_B.yaml").resolve())
        cfgB = replace(config, personas=replace(
            config.personas, table="persona_B", table_path=new_path))
        e05: list[float] = []
        e10: list[float] = []
        sys_ok = True
        for seed in seeds:
            world = build_world(cfgB, seed)
            attrs = apply_ablation(world.params, Ablation.NONE, cfgB.model.k_react)
            attrs_sys = apply_ablation(world.params, Ablation.SYS, cfgB.model.k_react)
            ctrl = run_arm(world, attrs, W=0.0, config=cfgB)
            ctrl_sys = run_arm(world, attrs_sys, W=0.0, config=cfgB)
            for W, acc in ((0.5, e05), (1.0, e10)):
                treat = run_arm(world, attrs, W=W, config=cfgB)
                acc.append(calculate_metrics(
                    ctrl, treat, world, cfgB, W_level=W, ablation="none")["E_rate"])
                treat_sys = run_arm(world, attrs_sys, W=W, config=cfgB)
                ms = calculate_metrics(
                    ctrl_sys, treat_sys, world, cfgB, W_level=W, ablation="ABL-SYS")
                if not (ms["E_rate"] == 0.0 and ms["flip_rate"] == 0.0 and ms["dO_mean"] == 0.0):
                    sys_ok = False
        m05, m10 = float(np.mean(e05)), float(np.mean(e10))
        pdb = bootstrap_paired_diff(e05, e10, B=cfgB.metrics.bootstrap_B, seed=cfgB.metrics.bootstrap_seed)
        nonmono_b = bool(pdb["significant"]) and pdb["mean"] < 0
        print(f"  persona_B: E(0.5)={m05:+.5f} E(1.0)={m10:+.5f} "
              f"d={pdb['mean']:+.5f} [{pdb['lo']:+.5f},{pdb['hi']:+.5f}] "
              f"非単調(対ペア有意)={'Y' if nonmono_b else 'n'} ABL-SYSゼロ={'Y' if sys_ok else 'N'}")


if __name__ == "__main__":
    main()
