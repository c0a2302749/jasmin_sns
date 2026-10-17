"""scripts/select_main_experiment.py — A4: 主実験のO0平均選定とW*確認。

手順:
  1. 探索 (seeds 0-29): mu in {0.5,0.6,0.7,0.75,0.8,0.85,0.9} の7点で W を0.05刻み走査し、
     |E_rate|最大の W* を決める。
     同時に介入なし賛成率 ctrl率 と初期意見クリップ率 n_clipped_O0/n を出す。
  2. 選定基準 (結果を見る前の事前固定):
       介入なし賛成率 in [0.4, 0.6] かつ 初期意見クリップ率 < 0.05。
  3. 確認 (seeds 60-89): 選んだ mu と W* で E(W*) と E(1.0) を対にし、
     絶対値差 |E(W*)|-|E(1.0)| の95%CI (B=10000) を出す。E_maj も出す。

実行: uv run python scripts/select_main_experiment.py
"""
from __future__ import annotations

import argparse
import datetime
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from analysis.stats import bootstrap_ci, bootstrap_paired_diff, summarize_emaj
from config import load
from run.arm import run_arm
from run.metrics import calculate_metrics
from runlog import log_header
from world.ablation import Ablation, apply_ablation
from world.build import build_world

CRITERION = "介入なし賛成率 in [0.4,0.6] かつ 初期意見クリップ率 < 0.05"


def scan(config, mu, seeds, W_list):
    """mu 上書き config で seeds 上の W別 E_rate/E_maj と ctrl率/clip率を一度に返す。"""
    cfg = replace(config, init_opinion=replace(config.init_opinion, mean=mu))
    e: dict[float, list[float]] = {w: [] for w in W_list}
    emaj: dict[float, list[int]] = {w: [] for w in W_list}
    ctrl_rates: list[float] = []
    clip = 0
    n = cfg.network.n
    theta = cfg.metrics.theta
    for seed in seeds:
        world = build_world(cfg, seed)
        clip += world.n_clipped_O0
        attrs = apply_ablation(world.params, Ablation.NONE, cfg.model.k_react)
        ctrl = run_arm(world, attrs, W=0.0, config=cfg)
        tc = ctrl.t_conv if ctrl.t_conv is not None else ctrl.O.shape[0] - 1
        ctrl_rates.append(float(np.mean(ctrl.O[tc] >= theta)))
        for W in W_list:
            treat = run_arm(world, attrs, W=W, config=cfg)
            m = calculate_metrics(ctrl, treat, world, cfg, W_level=W, ablation="none")
            e[W].append(m["E_rate"])
            emaj[W].append(m["E_maj"])
    return e, emaj, float(np.mean(ctrl_rates)), clip / (len(seeds) * n)


def main() -> None:
    ap = argparse.ArgumentParser(description="主実験のO0平均選定とW*確認")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--mus", default="0.5,0.6,0.7,0.75,0.8,0.85,0.9")
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--explore-start", type=int, default=0)
    ap.add_argument("--explore-n", type=int, default=30)
    ap.add_argument("--confirm-start", type=int, default=60)
    ap.add_argument("--confirm-n", type=int, default=30)
    args = ap.parse_args()

    base = load(args.config)
    B = base.metrics.bootstrap_B
    bs = base.metrics.bootstrap_seed
    mus = [float(x) for x in args.mus.split(",")]
    ex_seeds = list(range(args.explore_start, args.explore_start + args.explore_n))
    cf_seeds = list(range(args.confirm_start, args.confirm_start + args.confirm_n))
    W_grid = [round(x, 4) for x in np.arange(args.step, 1.0 + args.step / 2, args.step)]
    log_header(base, f"A4 explore={ex_seeds[0]}..{ex_seeds[-1]} confirm={cf_seeds[0]}..{cf_seeds[-1]} B={B}")

    print("=" * 100)
    print("A4. 主実験 O0平均の選定と W* 確認")
    print("=" * 100)
    print(f"事前固定した選定基準 (記録日 {datetime.date.today().isoformat()}): {CRITERION}")
    print("  固定前に観測済みの値 (A1 core走査, seeds 0-29, W=0.5/1.0):")
    print("    mu=0.8 : ctrl率 0.491, E(0.5)=-0.21033, E(1.0)=-0.11767")
    print("    mu=0.85: ctrl率 0.533, E(0.5)=-0.18267, E(1.0)=-0.10900")
    print(f"  探索 seeds={ex_seeds[0]}..{ex_seeds[-1]} ({len(ex_seeds)}), "
          f"確認 seeds={cf_seeds[0]}..{cf_seeds[-1]} ({len(cf_seeds)}), B={B}")
    print()

    chosen: dict[float, dict] = {}
    print("[探索] W 走査 (seeds 0-29, 0.05刻み)")
    print("  mu    ctrl率  clip率   基準  W*     |E(W*)|max  E(W*)      E(1.0)")
    for mu in mus:
        e, emaj, ctrl_rate, clip_rate = scan(base, mu, ex_seeds, W_grid)
        means = {W: float(np.mean(e[W])) for W in W_grid}
        Wstar = max(W_grid, key=lambda w: abs(means[w]))
        ok = (0.4 <= ctrl_rate <= 0.6) and (clip_rate < 0.05)
        chosen[mu] = {"Wstar": Wstar, "means": means, "ctrl": ctrl_rate,
                      "clip": clip_rate, "ok": ok, "explore": (e, emaj)}
        print(f"  {mu:.2f}  {ctrl_rate:.4f}  {clip_rate:.4f}  {'合格' if ok else '不合格':>4}  "
              f"{Wstar:.2f}  {abs(means[Wstar]):.5f}    {means[Wstar]:+.5f}  {means[1.0]:+.5f}")
        sel = sorted({0.1, 0.25, 0.5, Wstar, 0.75, 1.0})
        print("       E_rate: " + ", ".join(f"W={w:.2f}:{means[w]:+.5f}" for w in sel))

    print()
    print("[確認] seeds 60-89 で W* と W=1.0 (絶対値差, 対ペア, B=10000)")
    print("  mu    W*     E(W*)      E(1.0)     |E(W*)|-|E(1.0)| [95%CI]           有意  "
          "ctrl率  clip率  E_maj(W*)")
    for mu in mus:
        Wstar = chosen[mu]["Wstar"]
        e, emaj, ctrl_rate, clip_rate = scan(base, mu, cf_seeds, [Wstar, 1.0])
        e_star, e_tail = e[Wstar], e[1.0]
        pd = bootstrap_paired_diff([abs(v) for v in e_star], [abs(v) for v in e_tail], B=B, seed=bs)
        em = summarize_emaj(emaj[Wstar])
        ci_s = bootstrap_ci(e_star, B=B, seed=bs)
        ci_t = bootstrap_ci(e_tail, B=B, seed=bs)
        chosen[mu]["confirm"] = {"Wstar": Wstar, "pd": pd, "ctrl": ctrl_rate,
                                 "clip": clip_rate, "em": em, "ci_s": ci_s, "ci_t": ci_t}
        print(f"  {mu:.2f}  {Wstar:.2f}  {float(np.mean(e_star)):+.5f}  {float(np.mean(e_tail)):+.5f}  "
              f"{pd['mean']:+.5f} [{pd['lo']:+.5f},{pd['hi']:+.5f}]  "
              f"{'Y' if bool(pd['significant']) else 'n':>4}  "
              f"{ctrl_rate:.4f}  {clip_rate:.4f}  "
              f"mean={em['mean']:+.2f} p(-1)={em['p_minus1']:.2f} p(0)={em['p_0']:.2f} p(+1)={em['p_plus1']:.2f}")

    print()
    print("[参考] 確認シードでの E の95%CI")
    for mu in mus:
        c = chosen[mu]["confirm"]
        print(f"  mu={mu:.2f} W*={c['Wstar']:.2f}: E(W*)={c['ci_s']['mean']:+.5f}"
              f"[{c['ci_s']['lo']:+.5f},{c['ci_s']['hi']:+.5f}]  "
              f"E(1.0)={c['ci_t']['mean']:+.5f}[{c['ci_t']['lo']:+.5f},{c['ci_t']['hi']:+.5f}]")

    ok_mus = [mu for mu in mus if chosen[mu]["ok"]]
    print()
    if ok_mus:
        print(f"→ 選定基準を満たす候補: {', '.join(f'{mu:.2f}' for mu in ok_mus)}")
    else:
        print("→ 選定基準を満たす候補なし。基準の再検討が必要。")


if __name__ == "__main__":
    main()
