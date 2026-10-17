"""scripts/c_robustness.py — C: 規模・ネットワーク種類・persona構成比のロバストネス。

dim:
  n      : N in {50, 200, 500} (WS, k=4, p=0.15)。非単調性が規模で保たれるか。
  net    : WS / ER / BA (n=100)。トポロジーを変えても山が残るか。
  persona: persona A / B / C (n=100)。型構成比を変えても山が残るか。

主設定は主実験 mu=0.80。W は step 刻み、非単調は |E_rate| のピーク後に減少するかで判定。
ER/BA は build_world 後の neighbors だけを差し替え、O0/persona/order は同一に保つ。

実行: uv run python scripts/c_robustness.py --dim n --n-seeds 15 --step 0.1
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import networkx as nx
import numpy as np

import rng as rng_module
from analysis.stats import bootstrap_paired_diff
from config import load
from run.arm import run_arm
from run.metrics import calculate_metrics
from runlog import log_header
from world.ablation import Ablation, apply_ablation
from world.build import build_world


def neighbors_of(graph, n):
    out = []
    for i in range(n):
        arr = np.array(sorted(graph.neighbors(i)), dtype=np.int64)
        arr.flags.writeable = False
        out.append(arr)
    return tuple(out)


def build_variant_world(cfg, seed, net_type):
    """build_world を呼び、net_type が ws 以外なら neighbors のみ差し替える。"""
    world = build_world(cfg, seed)
    if net_type == "ws":
        return world
    n = cfg.network.n
    r = rng_module.streams(seed)["network"]
    if net_type == "er":
        p_edge = min(1.0, cfg.network.k / (n - 1))
        g = nx.erdos_renyi_graph(n, p_edge, seed=r)
    elif net_type == "ba":
        g = nx.barabasi_albert_graph(n, max(1, cfg.network.k // 2), seed=r)
    else:
        raise ValueError(net_type)
    return replace(world, neighbors=neighbors_of(g, n))


def scan(cfg, seeds, W_list, net_type="ws"):
    e: dict[float, list[float]] = {w: [] for w in W_list}
    weff: dict[float, list[float]] = {w: [] for w in W_list}
    for seed in seeds:
        world = build_variant_world(cfg, seed, net_type)
        attrs = apply_ablation(world.params, Ablation.NONE, cfg.model.k_react)
        ctrl = run_arm(world, attrs, W=0.0, config=cfg)
        for W in W_list:
            treat = run_arm(world, attrs, W=W, config=cfg)
            m = calculate_metrics(ctrl, treat, world, cfg, W_level=W, ablation="none")
            e[W].append(m["E_rate"])
            weff[W].append(float(np.mean(treat.W_eff)))
    return e, weff


def verdict(cfg, e, label):
    W_list = list(e.keys())
    means = {W: float(np.mean(e[W])) for W in W_list}
    peak = max(W_list, key=lambda w: abs(means[w]))
    mono = all(abs(means[a]) <= abs(means[b]) + 1e-12 for a, b in zip(W_list, W_list[1:]))
    pd = bootstrap_paired_diff(e[peak], e[W_list[-1]], B=cfg.metrics.bootstrap_B,
                               seed=cfg.metrics.bootstrap_seed)
    print(f"  {label:14s} peakW={peak:.2f} |E|={abs(means[peak]):.5f}  "
          f"d(E_peak-E_tail)={pd['mean']:+.5f}[{pd['lo']:+.5f},{pd['hi']:+.5f}]  "
          f"単調={mono}  非単調={'残る' if not mono else '消える'}")
    print("       " + " ".join(f"W={w:.2f}:{means[w]:+.4f}" for w in W_list))
    return means


def main() -> None:
    ap = argparse.ArgumentParser(description="C ロバストネス")
    ap.add_argument("--config", default="configs/main_o0_080.yaml")
    ap.add_argument("--dim", default="n", choices=["n", "net", "persona"])
    ap.add_argument("--o0mean", type=float, default=0.8)
    ap.add_argument("--n-seeds", type=int, default=15)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--step", type=float, default=0.1)
    ap.add_argument("--sizes", default="50,200,500")
    args = ap.parse_args()

    base = load(args.config)
    base = replace(base, init_opinion=replace(base.init_opinion, mean=args.o0mean))
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    W_list = [round(x, 4) for x in np.arange(args.step, 1.0 + args.step / 2, args.step)]
    log_header(base, f"C dim={args.dim} mu={args.o0mean} seeds={seeds[0]}..{seeds[-1]} step={args.step}")

    print("=" * 100)
    print(f"C ロバストネス: dim={args.dim} mu={args.o0mean} n_seeds={len(seeds)} W刻み={args.step}")
    print("=" * 100)

    if args.dim == "n":
        for n in [int(x) for x in args.sizes.split(",")]:
            cfg = replace(base, network=replace(base.network, n=n))
            e, _ = scan(cfg, seeds, W_list, "ws")
            print(f"  N={n}")
            verdict(cfg, e, f"N={n}")
    elif args.dim == "net":
        for net in ("ws", "er", "ba"):
            cfg = base
            e, weff = scan(cfg, seeds, W_list, net)
            print(f"  ネットワーク={net.upper()}")
            verdict(cfg, e, net.upper())
            print("       平均W_eff: " + " ".join(f"W={w:.2f}:{float(np.mean(weff[w])):.3f}" for w in W_list))
    elif args.dim == "persona":
        cfg_dir = Path(base.source_path).parent
        for name in ("persona_A", "persona_B", "persona_C"):
            path = str((cfg_dir / f"{name}.yaml").resolve())
            cfg = replace(base, personas=replace(base.personas, table=name, table_path=path))
            e, _ = scan(cfg, seeds, W_list, "ws")
            print(f"  {name}")
            verdict(cfg, e, name)


if __name__ == "__main__":
    main()
