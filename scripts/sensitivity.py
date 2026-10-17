"""scripts/sensitivity.py — 感度分析 (k_react, S, O0平均 + p, knet, std)。

実行: wsl bash -lc "cd ~/jasmin_sns && uv run python scripts/sensitivity.py --n-seeds 10"
  (本番は --n-seeds 30。W は {0.5, 1.0} の2点 + ABL-SYS ゼロ確認)
  部分実行: --dims p|knet|std|core|all (既定 all。重いので分割実行可)

基準: 結論の向き (E_rate(0.5)<0、ABL-SYS で効果消失) が保たれる範囲を特定。
非単調性は平均の大小ではなく、同一シードで対にした差の
bootstrap信頼区間 (bootstrap_paired_diff) で判定する
（persona_Bのような小差がノイズでYになるのを防ぐ）。
差の定義は d = E(0.5) - E(1.0) (両者とも負、非単調なら d<0)。
正典 p=0.15 (default.yaml)。論文表5.3の p=0.1 は旧値として扱う。
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


def eval_config(config, seeds, W_list=(0.5, 1.0)):
    out: dict[float, list[float]] = {w: [] for w in W_list}
    ctrl_rates: list[float] = []
    clip_counts: list[float] = []
    sys_zero_ok = True
    for seed in seeds:
        world = build_world(config, seed)
        attrs = apply_ablation(world.params, Ablation.NONE, config.model.k_react)
        attrs_sys = apply_ablation(world.params, Ablation.SYS, config.model.k_react)
        ctrl = run_arm(world, attrs, W=0.0, config=config)
        ctrl_sys = run_arm(world, attrs_sys, W=0.0, config=config)
        ctrl_rates.append(float(np.mean(ctrl.O[ctrl.t_conv or config.model.T_max] >= config.metrics.theta)) if ctrl.t_conv else float(np.mean(ctrl.O[-1] >= config.metrics.theta)))
        clip_counts.append(float(world.n_clipped_O0) / float(config.network.n))
        for W in W_list:
            treat = run_arm(world, attrs, W=W, config=config)
            m = calculate_metrics(ctrl, treat, world, config, W_level=W, ablation="none")
            out[W].append(m["E_rate"])
            treat_sys = run_arm(world, attrs_sys, W=W, config=config)
            ms = calculate_metrics(ctrl_sys, treat_sys, world, config,
                                   W_level=W, ablation="ABL-SYS")
            if not (ms["E_rate"] == 0.0 and ms["flip_rate"] == 0.0 and ms["dO_mean"] == 0.0):
                sys_zero_ok = False
    means = {w: float(np.mean(v)) for w, v in out.items()}
    return means, sys_zero_ok, out, float(np.mean(ctrl_rates))


def stream_separation_check(base, variant, seeds) -> tuple[bool, str]:
    """p を変えても init/persona/order の3配列が不変か (4.2.2の根拠の実証)。

    network ストリームのみが変化し、他3本が独立であることを確認する。
    neighbors は変わってよい。O0・update_order・theta_react が bit 一致すること。
    """
    for seed in seeds[:3]:
        w0 = build_world(base, seed)
        w1 = build_world(variant, seed)
        o0_ok = bool(np.array_equal(w0.O0, w1.O0))
        ord_ok = bool(np.array_equal(w0.update_order, w1.update_order))
        th0 = np.array([p.theta_react for p in w0.params])
        th1 = np.array([p.theta_react for p in w1.params])
        th_ok = bool(np.array_equal(th0, th1))
        if not (o0_ok and ord_ok and th_ok):
            return False, f"seed{seed}:O0={o0_ok},order={ord_ok},theta={th_ok}"
    return True, "O0/order/theta が3シードでbit一致"


def main() -> None:
    ap = argparse.ArgumentParser(description="感度分析 (k/S/O0平均 + p/knet/std)")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--dims", default="all",
                    choices=["all", "core", "p", "knet", "std"])
    # 分割実行用: --variant で単一条件のみ実行 (例: --variant knet=6)。
    # --raw-dir でシード別の生値を JSONL に追記し、中断・再開と後集計を可能にする。
    ap.add_argument("--variant", default=None)
    ap.add_argument("--raw-dir", default=None)
    args = ap.parse_args()

    raw_path = None
    done: set[tuple[str, int]] = set()
    if args.raw_dir is not None:
        raw_dir = Path(args.raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / "sens_raw.jsonl"
        if raw_path.exists():
            with raw_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        r = json.loads(line)
                        done.add((r["variant"], int(r["seed"])))

    base = load(args.config)
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    log_header(base, f"seeds={seeds[0]}..{seeds[-1]} ({len(seeds)}) dims={args.dims}")
    W_list = (0.5, 1.0)

    variants: list[tuple[str, object]] = []
    if args.dims in ("all", "core"):
        for k in (0.0, 0.5, 1.0, 1.5, 2.0):
            variants.append((f"k_react={k}",
                             replace(base, model=replace(base.model, k_react=k))))
        for s in (1.0, 1.5, 2.0, 2.5):
            variants.append((f"S={s}", replace(base, model=replace(base.model, S=s))))
        for mu in (0.3, 0.5, 0.7, 0.8, 0.85):
            variants.append((f"O0mean={mu}",
                             replace(base, init_opinion=replace(base.init_opinion, mean=mu))))
    if args.dims in ("all", "p"):
        # 正典 p=0.15 を含む5点 + 論文指定 p=0.1。1つずつ動かし他は既定に固定する。
        for p in (0.0, 0.05, 0.1, 0.15, 0.3, 1.0):
            variants.append((f"p={p}",
                             replace(base, network=replace(base.network, p=p))))
    if args.dims in ("all", "knet"):
        for knet in (2, 4, 6, 10):
            variants.append((f"knet={knet}",
                             replace(base, network=replace(base.network, k=knet))))
    if args.dims in ("all", "std"):
        for std in (0.05, 0.1, 0.15, 0.2):
            variants.append((f"std={std}",
                             replace(base, init_opinion=replace(base.init_opinion, std=std))))

    print("=" * 120)
    print(f"感度分析: seeds={seeds[0]}..{seeds[-1]} ({len(seeds)}) W={W_list}")
    print("基準: E_rate(0.5)<0 の向き + ABL-SYS ゼロ + 非単調(対ペア差が負に有意)")
    print("=" * 120)
    print("  条件          E(0.5)    E(1.0)    ctrl率  向き負?  非単調?  d=E.5-E1.0 [95%CI]       ABL-SYSゼロ?  分離?")
    for label, cfg in variants:
        if args.variant is not None and label != args.variant:
            continue
        # raw に記録済みのシードは飛ばす (再開対応)。未記録シードのみ実行する。
        todo_seeds = [s for s in seeds if (label, s) not in done]
        if not todo_seeds:
            print(f"  {label:12s}  (raw記録済みのためスキップ)")
            continue
        means, sys_ok, out, ctrl_rate = eval_config(cfg, todo_seeds, W_list)
        e05, e10 = means[0.5], means[1.0]
        neg = e05 < 0
        # P0: 差の定義は d = E(0.5) - E(1.0) (両者とも負、非単調なら d<0)。
        # 論文 5.4.7 の定義文 E(1.0)-E(0.5) とは符号が逆なので、論文側を d<0 に直すこと。
        # B はハードコードせず config.metrics.bootstrap_B に統一する。
        pd = bootstrap_paired_diff(out[0.5], out[1.0], B=cfg.metrics.bootstrap_B,
                                   seed=cfg.metrics.bootstrap_seed)
        nonmono = bool(pd["significant"]) and pd["mean"] < 0
        sep_txt = "-"
        if label.startswith("p="):
            sep_ok, _ = stream_separation_check(base, cfg, seeds)
            sep_txt = "Y" if sep_ok else "*N*"
        print(f"  {label:12s}  {e05:+.5f}  {e10:+.5f}  {ctrl_rate:.3f}  "
              f"{'Y' if neg else '*N*'}      {'Y' if nonmono else 'n'}       "
              f"{pd['mean']:+.5f} [{pd['lo']:+.5f},{pd['hi']:+.5f}]  "
              f"{'Y' if sys_ok else '*N*'}          {sep_txt}")
        if raw_path is not None:
            with raw_path.open("a", encoding="utf-8") as f:
                for s, v05, v10 in zip(todo_seeds, out[0.5], out[1.0]):
                    f.write(json.dumps({"variant": label, "seed": s,
                                        "E05": v05, "E10": v10}) + "\n")


if __name__ == "__main__":
    main()
