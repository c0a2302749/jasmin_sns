"""scripts/decompose_effect.py — 総効果の分解 (2x2 要因計画)。

実行: wsl bash -lc "cd ~/jasmin_sns && uv run python scripts/decompose_effect.py --n-seeds 10"
  本番は --n-seeds 30 --w-step 0.05。W は既定で粗い4点 (0.25/0.5/0.75/1.0)。

2x2 計画:
  C0 none            : 全経路 on。E_total (増幅を含む総効果)
  C1 ABL-SNS         : w_sns=0 (R経由除去。O経由 αS_O は残る)
  C2 ABL-ALPHA       : alpha=0 (O経由除去。R経由は残る)
  C3 ABL-SNS+ALPHA   : 両方0。参加者独立。E_direct (個人内効果)

分解 (条件内で ctrl=W0 との差を取る。条件間の比較はしない):
  増幅分 = E_total - E_direct
  R経路の寄与 = E_C2 - E_direct (目安)
  O経路の寄与 = E_C1 - E_direct (目安)
  交互作用 = E_total - E_C1 - E_C2 + E_direct (二値化は非線形のため必ず報告)
主指標は連続量 dO_mean、副指標 E_rate。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from analysis.stats import bootstrap_ci, bootstrap_paired_diff
from config import load
from run.arm import run_arm
from run.metrics import calculate_metrics
from runlog import log_header
from world.ablation import Ablation, apply_ablation
from world.build import build_world

CONDS: list[tuple[str, Ablation]] = [
    ("none", Ablation.NONE),
    ("ABL-SNS", Ablation.SNS),
    ("ABL-ALPHA", Ablation.ALPHA),
    ("ABL-SNS+ALPHA", Ablation.SNS_ALPHA),
]


def run_condition(config, seeds, cond: Ablation, W_list):
    """条件内で ctrl(W=0) との差を取り、シード別の E_rate/dO_mean を返す。

    戻り値は (e_rates, dOs, records)。records は raw保存用の
    [(seed, W, E_rate, dO_mean)] リスト。
    """
    e_rates: dict[float, list[float]] = {w: [] for w in W_list}
    dOs: dict[float, list[float]] = {w: [] for w in W_list}
    records: list[tuple[int, float, float, float]] = []
    for seed in seeds:
        world = build_world(config, seed)
        attrs = apply_ablation(world.params, cond, config.model.k_react)
        ctrl = run_arm(world, attrs, W=0.0, config=config)
        for W in W_list:
            treat = run_arm(world, attrs, W=W, config=config)
            m = calculate_metrics(ctrl, treat, world, config,
                                  W_level=W, ablation=cond.value)
            e_rates[W].append(m["E_rate"])
            dOs[W].append(m["dO_mean"])
            records.append((seed, W, m["E_rate"], m["dO_mean"]))
    return e_rates, dOs, records


def load_pool(raw_path: Path):
    """raw JSONL を {(cond, W): {seed: (E_rate, dO)}} に読み込む。"""
    pool: dict[tuple[str, float], dict[int, tuple[float, float]]] = defaultdict(dict)
    if raw_path.exists():
        with raw_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    pool[(r["cond"], float(r["W"]))][int(r["seed"])] = (
                        float(r["E_rate"]), float(r["dO"]))
    return pool


def main() -> None:
    ap = argparse.ArgumentParser(description="総効果の分解 (2x2)")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--w-step", type=float, default=0.25)
    ap.add_argument("--o0mean", type=float, default=None)
    # 分割実行用: --conds で条件を絞る (例: --conds none,ABL-SNS+ALPHA)。
    # --raw-dir で (cond,seed,W) の生値を JSONL に追記し、分割実行を合算表示する。
    ap.add_argument("--conds", default=None)
    ap.add_argument("--raw-dir", default=None)
    args = ap.parse_args()

    wanted = None
    if args.conds is not None:
        wanted = {c.strip() for c in args.conds.split(",") if c.strip()}
    conds = [(lb, cd) for lb, cd in CONDS if wanted is None or lb in wanted]
    if not conds:
        print(f"条件の指定が空: {args.conds}", file=sys.stderr)
        sys.exit(2)

    raw_path = None
    pool: dict = defaultdict(dict)
    if args.raw_dir is not None:
        raw_dir = Path(args.raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / "decomp_raw.jsonl"
        pool = load_pool(raw_path)

    config = load(args.config)
    if args.o0mean is not None:
        config = replace(config, init_opinion=replace(
            config.init_opinion, mean=args.o0mean))
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    log_header(config, f"seeds={seeds[0]}..{seeds[-1]} ({len(seeds)}) o0mean={config.init_opinion.mean}")
    W_list = [round(float(x), 4) for x in np.arange(0.0, 1.0 + args.w_step / 2, args.w_step)]
    W_list = [w for w in W_list if 0.0 < w <= 1.0]
    B = config.metrics.bootstrap_B
    bs = config.metrics.bootstrap_seed

    print("=" * 118)
    print(f"総効果の分解: seeds={seeds[0]}..{seeds[-1]} ({len(seeds)}) "
          f"W刻み={args.w_step} ({len(W_list)}点) B={B} o0mean={config.init_opinion.mean}")
    print("主指標 dO_mean / 副指標 E_rate。増幅分=E_total-E_direct (対ペア差の95%CI)")
    print("=" * 118)

    data = {}
    for label, cond in conds:
        # raw記録済みのシードは飛ばす (再開対応: 全Wが揃ったシードのみスキップ)
        if raw_path is not None:
            todo = [s for s in seeds
                    if any(s not in pool.get((label, w), {}) for w in W_list)]
        else:
            todo = list(seeds)
        if raw_path is not None and not todo:
            print(f"  [{label}] raw記録済みのためスキップ")
        elif not todo:
            print(f"  [{label}] 実行シードなしのためスキップ")
        else:
            e_rates, dOs, records = run_condition(config, todo, cond, W_list)
            data[label] = (e_rates, dOs)
            if raw_path is not None:
                with raw_path.open("a", encoding="utf-8") as f:
                    for s, w, e, d in records:
                        f.write(json.dumps({"cond": label, "seed": s, "W": w,
                                            "E_rate": e, "dO": d}) + "\n")
                pool = load_pool(raw_path)

    def paired(label_a, W, idx_a, label_b, idx_b):
        """両条件に共通するシードのみをシード順に並べて返す (対ペア用)。"""
        if raw_path is not None:
            da = pool.get((label_a, W), {})
            db = pool.get((label_b, W), {})
            common = sorted(set(da) & set(db))
            return [da[s][idx_a] for s in common], [db[s][idx_b] for s in common], len(common)
        ea, oa = data.get(label_a, ({}, {}))
        eb, ob = data.get(label_b, ({}, {}))
        la = ea.get(W, []) if idx_a == 0 else oa.get(W, [])
        lb = eb.get(W, []) if idx_b == 0 else ob.get(W, [])
        n = min(len(la), len(lb))
        return la[:n], lb[:n], n

    def series(label, W, idx):
        d = pool.get((label, W), {}) if raw_path is not None else {}
        if d:
            sds = sorted(d)
            return [d[s][idx] for s in sds], len(sds)
        e, o = data.get(label, ({}, {}))
        return (e.get(W, []), len(e.get(W, []))) if idx == 0 else (o.get(W, []), len(o.get(W, [])))

    print("  W      E_total   E_direct  増幅分dE[95%CI]            増幅割合   dO_total  dO_direct dO増幅[95%CI]   n")
    for W in W_list:
        e_tot, e_dir, n = paired("none", W, 0, "ABL-SNS+ALPHA", 0)
        dO_tot, dO_dir, _ = paired("none", W, 1, "ABL-SNS+ALPHA", 1)
        if n == 0:
            print(f"  {W:.2f}   (データなし)")
            continue
        pd_e = bootstrap_paired_diff(e_tot, e_dir, B=B, seed=bs)
        pd_d = bootstrap_paired_diff(dO_tot, dO_dir, B=B, seed=bs)
        mean_tot, mean_dir = float(np.mean(e_tot)), float(np.mean(e_dir))
        amp = pd_e["mean"]
        share = (amp / mean_tot) if abs(mean_tot) > 1e-12 else float("nan")
        print(f"  {W:.2f}   {mean_tot:+.5f}  {mean_dir:+.5f}  "
              f"{amp:+.5f} [{pd_e['lo']:+.5f},{pd_e['hi']:+.5f}]  "
              f"{share:6.2f}  {float(np.mean(dO_tot)):+.5f}  {float(np.mean(dO_dir)):+.5f}  "
              f"{pd_d['mean']:+.5f} [{pd_d['lo']:+.5f},{pd_d['hi']:+.5f}]  {n:3d}")

    print()
    print("  経路別の目安 (W=0.5/1.0 のみ): R経路=E_C2-E_direct / O経路=E_C1-E_direct / 交互作用")
    for W in (0.5, 1.0):
        if W not in W_list:
            continue
        if raw_path is not None:
            dicts = [pool.get((lb, W), {}) for lb in
                     ("none", "ABL-SNS", "ABL-ALPHA", "ABL-SNS+ALPHA")]
            common = sorted(set.intersection(*(set(d) for d in dicts))) if all(dicts) else []
            if not common:
                ns = "/".join(str(len(d)) for d in dicts)
                print(f"    W={W:.2f}: (データ不足 n={ns})")
                continue
            m = {lb: float(np.mean([dicts[i][s][0] for s in common]))
                 for i, lb in enumerate(("none", "ABL-SNS", "ABL-ALPHA", "ABL-SNS+ALPHA"))}
            nn = len(common)
        else:
            e_tot, n0 = series("none", W, 0)
            e_c1, n1 = series("ABL-SNS", W, 0)
            e_c2, n2 = series("ABL-ALPHA", W, 0)
            e_dir, n3 = series("ABL-SNS+ALPHA", W, 0)
            if min(n0, n1, n2, n3) == 0:
                print(f"    W={W:.2f}: (データ不足 n={n0}/{n1}/{n2}/{n3})")
                continue
            m = {"none": float(np.mean(e_tot)), "ABL-SNS": float(np.mean(e_c1)),
                 "ABL-ALPHA": float(np.mean(e_c2)), "ABL-SNS+ALPHA": float(np.mean(e_dir))}
            nn = min(n0, n1, n2, n3)
        r_path = m["ABL-ALPHA"] - m["ABL-SNS+ALPHA"]
        o_path = m["ABL-SNS"] - m["ABL-SNS+ALPHA"]
        ia = m["none"] - m["ABL-SNS"] - m["ABL-ALPHA"] + m["ABL-SNS+ALPHA"]
        print(f"    W={W:.2f}(n={nn}): R経路 {r_path:+.5f} / O経路 {o_path:+.5f} / "
              f"交互作用 {ia:+.5f} (E_total={m['none']:+.5f})")


if __name__ == "__main__":
    main()
