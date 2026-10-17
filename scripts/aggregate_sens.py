"""scripts/aggregate_sens.py — sens_raw.jsonl を集計し、表5.15用の行を出す。

実行: uv run python scripts/aggregate_sens.py --raw-dir logs/raw_sens --config configs/default.yaml
  期待シード数 (既定30) に満たない条件は *INCOMPLETE* と表示する。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from analysis.stats import bootstrap_paired_diff
from config import load


def main() -> None:
    ap = argparse.ArgumentParser(description="感度分析rawの集計")
    ap.add_argument("--raw-dir", default="logs/raw_sens")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n-seeds", type=int, default=30)
    args = ap.parse_args()

    config = load(args.config)
    B, bs = config.metrics.bootstrap_B, config.metrics.bootstrap_seed
    groups: dict[str, dict[int, tuple[float, float]]] = defaultdict(dict)
    raw_path = Path(args.raw_dir) / "sens_raw.jsonl"
    if raw_path.exists():
        with raw_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    groups[r["variant"]][int(r["seed"])] = (float(r["E05"]), float(r["E10"]))

    print(f"集計: {raw_path} B={B} 期待シード数={args.n_seeds}")
    print("  条件          n   E(0.5)    E(1.0)    非単調?  d=E.5-E1.0 [95%CI]")
    for label in sorted(groups):
        d = groups[label]
        n = len(d)
        flag = "" if n >= args.n_seeds else " *INCOMPLETE*"
        sds = sorted(d)
        e05 = [d[s][0] for s in sds]
        e10 = [d[s][1] for s in sds]
        m05, m10 = float(np.mean(e05)), float(np.mean(e10))
        pd = bootstrap_paired_diff(e05, e10, B=B, seed=bs)
        nm = "Y" if (bool(pd["significant"]) and pd["mean"] < 0) else "n"
        print(f"  {label:12s} {n:3d}  {m05:+.5f}  {m10:+.5f}  {nm}       "
              f"{pd['mean']:+.5f} [{pd['lo']:+.5f},{pd['hi']:+.5f}]{flag}")


if __name__ == "__main__":
    main()
