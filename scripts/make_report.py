#!/usr/bin/env python3
"""scripts/make_report.py — F-16: 本番runの summary から report.md と図を出す。

実行: uv run python scripts/make_report.py --run runs/<run_id> [--out reports/<run_id>]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analysis.report import make
from world.persona import load_persona_table


def main() -> None:
    ap = argparse.ArgumentParser(description="F-16 report.md と図の生成")
    ap.add_argument("--run", required=True, help="run_id ディレクトリ (manifest.json を含む)")
    ap.add_argument("--out", default=None, help="出力先 (既定: <run>/report)")
    args = ap.parse_args()

    run_dir = Path(args.run)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    cfg = manifest["config"]
    out_dir = Path(args.out) if args.out else (run_dir / "report")

    bounds: dict[str, float] = {}
    for row in load_persona_table(cfg["personas"]["table_path"]):
        b = float(row["w_sys"] + row["w_sns"] + row["b"])
        bounds[str(row["name"])] = max(bounds.get(str(row["name"]), 0.0), b)

    md = make(
        run_dir / "summary" / "part.parquet",
        out_dir,
        B=int(cfg["metrics"]["bootstrap_B"]),
        boot_seed=int(cfg["metrics"]["bootstrap_seed"]),
        T_max=int(cfg["model"]["T_max"]),
        n=int(cfg["network"]["n"]),
        persona_bounds=bounds,
    )
    print(f"report を生成しました: {md}")


if __name__ == "__main__":
    main()
