"""scripts/export_schema_csv.py — スキーマ準拠の summary CSV (seed×条件×W の全列) を出力する。

io_/schema.py の SUMMARY_SCHEMA を正典として列・型を固定し、指定シード範囲で
seed -> ablation -> W の要約を計算して CSV にする。trajectory は書かない
(CSV に必要な summary のみを直接計算する軽量版)。

出力:
  <out-dir>/schema_summary_seeds{start}-{end}_{config4hex}.csv
  <out-dir>/hashes.json  (git commit / configハッシュ / CSV sha256 / 行数)

実行: uv run python scripts/export_schema_csv.py --config configs/main_o0_080.yaml \
        --seed-start 60 --n-seeds 30
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import polars as pl

from config import load, run_id
from io_.schema import SUMMARY_SCHEMA
from run.arm import run_arm
from run.metrics import calculate_metrics
from runlog import log_header
from world.ablation import Ablation, apply_ablation
from world.build import build_world


def collect_summaries(config, seeds) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        world = build_world(config, seed)
        for condition in config.experiment.ablation_conditions:
            attrs = apply_ablation(world.params, Ablation(condition), config.model.k_react)
            ctrl = run_arm(world, attrs, W=0.0, config=config)
            for W in config.experiment.W_values:
                treat = ctrl if W == 0.0 else run_arm(world, attrs, W=W, config=config)
                rows.append(calculate_metrics(ctrl, treat, world, config,
                                              W_level=W, ablation=condition))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="SUMMARY_SCHEMA準拠のCSVを出力")
    ap.add_argument("--config", default="configs/main_o0_080.yaml")
    ap.add_argument("--seed-start", type=int, default=60)
    ap.add_argument("--n-seeds", type=int, default=30)
    ap.add_argument("--out-dir", default="artifacts")
    args = ap.parse_args()

    base = load(args.config)
    cfg = replace(base, experiment=replace(
        base.experiment, master_seed_start=args.seed_start, n_seeds=args.n_seeds))
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    digest = run_id(cfg).rsplit("_", 1)[-1]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_header(cfg, f"export_schema_csv seeds={seeds[0]}..{seeds[-1]} ({len(seeds)}) "
                    f"conds={cfg.experiment.ablation_conditions} W={cfg.experiment.W_values}")
    print("=" * 96)
    print(f"スキーマCSV出力: seeds={seeds[0]}..{seeds[-1]} ({len(seeds)}) config_hash={digest}")
    print("=" * 96)

    rows = collect_summaries(cfg, seeds)
    df = pl.DataFrame(rows, schema=SUMMARY_SCHEMA).sort(["seed", "ablation_condition", "W"])
    csv_path = out_dir / f"schema_summary_seeds{seeds[0]}-{seeds[-1]}_{digest}.csv"
    df.write_csv(csv_path)

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, cwd=Path(__file__).resolve().parents[1],
                                timeout=30).stdout.strip() or "unknown"
        dirty = bool(subprocess.run(["git", "status", "--short"], capture_output=True,
                                    text=True, cwd=Path(__file__).resolve().parents[1],
                                    timeout=30).stdout.strip())
    except Exception:
        commit, dirty = "unknown", True

    hashes = {
        "csv": csv_path.name,
        "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "rows": df.height,
        "columns": df.columns,
        "config": str(cfg.source_path),
        "config_hash": digest,
        "seed_start": seeds[0],
        "n_seeds": len(seeds),
        "git_commit": commit,
        "git_dirty": dirty,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / f"hashes_{digest}_seeds{seeds[0]}-{seeds[-1]}.json").write_text(
        json.dumps(hashes, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"rows={df.height} columns={len(df.columns)}")
    print(f"-> {csv_path}")
    print(json.dumps(hashes, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
