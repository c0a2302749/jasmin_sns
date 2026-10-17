#!/usr/bin/env python
"""scripts/select_persona.py — 設定の健全性確認: C2〜C4で確認し、C1は参考値。

C1（ベースラインの意見分化）は合否基準としない（5.3.4）。
β=0/α=0分解によりC1通過が同調由来でないと分かったため。
ペルソナ表の選定物語はやめ、persona_A/Bの両方を健全性確認する。
"""
from __future__ import annotations
import argparse, sys
from collections import defaultdict
from dataclasses import replace
from datetime import date
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from analysis.persona_select import criteria
from config import load
from run.arm import run_arm
from runlog import log_header
from world.ablation import Ablation, apply_ablation
from world.build import build_world

def evaluate_pattern(config, table_name: str) -> dict:
    config_dir = Path(config.source_path).parent
    new_table_path = str((config_dir / f"{table_name}.yaml").resolve())
    cfg = replace(config, personas=replace(config.personas, table=table_name, table_path=new_table_path))
    records = []
    theta_by_persona: dict[str, list[float]] = defaultdict(list)
    seed_start = cfg.experiment.master_seed_start
    for seed in range(seed_start, seed_start + cfg.experiment.n_seeds):
        world = build_world(cfg, seed)
        attrs = apply_ablation(world.params, Ablation.NONE, cfg.model.k_react)
        record = run_arm(world, attrs, W=0.0, config=cfg)
        records.append(record)
        for p in world.params:
            theta_by_persona[p.persona].append(p.theta_react)
    theta_arrays = {k: np.array(v) for k, v in theta_by_persona.items()}
    return criteria(records, theta_arrays, W_probe=0.5, k_react=cfg.model.k_react, theta=cfg.metrics.theta)

def main() -> None:
    parser = argparse.ArgumentParser(description="設定の健全性確認 (C2〜C4、C1は参考)")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--plan", default="docs/plan.md")
    args = parser.parse_args()
    config = load(args.config)
    log_header(config)
    result_a = evaluate_pattern(config, "persona_A")
    result_b = evaluate_pattern(config, "persona_B")

    def passed(r): return r["C2_pass"] and r["C3_pass"] and r["C4_pass"]
    if passed(result_a) and passed(result_b):
        chosen = "両表ともC2〜C4を満たす。主設定はpersona_A、persona_Bは感度分析に用いる"
    elif passed(result_a):
        chosen = "persona_Aのみ健全"
    elif passed(result_b):
        chosen = "persona_Bのみ健全"
    else:
        chosen = "両方不合格。基準値を見直して再実行する"

    lines = [f"\n## 設定の健全性確認 {date.today().isoformat()}", "",
             f"- パターンA: {result_a}", f"- パターンB: {result_b}",
             f"- C1は参考値（合否外）: A={result_a['C1_baseline_reproduction_rate']:.3f}, "
             f"B={result_b['C1_baseline_reproduction_rate']:.3f}",
             f"- 健全性: {chosen}"]
    with Path(args.plan).open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
