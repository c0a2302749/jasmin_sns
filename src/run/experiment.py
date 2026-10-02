"""run/experiment.py — F-14: 実験の組み合わせを列挙し、シード外側で回す。

設計決定参照:
- F-14, D-計画: シード外側ループ (seed -> ablation -> W) の組み合わせ探索
- 1.5: 前提崩壊時の防御的継続 (1ペアの失敗で全体を止めず記録)
- NFR-6: Parquet I/O 最適化 (summary/checks のオンメモリ集計と一括書き出し)
"""
from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

from config import Config, run_id
from io_.writer import (
    write_manifest,
    write_static,
    write_summaries_and_checks,
    write_trajectory,
)
from run.arm import run_arm
from run.checks import run_checks
from run.metrics import calculate_metrics
from world.ablation import Ablation, apply_ablation
from world.build import build_world


def run(config: Config, out_dir: str | Path) -> Path:
    """介入なし1本＋Wごとの介入ありを、seedとAblation条件を外側にして回す。

    Args:
        config (Config): 実験設定。
        out_dir (str | Path): 出力基本ディレクトリ（run_id 配下に書き出す）。

    Returns:
        Path: 生成された run_id ディレクトリのパス。
    """
    r_id = run_id(config)
    out_path = Path(out_dir) / r_id
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. 実行開始時に manifest を "running" 状態で書き出す
    failed_pairs: list[str] = []
    write_manifest(
        config=config,
        run_id=r_id,
        out_dir=out_path,
        status="running",
        warnings=list(config.warnings),
        failed_pairs=failed_pairs,
    )

    summaries: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []

    seed_start = config.experiment.master_seed_start
    debug = config.logging.debug_terms

    # 2. メイン実験ループ: seed (外側) -> ablation -> W
    for seed in range(seed_start, seed_start + config.experiment.n_seeds):
        try:
            # 共通世界の構築
            world = build_world(config, seed)
            write_static(world, out_path)

            for condition in config.experiment.ablation_conditions:
                attrs = apply_ablation(world.params, Ablation(condition), config.model.k_react)

                # 統制アーム (W=0.0) は各 Ablation 条件につき1回だけ実行する
                ctrl_record = run_arm(world, attrs, W=0.0, config=config, debug=debug)
                write_trajectory(
                    record=ctrl_record, seed=world.seed, W_level=0.0,
                    ablation=condition, arm="ctrl", config=config, out_dir=out_path
                )

                for W in config.experiment.W_values:
                    pair_id = f"seed_{seed}-abl_{condition}-W_{W:.2f}"
                    try:
                        # 処置アーム (W) の実行
                        if W == 0.0:
                            # W=0.0 の場合は統制アームと完全に同一のため再計算をスキップ
                            treat_record = ctrl_record
                        else:
                            treat_record = run_arm(world, attrs, W=W, config=config, debug=debug)
                            write_trajectory(
                                record=treat_record, seed=world.seed, W_level=W,
                                ablation=condition, arm="treat", config=config, out_dir=out_path
                            )

                        # 共通評価時刻 T* (A10) で揃えた指標と不変条件を計算
                        summary_row = calculate_metrics(
                            ctrl_record,
                            treat_record,
                            world,
                            config,
                            W_level=W,
                            ablation=condition,
                        )
                        check_rows = run_checks(
                            ctrl_record,
                            treat_record,
                            W_level=W,
                            ablation=condition,
                            world=world,
                            attrs=attrs,
                        )

                        summaries.append(summary_row)
                        checks.extend(check_rows)

                    except Exception as exc:
                        # ペア単位での失敗: traceback を manifest 用に保持しつつ、Parquet の checks にもエラー行を残す
                        err_msg = f"{pair_id}: {type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                        failed_pairs.append(err_msg)
                        checks.append(
                            {
                                "seed": seed,
                                "W": float(W),
                                "ablation_condition": condition,
                                "check_id": "PAIR_EXECUTION_FAILED",
                                "passed": False,
                                "detail": f"{type(exc).__name__}: {exc}",
                            }
                        )

        except Exception as e:
            # シード（世界生成）単位での致命的な失敗
            err_msg = f"seed_{seed}: {type(e).__name__}: {e}\n{traceback.format_exc()}"
            failed_pairs.append(err_msg)

    # 3. 集計結果と最終 manifest の書き出し
    write_summaries_and_checks(summaries, checks, out_path)
    
    status = "completed" if not failed_pairs else "completed_with_errors"
    write_manifest(
        config=config,
        run_id=r_id,
        out_dir=out_path,
        status=status,
        warnings=list(config.warnings),
        failed_pairs=failed_pairs,
    )

    return out_path