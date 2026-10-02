"""シミュレーション結果（static, trajectory, summary, checks, manifest）の書き出し (F-15)

設計決定参照:
- F-15, NFR-6: Polars による長形式列指向 Parquet 書き出し
- 3.2: Hive パーティション風ディレクトリ構造と 4 分離ファイルの出力
"""

from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any
import polars as pl

from config import Config
from io_.schema import (
    CHECKS_SCHEMA,
    STATIC_SCHEMA,
    SUMMARY_SCHEMA,
    TRAJECTORY_DEBUG_COLUMNS,
    TRAJECTORY_SCHEMA,
)
from run.arm import ArmRecord
from world.build import World


def write_static(world: World, out_dir: Path) -> None:
    """共通世界の固定属性（static）を Parquet に書き出す。1シードにつき1ファイル。"""
    static_dir = out_dir / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    out_path = static_dir / f"seed_{world.seed}.parquet"

    if out_path.exists():
        return  # 既に書かれている場合はスキップ

    rows = []
    for i, p in enumerate(world.params):
        rows.append(
            {
                "seed": world.seed,
                "participant_id": i,
                "persona": p.persona,
                "O_0": float(world.O0[i]),
                "b": p.b,
                "alpha": p.alpha,
                "beta": p.beta,
                "gamma": p.gamma,
                "theta_react": p.theta_react,
                "w_sys": p.w_sys,
                "w_sns": p.w_sns,
                "degree": len(world.neighbors[i]),
                "neighbors": world.neighbors[i].tolist(),
            }
        )

    df = pl.DataFrame(rows, schema=STATIC_SCHEMA)
    df.write_parquet(out_path)


def write_trajectory(
    record: ArmRecord,
    W_level: float,
    ablation: str,
    arm: str,
    seed: int,
    config: Config,
    out_dir: Path,
) -> None:
    """アームの時系列軌跡（trajectory）を Parquet に書き出す。

    間引きは logging.log_every で行い、t=0 と t=T_max は必ず記録する。
    S_O / S_R はターン t の更新時に計算されるため t<T_max の値しか存在せず、
    t=T_max では NaN を記録する。

    Args:
        record (ArmRecord): 書き出すアームの実行記録。
        W_level (float): 介入強度 W。
        ablation (str): 適用したAblation条件。
        arm (str): "ctrl" または "treat"。
        seed (int): マスターシード。
        config (Config): 実験設定。
        out_dir (Path): 出力ディレクトリ（run_id 配下）。
    """
    traj_dir = (
        out_dir
        / "trajectory"
        / f"W_{W_level:.2f}"
        / f"seed_{seed}"
        / f"ablation_{ablation}"
        / f"arm_{arm}"
    )
    traj_dir.mkdir(parents=True, exist_ok=True)
    out_path = traj_dir / "part.parquet"

    t_max = config.model.T_max
    log_every = config.logging.log_every
    debug_mode = config.logging.debug_terms
    n = config.network.n

    rows = []
    for t in range(t_max + 1):
        # ログ間引き (ただし t=0 と 最終ターン t_max は必須記録)
        if t != 0 and t != t_max and (t % log_every != 0):
            continue

        # S_O / S_R と debug の各項はターン t の更新時に計算されるため
        # t < T_max の値しか存在しない（t=T_max では NaN）
        computed = t < t_max

        for i in range(n):
            row = {
                "seed": seed,
                "W": W_level,
                "ablation_condition": ablation,
                "arm": arm,
                "turn": t,
                "participant_id": i,
                "O": float(record.O[t, i]),
                "R": float(record.R[t, i]),
                "W_eff": float(record.W_eff[i]),
                "S_O": float(record.S_O[t, i]) if computed else math.nan,
                "S_R": float(record.S_R[t, i]) if computed else math.nan,
            }

            if debug_mode and record.terms is not None:
                terms = record.terms
                row["term_alpha_SO"] = (
                    float(terms["term_alpha_SO"][t, i]) if computed else math.nan
                )
                row["term_beta_R"] = (
                    float(terms["term_beta_R"][t, i]) if computed else math.nan
                )
                row["term_gamma_diff"] = (
                    float(terms["term_gamma_diff"][t, i]) if computed else math.nan
                )
                row["pre_tanh"] = (
                    float(terms["pre_tanh"][t, i]) if computed else math.nan
                )

            rows.append(row)

    schema = dict(TRAJECTORY_SCHEMA)
    if debug_mode:
        schema.update(TRAJECTORY_DEBUG_COLUMNS)

    df = pl.DataFrame(rows, schema=schema)
    df.write_parquet(out_path)


def write_summaries_and_checks(
    summaries: list[dict[str, Any]],
    checks: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    """全ペアの summary および checks リストを一括で Parquet 出力する。"""
    summary_dir = out_dir / "summary"
    checks_dir = out_dir / "checks"
    summary_dir.mkdir(parents=True, exist_ok=True)
    checks_dir.mkdir(parents=True, exist_ok=True)

    if summaries:
        df_summary = pl.DataFrame(summaries, schema=SUMMARY_SCHEMA)
        df_summary.write_parquet(summary_dir / "part.parquet")

    if checks:
        df_checks = pl.DataFrame(checks, schema=CHECKS_SCHEMA)
        df_checks.write_parquet(checks_dir / "part.parquet")


def write_manifest(
    config: Config,
    run_id: str,
    out_dir: Path,
    status: str,
    warnings: list[str],
    failed_pairs: list[str],
) -> None:
    """実行全体のメタデータ manifest.json を書き出す。

    Args:
        config (Config): 実験設定（asdict でそのまま記録する）。
        run_id (str): 実行ID。
        out_dir (Path): 出力ディレクトリ（run_id 配下）。
        status (str): "running" / "completed" / "completed_with_errors"。
        warnings (list[str]): 設定読み込み時の警告メッセージ。
        failed_pairs (list[str]): 失敗したペア・シードの記録。
    """
    manifest_path = out_dir / "manifest.json"

    data = {
        "run_id": run_id,
        "schema_version": 1,
        "status": status,
        "config": asdict(config),
        "warnings": warnings,
        "failed_pairs": failed_pairs,
    }

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)