"""io/schema.py — ログ出力と Parquet 保存に関わる列名・データ型の正典。

設計決定参照:
- NFR-4: 論文記号 -> コード変数名 -> ログ列名 の完全一元化対応表（追跡可能性）
- NFR-6: 列指向形式 (Polars/PyArrow) による効率的集計のための型定義
- A5: 因果効果・分極化指標の定義列
"""
from typing import Final

import polars as pl

# pl.UInt32 等は DataType の「インスタンス」ではなく DataTypeClass（型のクラス）なので、
# pl.DataType 単独では宣言できない。polars._typing.PolarsDataType 相当をローカルに定義する。
type PolarsDataType = type[pl.DataType] | pl.DataType

# 論文記号 -> コード変数名 -> ログ列名 -> 設計決定 の対応正典 (NFR-4)
SYMBOL_TABLE: Final[list[dict[str, str]]] = [
    {"symbol": "i", "var": "participant_id", "column": "participant_id", "decision": "FR-P1"},
    {"symbol": "t", "var": "t_idx", "column": "turn", "decision": "FR-P5"},
    {"symbol": "O_i(t)", "var": "O_curr", "column": "O", "decision": "FR-P2, A4"},
    {"symbol": "R_i(t)", "var": "R_curr", "column": "R", "decision": "FR-P3, A1"},
    {"symbol": "W(t)", "var": "W", "column": "W", "decision": "FR-G1"},
    {"symbol": "W_eff,i(t)", "var": "w_eff", "column": "W_eff", "decision": "FR-P4, A13"},
    {"symbol": "S_O,i(t)", "var": "s_o", "column": "S_O", "decision": "FR-S2"},
    {"symbol": "S_R,i(t)", "var": "s_r", "column": "S_R", "decision": "FR-S3"},
    {"symbol": "O_{0,i}", "var": "O0", "column": "O_0", "decision": "A11"},
    {"symbol": "b_i", "var": "b", "column": "b", "decision": "A0, A7"},
    {"symbol": "α_i", "var": "alpha", "column": "alpha", "decision": "FR-P6"},
    {"symbol": "β_i", "var": "beta", "column": "beta", "decision": "FR-P6"},
    {"symbol": "γ_i", "var": "gamma", "column": "gamma", "decision": "FR-P6"},
    {"symbol": "θ_{react,i}", "var": "theta_react", "column": "theta_react", "decision": "FR-P7, A8, A12"},
    {"symbol": "w_{sys,i}", "var": "w_sys", "column": "w_sys", "decision": "A6, A7"},
    {"symbol": "w_{sns,i}", "var": "w_sns", "column": "w_sns", "decision": "A6, A7"},
    {"symbol": "E_rate", "var": "e_rate", "column": "E_rate", "decision": "A5"},
    {"symbol": "E_maj", "var": "e_maj", "column": "E_maj", "decision": "A5"},
    {"symbol": "flip_rate", "var": "flip_rate", "column": "flip_rate", "decision": "A5"},
    {"symbol": "ΔO_mean", "var": "dO_mean", "column": "dO_mean", "decision": "A5"},
]

# --- Polars スキーマ定義 (NFR-6) ---

STATIC_SCHEMA: Final[dict[str, PolarsDataType]] = {
    "seed": pl.UInt32,
    "participant_id": pl.UInt16,
    "persona": pl.Categorical,
    "O_0": pl.Float64,
    "b": pl.Float64,
    "alpha": pl.Float64,
    "beta": pl.Float64,
    "gamma": pl.Float64,
    "theta_react": pl.Float64,
    "w_sys": pl.Float64,
    "w_sns": pl.Float64,
    "degree": pl.UInt8,  # 隣接ノード数
    "neighbors": pl.List(pl.UInt16),
}

TRAJECTORY_SCHEMA: Final[dict[str, PolarsDataType]] = {
    "seed": pl.UInt32,
    "W": pl.Float64,
    "ablation_condition": pl.Categorical,
    "arm": pl.Categorical,  # 'treat' | 'ctrl'
    "turn": pl.UInt16,
    "participant_id": pl.UInt16,
    "O": pl.Float64,
    "R": pl.Float64,
    "W_eff": pl.Float64,
    "S_O": pl.Float64,
    "S_R": pl.Float64,
}

TRAJECTORY_DEBUG_COLUMNS: Final[dict[str, PolarsDataType]] = {
    "term_alpha": pl.Float64,
    "term_beta": pl.Float64,
    "term_gamma": pl.Float64,
    "pre_tanh": pl.Float64,
}

SUMMARY_SCHEMA: Final[dict[str, PolarsDataType]] = {
    "seed": pl.UInt32,
    "W": pl.Float64,
    "ablation_condition": pl.Categorical,
    "E_rate": pl.Float64,
    "E_maj": pl.Int8,
    "flip_rate": pl.Float64,
    "dO_mean": pl.Float64,
    "pol_within_treat": pl.Float64,
    "pol_within_ctrl": pl.Float64,
    "pol_between_treat": pl.Float64,
    "pol_between_ctrl": pl.Float64,
    "T_star": pl.UInt16,
    "t_conv_treat": pl.Int32,  # 未収束時は -1
    "t_conv_ctrl": pl.Int32,
    "n_clipped_O0": pl.UInt16,
    "n_clipped_R_treat": pl.UInt32,
    "n_clipped_R_ctrl": pl.UInt32,
    "n_reactance_active": pl.UInt16,
}

CHECKS_SCHEMA: Final[dict[str, PolarsDataType]] = {
    "seed": pl.UInt32,
    "W": pl.Float64,
    "ablation_condition": pl.Categorical,
    "check_id": pl.Categorical,
    "passed": pl.Boolean,
    "detail": pl.String,
}