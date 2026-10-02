"""tests/test_analysis.py — F-16 の統計とレポート生成。"""
from __future__ import annotations

import polars as pl

from analysis.report import make
from analysis.stats import bootstrap_ci


def test_bootstrap_ci_contains_mean_and_is_deterministic():
    vals = [-0.02, -0.03, -0.02, -0.04, -0.01]
    r1 = bootstrap_ci(vals, B=500, seed=12345)
    r2 = bootstrap_ci(vals, B=500, seed=12345)
    assert r1 == r2
    assert r1["lo"] <= r1["mean"] <= r1["hi"]


def test_bootstrap_ci_empty_and_single():
    r = bootstrap_ci([], B=100, seed=1)
    assert r["lo"] != r["lo"]  # nan
    r1 = bootstrap_ci([0.5], B=100, seed=1)
    assert r1 == {"mean": 0.5, "lo": 0.5, "hi": 0.5, "B": 100.0}


def test_report_make_writes_md_and_three_figs(tmp_path):
    rows = []
    for seed in range(6):
        for w in (0.0, 0.5, 1.0):
            rows.append({
                "seed": seed, "W": w, "ablation_condition": "none",
                "E_rate": -0.02 * w if seed % 2 == 0 else -0.01 * w,
                "E_maj": 0, "flip_rate": 0.0, "dO_mean": 0.0,
                "pol_within_treat": 0.0, "pol_within_ctrl": 0.0,
                "pol_between_treat": 0.0, "pol_between_ctrl": 0.0,
                "T_star": 10, "t_conv_treat": 10, "t_conv_ctrl": 10,
                "n_clipped_O0": 0, "n_clipped_R_treat": 0,
                "n_clipped_R_ctrl": 0, "n_reactance_active": 0,
            })
            rows.append({**rows[-1], "ablation_condition": "ABL-SYS", "E_rate": 0.0})
    from io_.schema import SUMMARY_SCHEMA
    (tmp_path / "summary").mkdir()
    pl.DataFrame(rows, schema=SUMMARY_SCHEMA).write_parquet(tmp_path / "summary" / "part.parquet")

    md = make(tmp_path / "summary" / "part.parquet", tmp_path / "report",
              B=200, boot_seed=7, T_max=50, n=20,
              persona_bounds={"a": 0.6, "b": 1.2})
    text = md.read_text(encoding="utf-8")
    assert "表1" in text and "ABL-SYS" in text
    for fig in ("fig_dose_response.png", "fig_ablation.png", "fig_clip.png"):
        assert (tmp_path / "report" / fig).exists()
