"""tests/test_remaining.py — 残課題 (CI対検定・FJ・単調性・C3_pass・型別dO) の回帰テスト。"""
from __future__ import annotations

import numpy as np

from analysis.persona_select import criteria
from analysis.stats import bootstrap_ci, bootstrap_paired_diff, summarize_emaj
from layers import governance
from run.arm import run_arm
from run.checks import run_checks
from run.metrics import calculate_metrics, persona_dO_means
from world.ablation import Ablation, apply_ablation
from world.build import build_world


def test_paired_diff_detects_nonzero_and_is_deterministic():
    a = [-0.03, -0.02, -0.04, -0.025, -0.035, -0.02]
    b = [-0.01, -0.015, -0.01, -0.02, -0.01, -0.012]
    r1 = bootstrap_paired_diff(a, b, B=500, seed=7)
    r2 = bootstrap_paired_diff(a, b, B=500, seed=7)
    assert r1 == r2
    assert r1["mean"] < 0
    assert r1["lo"] <= r1["mean"] <= r1["hi"]
    assert r1["significant"] == 1.0
    # 長さ不一致は nan
    r = bootstrap_paired_diff([1.0], [1.0, 2.0], B=100, seed=1)
    assert r["mean"] != r["mean"]


def test_summarize_emaj_counts():
    r = summarize_emaj([0, 0, -1, 0, 1, 0])
    assert r["n"] == 6.0
    assert r["p_0"] == 4 / 6
    assert r["p_minus1"] == 1 / 6
    assert r["p_plus1"] == 1 / 6


def test_monotone_decrease_passes_and_detects_violation(small_config):
    world = build_world(small_config, seed=3)
    attrs = apply_ablation(world.params, Ablation.NONE, small_config.model.k_react)
    ctrl = run_arm(world, attrs, W=0.0, config=small_config)
    treat = run_arm(world, attrs, W=0.5, config=small_config)
    checks = run_checks(ctrl, treat, W_level=0.5, ablation="none", world=world, attrs=attrs)
    mono = next(c for c in checks if c["check_id"] == "MONOTONE_DECREASE")
    assert mono["passed"] is True
    # 上下を入れ替えれば違反として検出できる (検査が空でない証拠)
    checks_rev = run_checks(treat, ctrl, W_level=0.5, ablation="none", world=world, attrs=attrs)
    mono_rev = next(c for c in checks_rev if c["check_id"] == "MONOTONE_DECREASE")
    assert mono_rev["passed"] is False


def test_c3_pass_auto(small_config):
    world = build_world(small_config, seed=5)
    attrs = apply_ablation(world.params, Ablation.NONE, small_config.model.k_react)
    rec = run_arm(world, attrs, W=0.0, config=small_config)
    from collections import defaultdict
    by_p: dict[str, list[float]] = defaultdict(list)
    for p in world.params:
        by_p[p.persona].append(p.theta_react)
    arrs = {k: np.array(v) for k, v in by_p.items()}
    out = criteria([rec], arrs, W_probe=0.5, k_react=small_config.model.k_react,
                   theta=small_config.metrics.theta)
    assert "C3_pass" in out and isinstance(out["C3_pass"], bool)
    vals = list(out["C3_reactance_discriminability_by_persona"].values())
    assert out["C3_pass"] == (max(vals) >= 0.50 and min(vals) <= 0.20)


def test_persona_dO_weighted_mean_equals_global(small_config):
    world = build_world(small_config, seed=6)
    attrs = apply_ablation(world.params, Ablation.NONE, small_config.model.k_react)
    ctrl = run_arm(world, attrs, W=0.0, config=small_config)
    treat = run_arm(world, attrs, W=0.5, config=small_config)
    m = calculate_metrics(ctrl, treat, world, small_config, 0.5, "none")
    pd = persona_dO_means(ctrl, treat, world, small_config)
    labels = np.array([p.persona for p in world.params])
    recon = sum(pd[nm] * np.sum(labels == nm) for nm in pd) / len(labels)
    assert abs(recon - m["dO_mean"]) < 1e-12
    assert set(pd) == set(labels.tolist())


def _fj_helpers():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "verify_fj", str(Path(__file__).resolve().parents[1] / "scripts" / "verify_fj.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.fj_closed_form, mod.fj_iterate


def test_fj_closed_form_matches_iteration(small_config):
    fj_closed_form, fj_iterate = _fj_helpers()
    world = build_world(small_config, seed=4)
    attrs = apply_ablation(world.params, Ablation.SNS, small_config.model.k_react)
    W_eff, _ = governance.calculate_w_eff(0.2, attrs.theta_react, attrs.k_react)
    R_const = np.clip(np.asarray(attrs.w_sys) * np.asarray(W_eff)
                      + np.asarray(attrs.b), 0.0, 1.0)
    O_star = fj_closed_form(world, attrs, R_const)
    O_iter, _ = fj_iterate(world, attrs, R_const)
    assert float(np.max(np.abs(O_star - O_iter))) < 1e-9
