"""tests/test_pair.py — ペア評価（指標計算と不変条件検査）の単体テスト。

設計決定参照:
- F-12, F-13: ペア指標の計算と不変条件の検査
- FR-E5, A6: Ablation（ABL-SYS, ABL-SNS）による経路分離の確認
- NFR-1: 決定論性・再現性の確認（TURN0_IDENTICAL, W0_IDENTICAL）
- D-分極化: 分極化指標の数理的正確性
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from run.arm import ArmRecord, run_arm
from run.checks import run_checks
from run.metrics import _polarization, calculate_metrics
from world.ablation import Ablation, apply_ablation
from world.build import build_world


def _synthetic_record(value: float, n: int, t_conv: int, n_turns: int = 3) -> ArmRecord:
    """全員が同一意見 value の合成 ArmRecord を作る（指標の単体検証用）。"""
    O = np.full((n_turns + 1, n), value, dtype=np.float64)
    R = np.full((n_turns + 1, n), value, dtype=np.float64)
    W_eff = np.full(n, value, dtype=np.float64)
    S_O = np.zeros((n_turns, n), dtype=np.float64)
    S_R = np.zeros((n_turns, n), dtype=np.float64)
    max_delta = np.zeros(n_turns, dtype=np.float64)
    return ArmRecord(
        O=O, R=R, W_eff=W_eff, S_O=S_O, S_R=S_R,
        max_delta=max_delta, t_conv=t_conv,
        n_clipped_R=0, n_reactance_active=0,
    )


def test_metrics_theta_is_used(small_config):
    """二値化に config.metrics.theta が使われる (A5)。

    ctrl=0.4・treat=0.6 の合成記録では、theta=0.5 で E_rate=1.0、
    theta=0.7 で E_rate=0.0 になる。
    """
    world = build_world(small_config, seed=1)
    n = small_config.network.n
    ctrl = _synthetic_record(0.4, n, t_conv=3)
    treat = _synthetic_record(0.6, n, t_conv=3)

    m50 = calculate_metrics(ctrl, treat, world, small_config, 0.5, "none")
    assert m50["E_rate"] == pytest.approx(1.0)

    cfg70 = replace(small_config, metrics=replace(small_config.metrics, theta=0.7))
    m70 = calculate_metrics(ctrl, treat, world, cfg70, 0.5, "none")
    assert m70["E_rate"] == pytest.approx(0.0)


def test_ORDER_VALID_passes_on_real_run_and_fails_on_broken_order(small_config):
    """ORDER_VALID が実 run で合格し、破損順序では不合格になる (NFR-1)。"""
    world = build_world(small_config, seed=12)
    attrs = apply_ablation(world.params, Ablation.NONE, small_config.model.k_react)
    ctrl = run_arm(world, attrs, W=0.0, config=small_config)
    treat = run_arm(world, attrs, W=0.0, config=small_config)

    checks = run_checks(ctrl, treat, W_level=0.0, ablation="none", world=world, attrs=attrs)
    valid = next(c for c in checks if c["check_id"] == "ORDER_VALID")
    assert valid["passed"] is True

    # 順列でない順序では不合格になること（検査が空でない証拠）
    broken = np.zeros_like(world.update_order)
    world_broken = replace(world, update_order=broken)
    checks_broken = run_checks(
        ctrl, treat, W_level=0.0, ablation="none", world=world_broken, attrs=attrs)
    valid_broken = next(c for c in checks_broken if c["check_id"] == "ORDER_VALID")
    assert valid_broken["passed"] is False


def test_calculate_polarization_D_polarization():
    """分極化指標計算の数理検証 (D-分極化)。"""
    # 均等に分かれたデータ
    O = np.array([0.1, 0.3, 0.7, 0.9])
    pol = _polarization(O, theta=0.5)

    # L = [0.1, 0.3] (mean=0.2, var=0.01), H = [0.7, 0.9] (mean=0.8, var=0.01)
    # pol_within = (2*0.01 + 2*0.01)/4 = 0.01
    # pol_between = 0.8 - 0.2 = 0.6
    assert pytest.approx(pol["pol_within"]) == 0.01
    assert pytest.approx(pol["pol_between"]) == 0.6
    
    # 片群が空のケース（全員賛成）
    O_all_high = np.array([0.7, 0.8, 0.9])
    pol_empty = _polarization(O_all_high, theta=0.5)
    # 片群空の場合は between は NaN
    assert math.isnan(pol_empty["pol_between"])


def test_NFR1_turn0_and_w0_bit_match(small_config):
    """W=0.0 条件でアーム間の TURN0 と全軌跡の bit 一致を検証 (NFR-1)。"""
    world = build_world(small_config, seed=12)
    attrs = apply_ablation(world.params, Ablation.NONE, small_config.model.k_react)
    
    # treat 側も介入強度 W=0.0 で実行する
    ctrl = run_arm(world, attrs, W=0.0, config=small_config)
    treat = run_arm(world, attrs, W=0.0, config=small_config)

    checks = run_checks(ctrl, treat, W_level=0.0, ablation="none", world=world, attrs=attrs)
    
    turn0_check = next(c for c in checks if c["check_id"] == "TURN0_IDENTICAL")
    assert turn0_check["passed"] is True
    
    w0_check = next(c for c in checks if c["check_id"] == "W0_IDENTICAL")
    assert w0_check["passed"] is True


def test_FR_E5_ablation_sys_bit_identical_and_effect_zero(small_config):
    """ABL-SYS 条件において ABL_SYS_BITWISE が合格し、因果効果 E_rate == 0 であることを検証 (FR-E5, A6)。"""
    world = build_world(small_config, seed=10)
    attrs_sys = apply_ablation(world.params, Ablation.SYS, small_config.model.k_react)

    ctrl = run_arm(world, attrs_sys, W=0.0, config=small_config)
    treat = run_arm(world, attrs_sys, W=0.8, config=small_config)

    # 1. チェック評価 (軌跡が bit 一致しているか)
    checks = run_checks(ctrl, treat, W_level=0.8, ablation="ABL-SYS", world=world, attrs=attrs_sys)
    sys_check = next(c for c in checks if c["check_id"] == "ABL_SYS_BITWISE")
    assert sys_check["passed"] is True

    # 2. 指標評価 (軌跡が一致しているため、介入効果は完全に 0.0 になる)
    metrics = calculate_metrics(
        ctrl,
        treat,
        world=world,
        config=small_config,
        W_level=0.8,
        ablation="ABL-SYS",
    )
    assert metrics["E_rate"] == 0.0
    assert metrics["flip_rate"] == 0.0
    assert metrics["dO_mean"] == 0.0


def test_FR_E5_ablation_sns_R_independent(small_config):
    """ABL-SNS (w_sns=0) では R が近傍影響を受けず孤立した式に一致することを検証 (FR-E5, A6)。"""
    world = build_world(small_config, seed=11)
    attrs_sns = apply_ablation(world.params, Ablation.SNS, small_config.model.k_react)

    ctrl = run_arm(world, attrs_sns, W=0.0, config=small_config)
    treat = run_arm(world, attrs_sns, W=0.5, config=small_config)

    checks = run_checks(ctrl, treat, W_level=0.5, ablation="ABL-SNS", world=world, attrs=attrs_sns)
    sns_check = next(c for c in checks if c["check_id"] == "ABL_SNS_R_INDEPENDENT")
    assert sns_check["passed"] is True