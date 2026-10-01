"""tests/test_arm.py — run/arm.py の単体テスト。"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

import rng
from run.arm import run_arm
from world.ablation import Ablation, apply_ablation
from world.build import build_world


def test_F11_output_shapes_are_consistent(small_config):
    world = build_world(small_config, seed=1)
    attrs = apply_ablation(world.params, Ablation.NONE, small_config.model.k_react)
    record = run_arm(world, attrs, W=0.5, config=small_config)

    n, T_max = small_config.network.n, small_config.model.T_max
    assert record.O.shape == (T_max + 1, n)
    assert record.R.shape == (T_max + 1, n)
    assert record.S_O.shape == (T_max, n)
    assert record.max_delta.shape == (T_max,)


def test_F11_A10_runs_full_T_max_even_after_convergence(small_config):
    """収束後も走行を止めない（A10帰結）。"""
    world = build_world(small_config, seed=1)
    attrs = apply_ablation(world.params, Ablation.NONE, small_config.model.k_react)
    record = run_arm(world, attrs, W=0.0, config=small_config)
    assert record.O.shape[0] == small_config.model.T_max + 1


def test_A2_jacobi_mode_is_order_invariant(small_config):
    """Jacobi型は参加者の処理順序を入れ替えても結果がbit一致する（A2根拠[工]）。"""
    cfg = replace(small_config, stopping=replace(small_config.stopping, update_mode="jacobi"))
    world = build_world(cfg, seed=2)
    attrs = apply_ablation(world.params, Ablation.NONE, cfg.model.k_react)

    record_original = run_arm(world, attrs, W=0.5, config=cfg)

    shuffled_order = world.update_order.copy()
    order_rng = rng.streams(0)["order"]
    for t in range(shuffled_order.shape[0]):
        order_rng.shuffle(shuffled_order[t])
    world_shuffled = replace(world, update_order=shuffled_order)

    record_shuffled = run_arm(world_shuffled, attrs, W=0.5, config=cfg)

    assert np.array_equal(record_original.O, record_shuffled.O)
    assert np.array_equal(record_original.R, record_shuffled.R)


def test_A2_gauss_seidel_is_order_dependent_in_general(small_config):
    """Gauss-Seidel型は一般に順序に依存する（Jacobiとの対比を確認する）。"""
    world = build_world(small_config, seed=3)
    attrs = apply_ablation(world.params, Ablation.NONE, small_config.model.k_react)
    record_original = run_arm(world, attrs, W=0.5, config=small_config)

    shuffled_order = world.update_order.copy()
    order_rng = rng.streams(1)["order"]
    for t in range(shuffled_order.shape[0]):
        order_rng.shuffle(shuffled_order[t])
    world_shuffled = replace(world, update_order=shuffled_order)
    record_shuffled = run_arm(world_shuffled, attrs, W=0.5, config=small_config)

    # Gauss-Seidel型はインプレース更新のため、処理順序を変えると結果が変わる。
    # seed=3 の組み合わせでは O・R ともに一致しないことを実測で確認済み。
    assert not np.array_equal(record_original.O, record_shuffled.O)
    assert not np.array_equal(record_original.R, record_shuffled.R)