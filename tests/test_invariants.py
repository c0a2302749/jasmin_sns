"""tests/test_invariants.py — シミュレーション基盤の決定論性とアーキテクチャ上の不変条件。

設計決定参照:
- NFR-1: 決定論性・再現性の保証（同一シードで同一結果）
- NFR-2: 乱数ストリームの分離（Wが変わっても世界生成が同一）
- A2: Jacobi 型では更新順序によらず結果が bit 一致する
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from run.arm import run_arm
from world.ablation import Ablation, apply_ablation
from world.build import build_world


def test_NFR1_same_seed_two_runs_are_bit_identical(small_config):
    """同一シード・同一設定で2回実行した結果が bit レベルで完全一致する (NFR-1)。"""
    world1 = build_world(small_config, seed=42)
    attrs1 = apply_ablation(world1.params, Ablation.NONE, small_config.model.k_react)
    record1 = run_arm(world1, attrs1, W=0.5, config=small_config)

    world2 = build_world(small_config, seed=42)
    attrs2 = apply_ablation(world2.params, Ablation.NONE, small_config.model.k_react)
    record2 = run_arm(world2, attrs2, W=0.5, config=small_config)

    assert np.array_equal(record1.O, record2.O)
    assert np.array_equal(record1.R, record2.R)


def test_NFR2_same_seed_different_W_shares_same_world(small_config):
    """W を変えて実行しても世界（World）が不変で、同シード再構築と一致する (NFR-2)。

    run_arm は Generator を受け取らず乱数を消費しないため、
    介入強度を変えた実行の前後で世界の配列は変わらない。
    """
    world = build_world(small_config, seed=8)
    attrs = apply_ablation(world.params, Ablation.NONE, small_config.model.k_react)

    # 実行前のスナップショット
    O0_before = world.O0.copy()
    R0_before = world.R0.copy()
    order_before = world.update_order.copy()
    neighbors_before = [nbr.copy() for nbr in world.neighbors]

    # W を変えて2本走らせても世界は変わらない
    run_arm(world, attrs, W=0.0, config=small_config)
    run_arm(world, attrs, W=0.9, config=small_config)

    np.testing.assert_array_equal(world.O0, O0_before)
    np.testing.assert_array_equal(world.R0, R0_before)
    np.testing.assert_array_equal(world.update_order, order_before)
    for nbr, before in zip(world.neighbors, neighbors_before, strict=True):
        np.testing.assert_array_equal(nbr, before)

    # 同シードで再構築した世界とも完全一致する
    world_again = build_world(small_config, seed=8)
    np.testing.assert_array_equal(world_again.O0, world.O0)
    np.testing.assert_array_equal(world_again.update_order, world.update_order)
    for n1, n2 in zip(world_again.neighbors, world.neighbors, strict=True):
        np.testing.assert_array_equal(n1, n2)


def test_A2_jacobi_mode_is_order_invariant(small_config):
    """Jacobi 更新モードでは、ターンの更新順序をシャッフルしても結果が bit 一致する (A2)。"""
    # config を Jacobi モードに上書き（frozen なので dataclasses.replace 経由）
    cfg = replace(
        small_config,
        stopping=replace(small_config.stopping, update_mode="jacobi"),
    )

    world1 = build_world(cfg, seed=99)
    attrs = apply_ablation(world1.params, Ablation.NONE, cfg.model.k_react)

    # 順序配列を反転させた別の世界を作る（初期状態などは同一、World は frozen なので replace）
    flipped_order = np.flip(world1.update_order, axis=1)
    flipped_order.flags.writeable = False
    world2 = replace(world1, update_order=flipped_order)

    record1 = run_arm(world1, attrs, W=0.5, config=cfg)
    record2 = run_arm(world2, attrs, W=0.5, config=cfg)

    # Jacobi ではターン開始時の状態のみを見て一括更新されるため、
    # 参加者の処理順序が異なっても計算結果に一切影響せず bit 一致する
    assert np.array_equal(record1.O, record2.O)
    assert np.array_equal(record1.R, record2.R)