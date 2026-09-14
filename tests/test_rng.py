"""tests/test_rng.py — F-02: 4本の乱数ストリームが独立に再現される。"""
from __future__ import annotations

import numpy as np

from src import rng
from src.rng import STREAM_NAMES


def test_F02_streams_has_four_named_generators():
    """指定された4つのストリームキーが正確に含まれることの検証。"""
    s = rng.streams(42)
    assert set(s.keys()) == set(STREAM_NAMES)
    assert len(s) == 4


def test_F02_same_master_seed_is_reproducible():
    """同一マスターシードから同一の乱数系列が生成されることの検証 (NFR-1)。"""
    s1 = rng.streams(12345)
    s2 = rng.streams(12345)

    for name in STREAM_NAMES:
        a = s1[name].random(10)
        b = s2[name].random(10)
        np.testing.assert_array_equal(a, b, err_msg=f"{name} ストリームが再現されない")


def test_F02_different_seeds_produce_different_outputs():
    """異なるマスターシードからは異なる乱数系列が生成されることの検証。"""
    s1 = rng.streams(100)
    s2 = rng.streams(200)

    for name in STREAM_NAMES:
        a = s1[name].random(10)
        b = s2[name].random(10)
        assert not np.array_equal(a, b)


def test_F02_streams_are_mutually_independent():
    """あるストリームを大量消費しても、他のストリームの生成結果に一切影響を与えないことの検証 (NFR-2)。"""
    master_seed = 9999

    # 1回目の生成: 通常通り各ストリームから取得
    s_a = rng.streams(master_seed)
    _ = s_a["network"].random(1000)  # 'network' ストリームを1000回消費して進める
    init_a = s_a["init"].random(5)
    persona_a = s_a["persona"].random(5)
    order_a = s_a["order"].integers(0, 100, size=5)

    # 2回目の生成: 'network' を一切消費せずに他を取得
    s_b = rng.streams(master_seed)
    init_b = s_b["init"].random(5)
    persona_b = s_b["persona"].random(5)
    order_b = s_b["order"].integers(0, 100, size=5)

    # 'network' の使い込みに関わらず、他ストリームの乱数列は完全一致しなければならない
    np.testing.assert_array_equal(init_a, init_b)
    np.testing.assert_array_equal(persona_a, persona_b)
    np.testing.assert_array_equal(order_a, order_b)
