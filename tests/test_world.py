"""tests/test_world.py — world/* モジュールの単体テスト。

設計決定参照:
- F-02〜F-06: 世界の構築と各コンポーネントの生成
- F-07: Ablation条件の適用と配列の不変化
- A0, A11, A14: 初期状態、クリップ、ネットワーク連結性
- NFR-1: 決定論的再現性と状態の不変化
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

import rng
from config import Config, load
from world.ablation import Ablation, apply_ablation
from world.build import build_world, init_opinions, update_orders
from world.network import NetworkGenerationError, build_neighbors
from world.persona import _persona_counts, assign, load_persona_table
import yaml


@pytest.fixture
def small_config(tmp_path) -> Config:
    """テスト用の最小構成設定ファイルを作成して読み込む。"""
    cfg_file = tmp_path / "test_config.yaml"
    persona_file = tmp_path / "test_persona.yaml"

    # Pythonの辞書として定義する（インデントはPythonのルールに従うだけでOK）
    cfg_data = {
        "model": {
            "S": 2.0,
            "k_react": 1.0,
            "T_max": 50,
            "T_min": 5,
            "eps": 1e-4,
        },
        "network": {
            "n": 20,
            "k": 2,
            "p": 0.1,
            "max_tries": 100,
        },
        "init_opinion": {
            "mean": 0.5,
            "std": 0.1,
        },
        "personas": {
            "table": "test_persona",
            "concentration_nu": 10.0,
        },
        "experiment": {
            "master_seed_start": 0,
            "n_seeds": 5,
            "W_values": [0.0, 0.5],
            "ablation_conditions": ["none", "ABL-SYS"],
        },
        "stopping": {
            "update_mode": "gauss_seidel",
        },
        "metrics": {
            "theta": 0.5,
            "bootstrap_B": 100,
            "bootstrap_seed": 42,
        },
        "logging": {
            "log_every": 1,
        }
    }

    persona_data = {
        "personas": [
            {
                "name": "conformist",
                "share": 0.5,
                "alpha": 0.7,
                "beta": 0.4,
                "gamma": 0.1,
                "w_sns": 0.7,
                "w_sys": 0.4,
                "b": 0.15,
                "mu_react": 0.7,
            },
            {
                "name": "cautious",
                "share": 0.5,
                "alpha": 0.4,
                "beta": 0.7,
                "gamma": 0.4,
                "w_sns": 0.4,
                "w_sys": 0.7,
                "b": 0.30,
                "mu_react": 0.7,
            },
        ]
    }

    # 辞書をYAML文字列に変換して書き込む
    with cfg_file.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_data, f, sort_keys=False)

    with persona_file.open("w", encoding="utf-8") as f:
        yaml.safe_dump(persona_data, f, sort_keys=False)

    return load(cfg_file)


def test_F03_A14_network_is_connected(small_config):
    """ネットワークが連結グラフになるまで再試行され、成功するかを検証 (F-03, A14)。"""
    streams = rng.streams(1)
    neighbors = build_neighbors(
        n=small_config.network.n,
        k=small_config.network.k,
        p=small_config.network.p,
        rng=streams["network"],
        max_tries=small_config.network.max_tries,
    )

    g = nx.Graph()
    g.add_nodes_from(range(small_config.network.n))
    for i, neigh in enumerate(neighbors):
        for j in neigh:
            g.add_edge(i, int(j))

    assert nx.is_connected(g)


def test_F04_A11_init_opinions_are_clipped_to_unit_interval(small_config):
    """初期意見が [0, 1] にクリップされていることを検証 (F-04, A11)。"""
    streams = rng.streams(1)
    O0, n_clipped = init_opinions(small_config, streams["init"])

    assert np.all((O0 >= 0.0) & (O0 <= 1.0))
    assert n_clipped >= 0


def test_F04_A11_clipped_count_matches_raw_excursions(small_config):
    """n_clipped が raw 生成値の範囲外件数と一致する (F-04, A11)。

    同一シードの別ストリームで raw を再現し、期待クリップ値・件数と比較する。
    """
    n = small_config.network.n
    mean, std = small_config.init_opinion.mean, small_config.init_opinion.std

    O0, n_clipped = init_opinions(small_config, rng.streams(1)["init"])
    raw = rng.streams(1)["init"].normal(mean, std, size=n)
    expected = np.clip(raw, 0.0, 1.0)

    np.testing.assert_array_equal(O0, expected)
    assert n_clipped == int(np.sum(raw != expected))


def test_F04_A11_forced_clipping_with_large_std(small_config):
    """std を大きくするとクリップが実際に発生し、件数が正しく数えられる (F-04, A11)。

    A11 の既定値 (mean=0.5, std=0.1) では ±5σ でほぼ発生しないため、
    std=5.0 の強制ケースでカウントロジック自体を検証する。
    """
    from dataclasses import replace

    wide = replace(small_config, init_opinion=replace(
        small_config.init_opinion, std=5.0))
    n = wide.network.n

    O0, n_clipped = init_opinions(wide, rng.streams(1)["init"])
    raw = rng.streams(1)["init"].normal(0.5, 5.0, size=n)

    assert n_clipped > 0
    assert n_clipped == int(np.sum((raw < 0.0) | (raw > 1.0)))
    assert np.all((O0 >= 0.0) & (O0 <= 1.0))


def test_F06_update_orders_are_permutations(small_config):
    """更新順序行列の各行が正しい順列になっているかを検証 (F-06)。"""
    streams = rng.streams(1)
    t_max = small_config.model.T_max
    n = small_config.network.n

    orders = update_orders(n, t_max, streams["order"])

    assert orders.shape == (t_max, n)
    for t in range(t_max):
        assert sorted(orders[t].tolist()) == list(range(n))


def test_F06_update_orders_dtype_frozen_and_reproducible(small_config):
    """更新順序は int64・writeable=False・同一シードで再現する (F-06, NFR-1)。"""
    t_max = small_config.model.T_max
    n = small_config.network.n

    orders = update_orders(n, t_max, rng.streams(1)["order"])
    again = update_orders(n, t_max, rng.streams(1)["order"])

    assert orders.dtype == np.int64
    assert not orders.flags.writeable
    np.testing.assert_array_equal(orders, again)
    with pytest.raises(ValueError):
        orders[0, 0] = -1


def test_F06_update_orders_differ_across_seeds(small_config):
    """異なるシードでは更新順序が変わる (F-06)。全行一致は実質ありえない。"""
    t_max = small_config.model.T_max
    n = small_config.network.n

    a = update_orders(n, t_max, rng.streams(100)["order"])
    b = update_orders(n, t_max, rng.streams(200)["order"])

    assert not np.array_equal(a, b)


def test_F02_F06_build_world_is_reproducible(small_config):
    """同一シードで構築された World が完全一致することを検証 (F-02, NFR-1)。"""
    w1 = build_world(small_config, seed=99)
    w2 = build_world(small_config, seed=99)

    assert w1.seed == w2.seed
    assert w1.n_clipped_O0 == w2.n_clipped_O0
    np.testing.assert_array_equal(w1.O0, w2.O0)
    np.testing.assert_array_equal(w1.R0, w2.R0)
    np.testing.assert_array_equal(w1.update_order, w2.update_order)

    for n1, n2 in zip(w1.neighbors, w2.neighbors, strict=True):
        np.testing.assert_array_equal(n1, n2)


def test_F07_ablation_does_not_mutate_params(small_config):
    """apply_ablation が元の params を変更せず、配列を不変化することを検証 (F-07, FR-E5)。"""
    w = build_world(small_config, seed=10)
    original_p0_wsys = w.params[0].w_sys

    # ABL-SYS 条件を適用
    attrs_abl = apply_ablation(w.params, condition=Ablation.SYS, k_react=1.0)

    # 生成された Attrs では w_sys が全 0 であること
    assert np.all(attrs_abl.w_sys == 0.0)

    # 元の params は変更されていないこと
    assert w.params[0].w_sys == original_p0_wsys

    # 生成された配列が不変化 (writeable=False) されていること
    with pytest.raises(ValueError):
        attrs_abl.w_sys[0] = 1.0


def test_A0_turn0_no_intervention_no_social_influence(small_config):
    """R_i(0) = Clip(b_i, 0, 1) であり、初期化条件が正しいことを検証 (A0)。"""
    w = build_world(small_config, seed=5)
    expected_R0 = np.clip(w.attrs.b, 0.0, 1.0)

    np.testing.assert_array_equal(w.R0, expected_R0)


def test_F07_ablation_sns_zeroes_only_w_sns(small_config):
    """ABL-SNS では w_sns のみが 0 になり、w_sys は残る (F-07, A6)。"""
    w = build_world(small_config, seed=10)
    before = [(p.w_sys, p.w_sns) for p in w.params]

    attrs_sns = apply_ablation(w.params, condition=Ablation.SNS, k_react=1.0)

    assert np.all(attrs_sns.w_sns == 0.0)
    np.testing.assert_array_equal(
        attrs_sns.w_sys, np.array([b[0] for b in before]))
    # 元paramsは不変
    assert [(p.w_sys, p.w_sns) for p in w.params] == before


def test_F07_ablation_react_zeroes_only_k_react(small_config):
    """ABL-REACT では k_react のみが 0 になり、theta_react 配列は残る (F-07, A9)。"""
    w = build_world(small_config, seed=10)

    attrs_react = apply_ablation(w.params, condition=Ablation.REACT, k_react=1.0)

    assert attrs_react.k_react == 0.0
    np.testing.assert_array_equal(attrs_react.theta_react, w.attrs.theta_react)
    np.testing.assert_array_equal(attrs_react.w_sys, w.attrs.w_sys)
    np.testing.assert_array_equal(attrs_react.w_sns, w.attrs.w_sns)


def test_F07_ablation_none_is_passthrough(small_config):
    """condition="none" でも値はそのまま配列化され、k_react が保持される (F-07)。"""
    w = build_world(small_config, seed=10)

    attrs_none = apply_ablation(w.params, condition=Ablation.NONE, k_react=1.0)

    assert attrs_none.condition == Ablation.NONE
    assert attrs_none.k_react == 1.0
    np.testing.assert_array_equal(
        attrs_none.w_sys, np.array([p.w_sys for p in w.params]))
    np.testing.assert_array_equal(
        attrs_none.theta_react, np.array([p.theta_react for p in w.params]))


def test_F07_A06_alpha_beta_gamma_untouched_by_any_ablation(small_config):
    """いずれのAblation条件でも alpha/beta/gamma 経路は残る (F-07, A6)。"""
    w = build_world(small_config, seed=10)

    for cond in (Ablation.NONE, Ablation.SYS, Ablation.SNS, Ablation.REACT):
        attrs = apply_ablation(w.params, condition=cond, k_react=1.0)
        np.testing.assert_array_equal(attrs.alpha, w.attrs.alpha)
        np.testing.assert_array_equal(attrs.beta, w.attrs.beta)
        np.testing.assert_array_equal(attrs.gamma, w.attrs.gamma)


def test_F05_persona_share_must_sum_to_one(tmp_path):
    """share合計 ≠ 1 のペルソナ表は ValueError (F-05)。"""
    bad = tmp_path / "bad_persona.yaml"
    with bad.open("w", encoding="utf-8") as f:
        yaml.safe_dump({"personas": [
            {"name": "a", "share": 0.5, "alpha": 0.7, "beta": 0.4,
             "gamma": 0.1, "w_sns": 0.7, "w_sys": 0.4, "b": 0.15,
             "mu_react": 0.7},
            {"name": "b", "share": 0.4, "alpha": 0.4, "beta": 0.7,
             "gamma": 0.4, "w_sns": 0.4, "w_sys": 0.7, "b": 0.30,
             "mu_react": 0.7},
        ]}, f, sort_keys=False)

    with pytest.raises(ValueError):
        load_persona_table(bad)


def test_F05_persona_counts_use_largest_remainder():
    """最大剰余法で端数を配分し、合計が n に一致する (F-05)。

    [0.5, 0.3, 0.2]×10 → [5, 3, 2]（端数なし）。
    三等分×10 → 端数1を先頭型へ配分 [4, 3, 3]（同値はテーブル順）。
    """
    table = [{"share": s} for s in (0.5, 0.3, 0.2)]
    assert _persona_counts(table, 10) == [5, 3, 2]

    thirds = [{"share": 1 / 3}] * 3
    counts = _persona_counts(thirds, 10)
    assert counts == [4, 3, 3]
    assert sum(counts) == 10


def test_F05_theta_react_in_unit_interval(small_config):
    """theta_react は全員 [0, 1] に収まる (F-05, FR-P7)。"""
    w = build_world(small_config, seed=7)

    assert all(0.0 <= p.theta_react <= 1.0 for p in w.params)
    assert np.all((w.attrs.theta_react >= 0.0) & (w.attrs.theta_react <= 1.0))


def test_F05_persona_assignment_is_reproducible(small_config):
    """同一シードの persona ストリームでは割当が完全一致する (F-05, NFR-1)。"""
    n = small_config.network.n

    p1 = assign(small_config, n, rng.streams(42)["persona"])
    p2 = assign(small_config, n, rng.streams(42)["persona"])

    assert [p.persona for p in p1] == [p.persona for p in p2]
    np.testing.assert_array_equal(
        [p.theta_react for p in p1], [p.theta_react for p in p2])


def test_F05_persona_counts_match_shares(small_config):
    """割当人数が share 通りになる (F-05)。2型×0.5×20人 → 各10人。"""
    params = assign(small_config, 20, rng.streams(3)["persona"])

    assert sorted(p.persona for p in params).count("conformist") == 10
    assert sorted(p.persona for p in params).count("cautious") == 10


def _config_with_persona_rows(small_config, tmp_path, rows, nu=10.0, name="edge_persona"):
    """指定行のペルソナ表を指すよう差し替えたConfigを返す。"""
    from dataclasses import replace

    table_file = tmp_path / f"{name}.yaml"
    with table_file.open("w", encoding="utf-8") as f:
        yaml.safe_dump({"personas": rows}, f, sort_keys=False)
    return replace(
        small_config,
        personas=replace(
            small_config.personas, table=name,
            table_path=str(table_file.resolve()), concentration_nu=nu),
    )


def _persona_row(name="x", share=1.0, mu_react=0.7):
    """単一型ペルソナ行の雛形を返す。"""
    return {"name": name, "share": share, "alpha": 0.7, "beta": 0.4,
            "gamma": 0.1, "w_sns": 0.7, "w_sys": 0.4, "b": 0.15,
            "mu_react": mu_react}


@pytest.mark.parametrize("mu", [0.0, 1.0])
def test_F05_mu_react_boundary_raises(small_config, tmp_path, mu):
    """mu_react が 0/1 の境界では Beta 形状母数が 0 になり ValueError (F-05, FR-P7)。"""
    cfg = _config_with_persona_rows(small_config, tmp_path, [_persona_row(mu_react=mu)])

    with pytest.raises(ValueError):
        assign(cfg, 4, rng.streams(1)["persona"])


def test_F05_non_positive_nu_raises(small_config, tmp_path):
    """concentration_nu <= 0 では Beta 生成が ValueError (F-05, FR-P7)。"""
    cfg = _config_with_persona_rows(
        small_config, tmp_path, [_persona_row()], nu=0.0)

    with pytest.raises(ValueError):
        assign(cfg, 4, rng.streams(1)["persona"])


def test_F03_A14_exhausted_tries_raise(small_config):
    """tries 枯渇時は NetworkGenerationError にラップされる (F-03, A14)。"""
    with pytest.raises(NetworkGenerationError):
        build_neighbors(
            n=small_config.network.n,
            k=small_config.network.k,
            p=small_config.network.p,
            rng=rng.streams(1)["network"],
            max_tries=0,
        )


def test_F03_neighbors_are_sorted_and_frozen(small_config):
    """隣接配列はソート済みかつ writeable=False (F-03, NFR-1)。"""
    neighbors = build_neighbors(
        n=small_config.network.n,
        k=small_config.network.k,
        p=small_config.network.p,
        rng=rng.streams(2)["network"],
        max_tries=small_config.network.max_tries,
    )

    assert len(neighbors) == small_config.network.n
    for i, nbr in enumerate(neighbors):
        assert nbr.tolist() == sorted(nbr.tolist())
        assert not nbr.flags.writeable
        assert i not in nbr.tolist()
        with pytest.raises(ValueError):
            nbr[0] = -1


def test_F03_same_seed_same_topology(small_config):
    """同一シードではトポロジが完全一致する (F-03, NFR-1)。"""
    kwargs = dict(n=small_config.network.n, k=small_config.network.k,
                  p=small_config.network.p,
                  max_tries=small_config.network.max_tries)
    n1 = build_neighbors(rng=rng.streams(11)["network"], **kwargs)
    n2 = build_neighbors(rng=rng.streams(11)["network"], **kwargs)

    for a, b in zip(n1, n2, strict=True):
        np.testing.assert_array_equal(a, b)


def test_NFR1_world_arrays_are_frozen(small_config):
    """O0/R0/update_order/neighbors/Attrs がすべて凍結されている (NFR-1)。"""
    w = build_world(small_config, seed=21)

    for arr in (w.O0, w.R0, w.update_order):
        assert not arr.flags.writeable
    for nbr in w.neighbors:
        assert not nbr.flags.writeable
    for arr in (w.attrs.alpha, w.attrs.beta, w.attrs.gamma, w.attrs.w_sys,
                w.attrs.w_sns, w.attrs.b, w.attrs.theta_react):
        assert not arr.flags.writeable


def test_F02_F06_build_world_reproducible_params_and_attrs(small_config):
    """同一シードでは params・attrs も完全一致し、attrs は none 基準 (NFR-1, A6)。"""
    w1 = build_world(small_config, seed=99)
    w2 = build_world(small_config, seed=99)

    assert [p.persona for p in w1.params] == [p.persona for p in w2.params]
    np.testing.assert_array_equal(
        [p.theta_react for p in w1.params], [p.theta_react for p in w2.params])
    for field in ("alpha", "beta", "gamma", "w_sys", "w_sns", "b", "theta_react"):
        np.testing.assert_array_equal(getattr(w1.attrs, field), getattr(w2.attrs, field))
    assert w1.attrs.condition == w2.attrs.condition == Ablation.NONE
    assert w1.attrs.k_react == w2.attrs.k_react == small_config.model.k_react
