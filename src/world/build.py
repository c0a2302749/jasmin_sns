"""world/build.py — F-02〜F-07: 初期意見・更新順序の生成と世界の組み立て。

設計決定参照:
- F-04: 初期意見 O0 の生成とクリップ
- F-06: 全ターン分の非同期更新順序アレイのあらかじめ生成
- A0, A11: ターン0の初期状態設定とクリップ
- A2: 非同期逐次更新の順序

build_world() は F-02〜F-06 を network → init → persona → order の順で呼ぶ
「乱数を消費する処理の集約点」。attrs は常に condition="none" のベースライン
として apply_ablation 経由で構築する（A6）。
これより後段（layers/, run/）では乱数を一切生成しない。
World（値を持ち回るクラス）もここに置く。
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

import rng as rng_module
from config import Config
from world.ablation import Attrs, Ablation, apply_ablation
from world.network import build_neighbors
from world.persona import ParticipantParams, assign


@dataclass(frozen=True)
class World:
    """シードで確定した後は触らない世界の状態（値を持ち回る不変クラス）。

    不変条件: O0, R0, update_order および neighbors の各配列は
    writeable=False で凍結され、neighbors のタプル自体も追加・差替不可。
    params の ParticipantParams は frozen、
    attrs の全配列は Attrs.__post_init__ で凍結される。

    Attributes:
        seed: マスターシード。
        neighbors: 隣接ノードのタプル（各要素は np.ndarray）。
        params: 参加者のペルソナパラメータのリスト。
        attrs: 計算高速化・一括処理のための属性配列コンテナ。
        O0: 初期意見配列（shape (n,)）。
        R0: 初期リスク認知配列（shape (n,)）。
        update_order: 全ターン分の更新順序配列（shape (T_max, n)）。
        n_clipped_O0: 初期意見クリップが発生した件数。
    """
    seed: int
    neighbors: tuple[np.ndarray, ...]
    params: list[ParticipantParams]
    attrs: Attrs                 # condition="none" の基準属性（A6）
    O0: np.ndarray               # shape (n,)
    R0: np.ndarray               # shape (n,)
    update_order: np.ndarray     # shape (T_max, n)
    n_clipped_O0: int


def init_opinions(config: Config, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """初期意見 O_0 を N(mean, std) から生成し [0,1] にクリップする（A11）。

    クリップ件数は raw != clipped の要素数と定義する。返却配列は
    writeable=False に凍結される。

    Args:
        config: 実験設定。
        rng: 'init' 用の乱数ストリーム。

    Returns:
        (クリップ済みO0配列, クリップが発生した件数) のタプル。
    """
    n = config.network.n
    raw = rng.normal(config.init_opinion.mean, config.init_opinion.std, size=n)
    clipped = np.clip(raw, 0.0, 1.0)

    n_clipped = int(np.sum(raw != clipped))

    # 状態の不変性を保証
    clipped.flags.writeable = False

    return clipped, n_clipped


def update_orders(n: int, T_max: int, rng: np.random.Generator) -> np.ndarray:
    """全ターン分の更新順序を生成する（FR-P5, A2）。

    Args:
        n: 参加者数。
        T_max: 最大ターン数。
        rng: 'order' 用の乱数ストリーム。

    Returns:
        shape (T_max, n) の配列。各行がそのターンの参加者処理順（0..n-1の順列）。
    """
    orders = np.empty((T_max, n), dtype=np.int64)
    base = np.arange(n)

    for t in range(T_max):
        orders[t] = rng.permutation(base)

    # 状態の不変性を保証
    orders.flags.writeable = False

    return orders


def build_world(config: Config, seed: int) -> World:
    """F-02〜F-06 をこの順で呼び、World を組み立てる。乱数を使うのはここまで。

    固定順序: network → init → persona → order。この順序を変えると
    ストリーム分離（NFR-2）の比較可能性が壊れる。attrs は常に
    condition="none" のベースライン（A6）。

    Args:
        config: 実験設定。
        seed: マスターシード。

    Raises:
        NetworkGenerationError: 連結グラフが max_tries 回で得られない場合。
        ValueError: ペルソナ表の share 合計が 1 でない場合。

    Returns:
        構築済みの不変な World オブジェクト。
    """
    streams = rng_module.streams(seed)

    # 1. ネットワーク構築 (F-03)
    neighbors = build_neighbors(
        n=config.network.n,
        k=config.network.k,
        p=config.network.p,
        rng=streams["network"],
        max_tries=config.network.max_tries,
    )

    # 2. 初期意見構築 (F-04)
    O0, n_clipped_O0 = init_opinions(config, streams["init"])

    # 3. ペルソナ割当 (F-05)
    params = assign(
        config=config,
        n=config.network.n,
        rng=streams["persona"],
    )

    # 4. 全ターンの更新順序生成 (F-06)
    update_order = update_orders(
        n=config.network.n,
        T_max=config.model.T_max,
        rng=streams["order"]
    )

    # 5. デフォルト (none) 条件での Attrs 構築と R0 の初期化 (A0)
    baseline_attrs = apply_ablation(
        params=params,
        condition=Ablation.NONE,
        k_react=config.model.k_react
    )

    # ターン0のRはガバナンス・社会的影響なしの個人固有バイアス（A0）
    R0 = np.clip(baseline_attrs.b, 0.0, 1.0)
    R0.flags.writeable = False

    return World(
        seed=seed,
        neighbors=neighbors,
        params=params,
        attrs=baseline_attrs,
        O0=O0,
        R0=R0,
        update_order=update_order,
        n_clipped_O0=n_clipped_O0,
    )
