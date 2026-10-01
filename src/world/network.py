"""world/network.py — F-03: 連結Watts-Strogatzを生成し、隣接リストを固定する。

設計決定参照:
- F-03: Watts-Strogatz (n, k, p) による情報伝播ネットワーク構築
- A14: 非連結グラフのハンドリング（tries=100 で連結グラフを再生成、失敗時は例外）
"""
from __future__ import annotations

import numpy as np
import networkx as nx


class NetworkGenerationError(RuntimeError):
    """max_tries回試しても連結グラフが得られなかった場合の例外。"""
    pass


def build_neighbors(
    n: int,
    k: int,
    p: float,
    rng: np.random.Generator,
    max_tries: int = 100,
) -> tuple[np.ndarray, ...]:
    """連結 Watts-Strogatz グラフを生成し、ノードごとの隣接インデックス配列リストを返す。

    Generator を seed として直接渡すため NetworkX >= 3.3 が必須。
    max_tries は A14 の tries=100 に対応する。返却する各配列はソート済みかつ
    writeable=False で凍結される。

    Args:
        n (int): ノード数（参加者数）
        k (int): 各ノードの初期近傍数（偶数）
        p (float): 再配線確率
        rng (np.random.Generator): 'network' 用の乱数ストリーム
        max_tries (int, optional): 連結グラフが得られるまでの最大試行回数(Defaults to 100)

    Raises:
        NetworkGenerationError: max_tries 回試行しても連結グラフが生成できなかった場合
            （nx.NetworkXError のみラップする。それ以外の例外はそのまま伝播する）。

    Returns:
        tuple[np.ndarray, ...]: インデックス i の要素がノード i の近傍ノードID配列
            (1D ndarray) であるタプル
    """
    try:
        # ネットワーク生成用の関数に直接 Generator を渡す (NetworkX >= 3.3 で対応)
        graph = nx.connected_watts_strogatz_graph(
            n=n, k=k, p=p, tries=max_tries, seed=rng
        )
    except nx.NetworkXError as e:
        raise NetworkGenerationError(
            f"連結 Watts-Strogatz グラフの生成に失敗しました (n={n}, k={k}, p={p}, tries={max_tries}): {e}"
        ) from e

    neighbors_list: list[np.ndarray] = []
    for i in range(n):
        nbrs = sorted(graph.neighbors(i))
        nbr_arr = np.array(nbrs, dtype=np.int64)
        nbr_arr.flags.writeable = False  # ネットワークのトポロジーを凍結
        neighbors_list.append(nbr_arr)

    return tuple(neighbors_list)
