"""rng.py — F-02: マスターシードから4本の乱数ストリームを派生させる。

設計決定参照:
- NFR-2（乱数ストリームの分離）
- FR-G3（ガバナンス層は乱数を消費しない）

Generator を生成するのはこのファイルだけにする。他のモジュールは
streams() が返した Generator を受け取って使うだけで、自前で
np.random.default_rng() を呼ばない。これによりAblation条件間で
乱数消費量が変わっても比較可能性が崩れない（NFR-2の根拠）。
"""
from __future__ import annotations

from typing import Final
import numpy as np

STREAM_NAMES: Final[tuple[str, ...]] = ("network", "init", "persona", "order")


def streams(master_seed: int) -> dict[str, np.random.Generator]:
    """マスターシードから network/init/persona/order の4本を独立に派生させる。

    numpy.random.SeedSequence.spawn を使うことで、各ストリームが
    どれだけ乱数を消費してもストリーム間で影響しない（NFR-2）。
    spawn の順序は STREAM_NAMES 順に固定されており、順序を変えると
    再現性が壊れるため変更してはならない。

    Args:
        master_seed (int): 親となるマスターシード。

    Returns:
        dict[str, np.random.Generator]:  キーが 'network', 'init', 'persona', 'order' である Generator の辞書。
    """
    seed_seq = np.random.SeedSequence(master_seed)
    children = seed_seq.spawn(len(STREAM_NAMES))
    return {
        name: np.random.default_rng(child)
        for name, child in zip(STREAM_NAMES, children)
    }