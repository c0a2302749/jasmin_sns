"""world/persona.py — F-05: ペルソナを割り当て、固定属性とθ_reactを生成する。

設計決定参照:
- FR-P6, A7, A8: ペルソナごとのパラメータ割り当て
- FR-P7: θ_react の Beta 分布からの連続生成

ParticipantParams（値を持ち回るクラス）もここに置く。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

from config import Config


@dataclass(frozen=True)
class ParticipantParams:
    """1参加者ぶんの固定属性（生成後は不変。NFR-1）
    Attributes:
        persona (str): ペルソナ型の名前
        alpha (float): 意見更新式の同調圧力係数
        beta (float): 意見更新式のリスク認知係数
        gamma (float): 意見更新式の初期意見固執係数
        w_sys (float): リスク認知式の制度的介入係数
        w_sns (float): リスク認知式の社会的影響係数
        b (float): リスク認知式のベースラインリスク認知
        theta_react (float): リスク認知式のリアクタンス係数
    """
    persona: str
    alpha: float
    beta: float
    gamma: float
    w_sys: float
    w_sns: float
    b: float
    theta_react: float


def load_persona_table(path: str | Path) -> list[dict]:
    """ペルソナ設定ファイル (YAML) を読み込む。

    Args:
        path (str | Path): configs/persona_A.yaml 等のパス。

    Raises:
        ValueError: persona share の合計が1ではない場合（atol=1e-6）。
        FileNotFoundError: ファイルが存在しない場合。
        KeyError: "personas" キーや各行の必須キー
            （name/share/alpha/beta/gamma/w_sys/w_sns/b/mu_react）が欠落している場合。

    Returns:
        list[dict]: 各ペルソナ型の設定辞書のリスト。
    """
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    table = raw["personas"]

    total_share = sum(p["share"] for p in table)
    if not np.isclose(total_share, 1.0, atol=1e-6):
        raise ValueError(f"persona share の合計が1ではない: {total_share}")

    return table


def _persona_counts(table: list[dict], n: int) -> list[int]:
    """share から各型の人数を決める（最大剰余法により端数規則を固定する）

    端数配分は小数部分の降順、同値は元のテーブル順（インデックス昇順）で
    1ずつ分配する。合計は常に n に一致する。

    Args:
        table (list[dict]): ペルソナ型の設定辞書のリスト。
        n (int): 参加者数。

    Returns:
        list[int]: 各型の人数リスト。table の順序に対応。
    """
    raw_counts = [p["share"] * n for p in table]
    base = [int(np.floor(c)) for c in raw_counts]
    remainder = n - sum(base)

    # 端数が大きい順にソートして1ずつ分配（同値の場合は元のインデックス順）
    fractions = [(c - b, -i) for i, (c, b) in enumerate(zip(raw_counts, base))]
    fractions.sort(reverse=True)

    for _, neg_idx in fractions[:remainder]:
        base[-neg_idx] += 1

    return base


def assign(
    config: Config, n: int, rng: np.random.Generator
) -> list[ParticipantParams]:
    """参加者n人分のParticipantParamsを生成する（FR-P6, FR-P7）。
    alpha/beta/gamma/w_sys/w_sns/b はペルソナ型ごとの固定値（A7）。
    theta_react のみ Beta(mu*nu, (1-mu)*nu) から連続的に生成する（FR-P7）。
    生成順序は persona ストリームのみを消費し、他のストリームに影響しない。
    n は config.network.n と一致させること。二重指定であり、異なる値を渡すと
    World の他配列（O0/update_order）と不整合になる。

    Args:
        config (Config): 実験設定（ペルソナのテーブルパスや集中度 nu を含む）。
        n (int): 参加者数。
        rng (np.random.Generator): 'persona' 用の乱数ストリーム。

    Raises:
        ValueError: share 合計が 1 でない場合、または concentration_nu <= 0 の場合。
        KeyError: ペルソナ行に必須キーが欠落している場合。
        ValueError: mu_react ∉ [0, 1] の場合（Beta 分布の定義域外）。
            mu_react が 0/1 の境界では Beta の形状母数が 0 になるため、
            scipy.stats.beta.rvs が失敗しうる。

    Returns:
        list[ParticipantParams]: 参加者n人分のParticipantParamsリスト。
    """
    persona_table = load_persona_table(config.personas.table_path)
    concentration_nu = config.personas.concentration_nu

    counts = _persona_counts(persona_table, n)

    persona_labels: list[str] = []
    for p, count in zip(persona_table, counts):
        persona_labels.extend([p["name"]] * count)

    # Watts-Strogatz ネットワーク上で特定の型が空間的に偏るのを防ぐため、
    # 参加者インデックスに対してペルソナ型をランダムにシャッフルする
    rng.shuffle(persona_labels)

    table_by_name = {p["name"]: p for p in persona_table}

    params: list[ParticipantParams] = []
    for name in persona_labels:
        p = table_by_name[name]
        mu = p["mu_react"]  # A8 に基づくキー名

        a_beta = mu * concentration_nu
        b_beta = (1 - mu) * concentration_nu
        theta_react = float(stats.beta.rvs(a_beta, b_beta, random_state=rng))

        # 万が一 [0, 1] をはみ出た場合の防御
        theta_react = float(np.clip(theta_react, 0.0, 1.0))

        params.append(
            ParticipantParams(
                persona=name,
                alpha=p["alpha"],
                beta=p["beta"],
                gamma=p["gamma"],
                w_sys=p["w_sys"],
                w_sns=p["w_sns"],
                b=p["b"],
                theta_react=theta_react,
            )
        )

    return params
