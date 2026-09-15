"""world/ablation.py — F-07: Ablation条件で属性を上書きし、属性配列を1回作る。

設計決定参照:
- FR-E5, A6: Ablation条件 (none, ABL-SYS, ABL-SNS, ABL-REACT)
- A9: k_react=0 によるリアクタンス無効化 (ABL-REACT)
- D-状態表現: Attrs 内の ndarray 配列は writeable = False で不変化する
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
import numpy as np

from world.persona import ParticipantParams


class Ablation(StrEnum):
    """Ablation条件を表す列挙型（FR-E5, A6, A9）。

    NONE: 上書きなしのベースライン。SYS: w_sys=0。SNS: w_sns=0。
    REACT: k_react=0（theta_react は残る）。いずれも alpha/beta/gamma には触れない。
    """
    NONE = "none"
    SYS = "ABL-SYS"
    SNS = "ABL-SNS"
    REACT = "ABL-REACT"


@dataclass(frozen=True)
class Attrs:
    """計算高速化・一括処理のための属性配列コンテナ。

    Ablation上書き後に1回だけ作成され、全配列は不変となる。
    Attributes:
        alpha (np.ndarray): 意見更新式の同調圧力係数
        beta (np.ndarray): 意見更新式のリスク認知係数
        gamma (np.ndarray): 意見更新式の初期意見固執係数
        w_sys (np.ndarray): リスク認知式の制度的介入係数
        w_sns (np.ndarray): リスク認知式の社会的影響係数
        b (np.ndarray): リスク認知式のベースラインリスク認知
        theta_react (np.ndarray): リスク認知式のリアクタンス係数
        k_react (float): リアクタンス減衰係数（ABL-REACTで0に上書きされる）
        condition (Ablation): 適用されたAblation条件
    """
    alpha: np.ndarray
    beta: np.ndarray
    gamma: np.ndarray
    w_sys: np.ndarray
    w_sns: np.ndarray
    b: np.ndarray
    theta_react: np.ndarray
    k_react: float
    condition: Ablation

    def __post_init__(self) -> None:
        """全属性配列を読み取り専用に設定して不変条件を保証する。
        """
        for arr in (
            self.alpha, self.beta, self.gamma,
            self.w_sys, self.w_sns, self.b, self.theta_react,
        ):
            arr.flags.writeable = False


def apply_ablation(
    params: list[ParticipantParams],
    condition: Ablation,
    k_react: float,
) -> Attrs:
    """Ablation条件に基づき属性配列を構築し、不変化した Attrs オブジェクトを返す。

    元の params リストおよび ParticipantParams オブジェクトは一切変更されない
    （dataclasses.replace 経由で複製してから上書きする）。
    condition="none" でも必ずこの関数を通し、属性配列を作る場所を1つに保つ。
    SYS は w_sys のみ、SNS は w_sns のみを 0 にし、alpha/beta/gamma には触れない（A6）。
    REACT は k_react のみを 0 にし、theta_react 配列は残す（A9）。

    Args:
        params (list[ParticipantParams]): 参加者の基本属性リスト。空リストは不可。
        condition (Ablation): 適用するAblation条件。
        k_react (float): 基本となるリアクタンス減衰係数。

    Raises:
        ValueError: params が空の場合（内部の zip(*...) が空シーケンスで失敗する）。

    Returns:
        Attrs: 上書き適用済みの属性配列群。
    """

    effective_k_react = 0.0 if condition == Ablation.REACT else k_react

    # 設計指示通り、replace()経由で上書きされた新しいリストを作成
    if condition == Ablation.SYS:
        params = [replace(p, w_sys=0.0) for p in params]
    elif condition == Ablation.SNS:
        params = [replace(p, w_sns=0.0) for p in params]

    (
        alphas, betas, gammas,
        w_syss, w_snss, bs, theta_reacts
    ) = zip(*[
        (p.alpha, p.beta, p.gamma, p.w_sys, p.w_sns, p.b, p.theta_react)
        for p in params
    ])

    return Attrs(
        alpha=np.array(alphas, dtype=np.float64),
        beta=np.array(betas, dtype=np.float64),
        gamma=np.array(gammas, dtype=np.float64),
        w_sys=np.array(w_syss, dtype=np.float64),
        w_sns=np.array(w_snss, dtype=np.float64),
        b=np.array(bs, dtype=np.float64),
        theta_react=np.array(theta_reacts, dtype=np.float64),
        k_react=effective_k_react,
        condition=condition,
    )
