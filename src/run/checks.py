"""run/checks.py — F-13: 経路分離などの不変条件を検査する。

設計決定参照:
- F-13: 不変条件を検査する
- NFR-1: 決定論性・再現性の検査（TURN0_IDENTICAL, W0_IDENTICAL）
- FR-E5: Ablation 条件における経路分離の検証（ABL_SYS_BITWISE, ABL_SNS_R_INDEPENDENT）
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from run.arm import ArmRecord
from world.ablation import Attrs
from world.build import World


def _hash_array(arr: np.ndarray) -> str:
    """ndarray のメモリバッファから SHA-256 ハッシュ文字列を算出する。"""
    return hashlib.sha256(arr.tobytes()).hexdigest()


def run_checks(
    ctrl: ArmRecord,
    treat: ArmRecord,
    W_level: float,
    ablation: str,
    world: World,
    attrs: Attrs,
) -> list[dict[str, Any]]:
    """対比較アームの不変条件を検証し、checks.parquet の元となるレコードリストを返す。

    io/schema.py の CHECKS_SCHEMA に沿った辞書を生成する。

    Args:
        ctrl (ArmRecord): 介入なしアームの実行記録。
        treat (ArmRecord): 介入ありアームの実行記録。
        W_level (float): 介入アームの介入強度 W。
        ablation (str): 適用したAblation条件。
        world (World): シード確定済みの世界。
        attrs (Attrs): Ablation適用済みの属性配列（ABL-SNS の式検証に用いる）。

    Returns:
        list[dict[str, Any]]: CHECKS_SCHEMA に合致する検査レコードのリスト。
    """
    results: list[dict[str, Any]] = []

    def add_result(check_id: str, passed: bool, detail: str = "") -> None:
        results.append(
            {
                "seed": world.seed,
                "W": float(W_level),
                "ablation_condition": ablation,
                "check_id": check_id,
                "passed": bool(passed),
                "detail": detail,
            }
        )

    # 1. 決定論性の確認（NFR-1）
    # update_order が軌跡と整合する形状で、各行が 0..n-1 の順列かを検証する。
    # ついでに来歴追跡用ハッシュを detail に記録する (NFR-4)。
    order = world.update_order
    expected_shape = (ctrl.O.shape[0] - 1, ctrl.O.shape[1])
    shape_ok = order.shape == expected_shape
    perm_ok = False
    if shape_ok:
        perm_ok = bool(np.all(np.sort(order, axis=1) == np.arange(order.shape[1])))
    add_result(
        "ORDER_VALID",
        shape_ok and perm_ok,
        f"shape={order.shape}, expected={expected_shape}, "
        f"all_permutations={perm_ok}, "
        f"update_order_hash={_hash_array(order)[:8]}",
    )

    o0_equal = bool(np.array_equal(ctrl.O[0], treat.O[0]))
    r0_equal = bool(np.array_equal(ctrl.R[0], treat.R[0]))
    add_result(
        "TURN0_IDENTICAL",
        o0_equal and r0_equal,
        f"O0_equal={o0_equal}, R0_equal={r0_equal}",
    )

    # W=0.0 条件においては、Wの有無で差が出ないため全ターンで bit 一致するはず（NFR-1）
    if W_level == 0.0 and ablation == "none":
        o_bit_equal = bool(np.array_equal(ctrl.O, treat.O))
        r_bit_equal = bool(np.array_equal(ctrl.R, treat.R))
        add_result(
            "W0_IDENTICAL",
            o_bit_equal and r_bit_equal,
            f"O_bitwise={o_bit_equal}, R_bitwise={r_bit_equal}",
        )

    # 2. 経路分離の確認（FR-E5）
    # ABL-SYS (w_sys=0): 制度的介入が無効なため、W の値にかかわらず treat と ctrl の全軌跡が bit 一致する
    if ablation == "ABL-SYS":
        o_bit_equal = bool(np.array_equal(ctrl.O, treat.O))
        r_bit_equal = bool(np.array_equal(ctrl.R, treat.R))
        add_result(
            "ABL_SYS_BITWISE",
            o_bit_equal and r_bit_equal,
            f"O_bitwise={o_bit_equal}, R_bitwise={r_bit_equal}",
        )

    # ABL-SNS (w_sns=0): 社会的影響（S_R）が無効なため、R は近傍に依存せず数式だけで閉じて計算できる
    if ablation in ("ABL-SNS", "ABL-SNS+ALPHA"):
        # R_i = Clip(w_sys_i * W_eff_i + b_i)
        expected_r = np.clip(
            attrs.w_sys[None, :] * treat.W_eff[None, :] + attrs.b[None, :], 
            0.0, 
            1.0
        )
        # ターン 1 以降の全履歴と比較できるよう broadcast する
        expected_r_full = np.broadcast_to(expected_r, treat.R[1:].shape)
        
        r_sns_pass = bool(np.allclose(treat.R[1:], expected_r_full, atol=1e-12))
        add_result(
            "ABL_SNS_R_INDEPENDENT",
            r_sns_pass,
            f"R_matches_isolated_formula={r_sns_pass}",
        )

    # ABL-ALPHA (alpha=0): O経由の同調項が無効なため、alpha 配列は全0であること。
    # ABL-SNS+ALPHA でも同様に alpha は全0であること。
    if ablation in ("ABL-ALPHA", "ABL-SNS+ALPHA"):
        alpha_zero = bool(np.all(attrs.alpha == 0.0))
        add_result(
            "ABL_ALPHA_ZERO",
            alpha_zero,
            f"alpha_all_zero={alpha_zero}",
        )

    # 3. 単調性: 介入は -βR 経由でのみ O を下げる (β>=0, R>=0) ため、
    # 介入あり O(treat) <= 介入なし O(ctrl) が全ターン・全参加者で成立するはず。
    # W=0 では自明に成立するため W>0 のときのみ判定する。tol=1e-9。
    if W_level > 0.0:
        max_violation = float(np.max(treat.O - ctrl.O))
        mono_pass = bool(max_violation <= 1e-9)
        add_result(
            "MONOTONE_DECREASE",
            mono_pass,
            f"max(O_treat-O_ctrl)={max_violation:.6e} (tol=1e-9)",
        )

    return results