"""tests/test_layers.py — layers/* の単体テスト。

状態配列を持たず、入力を与えて出力を見る純粋関数の振る舞いを検証する。
境界値やエッジケースを中心にテストする。

設計決定参照:
- FR-S2, FR-S3: 近傍集約 S_O, S_R
- FR-P3: R の更新とクリップ
- A3: S による安定性の切り替わり
- A9: k_react の振る舞い
- A13: W_eff の境界連続性
"""
from __future__ import annotations

import numpy as np
import pytest

from layers import governance, participant, sns


# ==========================================
# Governance Layer (W_eff) のテスト
# ==========================================

def test_governance_w_eff_boundary_continuity_A13():
    """W <= θ で W_eff == W となり、境界 W = θ で両側から連続であることを検証 (A13)。"""
    theta = np.array([0.4])
    k = 1.0
    
    # 閾値未満
    w_eff_below, active_below = governance.calculate_w_eff(0.3, theta, k)
    assert pytest.approx(w_eff_below[0]) == 0.3
    assert active_below == 0

    # 境界値周辺の連続性（等号処理の確認）
    below = governance.calculate_w_eff(0.4 - 1e-9, theta, k)[0]
    at = governance.calculate_w_eff(0.4, theta, k)[0]
    above = governance.calculate_w_eff(0.4 + 1e-9, theta, k)[0]
    
    np.testing.assert_allclose(below, at, atol=1e-6)
    np.testing.assert_allclose(above, at, atol=1e-6)


def test_governance_w_eff_k_behavior_A9():
    """リアクタンス減衰係数 k の値による振る舞いの違いを検証 (A9)。"""
    theta = np.array([0.3])
    
    # k = 1.0 のとき: W_eff = θ - (W - θ) となり、W = 2θ で 0 に達して鏡像的に減衰する
    w_eff_k1, active_k1 = governance.calculate_w_eff(W=0.6, theta_react=theta, k_react=1.0)
    assert pytest.approx(w_eff_k1[0], abs=1e-9) == 0.0
    assert active_k1 == 1

    # k = 0.0 のとき: W_eff = min(W, θ) となり飽和するのみ（リアクタンスの無効化）
    theta_k0 = np.array([0.4])
    for w in (0.1, 0.4, 0.9):
        expected = min(w, 0.4)
        w_eff_k0, _ = governance.calculate_w_eff(W=w, theta_react=theta_k0, k_react=0.0)
        assert pytest.approx(w_eff_k0[0]) == expected


# ==========================================
# SNS Layer (S_O, S_R) のテスト
# ==========================================

def test_sns_social_signals_normal_FR_S2_S3():
    """近傍の O, R 集約計算が意図通り行われるかを検証 (FR-S2, FR-S3)。"""
    # 参加者 0 の近傍が 1, 2 であるネットワーク
    neighbors = [np.array([1, 2])]
    O = np.array([0.2, 0.6, 0.8])
    R = np.array([0.1, 0.5, 0.9])
    
    s_o, s_r = sns.social_signals(0, neighbors, O, R)
    
    # 近傍 (1, 2) の O 平均は 0.7。S_O = 0.7 - 0.2 = 0.5
    assert pytest.approx(s_o) == 0.5
    # 近傍 (1, 2) の R 平均は (0.5 + 0.9) / 2 = 0.7
    assert pytest.approx(s_r) == 0.7


def test_sns_social_signals_isolated_FR_S4():
    """孤立ノード（近傍なし）の場合にゼロ除算を回避し 0.0 を返すかを検証 (A14の防御的実装)。"""
    neighbors = [np.array([], dtype=np.int64)]
    O = np.array([0.7])
    R = np.array([0.3])
    
    s_o, s_r = sns.social_signals(0, neighbors, O, R)
    
    assert s_o == 0.0
    assert s_r == 0.0


# ==========================================
# Participant Layer (update_R, update_O) のテスト
# ==========================================

def test_participant_update_R_clipping_FR_P3():
    """R の計算および [0, 1] へのクリップ動作の検証 (FR-P3)。"""
    # クリップされないケース: 0.5*0.4 + 0.5*0.2 + 0.1 = 0.2 + 0.1 + 0.1 = 0.4
    r_val, r_raw = participant.update_R(w_sys=0.5, w_eff=0.4, w_sns=0.5, S_R=0.2, b=0.1)
    assert pytest.approx(r_val) == 0.4
    assert pytest.approx(r_raw) == 0.4

    # 1.0 を超えてクリップされるケース
    r_val_high, r_raw_high = participant.update_R(w_sys=1.0, w_eff=1.0, w_sns=1.0, S_R=1.0, b=1.0)
    assert r_val_high == 1.0
    assert r_raw_high == 3.0  # クリップ前の値はそのまま保持される


def test_participant_update_O_unstimulated_S2_A3():
    """無刺激時 (S_O=0, R=0)・S=2.0 で O が不動点 0.5 に回帰する性質の検証 (A3)。"""
    # O0 = 0.5 のとき、無刺激では不動点 0.5 に留まる
    new_o, _ = participant.update_O(
        O_i=0.5, O0_i=0.5, alpha=0.5, beta=0.5, gamma=0.1, S_O=0.0, R_i=0.0, S=2.0
    )
    assert pytest.approx(new_o) == 0.5

    # 0.5 から離れている場合、S=2.0 では無刺激で 0.5 に向かって緩やかに引き寄せられる
    # （|O(t+1) - 0.5| < |O(t) - 0.5| が成立する）
    o_start = 0.9
    o_next, _ = participant.update_O(
        O_i=o_start, O0_i=o_start, alpha=0.0, beta=0.0, gamma=0.0, S_O=0.0, R_i=0.0, S=2.0
    )
    
    # 0.5 への距離が更新前よりも小さくなっていることを確認
    assert abs(o_next - 0.5) < abs(o_start - 0.5)


def test_participant_update_O_unstimulated_S3_A3():
    """無刺激時・S=3.0 で 0.5 から微小にずれた O が 0.5 から離れる（不安定化）性質の検証 (A3)。"""
    o_start = 0.5 + 1e-3
    o_next, _ = participant.update_O(
        O_i=o_start, O0_i=o_start, alpha=0.0, beta=0.0, gamma=0.0, S_O=0.0, R_i=0.0, S=3.0
    )
    
    # S > 2.0 のときは 0.5 が不安定化し、自発的に分極化へ向かう
    assert abs(o_next - 0.5) > abs(o_start - 0.5)