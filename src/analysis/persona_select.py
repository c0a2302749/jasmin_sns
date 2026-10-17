"""analysis/persona_select.py — F-17: ペルソナ選定（段階0）の基準C1〜C4を計算する。対応決定: A7（改）"""
from __future__ import annotations

import numpy as np

from layers import governance
from run.arm import ArmRecord


def _pol_between(O_t: np.ndarray, theta: float) -> float:
    high = O_t[O_t >= theta]
    low = O_t[O_t < theta]
    if high.size == 0 or low.size == 0:
        return 0.0
    return float(abs(high.mean() - low.mean()))


def criteria(
    no_intervention_records: list[ArmRecord],
    theta_react_by_persona: dict[str, np.ndarray],
    W_probe: float,
    k_react: float,
    theta: float = 0.5,
) -> dict:
    n_seeds = len(no_intervention_records)

    c1_hits = 0
    total_clips = 0
    total_cells = 0
    c4_hits = 0
    for rec in no_intervention_records:
        t_end = rec.t_conv if rec.t_conv is not None else rec.O.shape[0] - 1
        pol0 = _pol_between(rec.O[0], theta)
        pol_t = _pol_between(rec.O[t_end], theta)
        if pol_t > pol0:
            c1_hits += 1
        total_clips += rec.n_clipped_R
        total_cells += (rec.O.shape[0] - 1) * rec.O.shape[1]
        if rec.t_conv is not None:
            c4_hits += 1

    c1_rate = c1_hits / n_seeds
    c2_rate = total_clips / total_cells if total_cells else 0.0
    c4_rate = c4_hits / n_seeds

    c3: dict[str, float] = {}
    for persona, thetas in theta_react_by_persona.items():
        w_eff_vals, _ = governance.calculate_w_eff(W_probe, thetas, k_react)
        c3[persona] = float(np.mean(w_eff_vals < W_probe))

    # C3_pass の自動判定: 反発しやすい型 (>=50%) と反発しにくい型 (<=20%)
    # の弁別性。テーブル名に依存しないよう max/min で判定する
    # (persona_A: max=reactant 0.988, min=conformist 0.073 で合格)。
    c3_vals = list(c3.values())
    c3_pass = (
        bool(max(c3_vals) >= 0.50 and min(c3_vals) <= 0.20)
        if c3_vals else False
    )

    return {
        "C1_baseline_reproduction_rate": c1_rate,
        "C1_pass": c1_rate >= 0.80,
        "C2_ceiling_clip_rate": c2_rate,
        "C2_pass": c2_rate < 0.05,
        "C3_reactance_discriminability_by_persona": c3,
        "C3_pass": c3_pass,
        "C4_convergence_rate": c4_rate,
        "C4_pass": c4_rate >= 0.95,
    }