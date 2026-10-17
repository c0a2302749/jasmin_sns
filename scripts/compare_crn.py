"""scripts/compare_crn.py — 共通乱数法 (CRN) の分散削減効果 (5.3.6)。

実行: wsl bash -lc "cd ~/jasmin_sns && uv run python scripts/compare_crn.py --n-seeds 30"

(a) 共通シード: 同一 world で ctrl(W=0)/treat(W) を実行し E_rate を算出。
(b) 独立シード: ctrl は seed s、treat は seed perm[s] (1/2ずらし) の別 world
    で実行し、rate_treat - rate_ctrl を算出 (ペア数は同数)。
基準: (a) の標本分散 < (b) の標本分散 (不偏分散 ddof=1)。
分散比には、対を保ったままのブートストラップ95%CIを添える
（30シードでは標本ばらつきが大きいため）。
"""
from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from config import load
from run.arm import run_arm
from run.metrics import t_star
from runlog import log_header
from world.ablation import Ablation, apply_ablation
from world.build import build_world


def rate_at(O_Tstar: np.ndarray, theta: float) -> float:
    return float(np.mean(O_Tstar >= theta))


def ratio_ci(e_common: list[float], e_indep: list[float], B: int, seed: int,
             alpha: float = 0.05) -> tuple[float, float]:
    """対を保った復元抽出による分散比 var_c/var_i の percentile CI。

    シード乱数系列には触れない (random.Random のみ使用)。
    """
    n = len(e_common)
    rng = random.Random(seed)
    ratios: list[float] = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        sc = [e_common[i] for i in idx]
        si = [e_indep[i] for i in idx]
        vc = statistics.variance(sc) if n > 1 else 0.0
        vi = statistics.variance(si) if n > 1 else 0.0
        ratios.append(vc / vi if vi > 0 else float("nan"))
    ratios = sorted(r for r in ratios if r == r)
    if not ratios:
        return float("nan"), float("nan")
    lo = ratios[min(max(int((alpha / 2) * len(ratios)), 0), len(ratios) - 1)]
    hi = ratios[min(max(int((1 - alpha / 2) * len(ratios)), 0), len(ratios) - 1)]
    return float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser(description="共通乱数法の分散比較")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n-seeds", type=int, default=30)
    ap.add_argument("--seed-start", type=int, default=0)
    args = ap.parse_args()

    config = load(args.config)
    theta = config.metrics.theta
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    log_header(config, f"seeds={seeds[0]}..{seeds[-1]} ({len(seeds)}) theta={theta}")
    W_list = [w for w in config.experiment.W_values if w > 0.0] or [0.5]
    # 独立比較用の treat 側シードは ctrl 側と互いに素にする (P0: +15ずらしの重複を排除)。
    # 確認用 30-59, 予約 60-89 と衝突しないよう +1000 をずらす。
    INDEP_OFFSET = 1000
    indep_seeds = [s + INDEP_OFFSET for s in seeds]

    print("=" * 104)
    print("(a)共通 vs (b)独立 の E_rate 標本分散の比較 (比にbootstrap 95%CI)")
    print(f"seeds={seeds[0]}..{seeds[-1]} ({len(seeds)}) theta={theta} indep_offset={INDEP_OFFSET}")
    print("=" * 104)
    print("  W      var_common  var_indep   比[95%CI]              mean_common  mean_indep  判定")
    all_pass = True
    for W in W_list:
        # 世界を事前構築 (共通・独立の両方に使う。独立側は互いに素なシード集合)
        worlds = {s: build_world(config, s) for s in seeds}
        worlds_indep = {s: build_world(config, s) for s in indep_seeds}
        # (a) 共通
        e_common: list[float] = []
        for s in seeds:
            w = worlds[s]
            attrs = apply_ablation(w.params, Ablation.NONE, config.model.k_react)
            ctrl = run_arm(w, attrs, W=0.0, config=config)
            treat = run_arm(w, attrs, W=W, config=config)
            t = t_star(ctrl, treat, config)
            rc, rt = rate_at(ctrl.O[t], theta), rate_at(treat.O[t], theta)
            e_common.append(rt - rc)
        # (b) 独立: ctrl は seeds[i] の世界、treat は indep_seeds[i] の世界で
        # それぞれ同一世界内の T* で評価した rate の差を取る。60世界は互いに素。
        e_indep: list[float] = []
        for s_ctrl, s_treat in zip(seeds, indep_seeds):
            w_c, w_t = worlds[s_ctrl], worlds_indep[s_treat]
            a_c = apply_ablation(w_c.params, Ablation.NONE, config.model.k_react)
            a_t = apply_ablation(w_t.params, Ablation.NONE, config.model.k_react)
            ctrl_c = run_arm(w_c, a_c, W=0.0, config=config)
            treat_t = run_arm(w_t, a_t, W=W, config=config)
            # treat 側の rate は自身の世界での T* (ctrl_t との対) で評価する
            ctrl_t = run_arm(w_t, a_t, W=0.0, config=config)
            t_c = t_star(ctrl_c, ctrl_c, config)
            t_t = t_star(ctrl_t, treat_t, config)
            e_indep.append(rate_at(treat_t.O[t_t], theta) - rate_at(ctrl_c.O[t_c], theta))
        vc = float(np.var(e_common, ddof=1)) if len(seeds) > 1 else 0.0
        vi = float(np.var(e_indep, ddof=1)) if len(seeds) > 1 else 0.0
        ratio = vc / vi if vi > 0 else float("nan")
        lo, hi = ratio_ci(e_common, e_indep, B=config.metrics.bootstrap_B, seed=config.metrics.bootstrap_seed)
        ok = bool(vc < vi)
        all_pass &= ok
        print(f"  {W:.2f}   {vc:.6f}    {vi:.6f}    {ratio:.3f} [{lo:.3f},{hi:.3f}]  "
              f"{float(np.mean(e_common)):+.5f}     {float(np.mean(e_indep)):+.5f}      "
              f"{'PASS' if ok else 'FAIL'}")
    print(f"基準 (全Wで var_common < var_indep): {'PASS' if all_pass else 'FAIL'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
