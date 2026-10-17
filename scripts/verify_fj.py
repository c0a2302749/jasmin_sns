"""scripts/verify_fj.py — 線形版と Friedkin-Johnsen 閉形式の照合 (5.3.2)。

実行: wsl bash -lc "cd ~/jasmin_sns && uv run python scripts/verify_fj.py --config configs/default.yaml --n-seeds 5"

線形版の定義 (既定の検証条件: W=0.2):
    O_i(t+1) = O_i(t) + α_i * S_O,i(t) - β_i * R_i - γ_i * (O_i(t) - O0_i)
    S_O,i は本番と同一の layers/sns.social_signals で集約。tanh・Clip のみなし。
    R_i は ABL-SNS の閉形式 R = Clip(w_sys*W_eff + b) で時間一定とする
    (社会的経路を切断し、FJ の外生入力 u に対応させる)。

閉形式 (定常解 A O* = b):
    A_ii = α_i + γ_i,  A_ij = -α_i * M_ij (M は行確率隣接行列),
    b_i = γ_i * O0_i - β_i * R_i。
    A が正則なら O* = A^{-1} b が唯一の不動点。

基準: 反復 (Jacobi, 収束まで) と直接解の最大誤差が機械精度 (~1e-9)。
この照合は、集約を本番と共有する線形対応物の確認であり、
tanhとClipを含む更新式本体の検証ではない。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from config import load
from layers import sns
from runlog import log_header
from world.ablation import Ablation, apply_ablation
from world.build import build_world


def fj_closed_form(world, attrs, R_const: np.ndarray) -> np.ndarray:
    """A O* = b を直接解く。R_const は時間一定の外生入力。"""
    n = len(world.neighbors)
    M = np.zeros((n, n), dtype=np.float64)
    for i, nb in enumerate(world.neighbors):
        if nb.size > 0:
            M[i, nb] = 1.0 / nb.size
    alpha = np.asarray(attrs.alpha, dtype=np.float64)
    gamma = np.asarray(attrs.gamma, dtype=np.float64)
    beta = np.asarray(attrs.beta, dtype=np.float64)
    O0 = np.asarray(world.O0, dtype=np.float64)
    A = np.diag(alpha + gamma) - alpha[:, None] * M
    b = gamma * O0 - beta * R_const
    return np.linalg.solve(A, b)


def fj_iterate(world, attrs, R_const: np.ndarray, tol: float = 1e-13,
               max_iter: int = 20000) -> tuple[np.ndarray, int]:
    """線形版の Jacobi 反復。収束したら (O*, iters) を返す。

    近傍集約 S_O は本番と同じ layers/sns.social_signals を呼ぶ
    (集約経路を本番と共有し、tanh/Clip のみを線形化した対応物)。
    """
    n = len(world.neighbors)
    O = np.asarray(world.O0, dtype=np.float64).copy()
    alpha = np.asarray(attrs.alpha, dtype=np.float64)
    beta = np.asarray(attrs.beta, dtype=np.float64)
    gamma = np.asarray(attrs.gamma, dtype=np.float64)
    O0 = np.asarray(world.O0, dtype=np.float64)
    R_arr = np.asarray(R_const, dtype=np.float64)
    for it in range(1, max_iter + 1):
        O_new = O.copy()
        for i in range(n):
            s_o, _ = sns.social_signals(i, world.neighbors, O, R_arr)
            O_new[i] = O[i] + alpha[i] * s_o - beta[i] * R_const[i] - gamma[i] * (O[i] - O0[i])
        if float(np.max(np.abs(O_new - O))) < tol:
            return O_new, it
        O = O_new
    raise RuntimeError(f"線形反復が {max_iter} 回で収束しなかった")


def main() -> None:
    ap = argparse.ArgumentParser(description="線形版とFJ閉形式の照合")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--W", type=float, default=0.2)
    args = ap.parse_args()

    config = load(args.config)
    log_header(config, f"W={args.W} (検証条件)")
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    worst = 0.0
    all_ok = True
    for seed in seeds:
        world = build_world(config, seed)
        attrs = apply_ablation(world.params, Ablation.SNS, config.model.k_react)
        # ABL-SNS の R 閉形式 (時間一定)。W=0.2 では clip なしを期待するが、
        # clip されても同一の R_const を両者に与えるため照合は厳密に成立する。
        from layers import governance
        W_eff, _ = governance.calculate_w_eff(args.W, attrs.theta_react, attrs.k_react)
        R_const = np.clip(
            np.asarray(attrs.w_sys) * np.asarray(W_eff) + np.asarray(attrs.b), 0.0, 1.0)
        O_star = fj_closed_form(world, attrs, R_const)
        O_iter, iters = fj_iterate(world, attrs, R_const)
        err = float(np.max(np.abs(O_star - O_iter)))
        worst = max(worst, err)
        ok = err < 1e-9
        all_ok &= ok
        raw = np.asarray(attrs.w_sys) * np.asarray(W_eff) + np.asarray(attrs.b)
        n_clip = int(np.sum((raw > 1.0) | (raw < 0.0)))
        print(f"seed {seed}: iters={iters} max|direct-iter|={err:.3e} "
              f"{'PASS' if ok else 'FAIL'} n_clip_R={n_clip}")
    print(f"worst max err over {len(seeds)} seeds: {worst:.3e}")
    print("ALL PASS" if all_ok else "FAILURES PRESENT")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
