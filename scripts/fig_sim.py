#!/usr/bin/env python
"""scripts/fig_sim.py — 図5.2（意見の分布と軌跡）を自分の実装から描く。

参考実装 sim_ref.py の pair() インタフェースを踏襲しつつ、
意見配列は著者の実装 (world/build.py + run/arm.py) から得る。

- 初期意見 O0・介入なし軌跡 c・介入あり軌跡 a・評価時点 T* を pair() で返す。
- 図5.2は O0平均=0.8（主実験）・代表シード・W=0.5 で描く。
- (a) 意見の分布: t=0 / 介入なしT* / 介入ありT* の意見ヒストグラム（輪郭 step）。
- (b) 参加者の軌跡の例: 数名の意見のターン推移を細線で重ねる（0〜30ターン）。

出力: out/fig5_2_opinion_dynamics.{png,pdf} (300dpi, Noto Sans CJK JP)。
画像側タイトルは入れない（LaTeX caption と二重になるため）。
代わりにシード0の賛成率・E_rate を上部に記し、図と数値を対応させる。
実行: uv run python scripts/fig_sim.py --config configs/main_o0_080.yaml --seed 0 --W 0.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from config import load
from run.arm import run_arm
from run.metrics import t_star
from world.ablation import Ablation, apply_ablation
from world.build import build_world

# 日本語フォント (Noto Sans CJK JP) の登録
for _f in (Path.home() / ".fonts").glob("NotoSansCJK*.otf"):
    font_manager.fontManager.addfont(str(_f))
plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

# 他の図と揃える配色: 介入あり=青、介入なし=灰
C_TREAT = "tab:blue"
C_CTRL = "0.45"
C_INIT = "0.75"


def pair(seed: int, W: float, config):
    """自分の実装で介入あり/なしの軌跡と評価時点 T* を返す。

    Returns:
        (world, a, c, T): world は World、a/c は ArmRecord
        (a=介入あり W, c=介入なし W=0)、T は共通評価時点 T*。
    """
    world = build_world(config, seed)
    attrs = apply_ablation(world.params, Ablation.NONE, config.model.k_react)
    a = run_arm(world, attrs, W=W, config=config)       # 介入あり
    c = run_arm(world, attrs, W=0.0, config=config)     # 介入なし
    T = t_star(c, a, config)
    return world, a, c, T


def main() -> None:
    ap = argparse.ArgumentParser(description="図5.2 意見の分布と軌跡")
    ap.add_argument("--config", default="configs/main_o0_080.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--W", type=float, default=0.5)
    ap.add_argument("--out", default="out")
    ap.add_argument("--n-traj", type=int, default=12, help="軌跡を重ねる参加者数")
    ap.add_argument("--t-max-plot", type=int, default=30, help="(b) の横軸上限")
    args = ap.parse_args()

    config = load(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    world, a, c, T = pair(args.seed, args.W, config)
    theta = config.metrics.theta

    O0 = world.O0                       # 初期意見 (n,)
    O_ctrl = c.O[T]                     # 介入なし T* (n,)
    O_treat = a.O[T]                    # 介入あり T* (n,)
    traj_ctrl = c.O                     # 介入なし軌跡 (T_max+1, n)
    traj_treat = a.O                    # 介入あり軌跡 (T_max+1, n)

    rate_ctrl = float(np.mean(O_ctrl >= theta))
    rate_treat = float(np.mean(O_treat >= theta))
    e_rate = rate_treat - rate_ctrl
    print(f"seed={args.seed} W={args.W} T*={T}")
    print(f"初期意見平均={O0.mean():.3f}  介入なし平均={O_ctrl.mean():.3f}  介入あり平均={O_treat.mean():.3f}")
    print(f"介入なし賛成率={rate_ctrl:.4f}  介入あり賛成率={rate_treat:.4f}  E_rate={e_rate:+.5f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    # (a) 意見の分布（輪郭 step で重なりを見分けやすく）
    bins = np.linspace(0.0, 1.0, 26)  # 幅约0.038
    ax1.hist(O0, bins=bins, color=C_INIT, alpha=0.35,
             label="初期 (t=0)")
    ax1.hist(O_ctrl, bins=bins, histtype="step", color=C_CTRL, linewidth=1.6,
             label="介入なし (t=T*)")
    ax1.hist(O_treat, bins=bins, histtype="step", color=C_TREAT, linewidth=1.6,
             label=f"介入あり W={args.W:g} (t=T*)")
    ax1.axvline(theta, color="black", linewidth=0.9, linestyle="--",
                label="二値化閾値 0.5")
    ax1.set_xlabel("意見 $O$")
    ax1.set_ylabel("参加者数")
    ax1.set_title("(a) 意見の分布")
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(bottom=0)
    ax1.legend(fontsize=8, loc="upper left")

    # (b) 参加者の軌跡の例（十数人の細線を重ねる）
    n = traj_ctrl.shape[1]
    idx = np.linspace(0, n - 1, args.n_traj).astype(int)
    tp = np.arange(min(args.t_max_plot + 1, traj_ctrl.shape[0]))
    for i in idx:
        ax2.plot(tp, traj_ctrl[tp, i], color=C_CTRL, linewidth=0.8, alpha=0.55)
        ax2.plot(tp, traj_treat[tp, i], color=C_TREAT, linewidth=0.8, alpha=0.75)
    ax2.axvline(T, color="black", linewidth=0.9, linestyle=":")
    ax2.axhline(theta, color="gray", linewidth=0.8, linestyle="--")
    ax2.annotate(f"$T^*={T}$", xy=(T, 0.02), xytext=(T + 0.6, 0.04), fontsize=9)
    ax2.plot([], [], color=C_CTRL, linewidth=1.0, label="介入なし")
    ax2.plot([], [], color=C_TREAT, linewidth=1.0, label=f"介入あり W={args.W:g}")
    ax2.set_xlabel("ターン $t$")
    ax2.set_ylabel("意見 $O$")
    ax2.set_title("(b) 参加者の軌跡の例")
    ax2.set_xlim(0, args.t_max_plot)
    ax2.set_ylim(0.0, 1.0)
    ax2.legend(fontsize=8, loc="upper right")

    # 画像側に「図5.2」の題名は入れず、シード0の数値を上部に記して図と対応させる
    fig.suptitle(
        f"seed={args.seed}, W={args.W:g}, $T^*={T}$ : "
        f"介入なし賛成率 {rate_ctrl:.3f} / 介入あり {rate_treat:.3f} / E_rate {e_rate:+.3f}",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    png = out_dir / "fig5_2_opinion_dynamics.png"
    pdf = out_dir / "fig5_2_opinion_dynamics.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"saved: {png} / {pdf}")


if __name__ == "__main__":
    main()
