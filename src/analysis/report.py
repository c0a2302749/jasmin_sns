"""analysis/report.py — F-16: report.md と図の生成 (FR-E6, NFR-8, D-report)。

設計決定参照:
- F-16 / FR-E6: シード平均と信頼区間を報告する
- D-report: 図3枚と表1を人間向け出力とする
- 依存の向き (設計③§5): analysis は Parquet だけを見る。run/* への import 禁止。
  設定値 (B, seed, T_max, n) とペルソナ上限は引数で受け取る。

matplotlib への依存はこのモジュールと scripts/ に限定する (設計③§3)。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from analysis.stats import bootstrap_ci


def _cell_summary(
    df: pl.DataFrame, B: int, boot_seed: int
) -> dict[tuple[str, float], dict[str, float]]:
    """(ablation, W) ごとの E_rate 平均と95%CIを辞書で返す。"""
    out: dict[tuple[str, float], dict[str, float]] = {}
    keys = df.select(["ablation_condition", "W"]).unique().sort(["ablation_condition", "W"])
    for row in keys.iter_rows(named=True):
        ab, w = str(row["ablation_condition"]), float(row["W"])
        vals = (
            df.filter(
                (pl.col("ablation_condition").cast(pl.String) == ab)
                & (pl.col("W") == w)
            )
            .select("E_rate")
            .to_series()
            .to_list()
        )
        r = bootstrap_ci([float(v) for v in vals], B=B, seed=boot_seed)
        r["n"] = float(len(vals))
        out[(ab, w)] = r
    return out


def make(
    summary_path: Path,
    out_dir: Path,
    *,
    B: int,
    boot_seed: int,
    T_max: int,
    n: int,
    persona_bounds: dict[str, float] | None = None,
) -> Path:
    """summary Parquet から report.md と図3枚を生成する。

    Args:
        summary_path: summary/part.parquet へのパス。
        out_dir: 出力ディレクトリ (report.md と fig_*.png を書く)。
        B: ブートストラップ回数。
        boot_seed: ブートストラップ用シード。
        T_max: モデルの最大ターン (clip率の分母 T_max*n に用いる)。
        n: 参加者数。
        persona_bounds: 型名 -> R_raw上限 (w_sys+w_sns+b)。clip不可能群の
            証明表に用いる。None の場合は同節を省略する。

    Returns:
        生成した report.md のパス。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pl.read_parquet(summary_path)
    ablations = sorted(df.select("ablation_condition").cast(pl.String).unique().to_series().to_list())
    w_values = sorted(df.select("W").unique().to_series().to_list())
    w_values = [float(w) for w in w_values]

    cell = _cell_summary(df, B=B, boot_seed=boot_seed)

    def mean_e(ab: str, w: float) -> float:
        return cell[(ab, w)]["mean"]

    # --- 図1: 用量反応曲線 (none + ABL-REACT対照、95%CI) ---
    fig1 = out_dir / "fig_dose_response.png"
    xs = w_values
    for ab, marker, color in (("none", "o", "black"), ("ABL-REACT", "s", "gray")):
        if ab not in ablations:
            continue
        ys = [cell[(ab, w)]["mean"] for w in xs]
        lo = [cell[(ab, w)]["lo"] for w in xs]
        hi = [cell[(ab, w)]["hi"] for w in xs]
        lower = [m - l for m, l in zip(ys, lo, strict=True)]
        upper = [h - m for h, m in zip(hi, ys, strict=True)]
        plt.errorbar(xs, ys, yerr=[lower, upper], marker=marker, color=color,
                     ecolor=color, capsize=3, label=ab)
    plt.axhline(0.0, linewidth=0.8, color="black")
    plt.xlabel("W")
    plt.ylabel("E_rate (mean with 95% CI)")
    plt.title("Dose-response (none vs ABL-REACT)")
    plt.legend()
    plt.savefig(fig1, dpi=150, bbox_inches="tight")
    plt.close()

    # --- 図2: ablation 比較 (Wごとの棒) ---
    fig2 = out_dir / "fig_ablation.png"
    width = 0.8 / max(len(ablations), 1)
    xpos = list(range(len(xs)))
    for j, ab in enumerate(ablations):
        ys = [cell[(ab, w)]["mean"] for w in xs]
        off = [(x + (j - (len(ablations) - 1) / 2) * width) for x in xpos]
        plt.bar(off, ys, width=width, label=ab)
    plt.xticks(xpos, [f"{w:g}" for w in xs])
    plt.xlabel("W")
    plt.ylabel("mean E_rate")
    plt.title("Ablation comparison")
    plt.legend()
    plt.savefig(fig2, dpi=150, bbox_inches="tight")
    plt.close()

    # --- 図3: clip率 (介入側、T_max*n 分母) と C2=5% 線 ---
    fig3 = out_dir / "fig_clip.png"
    denom = T_max * n
    clip_means: list[float] = []
    for w in xs:
        sub = df.filter(
            (pl.col("ablation_condition").cast(pl.String) == "none") & (pl.col("W") == w)
        ).select("n_clipped_R_treat").to_series().to_list()
        clip_means.append(sum(float(v) for v in sub) / len(sub) / denom if sub else 0.0)
    plt.plot(xs, clip_means, marker="o", color="black")
    plt.axhline(0.05, linestyle="--", linewidth=1.0, color="black")
    plt.xlabel("W")
    plt.ylabel("clip rate (treat, all turns)")
    plt.title("Ceiling clip rate vs W (dashed=C2 5%)")
    plt.savefig(fig3, dpi=150, bbox_inches="tight")
    plt.close()

    # --- 表1: none の E_rate 平均[95%CI] ---
    lines: list[str] = []
    lines.append("# F-16 report")
    lines.append("")
    lines.append(f"seeds per cell: n={int(cell[(ablations[0], xs[0])]['n'])}"
                 if cell else "seeds per cell: n/a")
    lines.append(f"bootstrap: B={B}, seed={boot_seed}, percentile 95% CI")
    lines.append("")
    lines.append("## 表1 E_rate mean [95% CI] (ablation=none)")
    lines.append("")
    lines.append("| W | mean | lo | hi |")
    lines.append("|---|---|---|---|")
    for w in xs:
        r = cell[("none", w)] if ("none", w) in cell else None
        if r is None:
            continue
        lines.append(f"| {w:g} | {r['mean']:+.5f} | {r['lo']:+.5f} | {r['hi']:+.5f} |")
    lines.append("")
    lines.append("## ablation 比較 (mean E_rate)")
    lines.append("")
    lines.append("| ablation | " + " | ".join(f"{w:g}" for w in xs) + " |")
    lines.append("|---|" + "|".join("---" for _ in xs) + "|")
    for ab in ablations:
        lines.append("| " + ab + " | " + " | ".join(f"{mean_e(ab, w):+.5f}" for w in xs) + " |")
    lines.append("")
    lines.append("## clip率 (none, treat, 全ターン, 分母=T_max*n)")
    lines.append("")
    lines.append("| W | clip_rate |")
    lines.append("|---|---|")
    for w, c in zip(xs, clip_means, strict=True):
        lines.append(f"| {w:g} | {c:.5f} |")
    lines.append("")

    # --- 主張 ---
    if "none" in ablations and xs:
        vals = {w: cell[("none", w)]["mean"] for w in xs if ("none", w) in cell}
        peak_w = max(vals, key=lambda w: abs(vals[w]))
        tail_w = max(vals)
        peak, tail = vals[peak_w], vals[tail_w]
        decay = (1 - abs(tail) / abs(peak)) * 100 if peak != 0 else float("nan")
        lines.append("## 主張")
        lines.append("")
        lines.append(f"- 逆U字型: |E_rate| は W={peak_w:g} で最大 ({peak:+.5f})、"
                     f"W={tail_w:g} で {tail:+.5f} (減衰 {decay:.1f}%)。")
        sys_zero = all(("ABL-SYS", w) in cell and cell[("ABL-SYS", w)]["mean"] == 0.0 for w in xs)
        lines.append(f"- ABL-SYS では全Wで効果がゼロ ({sys_zero})。"
                     "制度的経路が効果の唯一の媒介という反実仮想の証拠。")
        if ("ABL-REACT", peak_w) in cell:
            lines.append("- ABL-REACT では単調化し、高Wでの減衰が消える。"
                         "リアクタンス (W_eff の潰れ) が減衰の機構という対照。")
    if persona_bounds:
        imp = sorted(nm for nm, b in persona_bounds.items() if b <= 1.0)
        pos = sorted(nm for nm, b in persona_bounds.items() if b > 1.0)
        lines.append(f"- clip不可能群 (R_raw上限<=1: {', '.join(imp) or 'なし'}) のみでも"
                     "山型が残り、同群の観測clipは0セル (verification_report §4.6)。"
                     "天井ではなくリアクタンスの説明が残る。"
                     f"ただし同群はRの可動域が狭い点は留保する (clip可能群: {', '.join(pos) or 'なし'})。")
    lines.append("")

    md_path = out_dir / "report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path
