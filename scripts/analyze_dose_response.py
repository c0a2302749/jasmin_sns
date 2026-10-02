"""scripts/analyze_dose_response.py — 非単調性の原因を機構まで切り分ける。

実行: uv run python scripts/analyze_dose_response.py --n-seeds 30 --step 0.05

問う問い:
  Q1. W を細かく刻んでも非単調性は本物か (頂点は本当に W=0.5 付近か)
  Q2. 機構は「W >= 2*theta_react で W_eff が 0 に潰れる」閉形式どおりか
  Q3. 低 theta_react 層だけが効果を失い、平均を引下げているのか
  Q4. 効果の消失は R の経路 (w_sys) を通じて起きているのか
  Q5. 介入ありで天井効果 (R の上側クリップ) が再発しないか (C2 の適用範囲)
  Q7. ペルソナ型別に用量反応を再集計し、clip が原理的に起きない群だけで
      山型が残るかを調べる（Phase A。設定変更なし、persona_A のまま）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from config import load
from layers import governance
from run.arm import run_arm
from run.metrics import calculate_metrics, t_star
from world.ablation import Ablation, apply_ablation
from world.build import build_world
from world.persona import load_persona_table


def reconstruct_r_raw(rec, attrs) -> np.ndarray:
    """R のクリップ前の生値をArmRecordから事後再構成して返す。shape (T_max, n)。

    R(t) の生値は R(t) = w_sys·W_eff + w_sns·S_R(t) + b であり、S_R(t) は
    S_R ヒストグラムにそのまま記録されている（arm.py:135 と arm.py:138 が同じ s_r を
    使っている）ため、ソースに手を入れずに再構成できる。
    天井効果（上側クリップ）を下側クリップと分離して数えるために使う。
    """
    return (
        attrs.w_sys[None, :] * rec.W_eff[None, :]
        + attrs.w_sns[None, :] * rec.S_R
        + attrs.b[None, :]
    )


def analytic_w_eff(W: float, theta: np.ndarray, k_react: float) -> np.ndarray:
    """W_eff 閉形式を W で微分可能な形に展開した解析解 (A13 の等価形)。

    W <= theta        : W_eff = W
    theta < W < 2theta: W_eff = (1+k)*theta - k*W
    W >= (1+k)*theta  : W_eff = 0
    """
    return np.maximum(0.0, np.minimum(W, (1.0 + k_react) * theta - k_react * W))


def main() -> None:
    ap = argparse.ArgumentParser(description="非単調性の原因分析")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n-seeds", type=int, default=30)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--step", type=float, default=0.05)
    args = ap.parse_args()

    config = load(args.config)
    n = config.network.n
    T_max = config.model.T_max
    k_react = config.model.k_react
    theta_metric = config.metrics.theta
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    W_grid = [round(x, 4) for x in np.arange(0.0, 1.0 + args.step / 2, args.step)]
    W_grid = [w for w in W_grid if 0.0 < w <= 1.0]
    total_cells = T_max * n          # C2 の実母集団（ターン 1..T_max の n 人）

    print("=" * 96)
    print("Q1/Q2. W の細密走査と閉形式の検証")
    print("=" * 96)
    print(f"config={args.config}  seeds={seeds[0]}..{seeds[-1]} ({len(seeds)})  "
          f"k_react={k_react}  W刻み={args.step}")
    print(f"C2 の分母 = T_max*n = {T_max}*{n} = {total_cells} セル/シード")

    # 蓄積器
    rows: dict[float, dict] = {
        w: {"E_rate": [], "flip": [], "dO": [], "mean_weff": [], "frac_zero": [],
            "anal_err": [], "mean_R": [], "frac_R_at_max": [],
            "clip_upper": [], "clip_lower": [], "n_clipped": []}
        for w in W_grid
    }
    # θ_react 三分割の層別集計（シード内で三分位で層を分割）
    tercile: dict[float, dict[str, list[float]]] = {
        w: {"lo_e": [], "mid_e": [], "hi_e": [],
            "lo_weff": [], "mid_weff": [], "hi_weff": [],
            "lo_R": [], "mid_R": [], "hi_R": [],
            "lo_zero": [], "mid_zero": [], "hi_zero": []}
        for w in W_grid
    }
    # Q7 用のペルソナ型別集計。clip 可否は型ごとに証明する。型の並びは
    # persona 表の記載順（設定の正典）に固定し、シャッフル順に依存しない。
    persona_names = [row["name"] for row in load_persona_table(config.personas.table_path)]
    by_persona: dict[float, dict[str, dict[str, list[float]]]] = {
        w: {name: {"flip": [], "zero": [], "up": [], "lo": [], "R": [], "n": []}
            for name in persona_names}
        for w in W_grid
    }
    raw_bound: dict[str, list[float]] = {name: [] for name in persona_names}

    for seed in seeds:
        world = build_world(config, seed)
        attrs = apply_ablation(world.params, Ablation.NONE, k_react)
        ctrl = run_arm(world, attrs, W=0.0, config=config)

        theta = attrs.theta_react
        q1, q2 = np.quantile(theta, [1 / 3, 2 / 3])
        lo = theta <= q1
        mid = (theta > q1) & (theta <= q2)
        hi = theta > q2

        labels = np.array([p.persona for p in world.params])
        persona_mask = {name: (labels == name) for name in persona_names}
        # R_raw の上限は W_eff <= 1 かつ S_R <= 1 の仮定のもとで
        # w_sys + w_sns + b で抑えられる。これが 1 以下なら、その型は
        # どの W・どのターンでも上側クリップが原理的に起きない。
        for name in persona_names:
            idx = persona_mask[name]
            if int(np.sum(idx)) > 0:
                bound = float(np.max(
                    attrs.w_sys[idx] + attrs.w_sns[idx] + attrs.b[idx]))
                raw_bound[name].append(bound)

        for W in W_grid:
            treat = run_arm(world, attrs, W=W, config=config)
            m = calculate_metrics(ctrl, treat, world, config, W_level=W, ablation="none")
            t = t_star(ctrl, treat, config)

            w_expected = analytic_w_eff(W, theta, k_react)
            anal_err = float(np.max(np.abs(treat.W_eff - w_expected)))

            O_c, O_t = ctrl.O[t], treat.O[t]
            flip_person = (O_t >= theta_metric) != (O_c >= theta_metric)
            dO_person = O_t - O_c
            R_t = treat.R[t]

            r = rows[W]
            r["E_rate"].append(m["E_rate"])
            r["flip"].append(m["flip_rate"])
            r["dO"].append(m["dO_mean"])
            r["mean_weff"].append(float(np.mean(treat.W_eff)))
            r["frac_zero"].append(float(np.mean(treat.W_eff == 0.0)))
            r["anal_err"].append(anal_err)
            r["mean_R"].append(float(np.mean(R_t)))
            r["frac_R_at_max"].append(float(np.mean(R_t >= 1.0 - 1e-12)))

            raw = reconstruct_r_raw(treat, attrs)
            r["clip_upper"].append(int(np.count_nonzero(raw > 1.0)))
            r["clip_lower"].append(int(np.count_nonzero(raw < 0.0)))
            r["n_clipped"].append(int(treat.n_clipped_R))

            tc = tercile[W]
            for key, mask in (("lo", lo), ("mid", mid), ("hi", hi)):
                tc[f"{key}_e"].append(float(np.mean(flip_person[mask])))
                tc[f"{key}_weff"].append(float(np.mean(treat.W_eff[mask])))
                tc[f"{key}_R"].append(float(np.mean(R_t[mask])))
                tc[f"{key}_zero"].append(float(np.mean(treat.W_eff[mask] == 0.0)))

            gp = by_persona[W]
            for name in persona_names:
                idx = persona_mask[name]
                count = int(np.sum(idx))
                if count == 0:
                    continue
                g = gp[name]
                g["flip"].append(float(np.mean(flip_person[idx])))
                g["zero"].append(float(np.mean(treat.W_eff[idx] == 0.0)))
                g["R"].append(float(np.mean(R_t[idx])))
                g["up"].append(int(np.count_nonzero(raw[:, idx] > 1.0)))
                g["lo"].append(int(np.count_nonzero(raw[:, idx] < 0.0)))
                g["n"].append(count)

    # --- 表1: W ごとの主指標 + 機構量 ---
    print()
    print("  W      E_rate      flip_rate   dO_mean     mean(W_eff)  W_eff=0の割合  "
          "mean(R)   R=1.0の割合  閉式誤差")
    print("  " + "-" * 100)
    best = None
    for W in W_grid:
        r = rows[W]
        e = float(np.mean(r["E_rate"]))
        marker = ""
        if best is None or abs(e) > abs(best[1]):
            best = (W, e)
        print(f"  {W:.2f}   {e:+.5f}   {float(np.mean(r['flip'])):.5f}    "
              f"{float(np.mean(r['dO'])):+.5f}    {float(np.mean(r['mean_weff'])):.4f}      "
              f"{float(np.mean(r['frac_zero'])):.4f}          {float(np.mean(r['mean_R'])):.4f}    "
              f"{float(np.mean(r['frac_R_at_max'])):.4f}        {max(r['anal_err']):.2e}")

    peak_W, peak_E = best
    e_values = {W: float(np.mean(rows[W]["E_rate"])) for W in W_grid}
    peak_actual = max(e_values, key=lambda w: abs(e_values[w]))
    print()
    print(f"  → |E_rate| 最大の W = {peak_actual}（{e_values[peak_actual]:+.8f}）")
    print("  → 全 W の E_rate 全文精度:")
    for W in W_grid:
        bar = "#" * int(abs(e_values[W]) * 2000)
        print(f"     W={W:.2f}  {e_values[W]:+.8f}  {bar}")
    # 単調減少局面の開始点（ピーク後の最初の減少）
    dec_start = None
    for a, b in zip(W_grid, W_grid[1:]):
        if b > peak_actual and abs(e_values[b]) < abs(e_values[a]) - 1e-12:
            dec_start = b
            break
    # ピーク→W=1.0 の減衰率
    tail = abs(e_values[max(W_grid)])
    print()
    print(f"  → 減少局面の開始 W = {dec_start}")
    print(f"  → ピーク |E_rate|={abs(e_values[peak_actual]):.5f} → W=1.0 |E_rate|={tail:.5f} "
          f"（減衰 {(1 - tail / abs(e_values[peak_actual])) * 100:.1f}%）")
    print(f"  → 全域で単調増加か: "
          f"{all(abs(e_values[a]) <= abs(e_values[b]) + 1e-12 for a, b in zip(W_grid, W_grid[1:]))}")
    print(f"  → 閉式 W_eff=min(W,(1+k)θ−kW) との最大乖離（全W・全シード）: "
          f"{max(max(rows[W]['anal_err']) for W in W_grid):.2e}")

    # --- Q2b: W_eff==0 ⟺ W >= (1+k)*theta の厳密検証 ---
    print()
    print("  [Q2b] 潰れの閾値の厳密検証: W_eff == 0  ⟺  W >= (1+k)*theta_react")
    viol_pos = viol_neg = 0
    for seed in seeds[:10]:
        world = build_world(config, seed)
        attrs = apply_ablation(world.params, Ablation.NONE, k_react)
        theta = attrs.theta_react
        for W in W_grid:
            weff, _ = governance.calculate_w_eff(
                W=W, theta_react=theta, k_react=k_react)
            is_zero = weff == 0.0
            predicted = W >= (1.0 + k_react) * theta
            if np.any(is_zero & ~predicted):
                viol_pos += 1
            if np.any(predicted & ~is_zero):
                viol_neg += 1
    print(f"     逆向きの例外（0 でないのに予測潰れ）: {viol_pos} 件")
    print(f"     逆向きの例外（0 なのに予測非潰れ）: {viol_neg} 件")
    print(f"     → 判定: {'完全一致' if viol_pos == 0 and viol_neg == 0 else '不一致あり'}")

    # --- 表2: θ_react 三分割層別の効果 ---
    print()
    print("=" * 96)
    print("Q3. θ_react 三分割別の効果と実効知覚（W_eff の潰れ方が層で違うか）")
    print("=" * 96)
    print(f"  ※層はシード内で θ_react の三分位で分割。lo=低θ（反発しやすい）/ hi=高θ（素直）")
    print()
    print("  W      低θ層                 中θ層                 高θ層")
    print("         転倒数  W_eff   R      転倒数  W_eff   R      転倒数  W_eff   R")
    print("  " + "-" * 84)
    for W in W_grid:
        tc = tercile[W]
        cells = []
        for key in ("lo", "mid", "hi"):
            cells.append(f"{float(np.mean(tc[f'{key}_e'])):.4f} {float(np.mean(tc[f'{key}_weff'])):.4f} "
                         f"{float(np.mean(tc[f'{key}_R'])):.4f}")
        print(f"  {W:.2f}   {cells[0]}   {cells[1]}   {cells[2]}")

    # --- 表3: 潰れ方が主要原因か ---
    print()
    print("=" * 96)
    print("Q4. 「W_eff 潰れ → R が上がらない → 効果が消える」の連鎖を W ごとに出力")
    print("=" * 96)
    print("  W      潰れ率(全体)  低θ層 潰れ率  高θ層 潰れ率  低θ層 mean(R)  全層 mean(R)  "
          "|E_rate|")
    print("  " + "-" * 84)
    for W in W_grid:
        r = rows[W]
        tc = tercile[W]
        print(f"  {W:.2f}   {float(np.mean(r['frac_zero'])):.4f}        "
              f"{float(np.mean(tc['lo_zero'])):.4f}       {float(np.mean(tc['hi_zero'])):.4f}      "
              f"{float(np.mean(tc['lo_R'])):.4f}        {float(np.mean(r['mean_R'])):.4f}     "
              f"{abs(e_values[W]):.5f}")

    print()
    print("=" * 96)
    print("解釈のための要点")
    print("=" * 96)
    zero_growth = [(W, float(np.mean(rows[W]["frac_zero"]))) for W in W_grid]
    print("  W_eff==0 の割合は W に対して単調増加するか: "
          f"{all(b[1] >= a[1] - 1e-12 for a, b in zip(zero_growth, zero_growth[1:]))}")
    print("  → " + ", ".join(f"W={W:.2f}:{v:.3f}" for W, v in zero_growth))
    lo_weff = [(W, float(np.mean(tercile[W]["lo_weff"]))) for W in W_grid]
    hi_weff = [(W, float(np.mean(tercile[W]["hi_weff"]))) for W in W_grid]
    print()
    print("  低θ層の mean(W_eff): " + ", ".join(f"W={W:.2f}:{v:.3f}" for W, v in lo_weff))
    print("  高θ層の mean(W_eff): " + ", ".join(f"W={W:.2f}:{v:.3f}" for W, v in hi_weff))
    print()
    print(f"  転倒数(全体) の |E_rate| ピーク W = {peak_actual}")
    peak_flip = max(W_grid, key=lambda w: float(np.mean(rows[w]["flip"])))
    print(f"  転倒数(個人) のピーク W       = {peak_flip}")
    print(f"  E_rate と flip_rate のピークは一致するか: {peak_actual == peak_flip}")
    print("  低θ層の平均転倒数: " + ", ".join(
        f"W={W:.2f}:{float(np.mean(tercile[W]['lo_e'])):.4f}" for W in W_grid))
    print("  高θ層の平均転倒数: " + ", ".join(
        f"W={W:.2f}:{float(np.mean(tercile[W]['hi_e'])):.4f}" for W in W_grid))

    # --- Q5: 天井効果の再現検証（C2 は W=0 のみを測っている） ---
    print()
    print("=" * 96)
    print("Q5. C2（天井効果なし, 閾値5%）は介入ありでも成立するか")
    print("=" * 96)
    print("  C2 の判定式は clip率 = n_clipped_R / total_cells < 0.05。")
    print("  total_cells は段階0実装では (T_max+1)*n だが、n_clipped_R は")
    print("  ターン 1..T_max しか数えないため正しい分母は T_max*n。以下は")
    print(f"  T_max*n = {total_cells} を分母に使い、上側クリップ（天井）と")
    print("  下側クリップ（床）を分離して数える。T* の一断面も併記する。")
    print()
    print("  自己検査: 再構成したクリップ総数が ArmRecord.n_clipped_R と一致するか")
    mismatch = 0
    for W in W_grid:
        r = rows[W]
        mismatch += sum(
            u + l - c for u, l, c in zip(r["clip_upper"], r["clip_lower"], r["n_clipped"])
        )
    print(f"     不一致セル数 = {mismatch}  → {'再構成は正確' if mismatch == 0 else '再構成にbug'}")
    print()

    print("  W      天井率(上側)  床率(下側)  C2のclip率  n_clipped_R  R=1.0率(T*)  5%判定")
    print("  " + "-" * 84)
    for W in W_grid:
        r = rows[W]
        up = float(np.mean(r["clip_upper"])) / total_cells
        lo = float(np.mean(r["clip_lower"])) / total_cells
        clip_rate = float(np.mean(r["n_clipped"])) / total_cells
        at1 = float(np.mean(r["frac_R_at_max"]))
        verdict = "OK" if clip_rate < 0.05 else "** 超過 **"
        print(f"  {W:.2f}   {up:.5f}       {lo:.5f}     {clip_rate:.5f}     "
              f"{float(np.mean(r['n_clipped'])):8.1f}      {at1:.5f}     {verdict}")

    print()
    worst = max(W_grid, key=lambda w: float(np.mean(rows[w]["n_clipped"])) / total_cells)
    worst_v = float(np.mean(rows[worst]["n_clipped"])) / total_cells
    worst_up = float(np.mean(rows[worst]["clip_upper"])) / total_cells
    exceed = [w for w in W_grid if float(np.mean(rows[w]["n_clipped"])) / total_cells >= 0.05]
    print(f"  → C2 の clip率 最大: W={worst:.2f} で {worst_v:.5f}（{worst_v * 100:.2f}%）"
          f"、うち天井（上側）が {worst_up:.5f}")
    print(f"  → C2 の 5% 閾値を超過する W: "
          f"{', '.join(f'{w:.2f}' for w in exceed) if exceed else 'なし'}")
    print(f"  → 天井が最初に現れる W: "
          f"{next((f'{w:.2f}' for w in W_grid if float(np.mean(rows[w]['clip_upper'])) > 0), 'なし')}")
    print(f"  → したがって C2 の『天井効果なし』は W=0 でのみ確認済みであり、")
    print(f"     介入ありでは最大 {worst_v * 100:.2f}% まで頭打ちが生じる。")
    # --- Q7: ペルソナ型別の用量反応（Phase A） ---
    print()
    print("=" * 96)
    print("Q7. clip が原理的に起きないペルソナ型だけで山型が残るか")
    print("=" * 96)
    print("  設定は persona_A のまま変えない。型の並びは persona 表の記載順。")
    print("  clip 可否は R_raw = w_sys*W_eff + w_sns*S_R + b に対して")
    print("  W_eff <= 1 かつ S_R <= 1 の仮定のもと、w_sys + w_sns + b <= 1 で証明する。")
    print()

    clip_impossible = [
        name for name in persona_names
        if len(raw_bound[name]) > 0 and max(raw_bound[name]) <= 1.0
    ]
    clip_possible = [name for name in persona_names if name not in clip_impossible]
    print(f"  clip 不可能群: {', '.join(clip_impossible) if clip_impossible else 'なし'}")
    print(f"  clip 可能群: {', '.join(clip_possible) if clip_possible else 'なし'}")
    print()
    print("  型       平均人数  R_raw上限  証明     観測clip率最大  "
          "転数ピーク(W)   転数(W=1.0)  減衰    W_eff=0率(W=1.0)")
    print("  " + "-" * 100)
    summary: dict[str, dict[str, float]] = {}
    for name in persona_names:
        flips: list[float] = []
        peak_w = float("nan")
        best = -1.0
        for w in W_grid:
            values = by_persona[w][name]["flip"]
            if not values:
                continue
            mean_flip = float(np.mean(values))
            flips.append(mean_flip)
            if mean_flip > best:
                best = mean_flip
                peak_w = w
        peak = max(flips) if flips else 0.0
        tail_values = by_persona[max(W_grid)][name]["flip"]
        tail = float(np.mean(tail_values)) if tail_values else 0.0
        zero_values = by_persona[max(W_grid)][name]["zero"]
        zero_tail = float(np.mean(zero_values)) if zero_values else 0.0

        clip_max = 0.0
        for w in W_grid:
            g = by_persona[w][name]
            if g["up"]:
                clip_max = max(
                    clip_max, float(sum(g["up"])) / (float(sum(g["n"])) * T_max))
        mean_n = float(np.mean(by_persona[W_grid[0]][name]["n"])) if by_persona[
            W_grid[0]][name]["n"] else 0.0
        bound = max(raw_bound[name]) if raw_bound[name] else float("nan")
        proof = "不可能" if name in clip_impossible else "可能"
        decay = (peak - tail) / peak if peak > 0 else 0.0
        summary[name] = {"peak": peak, "tail": tail}
        print(f"  {name:10s} {mean_n:8.1f}  {bound:8.3f}  {proof:6s}  "
              f"{clip_max:10.5f}  {peak:8.5f}({peak_w:4.2f})  {tail:10.5f}  "
              f"{decay * 100:5.1f}%  {zero_tail:14.4f}")

    def group_series(names: list[str], key: str) -> dict[float, float]:
        out: dict[float, float] = {}
        for w in W_grid:
            num = 0.0
            den = 0.0
            for name in names:
                g = by_persona[w][name]
                for value, count in zip(g[key], g["n"]):
                    if key in ("up", "lo"):
                        num += float(value)
                        den += float(count) * T_max
                    else:
                        num += float(value) * float(count)
                        den += float(count)
            out[w] = (num / den) if den > 0 else 0.0
        return out

    imp_flip = group_series(clip_impossible, "flip")
    imp_zero = group_series(clip_impossible, "zero")
    imp_clip = group_series(clip_impossible, "up")
    pos_flip = group_series(clip_possible, "flip")
    pos_zero = group_series(clip_possible, "zero")
    pos_clip = group_series(clip_possible, "up")

    print()
    print("  W      不可能群: 転数  W_eff=0率  clip率      可能群: 転数  W_eff=0率  clip率")
    print("  " + "-" * 92)
    for w in W_grid:
        print(f"  {w:.2f}   {imp_flip[w]:10.5f}  {imp_zero[w]:9.4f}  {imp_clip[w]:7.5f}   "
              f"{pos_flip[w]:10.5f}  {pos_zero[w]:9.4f}  {pos_clip[w]:7.5f}")

    imp_peak_w = max(W_grid, key=lambda w: imp_flip[w])
    imp_peak = imp_flip[imp_peak_w]
    imp_tail = imp_flip[max(W_grid)]
    imp_decline = imp_peak - imp_tail
    impossible_clip_total = sum(
        sum(by_persona[w][name]["up"]) for w in W_grid for name in clip_impossible)
    print()
    print(f"  → 不可能群の転数ピーク: W={imp_peak_w:.2f} で {imp_peak:.5f}、"
          f"W=1.0 で {imp_tail:.5f}（減少 {imp_decline:.5f}）")
    print(f"  → 不可能群の W_eff=0率(W=1.0): {imp_zero[max(W_grid)]:.4f}")
    print(f"  → 不可能群の観測clip総数: {impossible_clip_total} セル")
    if impossible_clip_total == 0 and imp_decline > 1e-12 and imp_zero[max(W_grid)] > 0:
        print("  → 判定: 山が残った。天井は原因になれない。")
        print("     同時に W_eff=0 も確認できたため、リアクタンスの説明が残る。")
        print("     ただし不可能群は R の可動域自体が狭い点は留保する。")
    elif imp_peak <= 1e-12 or imp_decline <= 1e-12:
        print("  → 判定: 判別不能。不可能群の転数が小さすぎるか単調である。")
        print("     これは『意見変容そのものが起きにくい型』という観測結果であり、")
        print("     Phase B（診断専用の clip バイパス）へ進む条件になる。")
    else:
        print("  → 判定: 不可群に clip が混入しているか集計に問題がある。要調査。")



if __name__ == "__main__":
    main()
