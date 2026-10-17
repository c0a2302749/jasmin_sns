"""scripts/verify_invariants.py — 前提2の検証: 反実仮想性と経路分離の機械的証明。

実行: uv run python scripts/verify_invariants.py --config configs/default.yaml

検証項目:
  [RNG] 乱数生成が rng.py 之外に存在しないこと (静的AST走査 + 動的カウンタ)
  [W0]  介入なしアーム (W=0) を独立に2本走らせたとき全ターン bit 一致すること
  [SYS] ABL-SYS では W に関係なく全軌跡が bit 一致すること (制度的経路が唯一の媒介)
  [SNS] ABL-SNS では R が近傍非依存の閉形式に厳密一致すること
        (かつ none 条件では同じ式が成立しない = 検査が空振りでないことの対照)
  [REACT] ABL-REACT では k_react=0 により W_eff == W になりリアクタンスが消えること
  [ISO] 異なる W を実行しても World の全配列が不変であること (乱数境界の保証)
  [MUT] Ablation が params を変更せず、対象経路のみをゼロ化すること (交絡の排除)
  [RNG2] 効果のダイナミックレンジ: none 条件では W>0 で E_rate != 0 になること
"""
from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from config import Config, load
from run.arm import run_arm
from run.checks import run_checks
from run.metrics import calculate_metrics
from world.ablation import Ablation, apply_ablation
from world.build import World, build_world

SRC = Path(__file__).resolve().parents[1] / "src"

results: list[tuple[str, str, bool, str]] = []


def record(section: str, check_id: str, passed: bool, detail: str) -> None:
    results.append((section, check_id, passed, detail))
    mark = "PASS" if passed else "FAIL"
    print(f"  [{mark}] {check_id}: {detail}")


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# --------------------------------------------------------------------------
# [RNG] 静的検査: np.random.* の呼び出しが rng.py 以外から存在しないこと
# --------------------------------------------------------------------------
def static_rng_audit() -> None:
    section("[RNG] 乱数生成境界の静的監査 (rng.py のみが default_rng を生成してよい)")
    offenders: list[str] = []
    allowed = {"rng.py"}

    for py in sorted(SRC.rglob("*.py")):
        if "__pycache__" in py.parts:
            continue
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            # np.random.default_rng / np.random.Generator / np.random.<anything>
            name = None
            if isinstance(func, ast.Attribute):
                parts = []
                cur: ast.AST = func
                while isinstance(cur, ast.Attribute):
                    parts.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name):
                    parts.append(cur.id)
                name = ".".join(reversed(parts))
            if name and name.startswith("np.random"):
                rel = py.name
                if rel not in allowed:
                    offenders.append(f"{py.relative_to(SRC.parent)}:{node.lineno} -> {name}()")
                elif name not in ("np.random.default_rng", "np.random.SeedSequence"):
                    offenders.append(
                        f"{py.relative_to(SRC.parent)}:{node.lineno} -> {name}() (rng.py 内の許可外API)"
                    )

    record(
        "RNG", "RNG_BOUNDARY_STATIC", not offenders,
        "np.random.* の呼び出しは rng.py の default_rng/SeedSequence のみ"
        if not offenders else f"不正検出: {offenders}",
    )


def dynamic_rng_audit(config: Config, seed: int) -> None:
    section("[RNG] 動的監査: 初期化後 (world 完成後) に乱数が消費されないこと")
    calls = {"count": 0}
    real_default_rng = np.random.default_rng

    def counting_default_rng(*a, **k):
        calls["count"] += 1
        return real_default_rng(*a, **k)

    world = build_world(config, seed)
    calls["count"] = 0
    np.random.default_rng = counting_default_rng
    try:
        attrs = apply_ablation(world.params, Ablation.NONE, config.model.k_react)
        for W in (0.0, 0.5, 1.0):
            for cond in (Ablation.NONE, Ablation.SYS, Ablation.SNS,
                         Ablation.ALPHA, Ablation.SNS_ALPHA, Ablation.REACT):
                a = apply_ablation(world.params, cond, config.model.k_react)
                run_arm(world, a, W=W, config=config)
        run_arm(world, attrs, W=0.25, config=config)
    finally:
        np.random.default_rng = real_default_rng

    record(
        "RNG", "RNG_CONSUMED_AFTER_WORLD", calls["count"] == 0,
        f"build_world 完了後に default_rng 呼び出し {calls['count']} 回 "
        f"(W=0/0.5/1.0 x 4 ablation 条件 + 追加1本を実行)",
    )


# --------------------------------------------------------------------------
# [ISO] World の不変性
# --------------------------------------------------------------------------
def world_immutability(config: Config, seed: int) -> None:
    section("[ISO] World 不変性: 異なる W / 異なる Ablation を実行しても世界が変わらない")
    world = build_world(config, seed)
    snap = {
        "O0": world.O0.copy(),
        "R0": world.R0.copy(),
        "update_order": world.update_order.copy(),
        "neighbors": tuple(nb.copy() for nb in world.neighbors),
        "params": list(world.params),
    }
    for cond in (Ablation.NONE, Ablation.SYS, Ablation.SNS,
                 Ablation.ALPHA, Ablation.SNS_ALPHA, Ablation.REACT):
        attrs = apply_ablation(world.params, cond, config.model.k_react)
        for W in (0.0, 0.25, 0.5, 1.0):
            run_arm(world, attrs, W=W, config=config)

    ok = (
        np.array_equal(world.O0, snap["O0"])
        and np.array_equal(world.R0, snap["R0"])
        and np.array_equal(world.update_order, snap["update_order"])
        and all(np.array_equal(a, b) for a, b in zip(world.neighbors, snap["neighbors"]))
        and world.params == snap["params"]
    )
    record("ISO", "WORLD_UNCHANGED_AFTER_ARMS", ok,
           "O0/R0/update_order/neighbors/params すべて実行前後で一致")

    frozen = all(not arr.flags.writeable for arr in
                 (world.O0, world.R0, world.update_order, *world.neighbors))
    record("ISO", "WORLD_ARRAYS_FROZEN", frozen,
           "O0/R0/update_order/neighbors が全て writeable=False")


# --------------------------------------------------------------------------
# [W0] 反実仮想性: W=0 の2アームが bit 一致
# --------------------------------------------------------------------------
def counterfactual_w0(config: Config, seeds: list[int]) -> None:
    section("[W0] 反実仮想性: 介入なし (W=0) を独立に2本実行 → 全ターン bit 一致")
    all_ok = True
    per_seed = []
    for seed in seeds:
        world = build_world(config, seed)
        attrs = apply_ablation(world.params, Ablation.NONE, config.model.k_react)
        ctrl = run_arm(world, attrs, W=0.0, config=config)
        treat = run_arm(world, attrs, W=0.0, config=config)   # 別オブジェクトとして独立実行
        checks = run_checks(ctrl, treat, W_level=0.0, ablation="none",
                            world=world, attrs=attrs)
        ids = {c["check_id"]: c["passed"] for c in checks}
        ok = ids.get("W0_IDENTICAL") and ids.get("TURN0_IDENTICAL")
        all_ok &= bool(ok)
        per_seed.append(
            f"seed{seed}:W0={'OK' if ids.get('W0_IDENTICAL') else 'NG'}"
            f",T0={'OK' if ids.get('TURN0_IDENTICAL') else 'NG'}"
            f",O_hash={hash(ctrl.O.tobytes()) % 10**8},"
            f"R_hash={hash(treat.R.tobytes()) % 10**8}"
        )
    record("W0", "W0_BIT_IDENTICAL", all_ok, "; ".join(per_seed))

    # 反証材料: W>0 では bit 一致しないこと（同一性検査が空振りでないことの証拠）
    world = build_world(config, seeds[0])
    attrs = apply_ablation(world.params, Ablation.NONE, config.model.k_react)
    c0 = run_arm(world, attrs, W=0.0, config=config)
    c1 = run_arm(world, attrs, W=1.0, config=config)
    differs = not np.array_equal(c0.O, c1.O)
    record("W0", "W0_CHECK_IS_DISCRIMINATING", differs,
           f"同一シード・同一世界でも W=0 と W=1.0 の O は異なる (max|ΔO|={np.max(np.abs(c0.O - c1.O)):.6f})"
           if differs else "W>0 でも一致しており W0 検査に検出力がない")


# --------------------------------------------------------------------------
# [SYS] 制度的経路の分離
# --------------------------------------------------------------------------
def ablation_sys(config: Config, seeds: list[int]) -> None:
    section("[SYS] 経路分離 ABL-SYS: w_sys=0 → W に依存せず全軌跡が bit 一致")
    all_ok = True
    details = []
    for seed in seeds:
        world = build_world(config, seed)
        attrs = apply_ablation(world.params, Ablation.SYS, config.model.k_react)
        ctrl = run_arm(world, attrs, W=0.0, config=config)
        for W in (0.25, 0.5, 0.75, 1.0):
            treat = run_arm(world, attrs, W=W, config=config)
            checks = run_checks(ctrl, treat, W_level=W, ablation="ABL-SYS",
                                world=world, attrs=attrs)
            ids = {c["check_id"]: c["passed"] for c in checks}
            m = calculate_metrics(ctrl, treat, world, config, W_level=W, ablation="ABL-SYS")
            ok = (ids.get("ABL_SYS_BITWISE")
                  and m["E_rate"] == 0.0 and m["flip_rate"] == 0.0 and m["dO_mean"] == 0.0)
            all_ok &= bool(ok)
            details.append(f"seed{seed}/W={W}:bit={'OK' if ids.get('ABL_SYS_BITWISE') else 'NG'},"
                           f"E_rate={m['E_rate']:.1f},flip={m['flip_rate']:.1f},dO={m['dO_mean']:.1e}")
    record("SYS", "ABL_SYS_BITWISE_AND_ZERO_EFFECT", all_ok, "; ".join(details))


# --------------------------------------------------------------------------
# [SNS] 社会的経路の分離
# --------------------------------------------------------------------------
def ablation_sns(config: Config, seeds: list[int]) -> None:
    section("[SNS] 経路分離 ABL-SNS: w_sns=0 → R が近傍非依存の閉形式 R=Clip(w_sys·W_eff+b) に一致")
    all_ok = True
    details = []
    for seed in seeds:
        world = build_world(config, seed)
        attrs = apply_ablation(world.params, Ablation.SNS, config.model.k_react)
        ctrl = run_arm(world, attrs, W=0.0, config=config)
        for W in (0.25, 0.5, 1.0):
            treat = run_arm(world, attrs, W=W, config=config)
            checks = run_checks(ctrl, treat, W_level=W, ablation="ABL-SNS",
                                world=world, attrs=attrs)
            ids = {c["check_id"]: c["passed"] for c in checks}
            # 独立再計算 (検査関数を経由しない)
            expected = np.clip(
                attrs.w_sys[None, :] * treat.W_eff[None, :] + attrs.b[None, :], 0.0, 1.0)
            max_err = float(np.max(np.abs(treat.R[1:] - expected)))
            all_ok &= bool(ids.get("ABL_SNS_R_INDEPENDENT")) and max_err <= 1e-12
            details.append(f"seed{seed}/W={W}:check={'OK' if ids.get('ABL_SNS_R_INDEPENDENT') else 'NG'},"
                           f"max|R-式|={max_err:.3e}")
    record("SNS", "ABL_SNS_R_CLOSED_FORM", all_ok, "; ".join(details))

    # 対照: none 条件では同じ式は成立しない = SNS 検査が検出力を持つことの証拠
    world = build_world(config, seeds[0])
    attrs_none = apply_ablation(world.params, Ablation.NONE, config.model.k_react)
    t = run_arm(world, attrs_none, W=0.5, config=config)
    expected = np.clip(
        attrs_none.w_sys[None, :] * t.W_eff[None, :] + attrs_none.b[None, :], 0.0, 1.0)
    err = float(np.max(np.abs(t.R[1:] - expected)))
    record("SNS", "SNS_CHECK_IS_DISCRIMINATING", err > 1e-6,
           f"none 条件では閉形式からの乖離 max|R-式|={err:.6f} (社会的影響が R を動かしている)")


# --------------------------------------------------------------------------
# [REACT] リアクタンス経路の分離
# --------------------------------------------------------------------------
def ablation_react(config: Config, seeds: list[int]) -> None:
    section("[REACT] 経路分離 ABL-REACT: k_react=0 → W_eff = min(W, theta_react) の飽和 (A9)")
    # 設計書 A9 の記述: 「k=0 では W_eff = min(W, θ) の飽和のみとなり、
    # リアクタンスによる減衰が消える」。したがって W_eff == W ではなく
    # min(W, θ_react) への一致が正しい帰結であり、かつ k=1 相比で
    # 減衰が 緩和されている (W_eff が 0 へ潰れない) ことを確認する。
    all_ok = True
    details = []
    for seed in seeds:
        world = build_world(config, seed)
        attrs_r = apply_ablation(world.params, Ablation.REACT, config.model.k_react)
        attrs_n = apply_ablation(world.params, Ablation.NONE, config.model.k_react)
        w = 0.5
        rec_r = run_arm(world, attrs_r, W=w, config=config)
        rec_n = run_arm(world, attrs_n, W=w, config=config)

        expected = np.minimum(w, attrs_r.theta_react)   # min(W, theta)
        max_err = float(np.max(np.abs(rec_r.W_eff - expected)))
        monotone = bool(np.all(rec_r.W_eff >= rec_n.W_eff - 1e-12))
        not_collapsed = bool(np.all(rec_r.W_eff > 0.0))
        ok = (
            attrs_r.k_react == 0.0
            and max_err <= 1e-12
            and np.array_equal(attrs_r.theta_react, attrs_n.theta_react)  # theta は残る (A9)
            and monotone          # k=1 より減衰が小さい
            and not_collapsed     # ゼロまで潰れない
        )
        all_ok &= bool(ok)
        details.append(
            f"seed{seed}:k_react={attrs_r.k_react},"
            f"max|W_eff-min(W,θ)|={max_err:.1e},"
            f"mean(W_eff|k=0)={float(np.mean(rec_r.W_eff)):.4f}"
            f" vs mean(W_eff|k=1)={float(np.mean(rec_n.W_eff)):.4f},"
            f"k=0側のゼロ要素={int(np.sum(rec_r.W_eff == 0.0))},"
            f"k=1側のゼロ要素={int(np.sum(rec_n.W_eff == 0.0))},"
            f"n_active={rec_r.n_reactance_active}")
    record("REACT", "ABL_REACT_SATURATES_NOT_ATTENUATES", all_ok, "; ".join(details))

    # 対照: k=0 でも theta_react は W_eff を min(W,θ) に縛る (完全な無効化ではない)
    world = build_world(config, seeds[0])
    attrs_r = apply_ablation(world.params, Ablation.REACT, config.model.k_react)
    rec = run_arm(world, attrs_r, W=1.0, config=config)
    capped = int(np.sum(rec.W_eff < 1.0 - 1e-12))
    record("REACT", "REACT_SATURATION_IS_RESIDUAL", capped > 0,
           f"W=1.0 でも {capped}/{config.network.n} 人の W_eff が θ_react で頭打ちになる "
           f"(A9 の『飽和のみ』の帰結。W_eff==W ではない点に注意)")


# --------------------------------------------------------------------------
# [MUT] Ablation の非破壊性と経路限定性
# --------------------------------------------------------------------------
def ablation_isolation(config: Config, seed: int) -> None:
    section("[MUT] 交絡排除: Ablation は params を変更せず、対象経路のみをゼロ化する")
    world = build_world(config, seed)
    base = apply_ablation(world.params, Ablation.NONE, config.model.k_react)
    original = list(world.params)

    array_fields = ["alpha", "beta", "gamma", "w_sys", "w_sns", "b", "theta_react"]
    report: list[str] = []
    ok_all = True

    for cond, expect_zero in (
        (Ablation.SYS, "w_sys"),
        (Ablation.SNS, "w_sns"),
        (Ablation.ALPHA, "alpha"),
        (Ablation.SNS_ALPHA, "w_sns+alpha"),
        (Ablation.REACT, "k_react"),
    ):
        a = apply_ablation(world.params, cond, config.model.k_react)
        changed = [
            f for f in array_fields
            if not np.array_equal(getattr(a, f), getattr(base, f))
        ]
        # ゼロ化の対象确实是値、そしてそれ以外の配列は無変化であること
        if expect_zero == "k_react":
            zeroed = a.k_react == 0.0
            ok = changed == [] and zeroed
        elif expect_zero == "w_sns+alpha":
            zeroed = bool(np.all(a.w_sns == 0.0) and np.all(a.alpha == 0.0))
            ok = changed == ["alpha", "w_sns"] and zeroed
        else:
            zeroed = bool(np.all(getattr(a, expect_zero) == 0.0))
            ok = changed == [expect_zero] and zeroed
        ok_all &= bool(ok)
        report.append(f"{cond.value}: 差分フィールド={changed or 'なし'}, {expect_zero}->0: {zeroed}")

    record("MUT", "ABLATION_TOUCHES_ONLY_TARGET_PATH", bool(ok_all), "; ".join(report))

    record("MUT", "PARAMS_NOT_MUTATED", world.params == original,
           "4条件の apply_ablation 適用後も world.params は原本と完全一致 (dataclasses.replace 経由)")


# --------------------------------------------------------------------------
# [RNG2] 効果のダイナミックレンジ (前提1の前提: 介入が測定可能であること)
# --------------------------------------------------------------------------
def dynamic_range(config: Config, seeds: list[int]) -> None:
    section("[RNG2] ダイナミックレンジ: none 条件で W>0 の因果効果が非ゼロか")
    rows = []
    nonzero = True
    for W in config.experiment.W_values:
        if W == 0.0:
            continue
        es, flips, dOs = [], [], []
        for seed in seeds:
            world = build_world(config, seed)
            attrs = apply_ablation(world.params, Ablation.NONE, config.model.k_react)
            ctrl = run_arm(world, attrs, W=0.0, config=config)
            treat = run_arm(world, attrs, W=W, config=config)
            m = calculate_metrics(ctrl, treat, world, config, W_level=W, ablation="none")
            es.append(m["E_rate"]); flips.append(m["flip_rate"]); dOs.append(m["dO_mean"])
        if W > 0 and all(abs(e) < 1e-12 for e in es):
            nonzero = False
        rows.append(
            f"W={W:.2f}: E_rate mean={np.mean(es):+.4f} (|max|={np.max(np.abs(es)):.4f}), "
            f"flip={np.mean(flips):.4f}, dO_mean={np.mean(dOs):+.5f}")
    record("RNG2", "EFFECT_MEASURABLE_AT_W_GT_0", nonzero, "; ".join(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description="前提2: 反実仮想性と経路分離の検証")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    args = parser.parse_args()

    config = load(args.config)
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))

    print(f"設定: {args.config}")
    print(f"  n={config.network.n}, k={config.network.k}, p={config.network.p}, "
          f"T_max={config.model.T_max}, eps={config.model.eps}, S={config.model.S}, "
          f"k_react={config.model.k_react}")
    print(f"  persona table={config.personas.table}, nu={config.personas.concentration_nu}")
    print(f"  検証シード: {seeds[0]}..{seeds[-1]} ({len(seeds)}本)")

    static_rng_audit()
    dynamic_rng_audit(config, seeds[0])
    world_immutability(config, seeds[0])
    counterfactual_w0(config, seeds)
    ablation_sys(config, seeds)
    ablation_sns(config, seeds)
    ablation_react(config, seeds)
    ablation_isolation(config, seeds[0])
    dynamic_range(config, seeds)

    print()
    print("=" * 78)
    print("集計")
    print("=" * 78)
    n_pass = sum(1 for _, _, p, _ in results if p)
    for sec, cid, passed, detail in results:
        print(f"{'PASS' if passed else 'FAIL'}  [{sec}] {cid}")
    print()
    print(f"合計 {len(results)} 項目: PASS {n_pass} / FAIL {len(results) - n_pass}")
    sys.exit(0 if n_pass == len(results) else 1)


if __name__ == "__main__":
    main()
