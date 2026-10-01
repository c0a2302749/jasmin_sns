"""tests/test_config.py — F-01: 設定の読込・検査・実行ID。

設計決定参照:
- F-01: 設定を読み、検査し、実行IDを作る
- NFR-5: 設定の外部化（値の正典はYAML、コード内にデフォルトなし）
- NFR-7: persona表パスはconfigファイルの親ディレクトリ基準で解決
- A3: S > 2 は bistable 警告 / A9: k_react 感度範囲 [0, 2] / A2: update_mode
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from config import Config, ConfigError, load, run_id


def _valid_cfg_data() -> dict[str, Any]:
    """検査を素通りする最小構成の設定辞書を返す。"""
    return {
        "model": {"S": 2.0, "k_react": 1.0, "T_max": 50, "T_min": 5, "eps": 1e-4},
        "network": {"n": 20, "k": 2, "p": 0.1, "max_tries": 100},
        "init_opinion": {"mean": 0.5, "std": 0.1},
        "personas": {"table": "test_persona", "concentration_nu": 10.0},
        "experiment": {
            "master_seed_start": 0,
            "n_seeds": 5,
            "W_values": [0.0, 0.5],
            "ablation_conditions": ["none", "ABL-SYS"],
        },
        "stopping": {"update_mode": "gauss_seidel"},
        "metrics": {"theta": 0.5, "bootstrap_B": 100, "bootstrap_seed": 42},
        "logging": {"log_every": 1},
    }


def _valid_persona_data() -> dict[str, Any]:
    """share合計が1のペルソナ表を返す。"""
    return {
        "personas": [
            {
                "name": "conformist", "share": 0.5, "alpha": 0.7, "beta": 0.4,
                "gamma": 0.1, "w_sns": 0.7, "w_sys": 0.4, "b": 0.15,
                "mu_react": 0.7,
            },
            {
                "name": "cautious", "share": 0.5, "alpha": 0.4, "beta": 0.7,
                "gamma": 0.4, "w_sns": 0.4, "w_sys": 0.7, "b": 0.30,
                "mu_react": 0.7,
            },
        ]
    }


def _write_pair(tmp_path: Path, cfg_data: dict[str, Any],
                persona_data: dict[str, Any] | None = None,
                name: str = "test_config") -> Path:
    """config + persona YAMLのペアを書き込み、configパスを返す。"""
    cfg_file = tmp_path / f"{name}.yaml"
    persona_file = tmp_path / f"{cfg_data['personas']['table']}.yaml"
    with cfg_file.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_data, f, sort_keys=False)
    with persona_file.open("w", encoding="utf-8") as f:
        yaml.safe_dump(persona_data or _valid_persona_data(), f, sort_keys=False)
    return cfg_file


@pytest.fixture
def cfg_file(tmp_path) -> Path:
    """検査を素通りする設定ファイルペアを作成する。"""
    return _write_pair(tmp_path, _valid_cfg_data())


def test_F01_load_valid_config(cfg_file):
    """正常系: 全フィールドが読み込まれ、警告なしでConfigが返る (F-01, NFR-5)。"""
    cfg = load(cfg_file)

    assert isinstance(cfg, Config)
    assert cfg.model.S == 2.0
    assert cfg.network.n == 20
    assert cfg.experiment.W_values == (0.0, 0.5)
    assert cfg.stopping.update_mode == "gauss_seidel"
    assert cfg.warnings == ()


@pytest.mark.parametrize("patch", [
    {"model": {"S": 0.0, "k_react": 1.0, "T_max": 50, "T_min": 5, "eps": 1e-4}},
    {"model": {"S": -1.5, "k_react": 1.0, "T_max": 50, "T_min": 5, "eps": 1e-4}},
])
def test_F01_rejects_non_positive_S(tmp_path, patch):
    """S <= 0 は ConfigError (F-01, A3)。"""
    data = _valid_cfg_data() | patch
    with pytest.raises(ConfigError):
        load(_write_pair(tmp_path, data))


@pytest.mark.parametrize("k", [3, 20, 22])
def test_F01_rejects_bad_k(tmp_path, k):
    """k が奇数または n 以上は ConfigError (F-01)。n=20 に対する k=3(奇数)/20/22 を検証。"""
    data = _valid_cfg_data()
    data["network"] = {"n": 20, "k": k, "p": 0.1, "max_tries": 100}
    with pytest.raises(ConfigError):
        load(_write_pair(tmp_path, data))


@pytest.mark.parametrize("p", [-0.1, 1.5])
def test_F01_rejects_p_out_of_range(tmp_path, p):
    """p ∉ [0, 1] は ConfigError (F-01)。"""
    data = _valid_cfg_data()
    data["network"] = {"n": 20, "k": 2, "p": p, "max_tries": 100}
    with pytest.raises(ConfigError):
        load(_write_pair(tmp_path, data))


@pytest.mark.parametrize("w", [-0.25, 1.25])
def test_F01_rejects_W_out_of_range(tmp_path, w):
    """W_values の要素 ∉ [0, 1] は ConfigError (F-01)。"""
    data = _valid_cfg_data()
    data["experiment"] = {
        "master_seed_start": 0, "n_seeds": 5, "W_values": [w],
        "ablation_conditions": ["none"],
    }
    with pytest.raises(ConfigError):
        load(_write_pair(tmp_path, data))


def test_F01_rejects_unknown_ablation(tmp_path):
    """未知のablation条件は ConfigError (F-01, FR-E5)。"""
    data = _valid_cfg_data()
    data["experiment"] = {
        "master_seed_start": 0, "n_seeds": 5, "W_values": [0.0],
        "ablation_conditions": ["ABL-UNKNOWN"],
    }
    with pytest.raises(ConfigError):
        load(_write_pair(tmp_path, data))


def test_F01_rejects_unknown_update_mode(tmp_path):
    """update_mode ∉ {gauss_seidel, jacobi} は ConfigError (F-01, A2)。"""
    data = _valid_cfg_data() | {"stopping": {"update_mode": "euler"}}
    with pytest.raises(ConfigError):
        load(_write_pair(tmp_path, data))


def test_F01_rejects_missing_config_file(tmp_path):
    """存在しない設定ファイルは ConfigError (F-01)。"""
    with pytest.raises(ConfigError):
        load(tmp_path / "no_such_config.yaml")


def test_F01_rejects_missing_persona_table(tmp_path):
    """persona表YAMLが存在しない場合は ConfigError (F-01, NFR-7)。"""
    data = _valid_cfg_data()
    cfg_file = tmp_path / "test_config.yaml"
    with cfg_file.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    # persona側をわざと書き込まない
    with pytest.raises(ConfigError):
        load(cfg_file)


def test_F01_warns_bistable_S(tmp_path):
    """S > 2 は止めずに bistable 警告を Config.warnings に格納 (F-01, A3)。"""
    data = _valid_cfg_data()
    data["model"] = {"S": 2.5, "k_react": 1.0, "T_max": 50, "T_min": 5, "eps": 1e-4}
    cfg = load(_write_pair(tmp_path, data))

    assert cfg.model.S == 2.5
    assert any("bistable" in w for w in cfg.warnings)


def test_F01_warns_k_react_out_of_sensitivity_range(tmp_path):
    """k_react ∉ [0, 2] は止めずに警告 (F-01, A9)。"""
    data = _valid_cfg_data()
    data["model"] = {"S": 2.0, "k_react": 3.0, "T_max": 50, "T_min": 5, "eps": 1e-4}
    cfg = load(_write_pair(tmp_path, data))

    assert any("k_react" in w for w in cfg.warnings)


def test_F01_warns_small_n_seeds(tmp_path):
    """n_seeds < 5 は止めずに警告 (F-01)。"""
    data = _valid_cfg_data()
    data["experiment"] = {
        "master_seed_start": 0, "n_seeds": 2, "W_values": [0.0],
        "ablation_conditions": ["none"],
    }
    cfg = load(_write_pair(tmp_path, data))

    assert any("n_seeds" in w for w in cfg.warnings)


def test_F01_persona_path_resolves_relative_to_config_parent(tmp_path, monkeypatch):
    """persona表パスはcwdではなくconfigファイルの親基準で解決される (NFR-7)。

    サブディレクトリにペアを置き、cwdを無関係な場所へ移してloadする。
    """
    sub = tmp_path / "exp1"
    sub.mkdir()
    cfg_file = _write_pair(sub, _valid_cfg_data())

    monkeypatch.chdir(tmp_path)  # configのある場所とは別のcwd
    cfg = load(cfg_file)

    assert cfg.personas.table_path == str((sub / "test_persona.yaml").resolve())
    assert Path(cfg.personas.table_path).exists()


def test_F01_run_id_format(cfg_file):
    """run_id は "run_YYYYMMDD_HHMMSS_XXXX" 形式 (F-01)。"""
    rid = run_id(load(cfg_file))

    assert re.fullmatch(r"run_\d{8}_\d{6}_[0-9a-f]{4}", rid) is not None


def test_F01_run_id_digest_reflects_config(tmp_path):
    """設定が変わればハッシュ部（末尾4桁）が変わる。"""
    cfg = load(_write_pair(tmp_path, _valid_cfg_data()))
    data = _valid_cfg_data()
    data["model"] = {"S": 1.5, "k_react": 1.0, "T_max": 50, "T_min": 5, "eps": 1e-4}
    cfg2 = load(_write_pair(tmp_path, data, name="other_config"))

    assert run_id(cfg)[-4:] != run_id(cfg2)[-4:]


def test_F01_run_id_digest_ignores_path_and_warnings(tmp_path):
    """同一内容を別パスから読んでもハッシュ部は変わらない。

    warnings / source_path / personas.table_path はハッシュ対象外。
    """
    from dataclasses import replace

    cfg = load(_write_pair(tmp_path, _valid_cfg_data(), name="config_a"))
    sub = tmp_path / "elsewhere"
    sub.mkdir()
    cfg_same = load(_write_pair(sub, _valid_cfg_data(), name="config_b"))

    assert run_id(cfg)[-4:] == run_id(cfg_same)[-4:]

    cfg_noisy = replace(cfg, warnings=("dummy warning",), source_path="other")
    assert run_id(cfg)[-4:] == run_id(cfg_noisy)[-4:]


def _model_patch(tmp_path, **overrides):
    """model セクションの一部を上書きした設定ペアを書き込む。"""
    data = _valid_cfg_data()
    data["model"] = {**data["model"], **overrides}
    return _write_pair(tmp_path, data)


@pytest.mark.parametrize("t_max", [0, -3])
def test_F01_rejects_non_positive_T_max(tmp_path, t_max):
    """T_max < 1 は ConfigError (F-01)。"""
    with pytest.raises(ConfigError):
        load(_model_patch(tmp_path, T_max=t_max))


def test_F01_rejects_T_min_above_T_max(tmp_path):
    """T_min > T_max は ConfigError (F-01)。"""
    with pytest.raises(ConfigError):
        load(_model_patch(tmp_path, T_min=60, T_max=50))


@pytest.mark.parametrize("eps", [0.0, -1e-4])
def test_F01_rejects_non_positive_eps(tmp_path, eps):
    """eps <= 0 は ConfigError (F-01)。"""
    with pytest.raises(ConfigError):
        load(_model_patch(tmp_path, eps=eps))


@pytest.mark.parametrize("tries", [0, -1])
def test_F01_rejects_non_positive_max_tries(tmp_path, tries):
    """max_tries < 1 は ConfigError (F-01, A14)。"""
    data = _valid_cfg_data()
    data["network"] = {"n": 20, "k": 2, "p": 0.1, "max_tries": tries}
    with pytest.raises(ConfigError):
        load(_write_pair(tmp_path, data))


def test_F01_rejects_negative_std(tmp_path):
    """init_opinion.std < 0 は ConfigError (F-01, F-04)。"""
    data = _valid_cfg_data()
    data["init_opinion"] = {"mean": 0.5, "std": -0.1}
    with pytest.raises(ConfigError):
        load(_write_pair(tmp_path, data))


@pytest.mark.parametrize("seeds", [0, -2])
def test_F01_rejects_non_positive_n_seeds(tmp_path, seeds):
    """n_seeds < 1 は ConfigError (F-01)。"""
    data = _valid_cfg_data()
    data["experiment"] = {
        "master_seed_start": 0, "n_seeds": seeds, "W_values": [0.0],
        "ablation_conditions": ["none"],
    }
    with pytest.raises(ConfigError):
        load(_write_pair(tmp_path, data))


def test_F01_rejects_missing_section(tmp_path):
    """セクション欠落は KeyError ではなく ConfigError (F-01)。"""
    data = _valid_cfg_data()
    del data["model"]
    with pytest.raises(ConfigError):
        load(_write_pair(tmp_path, data))


def test_F01_rejects_missing_key(tmp_path):
    """キー欠落は KeyError ではなく ConfigError (F-01)。"""
    data = _valid_cfg_data()
    del data["model"]["S"]
    with pytest.raises(ConfigError):
        load(_write_pair(tmp_path, data))


def test_F01_rejects_non_numeric_value(tmp_path):
    """非数値は ValueError ではなく ConfigError (F-01)。"""
    with pytest.raises(ConfigError):
        load(_model_patch(tmp_path, S="abc"))


def test_F01_rejects_extra_key(tmp_path):
    """未知のキーは TypeError ではなく ConfigError (F-01, NFR-5)。"""
    with pytest.raises(ConfigError):
        load(_model_patch(tmp_path, unknown_key=1))


def test_F01_rejects_empty_file(tmp_path):
    """空YAMLは ConfigError (F-01)。"""
    cfg_file = tmp_path / "test_config.yaml"
    cfg_file.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError):
        load(cfg_file)
