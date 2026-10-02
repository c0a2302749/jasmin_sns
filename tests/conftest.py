"""tests/conftest.py — 全テスト共有fixture。"""
from __future__ import annotations

import pytest
import yaml

from config import Config, load


@pytest.fixture
def small_config(tmp_path) -> Config:
    """テスト用の最小構成設定ファイルを作成して読み込む。"""
    cfg_file = tmp_path / "test_config.yaml"
    persona_file = tmp_path / "test_persona.yaml"

    cfg_data = {
        "model": {
            "S": 2.0,
            "k_react": 1.0,
            "T_max": 50,
            "T_min": 5,
            "eps": 1e-4,
        },
        "network": {
            "n": 20,
            "k": 2,
            "p": 0.1,
            "max_tries": 100,
        },
        "init_opinion": {
            "mean": 0.5,
            "std": 0.1,
        },
        "personas": {
            "table": "test_persona",
            "concentration_nu": 10.0,
        },
        "experiment": {
            "master_seed_start": 0,
            "n_seeds": 5,
            "W_values": [0.0, 0.5],
            "ablation_conditions": ["none", "ABL-SYS"],
        },
        "stopping": {
            "update_mode": "gauss_seidel",
        },
        "metrics": {
            "theta": 0.5,
            "bootstrap_B": 100,
            "bootstrap_seed": 42,
        },
        "logging": {
            "log_every": 1,
            "debug_terms": False,
        },
    }

    persona_data = {
        "personas": [
            {
                "name": "conformist",
                "share": 0.5,
                "alpha": 0.7,
                "beta": 0.4,
                "gamma": 0.1,
                "w_sns": 0.7,
                "w_sys": 0.4,
                "b": 0.15,
                "mu_react": 0.7,
            },
            {
                "name": "cautious",
                "share": 0.5,
                "alpha": 0.4,
                "beta": 0.7,
                "gamma": 0.4,
                "w_sns": 0.4,
                "w_sys": 0.7,
                "b": 0.30,
                "mu_react": 0.7,
            },
        ]
    }

    with cfg_file.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_data, f, sort_keys=False)

    with persona_file.open("w", encoding="utf-8") as f:
        yaml.safe_dump(persona_data, f, sort_keys=False)

    return load(cfg_file)
