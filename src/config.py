"""config.py — F-01: 設定を読み、検査し、実行IDを作る。

対応決定:
- NFR-5: 設定の外部化（値の正典は configs/*.yaml とし、コード内にデフォルト値を持たない）
- A3: S の値域と警告 (regime: bistable)
- 1.4: 検査ルール (エラー／警告の2段階判定)
"""
from __future__ import annotations

import datetime
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """検査で止めるべき致命的な設定不備。"""
    pass


@dataclass(frozen=True)
class ModelConfig:
    """モデルの基本パラメータを保持するデータクラス。

    Attributes:
        S: スケーリング係数。
        k_react: リアクタンス減衰係数。
        T_max: シミュレーションの上限ターン数。
        T_min: シミュレーションの下限ターン数。
        eps: 収束判定に使用する許容誤差。
    """
    S: float
    k_react: float
    T_max: int
    T_min: int
    eps: float


@dataclass(frozen=True)
class NetworkConfig:
    """ネットワークの生成パラメータを保持するデータクラス。

    Attributes:
        n: 参加者（ノード）の総数。
        k: 初期リングネットワークにおける各ノードの次数（偶数）。
        p: エッジのつなぎ替え確率。
        max_tries: 連結グラフを生成するための最大試行回数。
    """
    n: int
    k: int
    p: float
    max_tries: int


@dataclass(frozen=True)
class InitOpinionConfig:
    """初期意見の生成パラメータを保持するデータクラス。

    Attributes:
        mean: 正規分布の平均。
        std: 正規分布の標準偏差。
    """
    mean: float
    std: float


@dataclass(frozen=True)
class PersonaConfig:
    """ペルソナ型の設定を保持するデータクラス。

    Attributes:
        table: 読み込むペルソナ設定ファイル（YAML）の名前。
        concentration_nu: Beta分布の集中度パラメータ。
        table_path: 解決済みのペルソナ設定ファイルの絶対または相対パス。
    """
    table: str
    concentration_nu: float
    table_path: str = ""


@dataclass(frozen=True)
class ExperimentConfig:
    """実験の実行条件を保持するデータクラス。

    Attributes:
        master_seed_start: マスターシードの開始値。
        n_seeds: 実行するシードの数。
        W_values: 検証する介入強度（W）のタプル。
        ablation_conditions: 実行するAblation条件のタプル（例: "none", "ABL-SYS"等）。
    """
    master_seed_start: int
    n_seeds: int
    W_values: tuple[float, ...]
    ablation_conditions: tuple[str, ...]


@dataclass(frozen=True)
class StoppingConfig:
    """シミュレーションの更新・停止設定を保持するデータクラス。

    Attributes:
        update_mode: 状態配列の更新方式（"gauss_seidel" または "jacobi"）。
    """
    update_mode: str


@dataclass(frozen=True)
class MetricsConfig:
    """評価指標の計算設定を保持するデータクラス。

    Attributes:
        theta: 二値化のための閾値。
        bootstrap_B: ブートストラップ法でのリサンプリング回数。
        bootstrap_seed: ブートストラップ法に使用する乱数シード。
    """
    theta: float
    bootstrap_B: int
    bootstrap_seed: int


@dataclass(frozen=True)
class LoggingConfig:
    """ログ出力に関する設定を保持するデータクラス。

    Attributes:
        log_every: 状態を出力するターンの間隔。
    """
    log_every: int


@dataclass(frozen=True)
class Config:
    """システム全体の設定を保持するルートのデータクラス。

    Attributes:
        model: モデルパラメータの設定。
        network: ネットワーク生成の設定。
        init_opinion: 初期意見生成の設定。
        personas: ペルソナ設定。
        experiment: 実験条件の設定。
        stopping: 更新方式の設定。
        metrics: 評価指標の設定。
        logging: ログ出力の設定。
        warnings: 読み込み時に発生した警告メッセージのタプル。
        source_path: 読み込んだ設定ファイルのパス。
    """
    model: ModelConfig
    network: NetworkConfig
    init_opinion: InitOpinionConfig
    personas: PersonaConfig
    experiment: ExperimentConfig
    stopping: StoppingConfig
    metrics: MetricsConfig
    logging: LoggingConfig
    warnings: tuple[str, ...] = field(default_factory=tuple)
    source_path: str = ""


def _check_and_warn(cfg_dict: dict[str, Any]) -> list[str]:
    """止める／警告するの2段階検査（NFR-5, A3, 1.4）。

    読み込んだ設定辞書に対して値の妥当性を検証し、致命的なエラーがある場合は
    ConfigError を送出する。許容可能だが想定外の値については警告メッセージを生成する。

    Args:
        cfg_dict: YAMLから読み込んだ設定の辞書。

    Raises:
        ConfigError: model.S が正の実数でない場合。
        ConfigError: network.n が正の整数でない場合。
        ConfigError: network.k が奇数、または n 以上の場合。
        ConfigError: network.p が [0, 1] の範囲外の場合。
        ConfigError: experiment.W_values の要素が [0, 1] の範囲外の場合。
        ConfigError: experiment.ablation_conditions に未知の条件が含まれる場合。
        ConfigError: stopping.update_mode が "gauss_seidel" か "jacobi" でない場合。

    Returns:
        警告メッセージのリスト。
    """
    # 検査と警告
    warnings: list[str] = []

    # --- Model ---
    model = cfg_dict["model"]
    S = float(model["S"])
    if S <= 0:
        raise ConfigError(f"model.S は正の実数でなければならない: {S}")
    if S > 2.0:
        warnings.append(
            f"model.S={S} は推奨範囲(0<S<=2)を超えている。無刺激時にも自発的分極が生じるレジームとして扱う (regime: bistable, A3)。"
        )

    k_react = float(model["k_react"])
    if not (0 <= k_react <= 2):
        warnings.append(f"model.k_react={k_react} は感度分析範囲[0,2]の外（A9）。")

    # --- Network ---
    net = cfg_dict["network"]
    n, k, p = int(net["n"]), int(net["k"]), float(net["p"])
    if n <= 0:
        raise ConfigError(f"network.n は正の整数でなければならない: {n}")
    if k % 2 != 0 or k >= n:
        raise ConfigError(f"network.k は偶数かつ n 未満でなければならない: k={k}, n={n}")
    if not (0.0 <= p <= 1.0):
        raise ConfigError(f"network.p は [0, 1] の範囲でなければならない: {p}")

    # --- Experiment ---
    exp = cfg_dict["experiment"]
    for w in exp["W_values"]:
        if not (0.0 <= float(w) <= 1.0):
            raise ConfigError(f"experiment.W_values の値は[0,1]でなければならない: {w}")

    valid_ablations = {"none", "ABL-SYS", "ABL-SNS", "ABL-REACT"}
    for cond in exp["ablation_conditions"]:
        if cond not in valid_ablations:
            raise ConfigError(f"未知のAblation条件が含まれています: {cond}")

    if int(exp["n_seeds"]) < 5:
        warnings.append(
            f"experiment.n_seeds={exp['n_seeds']} < 5: 統計的信頼区間の計算精度が不十分な可能性がある。"
        )

    # --- Stopping ---
    stopping = cfg_dict["stopping"]
    if stopping["update_mode"] not in ("gauss_seidel", "jacobi"):
        raise ConfigError(
            f"stopping.update_mode は gauss_seidel か jacobi のみ（A2）: {stopping['update_mode']}"
        )

    return warnings


def load(path: str | Path) -> Config:
    """YAMLを読み込み、検査してConfigを返す。警告はConfig.warningsに含める。

    Args:
        path: 読み込むYAML設定ファイルのパス。

    Raises:
        ConfigError: 指定された設定ファイルが存在しない場合。
        ConfigError: 指定されたペルソナ設定ファイルが存在しない場合。

    Returns:
        読み込み、検証、および解決された設定を含むConfigオブジェクト。
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"設定ファイルが見つからない: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    warnings = _check_and_warn(raw)

    # persona表のパスは、configファイルと同じディレクトリからの相対で解決する
    # （cwdに依存させない。NFR-7: 可搬性）
    table_name = raw["personas"]["table"]
    table_path = str((path.parent / f"{table_name}.yaml").resolve())
    if not Path(table_path).exists():
        raise ConfigError(f"persona表が見つからない: {table_path}")

    return Config(
        model=ModelConfig(**raw["model"]),
        network=NetworkConfig(**raw["network"]),
        init_opinion=InitOpinionConfig(**raw["init_opinion"]),
        personas=PersonaConfig(
            table=table_name,
            concentration_nu=raw["personas"]["concentration_nu"],
            table_path=table_path,
        ),
        experiment=ExperimentConfig(
            master_seed_start=raw["experiment"]["master_seed_start"],
            n_seeds=raw["experiment"]["n_seeds"],
            W_values=tuple(raw["experiment"]["W_values"]),
            ablation_conditions=tuple(
                raw["experiment"]["ablation_conditions"]),
        ),
        stopping=StoppingConfig(**raw["stopping"]),
        metrics=MetricsConfig(**raw["metrics"]),
        logging=LoggingConfig(**raw["logging"]),
        warnings=tuple(warnings),
        source_path=str(path),
    )


def run_id(config: Config) -> str:
    """日時と設定ハッシュから一意な実行IDを作る。

    Args:
        config: ハッシュ生成の元となるConfigオブジェクト。

    Returns:
        "run_YYYYMMDD_HHMMSS_XXXX" 形式の実行ID文字列。
    """
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # config 全体をハッシュ化（4桁）
    payload = json.dumps(asdict(config), sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:4]

    return f"run_{now}_{digest}"
