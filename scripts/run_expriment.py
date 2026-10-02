#!/usr/bin/env python3
"""scripts/run_experiment.py — 段階1: 本実験を実行する (CLI エントリーポイント)"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# プロジェクトルートの src をインポート可能にする（src はパッケージ直下）
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import load
from run.experiment import run


def main() -> None:
    parser = argparse.ArgumentParser(description="JASMIN_SNS 3層反実仮想シミュレーション実行スクリプト")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/default.yaml",
        help="設定 YAML ファイルへのパス (既定: configs/default.yaml)",
    )
    parser.add_argument(
        "--out",
        "-o",
        type=str,
        default="runs",
        help="出力基本ディレクトリ (既定: runs/)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"エラー: 設定ファイルが見つかりません: {config_path}", file=sys.stderr)
        sys.exit(1)

    print(f"設定ファイルを読み込んでいます: {config_path}")
    config = load(config_path)
    
    # 実行前に設定のバリデーション警告をユーザーに提示する
    for w in config.warnings:
        print(f"[warning] {w}")

    print("シミュレーション実験を開始します...")
    
    # experiment.py の run 関数へ処理を委譲
    out_path = run(config=config, out_dir=args.out)
    
    print(f"実験が完了しました。出力先: {out_path}")


if __name__ == "__main__":
    main()