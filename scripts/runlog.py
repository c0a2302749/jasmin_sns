"""scripts/runlog.py — 再現性ヘッダの共通ヘルパー。

分析スクリプトの先頭出力に gitコミットID・作業ツリー状態・設定ファイルを記録する
（ログに再現情報を残す要件用）。シミュレーションのRNGには触れない。
"""
from __future__ import annotations

import subprocess
from pathlib import Path


def log_header(config=None, extra: str = "") -> str:
    """再現性ヘッダをstdoutに出し、本文も返す。

    Args:
        config: Config（source_path を記録する。None可）。
        extra: 追加で記録する1行（シード範囲等）。
    """
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=root, timeout=30).stdout.strip() or "unknown"
        dirty = subprocess.run(
            ["git", "status", "--short"], capture_output=True, text=True,
            cwd=root, timeout=30).stdout.strip()
    except Exception:
        commit, dirty = "unknown", ""
    lines = [
        f"commit={commit}" + (" DIRTY" if dirty else " clean"),
        f"config={getattr(config, 'source_path', '-')}",
    ]
    if dirty:
        lines.append("uncommitted:\n" + dirty)
    if extra:
        lines.append(extra)
    text = "\n".join(lines)
    print(text)
    print("-" * 96)
    return text
