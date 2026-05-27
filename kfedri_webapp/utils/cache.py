"""
실시간 예측 결과 일별 캐시 유틸리티.
첫 번째 사용자가 API를 호출하면 data/rt_cache/ 에 저장,
이후 같은 날 요청은 캐시 파일을 읽어 API 호출을 생략합니다.
"""

from __future__ import annotations

import os
import pandas as pd
from datetime import date as _date, datetime
from pathlib import Path
from typing import Optional

_STEM_LABEL = {"lgbm_v3a": "LightGBM v3a", "xgb_v3a": "XGBoost v3a"}
_LABEL_STEM = {v: k for k, v in _STEM_LABEL.items()}


def _cache_dir(data_dir: Path) -> Path:
    d = data_dir / "rt_cache"
    d.mkdir(exist_ok=True)
    return d


def _today_str() -> str:
    return _date.today().strftime("%Y%m%d")


def cache_path(data_dir: Path, stem: str) -> Path:
    return _cache_dir(data_dir) / f"{stem}_{_today_str()}.csv"


def load_today(data_dir: Path, model_labels: list) -> Optional[dict]:
    result = {}
    for label in model_labels:
        stem = _LABEL_STEM.get(label, label.replace(" ", "_").lower())
        p = cache_path(data_dir, stem)
        if not p.exists():
            return None
        df = pd.read_csv(p, parse_dates=["date"])
        result[label] = df
    return result if result else None


def save_today(data_dir: Path, all_results: dict) -> None:
    today = _today_str()
    cdir = _cache_dir(data_dir)
    for label, df in all_results.items():
        stem = _LABEL_STEM.get(label, label.replace(" ", "_").lower())
        df.to_csv(cache_path(data_dir, stem), index=False)
    for old in cdir.glob("*.csv"):
        if today not in old.name:
            old.unlink(missing_ok=True)


def cache_mtime(data_dir: Path, model_labels: list) -> str:
    for label in model_labels:
        stem = _LABEL_STEM.get(label, label.replace(" ", "_").lower())
        p = cache_path(data_dir, stem)
        if p.exists():
            t = datetime.fromtimestamp(os.path.getmtime(p))
            return t.strftime("%H:%M")
    return ""


def _weather_cache_path(data_dir: Path) -> Path:
    return _cache_dir(data_dir) / f"weather_{_today_str()}.csv"


def save_weather(data_dir: Path, raw_df: pd.DataFrame) -> None:
    raw_df.to_csv(_weather_cache_path(data_dir), index=False)


def load_weather(data_dir: Path) -> Optional[pd.DataFrame]:
    p = _weather_cache_path(data_dir)
    if not p.exists():
        return None
    return pd.read_csv(p)
