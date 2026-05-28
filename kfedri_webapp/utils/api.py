"""
K-FEDRI v3a 실시간 예측 파이프라인
- 기상청 API허브 ASOS 일자료 호출
- v2 기상 피처 11개 계산
- 임상(imsang) 7개 + 지형(dem) 11개 정적 피처 병합
- v3a 모델 (.pkl) 로드 → predict_proba
"""

import math
import time
import joblib
import requests
import pandas as pd
import numpy as npa
from datetime import datetime, timedelta
from pathlib import Path

# ── 상수 ──────────────────────────────────────────────────────────
API_URL       = "https://apihub.kma.go.kr/api/typ01/url/kma_sfcdd3.php"
LOOKBACK_DAYS = 31          # 이동평균 계산에 필요한 과거 일수 (30일 rolling + 1)
REQUEST_SLEEP = 0.15        # 지점 간 딜레이 (초) - KMA 과부하 방지

API_COLUMNS = [
    "TM", "STN",
    "WS_AVG", "WR_DAY", "WD_MAX", "WS_MAX", "WS_MAX_TM",
    "WD_INS", "WS_INS", "WS_INS_TM",
    "TA_AVG", "TA_MAX", "TA_MAX_TM", "TA_MIN", "TA_MIN_TM",
    "TD_AVG", "TS_AVG", "TG_MIN",
    "HM_AVG", "HM_MIN", "HM_MIN_TM",
    "PV_AVG", "EV_S", "EV_L", "FG_DUR",
    "PA_AVG", "PS_AVG", "PS_MAX", "PS_MAX_TM", "PS_MIN", "PS_MIN_TM",
    "CA_TOT",
    "SS_DAY", "SS_DUR", "SS_CMB",
    "SI_DAY", "SI_60M_MAX", "SI_60M_MAX_TM",
    "RN_DAY", "RN_D99", "RN_DUR",
    "RN_60M_MAX", "RN_60M_MAX_TM",
    "RN_10M_MAX", "RN_10M_MAX_TM",
    "RN_POW_MAX", "RN_POW_MAX_TM",
    "SD_NEW", "SD_NEW_TM", "SD_MAX", "SD_MAX_TM",
    "TE_05", "TE_10", "TE_15", "TE_30", "TE_50",
]

# v3a 피처 순서 (학습 시와 동일해야 함)
V3A_FEATURE_COLS = [
    "HeatRisk", "DrynessRisk", "WindRisk",
    "Dry7", "Dry30", "LongDryRisk",
    "SolarRisk", "SnowRisk",
    "Season_sin", "Season_cos", "WeekendRisk",
    "imsang_ForestRatio", "imsang_ConiferRatio", "imsang_BroadleafRatio",
    "imsang_MixedRatio", "imsang_BambooRatio", "imsang_NonForestRatio",
    "imsang_ForestRiskScore",
    "dem_MeanElevation", "dem_MaxElevation", "dem_MinElevation",
    "dem_MeanSlope", "dem_MaxSlope",
    "dem_NorthRatio", "dem_EastRatio", "dem_SouthRatio",
    "dem_WestRatio", "dem_FlatRatio",
    "dem_TerrainRiskScore",
]


# ── API 호출 ───────────────────────────────────────────────────────

def _get_date_range(days: int = LOOKBACK_DAYS, start_date=None):
    end = datetime.now() - timedelta(days=1)
    if start_date is not None:
        # start_date: datetime 또는 date 객체
        start = datetime.combine(start_date, datetime.min.time()) \
                if not isinstance(start_date, datetime) else start_date
    else:
        start = end - timedelta(days=days - 1)
    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")


def _fetch_all_at_once(tm1: str, tm2: str, api_key: str) -> pd.DataFrame:
    """stn=0 으로 전체 지점 일괄 호출 — API 1번으로 완료."""
    params = {
        "tm1": tm1, "tm2": tm2,
        "stn": "0",
        "help": "0",
        "authKey": api_key,
    }
    resp = requests.get(API_URL, params=params, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}")

    data_lines = [
        ln.strip() for ln in resp.text.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    rows = [ln.split() for ln in data_lines if len(ln.split()) == len(API_COLUMNS)]
    if not rows:
        raise RuntimeError("API 응답에 데이터 행이 없습니다.")

    return pd.DataFrame(rows, columns=API_COLUMNS)


def fetch_all_stations(
    station_ids: list,
    api_key: str,
    lookback_days: int = LOOKBACK_DAYS,
    start_date=None,
    progress_callback=None,
) -> tuple[pd.DataFrame, list]:
    """
    stn=0 으로 전체 지점을 1번 호출해 반환.
    Returns: (raw_df, failed_ids)
    """
    tm1, tm2 = _get_date_range(lookback_days, start_date=start_date)

    if progress_callback:
        progress_callback(0, 1)

    df = _fetch_all_at_once(tm1, tm2, api_key)

    if progress_callback:
        progress_callback(1, 1)

    # 요청한 지점만 필터링 & 미반환 지점 = failed
    df["STN"] = df["STN"].astype(int)
    df = df[df["STN"].isin(station_ids)].copy()
    returned = set(df["STN"].unique())
    failed   = [sid for sid in station_ids if sid not in returned]

    return df, failed


# ── 전처리 ────────────────────────────────────────────────────────

def _to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in API_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _clean_missing(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in API_COLUMNS:
        if col not in out.columns:
            continue
        if col in ("RN_DAY", "SD_MAX", "SD_NEW"):
            out[col] = out[col].where(out[col] > -8.999, 0)
        elif col not in ("TM", "STN"):
            out[col] = out[col].where(out[col] > -8.999, np.nan)
    return out


def _impute_si_day(df: pd.DataFrame) -> pd.DataFrame:
    """SI_DAY 결측을 같은 날짜 중앙값으로 보정."""
    out = df.copy()
    date_med = out.groupby("date")["SI_DAY"].transform("median")
    global_med = out["SI_DAY"].median()
    out["SI_DAY"] = out["SI_DAY"].fillna(date_med).fillna(global_med)
    return out


# ── 기상 피처 계산 (v2 로직 그대로) ──────────────────────────────

def build_weather_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    out = _to_numeric(raw_df)
    out = _clean_missing(out)

    out["date"]       = pd.to_datetime(out["TM"].astype(str), format="%Y%m%d", errors="coerce")
    out["station_id"] = out["STN"].astype(int)
    out = out.sort_values(["station_id", "date"]).reset_index(drop=True)

    out = _impute_si_day(out)

    # 기본 피처
    out["HeatRisk"]    = out["TA_MAX"]
    out["DrynessRisk"] = 100 - out["HM_MIN"]
    out["WindRisk"]    = out["WS_MAX"] * 3.6

    # 이동합 (전날까지 shift(1))
    grp = out.groupby("station_id")
    out["Precip_7d"]  = grp["RN_DAY"].transform(lambda s: s.shift(1).rolling(7,  min_periods=1).sum())
    out["Precip_30d"] = grp["RN_DAY"].transform(lambda s: s.shift(1).rolling(30, min_periods=1).sum())
    out["Solar_7d"]   = grp["SI_DAY"].transform(lambda s: s.shift(1).rolling(7,  min_periods=1).sum())

    out["Dry7"]     = 1 / (1 + out["Precip_7d"])
    out["Dry30"]    = 1 / (1 + out["Precip_30d"])
    out["SolarRisk"] = out["Solar_7d"]
    out["SnowRisk"]  = (out["SD_MAX"] > 0).astype(int)

    # 연속 무강수일수 (전날까지 누적)
    long_dry = []
    for _, grp_df in out.groupby("station_id", sort=False):
        streak, vals = 0, []
        for rn in grp_df.sort_values("date")["RN_DAY"]:
            vals.append(streak)
            streak = streak + 1 if (pd.notna(rn) and rn < 1) else 0
        long_dry.extend(vals)
    out["LongDryRisk"] = long_dry

    # 계절 / 주말
    out["doy"]        = out["date"].dt.dayofyear
    out["Season_sin"] = np.sin(2 * math.pi * out["doy"] / 365.25)
    out["Season_cos"] = np.cos(2 * math.pi * out["doy"] / 365.25)
    out["WeekendRisk"] = out["date"].dt.dayofweek.isin([5, 6]).astype(int)

    return out


# ── v3a 피처 병합 (임상 + 지형 정적 피처 추가) ─────────────────────

def build_v3a_features(
    weather_df: pd.DataFrame,
    imsang_df: pd.DataFrame,
    dem_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    기상 피처 + 임상(imsang_*) + 지형(dem_*) 병합 → 29개 피처 완성.
    imsang/dem CSV의 컬럼명에 자동으로 prefix 추가.
    """
    imsang_cols = [
        "ForestRatio", "ConiferRatio", "BroadleafRatio",
        "MixedRatio", "BambooRatio", "NonForestRatio", "ForestRiskScore",
    ]
    dem_cols = [
        "MeanElevation", "MaxElevation", "MinElevation",
        "MeanSlope", "MaxSlope",
        "NorthRatio", "EastRatio", "SouthRatio", "WestRatio", "FlatRatio",
        "TerrainRiskScore",
    ]

    imsang_sub = imsang_df[["station_id"] + imsang_cols].copy()
    imsang_sub.columns = ["station_id"] + [f"imsang_{c}" for c in imsang_cols]

    dem_sub = dem_df[["station_id"] + dem_cols].copy()
    dem_sub.columns = ["station_id"] + [f"dem_{c}" for c in dem_cols]

    out = (
        weather_df
        .merge(imsang_sub, on="station_id", how="left")
        .merge(dem_sub,    on="station_id", how="left")
    )
    return out


# ── 예측 ──────────────────────────────────────────────────────────

def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"모델 파일 없음: {model_path}")
    return joblib.load(model_path)


def _is_logreg(model) -> bool:
    """LogisticRegression 여부 판별 — 스케일링 적용 대상."""
    return type(model).__name__ == "LogisticRegression"


def _fit_scaler(features_df: pd.DataFrame, feature_cols: list):
    """LOOKBACK_DAYS 분량 features 전체로 StandardScaler fit.
    학습 scaler(2022-2024 fit)와 완전 동일하지는 않으나,
    정적 feature(imsang/dem)는 동일 분포이고 기상 feature도 다수 표본으로 근사."""
    from sklearn.preprocessing import StandardScaler
    valid_all = features_df.dropna(subset=feature_cols)
    scaler = StandardScaler()
    scaler.fit(valid_all[feature_cols].values)
    return scaler


def predict_today(
    features_df: pd.DataFrame,
    model,
    feature_cols: list = V3A_FEATURE_COLS,
) -> pd.DataFrame:
    """
    최신 날짜 행만 필터링 → predict_proba → 결과 반환.
    LogReg 모델인 경우 features_df 전체(~31일치)로 fit한 StandardScaler 적용.
    Returns DataFrame with columns: station_id, date, proba
    """
    latest_date = features_df["date"].max()
    today_df = features_df[features_df["date"] == latest_date].copy()

    # 피처 결측 행 제거
    valid = today_df.dropna(subset=feature_cols)

    if valid.empty:
        raise RuntimeError(f"{latest_date.date()} 날짜에 예측 가능한 지점이 없습니다.")

    X = valid[feature_cols].values

    # LogReg는 StandardScaler 필요 — lookback 전체로 fit
    if _is_logreg(model):
        scaler = _fit_scaler(features_df, feature_cols)
        X = scaler.transform(X)

    proba = model.predict_proba(X)[:, 1]

    result = valid[["station_id", "date"]].copy()
    result["proba"] = proba
    result["date"]  = latest_date
    return result.reset_index(drop=True)


# ── 편의 함수: 전체 파이프라인 한 번에 ───────────────────────────────

def run_realtime_pipeline(
    api_key: str,
    station_ids: list,
    imsang_df: pd.DataFrame,
    dem_df: pd.DataFrame,
    model_path: Path,
    progress_callback=None,
) -> pd.DataFrame:
    """
    API 호출 → 피처 생성 → 모델 예측 → 오늘 날짜 예측 결과 반환.
    """
    model = load_model(model_path)

    raw_df, failed = fetch_all_stations(
        station_ids, api_key,
        progress_callback=progress_callback,
    )

    weather_df  = build_weather_features(raw_df)
    features_df = build_v3a_features(weather_df, imsang_df, dem_df)
    result_df   = predict_today(features_df, model)

    return result_df, failed


# ── 모델 파일명 → CSV 컬럼명 매핑 ─────────────────────────────────
MODEL_COL_MAP = {
    "lgbm_v3a": "v3a_LightGBM_proba",
    "xgb_v3a":  "v3a_XGBoost_proba",
    "lgr_v3a":  "v3a_LogReg_proba",
}


def predict_all_dates(
    features_df: pd.DataFrame,
    model,
    feature_cols: list = V3A_FEATURE_COLS,
) -> pd.DataFrame:
    """
    features_df의 모든 날짜에 대해 predict_proba.
    (predict_today와 달리 최신 날짜만 선택하지 않음)
    Returns: station_id, date, proba
    """
    valid = features_df.dropna(subset=feature_cols).copy()
    if valid.empty:
        raise RuntimeError("예측 가능한 행이 없습니다.")
    proba = model.predict_proba(valid[feature_cols].values)[:, 1]
    result = valid[["station_id", "date"]].copy()
    result["proba"] = proba
    return result.reset_index(drop=True)


def run_extension_pipeline(
    api_key: str,
    station_ids: list,
    imsang_df: pd.DataFrame,
    dem_df: pd.DataFrame,
    model_paths: dict,          # {model_stem: Path}  예: {"lgbm_v3a": Path(...)}
    since_date,                 # date 또는 datetime: 이 날짜부터 새로 예측
    progress_callback=None,
) -> tuple[pd.DataFrame, list]:
    """
    since_date 이후 날짜를 API로 가져와 v3a 모델 예측 후 반환.
    피처 계산을 위해 since_date 이전 LOOKBACK_DAYS만큼도 함께 호출.

    Returns:
      result_df  : station_id, date, {v3a_LightGBM_proba}, {v3a_XGBoost_proba}, ...
      failed_ids : API 호출 실패 지점 목록
    """
    since_dt = (
        datetime.combine(since_date, datetime.min.time())
        if not isinstance(since_date, datetime) else since_date
    )
    fetch_start = since_dt - timedelta(days=LOOKBACK_DAYS)

    raw_df, failed = fetch_all_stations(
        station_ids, api_key,
        start_date=fetch_start,
        progress_callback=progress_callback,
    )

    weather_df  = build_weather_features(raw_df)
    features_df = build_v3a_features(weather_df, imsang_df, dem_df)

    # since_date 이후 행만 예측 대상으로 필터
    new_df = features_df[features_df["date"] >= since_dt].copy()
    if new_df.empty:
        raise RuntimeError(
            f"{since_dt.date()} 이후 데이터가 없습니다. "
            "API 최신 데이터 반영 시까지 기다려주세요."
        )

    valid  = new_df.dropna(subset=V3A_FEATURE_COLS)
    result = valid[["station_id", "date"]].copy().reset_index(drop=True)

    # LogReg용 스케일러 — features_df 전체(LOOKBACK_DAYS + 연장기간)로 fit
    # 트리 모델(LGBM·XGB)에는 영향 없으므로 LogReg 있을 때만 lazy-fit
    _scaler = None
    X_raw = valid[V3A_FEATURE_COLS].values

    for stem, model_path in model_paths.items():
        col = MODEL_COL_MAP.get(stem, f"{stem}_proba")
        try:
            mdl = load_model(model_path)
            if _is_logreg(mdl):
                if _scaler is None:
                    _scaler = _fit_scaler(features_df, V3A_FEATURE_COLS)
                X_use = _scaler.transform(X_raw)
            else:
                X_use = X_raw
            proba = mdl.predict_proba(X_use)[:, 1]
            result[col] = proba
        except Exception as e:
            result[col] = np.nan

    return result, failed
