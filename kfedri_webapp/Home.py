import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from utils.style import apply_dark_theme

st.set_page_config(
    page_title="K-FEDRI | 산불 위험도 예측",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_dark_theme()

DATA_DIR = Path(__file__).parent / "data"


@st.cache_data
def load_data():
    stations = pd.read_csv(DATA_DIR / "asos_stations.csv")
    dem = pd.read_csv(DATA_DIR / "asos_dem_features.csv")
    imsang = pd.read_csv(DATA_DIR / "asos_imsangdo_features.csv")

    dem_cols = ["station_id", "MeanElevation", "MeanSlope", "MaxSlope",
                "TerrainRiskScore", "DominantAspect",
                "NorthRatio", "EastRatio", "SouthRatio", "WestRatio", "FlatRatio"]
    imsang_cols = ["station_id", "ForestRatio", "ConiferRatio", "BroadleafRatio",
                   "MixedRatio", "BambooRatio", "NonForestRatio",
                   "ForestRiskScore", "DominantForest"]

    merged = (
        stations[["station_id", "station_name", "region", "lat", "lon", "altitude_m"]]
        .merge(dem[dem_cols], on="station_id", how="left")
        .merge(imsang[imsang_cols], on="station_id", how="left")
    )

    fmi = merged["ForestRiskScore"].fillna(0)
    tmi = merged["TerrainRiskScore"].fillna(0)
    fmi_norm = (fmi - fmi.min()) / (fmi.max() - fmi.min() + 1e-8)
    tmi_norm = (tmi - tmi.min()) / (tmi.max() - tmi.min() + 1e-8)
    merged["StaticRisk"] = (fmi_norm * 0.65 + tmi_norm * 0.35) * 100

    merged["RiskLevel"] = pd.cut(
        merged["StaticRisk"],
        bins=[0, 25, 50, 75, 101],
        labels=["낮음", "보통", "높음", "매우높음"],
    )
    return merged


data = load_data()

# ── 헤더 ──────────────────────────────────────────────────────────
st.title("🔥 K-FEDRI 산불 위험도 예측 시스템")
st.markdown(
    "**K**orean **F**orest fir**E D**anger **R**ating **I**ndex v3a &nbsp;|&nbsp; "
    "기상청 ASOS 97개 지점 &nbsp;|&nbsp; 임상도·DEM 피처 통합 &nbsp;|&nbsp; "
    "LightGBM / XGBoost ML 모델"
)
st.divider()

# ── 핵심 지표 ─────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("관측 지점", "97개", "전국 ASOS")
with c2:
    st.metric("최고 ROC-AUC", "0.8781", "XGBoost v3a ↑")
with c3:
    st.metric("Top5% Recall", "0.420", "LightGBM v3a ↑")
with c4:
    st.metric("입력 피처", "29개", "기상+임상+지형")
with c5:
    high_risk = int((data["StaticRisk"] >= 75).sum())
    st.metric("고위험 지점", f"{high_risk}개", "지형·임상 기준")

st.divider()

# ── 지역별 요약 + 위험 등급 분포 ──────────────────────────────────
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("지역별 평균 위험지수")
    region_tbl = (
        data.groupby("region")[["StaticRisk", "ForestRiskScore", "TerrainRiskScore"]]
        .mean()
        .round(2)
        .rename(columns={
            "StaticRisk": "종합위험지수",
            "ForestRiskScore": "FMI (임상)",
            "TerrainRiskScore": "TMI (지형)",
        })
        .sort_values("종합위험지수", ascending=False)
    )
    st.dataframe(region_tbl, use_container_width=True)

with col_right:
    st.subheader("위험 등급 분포")
    risk_counts = data["RiskLevel"].value_counts()
    fig_pie = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        color=risk_counts.index,
        color_discrete_map={
            "낮음": "#22c55e",
            "보통": "#eab308",
            "높음": "#f97316",
            "매우높음": "#ef4444",
        },
        hole=0.35,
    )
    fig_pie.update_layout(margin=dict(t=10, b=0, l=0, r=0), height=270,
                          legend=dict(orientation="h", y=-0.1))
    st.plotly_chart(fig_pie, use_container_width=True)

st.divider()

# ── 기능 안내 카드 ─────────────────────────────────────────────────
st.subheader("📌 주요 기능")
g1, g2, g3, g4 = st.columns(4)
with g1:
    st.info("**🗺️ 위험도 지도** (사이드바 → 1번)\n\n"
            "97개 지점 위험도를 한국 지도 위에 색상으로 시각화합니다.\n"
            "마커 클릭 시 지점 상세 정보 확인 가능합니다.")
with g2:
    st.info("**📊 지점 분석** (사이드바 → 2번)\n\n"
            "지점별 임상 구성, 지형 방위, FMI/TMI 지수를 분석합니다.\n"
            "전국 순위 및 레이더 차트를 제공합니다.")
with g3:
    st.info("**🤖 모델 성능** (사이드바 → 3번)\n\n"
            "v2 vs v3a, LogReg·LightGBM·XGBoost 비교 결과와\n"
            "피처 중요도 분석 결과를 확인합니다.")
    with g4:
        st.info("** 시계열 분석** (사이드바 → 4번)\n\n"
                "지역, 지점별로 예측 모델의 예측 확률이 시간 경과에 따라 어떻게 변하는지 시각화합니다.\n"
                "여러 예측 모델들 중 하나를 고를 수 있습니다.")

st.caption(
    "⚠️ 현재 화면의 위험지수는 **정적 임상·지형 데이터** 기준입니다. "
    "기상 API 연동 및 모델 파일(.pkl) 등록 후 일별 실시간 예측이 활성화됩니다."
)
