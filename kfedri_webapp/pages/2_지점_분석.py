import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="지점 분석", page_icon="📊", layout="wide")
st.title("📊 ASOS 지점별 상세 분석")

DATA_DIR = Path(__file__).parents[1] / "data"


@st.cache_data
def load_data():
    stations = pd.read_csv(DATA_DIR / "asos_stations.csv")
    dem = pd.read_csv(DATA_DIR / "asos_dem_features.csv")
    imsang = pd.read_csv(DATA_DIR / "asos_imsangdo_features.csv")

    merged = (
        stations[["station_id", "station_name", "region", "lat", "lon", "altitude_m"]]
        .merge(dem, on="station_id", how="left", suffixes=("", "_dem"))
        .merge(imsang, on="station_id", how="left", suffixes=("", "_imsang"))
    )
    return merged, dem, imsang


data, dem_all, imsang_all = load_data()

# ── 지점 선택 ─────────────────────────────────────────────────────
col_region, col_station = st.columns([1, 2])
with col_region:
    regions = ["전체"] + sorted(data["region"].dropna().unique().tolist())
    sel_region = st.selectbox("시도", regions)

with col_station:
    filtered = data if sel_region == "전체" else data[data["region"] == sel_region]
    station_opts = {
        f"{r['station_name']} ({int(r['station_id'])})": int(r["station_id"])
        for _, r in filtered.iterrows()
    }
    sel_name = st.selectbox("ASOS 지점", list(station_opts.keys()))
    sel_id = station_opts[sel_name]

row = data[data["station_id"] == sel_id].iloc[0]

# ── 지점 기본 정보 ────────────────────────────────────────────────
st.divider()
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("지점명", row["station_name"])
with c2:
    st.metric("지점번호", int(row["station_id"]))
with c3:
    st.metric("지역", row.get("region", "N/A"))
with c4:
    alt = row.get("altitude_m", float("nan"))
    st.metric("해발고도 (ASOS)", f"{alt:.0f}m" if not pd.isna(alt) else "N/A")
with c5:
    mean_elev = row.get("MeanElevation", float("nan"))
    st.metric("평균고도 (DEM 5km)", f"{mean_elev:.0f}m" if not pd.isna(mean_elev) else "N/A")

st.divider()

# ── 임상 & 지형 차트 ──────────────────────────────────────────────
col_l, col_r = st.columns(2)

with col_l:
    st.subheader("🌲 임상 구성 (5km 버퍼)")

    forest_vals = {
        "침엽수 (Conifer)": row.get("ConiferRatio", 0) or 0,
        "활엽수 (Broadleaf)": row.get("BroadleafRatio", 0) or 0,
        "혼효림 (Mixed)": row.get("MixedRatio", 0) or 0,
        "죽림 (Bamboo)": row.get("BambooRatio", 0) or 0,
        "비산림": row.get("NonForestRatio", 0) or 0,
    }

    fig_pie = px.pie(
        values=list(forest_vals.values()),
        names=list(forest_vals.keys()),
        color_discrete_sequence=["#16a34a", "#86efac", "#4ade80", "#a7f3d0", "#d1d5db"],
        hole=0.3,
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    fig_pie.update_layout(margin=dict(t=10, b=10, l=0, r=0), height=280,
                          showlegend=False)
    st.plotly_chart(fig_pie, use_container_width=True)

    fmi = row.get("ForestRiskScore", 0) or 0
    fmi_avg = imsang_all["ForestRiskScore"].mean()
    st.metric(
        "임상위험지수 (FMI)",
        f"{fmi:.3f}",
        delta=f"전국 평균 대비 {fmi - fmi_avg:+.3f}",
        delta_color="inverse",
    )
    dom_forest = row.get("DominantForest", "N/A")
    st.caption(f"우세 임상: **{dom_forest}** | 산림 면적 비율: {row.get('ForestRatio', 0)*100:.1f}%")

with col_r:
    st.subheader("⛰️ 지형 방위 분포 (5km 버퍼)")

    directions = ["북향(N)", "동향(E)", "남향(S)", "서향(W)", "평지"]
    aspect_vals = [
        row.get("NorthRatio", 0) or 0,
        row.get("EastRatio", 0) or 0,
        row.get("SouthRatio", 0) or 0,
        row.get("WestRatio", 0) or 0,
        row.get("FlatRatio", 0) or 0,
    ]

    # 레이더 차트 (닫힌 형태)
    theta = directions + [directions[0]]
    r_vals = aspect_vals + [aspect_vals[0]]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=r_vals,
        theta=theta,
        fill="toself",
        fillcolor="rgba(234, 88, 12, 0.20)",
        line=dict(color="#ea580c", width=2),
        name=row["station_name"],
    ))
    # 전국 평균
    avg_vals = [
        dem_all["NorthRatio"].mean(), dem_all["EastRatio"].mean(),
        dem_all["SouthRatio"].mean(), dem_all["WestRatio"].mean(),
        dem_all["FlatRatio"].mean(),
    ]
    fig_radar.add_trace(go.Scatterpolar(
        r=avg_vals + [avg_vals[0]],
        theta=theta,
        fill="toself",
        fillcolor="rgba(148, 163, 184, 0.15)",
        line=dict(color="#94a3b8", width=1, dash="dot"),
        name="전국 평균",
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 0.65])),
        margin=dict(t=20, b=20, l=40, r=40),
        height=280,
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    tmi = row.get("TerrainRiskScore", 0) or 0
    tmi_avg = dem_all["TerrainRiskScore"].mean()
    st.metric(
        "지형위험지수 (TMI)",
        f"{tmi:.3f}",
        delta=f"전국 평균 대비 {tmi - tmi_avg:+.3f}",
        delta_color="inverse",
    )
    dom_aspect = row.get("DominantAspect", "N/A")
    mean_slope = row.get("MeanSlope", 0) or 0
    st.caption(f"우세 방위: **{dom_aspect}** | 평균 경사: {mean_slope:.1f}°")

# ── 지형 상세 & 전국 순위 ──────────────────────────────────────────
st.divider()
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("📈 지형 통계")
    terrain_df = pd.DataFrame({
        "항목": ["평균고도", "최고고도", "최저고도", "평균경사", "최대경사"],
        "값": [
            f"{row.get('MeanElevation', 0):.1f} m",
            f"{row.get('MaxElevation', 0):.1f} m",
            f"{row.get('MinElevation', 0):.1f} m",
            f"{row.get('MeanSlope', 0):.2f} °",
            f"{row.get('MaxSlope', 0):.2f} °",
        ],
    })
    st.dataframe(terrain_df, use_container_width=True, hide_index=True)

with col_b:
    st.subheader("🏆 전국 순위 (97개 지점 기준)")
    fmi_rank = int((imsang_all["ForestRiskScore"] > fmi).sum()) + 1
    tmi_rank = int((dem_all["TerrainRiskScore"] > tmi).sum()) + 1

    rc1, rc2 = st.columns(2)
    with rc1:
        st.metric("임상위험 (FMI) 순위", f"{fmi_rank}위",
                  delta=f"상위 {fmi_rank/97*100:.0f}%", delta_color="off")
    with rc2:
        st.metric("지형위험 (TMI) 순위", f"{tmi_rank}위",
                  delta=f"상위 {tmi_rank/97*100:.0f}%", delta_color="off")

    # 전국 분포에서 현재 지점 위치 표시
    fig_hist = px.histogram(
        imsang_all, x="ForestRiskScore", nbins=20,
        title="전국 임상위험지수(FMI) 분포",
        labels={"ForestRiskScore": "FMI"},
        color_discrete_sequence=["#86efac"],
    )
    fig_hist.add_vline(x=fmi, line_color="#ef4444", line_width=2,
                       annotation_text=f"현재: {fmi:.2f}",
                       annotation_position="top right")
    fig_hist.update_layout(height=200, margin=dict(t=30, b=0, l=0, r=0),
                           showlegend=False, bargap=0.05)
    st.plotly_chart(fig_hist, use_container_width=True)
