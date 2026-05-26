import streamlit as st
import pandas as pd
import numpy as np
import folium
import plotly.express as px
from streamlit_folium import st_folium
from pathlib import Path
from utils.style import apply_dark_theme

st.set_page_config(page_title="위험도 지도", page_icon="🗺️", layout="wide")
apply_dark_theme()
st.title("🗺️ 전국 산불 위험도 지도")

DATA_DIR = Path(__file__).parents[1] / "data"


@st.cache_data
def load_data():
    stations = pd.read_csv(DATA_DIR / "asos_stations.csv")
    dem = pd.read_csv(DATA_DIR / "asos_dem_features.csv")
    imsang = pd.read_csv(DATA_DIR / "asos_imsangdo_features.csv")

    merged = (
        stations[["station_id", "station_name", "region", "lat", "lon", "altitude_m"]]
        .merge(
            dem[["station_id", "TerrainRiskScore", "MeanElevation",
                 "MeanSlope", "DominantAspect"]],
            on="station_id", how="left",
        )
        .merge(
            imsang[["station_id", "ForestRiskScore", "ConiferRatio",
                    "BroadleafRatio", "DominantForest"]],
            on="station_id", how="left",
        )
    )

    fmi = merged["ForestRiskScore"].fillna(0)
    tmi = merged["TerrainRiskScore"].fillna(0)
    fmi_norm = (fmi - fmi.min()) / (fmi.max() - fmi.min() + 1e-8)
    tmi_norm = (tmi - tmi.min()) / (tmi.max() - tmi.min() + 1e-8)
    merged["risk_score"] = fmi_norm * 0.65 + tmi_norm * 0.35
    return merged


data = load_data()

# ── 사이드바 ───────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("🔍 필터")
    regions = ["전체"] + sorted(data["region"].dropna().unique().tolist())
    selected_region = st.selectbox("시도 선택", regions)
    show_labels = st.checkbox("지점명 표시", value=False)
    st.divider()
    st.caption("색상 기준 (임상+지형 복합위험지수)")
    st.markdown(
        "<span style='color:#22c55e'>●</span> 낮음 &nbsp; "
        "<span style='color:#eab308'>●</span> 보통 &nbsp; "
        "<span style='color:#f97316'>●</span> 높음 &nbsp; "
        "<span style='color:#ef4444'>●</span> 매우높음",
        unsafe_allow_html=True,
    )

# 데이터 필터
display = data if selected_region == "전체" else data[data["region"] == selected_region]


def risk_color(val):
    if pd.isna(val):
        return "#94a3b8"
    if val >= 0.75:
        return "#ef4444"
    if val >= 0.50:
        return "#f97316"
    if val >= 0.25:
        return "#eab308"
    return "#22c55e"


def risk_label(val):
    if pd.isna(val):
        return "정보없음"
    if val >= 0.75:
        return "매우높음"
    if val >= 0.50:
        return "높음"
    if val >= 0.25:
        return "보통"
    return "낮음"


# ── 지도 생성 ─────────────────────────────────────────────────────
center_lat = display["lat"].mean() if not display.empty else 36.5
center_lon = display["lon"].mean() if not display.empty else 127.8
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=7 if selected_region == "전체" else 9,
    tiles="CartoDB dark_matter",
)

for _, row in display.iterrows():
    if pd.isna(row["lat"]) or pd.isna(row["lon"]):
        continue

    color = risk_color(row["risk_score"])
    label = risk_label(row["risk_score"])

    fmi_val = row.get("ForestRiskScore", float("nan"))
    tmi_val = row.get("TerrainRiskScore", float("nan"))
    elev_val = row.get("MeanElevation", float("nan"))

    fmi_str = f"{fmi_val:.2f}" if not pd.isna(fmi_val) else "N/A"
    tmi_str = f"{tmi_val:.2f}" if not pd.isna(tmi_val) else "N/A"
    elev_str = f"{elev_val:.0f}m" if not pd.isna(elev_val) else "N/A"
    dom_forest = row.get("DominantForest", "N/A") or "N/A"

    popup_html = f"""
    <div style="font-family: sans-serif; min-width: 210px; font-size: 13px;">
        <b style="font-size:15px;">{row['station_name']} (#{int(row['station_id'])})</b><br>
        <span style="color:#64748b;">{row['region']}</span>
        <hr style="margin:6px 0; border-color:#e2e8f0;">
        <b>위험 등급:</b>
        <span style="color:{color}; font-weight:bold;">{label}</span><br>
        <b>임상위험 (FMI):</b> {fmi_str}<br>
        <b>지형위험 (TMI):</b> {tmi_str}<br>
        <b>우세임상:</b> {dom_forest}<br>
        <b>평균고도:</b> {elev_str}
    </div>
    """

    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=8,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.85,
        weight=1.5,
        popup=folium.Popup(popup_html, max_width=260),
        tooltip=f"{row['station_name']} | {label} ({row['risk_score']:.2f})",
    ).add_to(m)

    if show_labels:
        folium.Marker(
            location=[row["lat"], row["lon"]],
            icon=folium.DivIcon(
                html=f'<div style="font-size:9px; font-weight:bold; '
                     f'white-space:nowrap; color:#1e293b; '
                     f'text-shadow:1px 1px 2px white;">{row["station_name"]}</div>',
                icon_size=(60, 12),
                icon_anchor=(0, 0),
            ),
        ).add_to(m)

# 범례
legend_html = """
<div style="position:fixed; bottom:30px; left:30px; z-index:1000;
     background:balck; padding:12px 16px; border-radius:8px;
     color:rgb(255,255,255);
     box-shadow:0 2px 8px rgba(0,0,0,0.18); font-family:sans-serif; font-size:13px;">
  <b>위험 등급</b><br>
  <span style="color:#22c55e; font-size:16px;">●</span> 낮음 (0~25)<br>
  <span style="color:#eab308; font-size:16px;">●</span> 보통 (25~50)<br>
  <span style="color:#f97316; font-size:16px;">●</span> 높음 (50~75)<br>
  <span style="color:#ef4444; font-size:16px;">●</span> 매우높음 (75~100)
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

st_folium(m, width=None, height=600, use_container_width=True)

# ── 지도 하단 통계 ─────────────────────────────────────────────────
st.divider()
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("표시 지점", f"{len(display)}개")
with c2:
    high = int((display["risk_score"] >= 0.75).sum())
    st.metric("매우높음 지점", f"{high}개")
with c3:
    med = int(((display["risk_score"] >= 0.50) & (display["risk_score"] < 0.75)).sum())
    st.metric("높음 지점", f"{med}개")
with c4:
    avg = display["risk_score"].mean()
    st.metric("평균 위험지수", f"{avg:.2f}")

# 상위 5개 지점
st.subheader("⚠️ 고위험 지점 Top 5")
top5 = display.nlargest(5, "risk_score")[
    ["station_name", "region", "risk_score", "ForestRiskScore", "TerrainRiskScore"]
].rename(columns={
    "station_name": "지점명",
    "region": "지역",
    "risk_score": "종합위험지수",
    "ForestRiskScore": "FMI (임상)",
    "TerrainRiskScore": "TMI (지형)",
})
top5["종합위험지수"] = top5["종합위험지수"].round(3)
top5["FMI (임상)"] = top5["FMI (임상)"].round(2)
top5["TMI (지형)"] = top5["TMI (지형)"].round(2)
st.dataframe(top5, use_container_width=True, hide_index=True)

st.divider()

# ── 지역별 평균 위험지수 + 등급 분포 ──────────────────────────────
col_tbl, col_pie = st.columns([3, 2])

with col_tbl:
    st.subheader("📊 지역별 평균 위험지수")
    region_tbl = (
        display.groupby("region")[["risk_score", "ForestRiskScore", "TerrainRiskScore"]]
        .mean()
        .round(3)
        .rename(columns={
            "risk_score":        "종합위험지수",
            "ForestRiskScore":   "FMI (임상)",
            "TerrainRiskScore":  "TMI (지형)",
        })
        .sort_values("종합위험지수", ascending=False)
        .rename_axis("지역")
        .reset_index()
    )
    st.dataframe(region_tbl, use_container_width=True, hide_index=True)

with col_pie:
    st.subheader("📈 위험 등급 분포")
    display["RiskLevel"] = pd.cut(
        display["risk_score"],
        bins=[0, 0.25, 0.50, 0.75, 1.01],
        labels=["낮음", "보통", "높음", "매우높음"],
    )
    risk_counts = display["RiskLevel"].value_counts()
    fig_pie = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        color=risk_counts.index,
        color_discrete_map={
            "낮음":   "#22c55e",
            "보통":   "#eab308",
            "높음":   "#f97316",
            "매우높음": "#ef4444",
        },
        hole=0.35,
    )
    fig_pie.update_layout(
        margin=dict(t=10, b=0, l=0, r=0),
        height=280,
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig_pie, use_container_width=True)

st.caption(
    "⚠️ 현재 지도는 임상도·DEM 기반 **정적 위험지수**입니다. "
    "기상 API 연동 시 일별 실시간 예측값으로 자동 업데이트됩니다."
)
