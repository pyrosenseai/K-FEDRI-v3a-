import streamlit as st
import pandas as pd
import numpy as np
import folium
import plotly.express as px
from streamlit_folium import st_folium
from pathlib import Path
from utils.style import apply_dark_theme
from utils.api import (fetch_all_stations, build_weather_features,
                       build_v3a_features, predict_today, load_model)

st.set_page_config(page_title="위험도 지도", page_icon="🗺️", layout="wide")
apply_dark_theme()
st.title("🗺️ 전국 산불 위험도 지도")

DATA_DIR   = Path(__file__).parents[1] / "data"
MODELS_DIR = Path(__file__).parents[1] / "models"
PRED_PATH  = DATA_DIR / "v3_predictions.csv"

_avail_models = {}
for _stem in ["lgbm_v3a", "xgb_v3a"]:
    _p = MODELS_DIR / f"{_stem}.pkl"
    if _p.exists():
        _avail_models[_stem] = _p

_STEM_LABEL   = {"lgbm_v3a": "LightGBM v3a", "xgb_v3a": "XGBoost v3a"}
HAS_PRED      = PRED_PATH.exists()
HAS_MODEL     = len(_avail_models) > 0
HAS_APIKEY    = "kma" in st.secrets and "api_key" in st.secrets.get("kma", {})
CAN_REALTIME  = HAS_MODEL and HAS_APIKEY


@st.cache_data
def load_static():
    stations = pd.read_csv(DATA_DIR / "asos_stations.csv")
    dem      = pd.read_csv(DATA_DIR / "asos_dem_features.csv")
    imsang   = pd.read_csv(DATA_DIR / "asos_imsangdo_features.csv")
    merged = (
        stations[["station_id", "station_name", "region", "lat", "lon", "altitude_m"]]
        .merge(dem[["station_id", "TerrainRiskScore", "MeanElevation",
                    "MeanSlope", "DominantAspect"]], on="station_id", how="left")
        .merge(imsang[["station_id", "ForestRiskScore", "ConiferRatio",
                       "BroadleafRatio", "DominantForest"]], on="station_id", how="left")
    )
    fmi = merged["ForestRiskScore"].fillna(0)
    tmi = merged["TerrainRiskScore"].fillna(0)
    fmi_norm = (fmi - fmi.min()) / (fmi.max() - fmi.min() + 1e-8)
    tmi_norm = (tmi - tmi.min()) / (tmi.max() - tmi.min() + 1e-8)
    merged["risk_score"] = fmi_norm * 0.65 + tmi_norm * 0.35
    return merged


@st.cache_data
def load_preds():
    return pd.read_csv(PRED_PATH, parse_dates=["date"])


def col_label(c):
    c_no = c.replace("_proba", "")
    if c_no.startswith("v2"): return f"v2  {c_no.split('_')[-1]}"
    if c_no.startswith("v3"): return f"v3a {c_no.split('_')[-1]}"
    return c


static_data = load_static()

# ── 사이드바 ───────────────────────────────────────────────────────
run_map_btn  = False
sel_date     = None
model_choice = None
pred_source  = None
rt_model_sel = None

with st.sidebar:
    st.subheader("🔍 설정")

    data_mode = st.radio(
        "지도 모드",
        ["정적 위험지수", "예측 발생 확률"],
        help="정적: 임상·지형 기반 고정 위험지수 | 예측: ML 모델 발생 확률",
    )
    st.divider()

    if data_mode == "예측 발생 확률":
        rt_available = "rt_all_results" in st.session_state
        source_opts  = []
        if HAS_PRED:
            source_opts.append("과거 날짜 선택")
        if rt_available or CAN_REALTIME:
            source_opts.append("실시간 API")

        if not source_opts:
            st.warning("v3_predictions.csv 또는\n모델+API 키가 필요합니다.")
        elif len(source_opts) == 1:
            pred_source = source_opts[0]
        else:
            pred_source = st.radio("예측 소스", source_opts)

        if pred_source == "과거 날짜 선택":
            preds_df   = load_preds()
            proba_cols = [c for c in preds_df.columns if "proba" in c.lower()]
            model_choice = st.selectbox("모델", proba_cols, format_func=col_label)
            date_min = preds_df["date"].min().date()
            date_max = preds_df["date"].max().date()
            sel_date = st.date_input("날짜", value=date_max,
                                     min_value=date_min, max_value=date_max)

        elif pred_source == "실시간 API":
            rt_results = st.session_state.get("rt_all_results", {})
            if rt_results:
                rt_model_sel = st.selectbox("모델", list(rt_results.keys()))
                _rt_date = list(rt_results.values())[0]["date"].max().date()
                st.caption(f"예측 날짜: {_rt_date}")
                st.caption("홈 화면 예측 결과를 사용 중입니다.")
            elif CAN_REALTIME:
                st.info("홈 화면 '⚡ 실시간 예측'을 먼저\n실행하거나 아래 버튼으로 직접 실행하세요.")
                run_map_btn = st.button("🔄 전국 예측 실행",
                                        use_container_width=True, type="primary")
            else:
                st.warning("실시간 예측을 위해 모델 파일과\nAPI 키 등록이 필요합니다.")

        st.divider()

    regions = ["전체"] + sorted(static_data["region"].dropna().unique().tolist())
    selected_region = st.selectbox("시도 선택", regions)
    show_labels = st.checkbox("지점명 표시", value=False)
    st.divider()

    if data_mode == "정적 위험지수":
        st.caption("색상 기준 (임상+지형 복합위험지수)")
        st.markdown(
            "<span style='color:#22c55e'>●</span> 낮음 &nbsp; "
            "<span style='color:#eab308'>●</span> 보통 &nbsp; "
            "<span style='color:#f97316'>●</span> 높음 &nbsp; "
            "<span style='color:#ef4444'>●</span> 매우높음",
            unsafe_allow_html=True,
        )
    else:
        st.caption("색상 기준 (모델 발생 확률)")
        st.markdown(
            "<span style='color:#22c55e'>●</span> ~25% &nbsp; "
            "<span style='color:#eab308'>●</span> 25~50% &nbsp; "
            "<span style='color:#f97316'>●</span> 50~70% &nbsp; "
            "<span style='color:#ef4444'>●</span> 70%+",
            unsafe_allow_html=True,
        )

# ── 실시간 API 실행 (사이드바 밖) ─────────────────────────────────
if (data_mode == "예측 발생 확률"
        and pred_source == "실시간 API"
        and run_map_btn
        and CAN_REALTIME):
    try:
        imsang_df = pd.read_csv(DATA_DIR / "asos_imsangdo_features.csv")
        dem_df    = pd.read_csv(DATA_DIR / "asos_dem_features.csv")
        api_key   = st.secrets["kma"]["api_key"]
        stn_ids   = static_data["station_id"].tolist()

        progress_bar = st.progress(0, text="API 호출 중…")

        def on_progress(done, total):
            progress_bar.progress(done / total, text=f"API 호출 중… {done}/{total}")

        with st.spinner(f"전국 예측 중 ({len(_avail_models)}개 모델, 약 20초)"):
            raw_df, failed = fetch_all_stations(stn_ids, api_key,
                                                progress_callback=on_progress)
            weather_df  = build_weather_features(raw_df)
            features_df = build_v3a_features(weather_df, imsang_df, dem_df)

            all_results = {}
            for mname, mpath in _avail_models.items():
                mdl = load_model(mpath)
                res = predict_today(features_df, mdl)
                all_results[_STEM_LABEL[mname]] = res

        progress_bar.empty()
        st.session_state["rt_all_results"] = all_results
        st.session_state["rt_failed"]      = failed
        st.rerun()
    except Exception as e:
        st.error(f"예측 오류: {e}")

# ── map_df 구성 ────────────────────────────────────────────────────
is_pred  = (data_mode == "예측 발생 확률")
map_df   = static_data.copy()
date_label = ""

if is_pred and pred_source == "과거 날짜 선택" and sel_date and model_choice:
    preds_df  = load_preds()
    day_preds = (
        preds_df[preds_df["date"] == pd.Timestamp(sel_date)]
        [["station_id", model_choice]]
        .rename(columns={model_choice: "proba"})
    )
    map_df     = map_df.merge(day_preds, on="station_id", how="left")
    map_df["map_score"] = map_df["proba"]
    date_label = f"{sel_date}  ·  {col_label(model_choice)}"

elif is_pred and pred_source == "실시간 API":
    rt_results = st.session_state.get("rt_all_results", {})
    if rt_results and rt_model_sel and rt_model_sel in rt_results:
        res_df = rt_results[rt_model_sel][["station_id", "date", "proba"]]
        map_df = map_df.merge(res_df[["station_id", "proba"]], on="station_id", how="left")
        map_df["map_score"] = map_df["proba"]
        _rt_d  = res_df["date"].max().date()
        date_label = f"{_rt_d}  ·  {rt_model_sel}  ·  실시간"
    else:
        map_df["map_score"] = np.nan

else:
    map_df["map_score"] = map_df["risk_score"]

display = (
    map_df if selected_region == "전체"
    else map_df[map_df["region"] == selected_region]
)


# ── 색상 / 등급 함수 ───────────────────────────────────────────────
def risk_color(val, pred=False):
    if pd.isna(val): return "#94a3b8"
    if pred:
        if val >= 0.70: return "#ef4444"
        if val >= 0.50: return "#f97316"
        if val >= 0.25: return "#eab308"
        return "#22c55e"
    else:
        if val >= 0.75: return "#ef4444"
        if val >= 0.50: return "#f97316"
        if val >= 0.25: return "#eab308"
        return "#22c55e"


def risk_label(val, pred=False):
    if pd.isna(val): return "정보없음"
    if pred:
        if val >= 0.70: return "매우높음"
        if val >= 0.50: return "높음"
        if val >= 0.25: return "보통"
        return "낮음"
    else:
        if val >= 0.75: return "매우높음"
        if val >= 0.50: return "높음"
        if val >= 0.25: return "보통"
        return "낮음"


# ── 지도 생성 ─────────────────────────────────────────────────────
if is_pred and date_label:
    st.caption(f"📅 표시 중: **{date_label}**")

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

    val   = row["map_score"]
    color = risk_color(val, pred=is_pred)
    label = risk_label(val, pred=is_pred)

    fmi_str = (f"{row.get('ForestRiskScore', float('nan')):.2f}"
               if not pd.isna(row.get("ForestRiskScore", float("nan"))) else "N/A")
    tmi_str = (f"{row.get('TerrainRiskScore', float('nan')):.2f}"
               if not pd.isna(row.get("TerrainRiskScore", float("nan"))) else "N/A")
    elev_str = (f"{row.get('MeanElevation', float('nan')):.0f}m"
                if not pd.isna(row.get("MeanElevation", float("nan"))) else "N/A")
    dom_forest = row.get("DominantForest", "N/A") or "N/A"

    if is_pred and not pd.isna(val):
        score_line  = (f"<b>발생 확률:</b> "
                       f"<span style='color:{color}; font-weight:bold;'>"
                       f"{val*100:.1f}%</span><br>")
        tooltip_txt = f"{row['station_name']} | {val*100:.1f}% | {label}"
    else:
        sv = val if not pd.isna(val) else 0
        score_line  = (f"<b>위험지수:</b> "
                       f"<span style='color:{color}; font-weight:bold;'>"
                       f"{label} ({sv:.2f})</span><br>")
        tooltip_txt = f"{row['station_name']} | {label} ({sv:.2f})"

    popup_html = f"""
    <div style="font-family:sans-serif; min-width:210px; font-size:13px;">
        <b style="font-size:15px;">{row['station_name']} (#{int(row['station_id'])})</b><br>
        <span style="color:#64748b;">{row['region']}</span>
        <hr style="margin:6px 0; border-color:#e2e8f0;">
        {score_line}
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
        tooltip=tooltip_txt,
    ).add_to(m)

    if show_labels:
        folium.Marker(
            location=[row["lat"], row["lon"]],
            icon=folium.DivIcon(
                html=(f'<div style="font-size:9px; font-weight:bold; '
                      f'white-space:nowrap; color:#1e293b; '
                      f'text-shadow:1px 1px 2px white;">{row["station_name"]}</div>'),
                icon_size=(60, 12),
                icon_anchor=(0, 0),
            ),
        ).add_to(m)

legend_html = """
<div style="position:fixed; bottom:30px; left:30px; z-index:1000;
     background:#334155; padding:12px 16px; border-radius:8px;
     color:rgb(255,255,255);
     box-shadow:0 2px 8px rgba(0,0,0,0.18); font-family:sans-serif; font-size:13px;">
  <b>위험 등급</b><br>
  <span style="color:#22c55e; font-size:16px;">●</span> 낮음<br>
  <span style="color:#eab308; font-size:16px;">●</span> 보통<br>
  <span style="color:#f97316; font-size:16px;">●</span> 높음<br>
  <span style="color:#ef4444; font-size:16px;">●</span> 매우높음
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))
st_folium(m, width=None, height=600, use_container_width=True)

# ── 하단 통계 ─────────────────────────────────────────────────────
st.divider()
valid = display[display["map_score"].notna()]

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("표시 지점", f"{len(display)}개")
with c2:
    thresh_hi = 0.70 if is_pred else 0.75
    high = int((valid["map_score"] >= thresh_hi).sum())
    st.metric("매우높음" + (" (≥70%)" if is_pred else ""), f"{high}개")
with c3:
    med = int(((valid["map_score"] >= 0.50) & (valid["map_score"] < thresh_hi)).sum())
    st.metric("높음" + (" (50~70%)" if is_pred else ""), f"{med}개")
with c4:
    avg = valid["map_score"].mean()
    avg_str = f"{avg*100:.1f}%" if is_pred else f"{avg:.2f}"
    st.metric(
        "평균 발생 확률" if is_pred else "평균 위험지수",
        avg_str,
        help="모델 점수 백분율 (실제 발생확률 아님)" if is_pred else None,
    )

# ── Top 5 ─────────────────────────────────────────────────────────
st.subheader("⚠️ 고위험 지점 Top 5")
if is_pred and "proba" in valid.columns:
    top5 = valid.nlargest(5, "map_score")[
        ["station_name", "region", "proba", "ForestRiskScore", "TerrainRiskScore"]
    ].rename(columns={
        "station_name": "지점명", "region": "지역",
        "proba": "발생 확률", "ForestRiskScore": "FMI", "TerrainRiskScore": "TMI",
    })
    top5["발생 확률"] = (top5["발생 확률"] * 100).round(1).astype(str) + "%"
    top5["FMI"] = top5["FMI"].round(2)
    top5["TMI"] = top5["TMI"].round(2)
else:
    top5 = display.nlargest(5, "map_score")[
        ["station_name", "region", "risk_score", "ForestRiskScore", "TerrainRiskScore"]
    ].rename(columns={
        "station_name": "지점명", "region": "지역",
        "risk_score": "종합위험지수", "ForestRiskScore": "FMI (임상)", "TerrainRiskScore": "TMI (지형)",
    })
    top5["종합위험지수"] = top5["종합위험지수"].round(3)
    top5["FMI (임상)"] = top5["FMI (임상)"].round(2)
    top5["TMI (지형)"] = top5["TMI (지형)"].round(2)
st.dataframe(top5, use_container_width=True, hide_index=True)

st.divider()

# ── 지역별 평균 + 등급 분포 ─────────────────────────────────────────
col_tbl, col_pie = st.columns([3, 2])

with col_tbl:
    st.subheader("📊 지역별 평균" + (" 발생 확률" if is_pred else " 위험지수"))
    if is_pred and "proba" in valid.columns:
        region_tbl = (
            valid.groupby("region")["proba"].mean().reset_index()
            .rename(columns={"region": "지역", "proba": "평균 발생 확률"})
            .sort_values("평균 발생 확률", ascending=False)
        )
        region_tbl["평균 발생 확률"] = (
            (region_tbl["평균 발생 확률"] * 100).round(1).astype(str) + "%"
        )
    else:
        region_tbl = (
            valid.groupby("region")[["risk_score", "ForestRiskScore", "TerrainRiskScore"]]
            .mean().round(3)
            .rename(columns={
                "risk_score": "종합위험지수",
                "ForestRiskScore": "FMI (임상)",
                "TerrainRiskScore": "TMI (지형)",
            })
            .sort_values("종합위험지수", ascending=False)
            .rename_axis("지역").reset_index()
        )
    st.dataframe(region_tbl, use_container_width=True, hide_index=True)

with col_pie:
    st.subheader("📈 위험 등급 분포")
    bins   = [0, 0.25, 0.50, 0.70, 1.01] if is_pred else [0, 0.25, 0.50, 0.75, 1.01]
    v_copy = valid.copy()
    v_copy["RiskLevel"] = pd.cut(
        v_copy["map_score"], bins=bins,
        labels=["낮음", "보통", "높음", "매우높음"],
    )
    risk_counts = v_copy["RiskLevel"].value_counts()
    fig_pie = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        color=risk_counts.index,
        color_discrete_map={
            "낮음": "#22c55e", "보통": "#eab308",
            "높음": "#f97316", "매우높음": "#ef4444",
        },
        hole=0.35,
    )
    fig_pie.update_layout(
        margin=dict(t=10, b=0, l=0, r=0),
        height=280,
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig_pie, use_container_width=True)

if is_pred:
    st.caption(
        "⚠️ 발생 확률은 모델 점수로 실제 발생확률이 아닌 상대적 위험 순위 비교에 활용하세요."
    )
else:
    st.caption(
        "⚠️ 현재 지도는 임상도·DEM 기반 정적 위험지수입니다. "
        "'예측 발생 확률' 모드로 전환하면 ML 모델 예측값이 표시됩니다."
    )
