import streamlit as st
import pandas as pd
import numpy as np
import folium
import plotly.express as px
from streamlit_folium import st_folium
from pathlib import Path
from datetime import timedelta
from utils.style import apply_dark_theme

st.set_page_config(page_title="위험도 지도", page_icon="🗺️", layout="wide")
apply_dark_theme()
st.title("🗺️ 전국 산불 위험도 지도")

DATA_DIR   = Path(__file__).parents[1] / "data"
MODELS_DIR = Path(__file__).parents[1] / "models"
PRED_PATH  = DATA_DIR / "v3_predictions.csv"

_avail_models = {}
for _stem in ["lgbm_v3a", "xgb_v3a", "lgr_v3a"]:
    _p = MODELS_DIR / f"{_stem}.pkl"
    if _p.exists():
        _avail_models[_stem] = _p

_STEM_LABEL  = {"lgbm_v3a": "LightGBM v3a", "xgb_v3a": "XGBoost v3a", "lgr_v3a": "LogReg v3a"}
HAS_PRED     = PRED_PATH.exists()
HAS_MODEL    = len(_avail_models) > 0
HAS_APIKEY   = "kma" in st.secrets and "api_key" in st.secrets.get("kma", {})
CAN_EXTEND   = HAS_MODEL and HAS_APIKEY

if not HAS_PRED:
    st.warning("**v3_predictions.csv** 파일이 없습니다. `step6.py` 실행 후 `data/` 폴더에 추가하세요.")
    st.stop()


@st.cache_data
def load_static():
    stations = pd.read_csv(DATA_DIR / "asos_stations.csv")
    dem      = pd.read_csv(DATA_DIR / "asos_dem_features.csv")
    imsang   = pd.read_csv(DATA_DIR / "asos_imsangdo_features.csv")
    return (
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


@st.cache_data
def load_preds():
    return pd.read_csv(PRED_PATH, parse_dates=["date"])


def col_label(c):
    c_no = c.replace("_proba", "")
    if c_no.startswith("v2"): return f"v2  {c_no.split('_')[-1]}"
    if c_no.startswith("v3"): return f"v3a {c_no.split('_')[-1]}"
    return c


static_data = load_static()
preds_df    = load_preds()
proba_cols  = [c for c in preds_df.columns if "proba" in c.lower()]

# 2026년 API 연장 캐시 키 (page 4와 동일 규칙)
_preds_max   = preds_df["date"].max()
_since_date  = _preds_max + timedelta(days=1)
_ext_key     = f"ext_preds_{_since_date.date()}"

# 1페이지 "2026년 예측" 모드는 2026-01-01 이후 데이터만 표시
_2026_START  = pd.Timestamp("2026-01-01")

# ── 사이드바 ───────────────────────────────────────────────────────
run_ext_btn  = False
sel_date     = None
model_choice = None

with st.sidebar:
    st.subheader("🔍 설정")

    data_mode = st.radio(
        "예측 기간",
        ["2025년 예측", "2026년 예측 (API)"],
        help=(
            "2025년: 학습된 v3a 모델의 Hold-out Test 예측 결과 (2025-01-01 ~ 2025-09-11)\n\n"
            "2026년: 기상청 API 실시간 확장 예측 (2026-01-01 이후)"
        ),
    )
    st.divider()

    if data_mode == "2025년 예측":
        model_choice = st.selectbox("모델", proba_cols, format_func=col_label)
        date_min = preds_df["date"].min().date()
        date_max = preds_df["date"].max().date()
        sel_date = st.date_input(
            "날짜",
            value=date_max,
            min_value=date_min,
            max_value=date_max,
        )

    elif data_mode == "2026년 예측 (API)":
        _ext_data = st.session_state.get(_ext_key)

        # 2026년 1월 1일 이후 데이터만 추출
        _ext_2026 = (
            _ext_data[_ext_data["date"] >= _2026_START].copy()
            if (_ext_data is not None and not _ext_data.empty)
            else None
        )

        if _ext_2026 is not None and not _ext_2026.empty:
            # ── 캐시 있음 (2026년 데이터 포함): 날짜·모델 선택 ──────
            _ext_proba_cols = [c for c in _ext_2026.columns if "proba" in c.lower()]
            model_choice = st.selectbox(
                "모델", _ext_proba_cols, format_func=col_label,
                help="API 연장 기간은 v3a LightGBM / XGBoost만 제공됩니다.",
            )
            _ext_min_d = _ext_2026["date"].min().date()
            _ext_max_d = _ext_2026["date"].max().date()
            sel_date = st.date_input(
                "날짜",
                value=_ext_max_d,
                min_value=_ext_min_d,
                max_value=_ext_max_d,
            )
            st.caption(f"2026년 API 데이터: {_ext_min_d} ~ {_ext_max_d}")
        elif _ext_data is not None and not _ext_data.empty:
            # ── API 데이터 있지만 2026년 데이터 없음 ─────────────────
            st.warning(
                f"API 데이터가 2025년({_ext_data['date'].max().date()})까지만 있습니다.\n\n"
                "API를 다시 불러오면 2026년 데이터가 포함될 수 있습니다."
            )
            run_ext_btn = st.button(
                "🔄 API 다시 불러오기",
                use_container_width=True,
                type="primary",
            )
        elif CAN_EXTEND:
            st.info(
                "4페이지에서 **📡 API 연장 예측 포함**을 체크하거나,\n"
                "아래 버튼으로 직접 불러오세요."
            )
            run_ext_btn = st.button(
                "🔄 API 연장 데이터 불러오기",
                use_container_width=True,
                type="primary",
            )
        else:
            st.warning("API 연장을 위해 모델 파일(.pkl)과\nAPI 키 등록이 필요합니다.")

    st.divider()
    regions = ["전체"] + sorted(static_data["region"].dropna().unique().tolist())
    selected_region = st.selectbox("시도 선택", regions)
    show_labels = st.checkbox("지점명 표시", value=False)
    st.divider()
    st.caption("색상 기준 (모델 발생 확률)")
    st.markdown(
        "🟢 &lt; 25% &nbsp; "
        "🟡 25–50% &nbsp; "
        "🟠 50–70% &nbsp; "
        "🔴 ≥ 70%"
    )

# ── 2026년 API 연장 실행 (사이드바 밖) ────────────────────────────
if run_ext_btn and CAN_EXTEND:
    try:
        from utils.api import run_extension_pipeline
        _imsang = pd.read_csv(DATA_DIR / "asos_imsangdo_features.csv")
        _dem    = pd.read_csv(DATA_DIR / "asos_dem_features.csv")
        _key    = st.secrets["kma"]["api_key"]

        with st.spinner(f"API 연장 예측 중… ({_since_date.date()} ~ 어제)"):
            ext_df, ext_failed = run_extension_pipeline(
                api_key=_key,
                station_ids=static_data["station_id"].tolist(),
                imsang_df=_imsang,
                dem_df=_dem,
                model_paths=_avail_models,
                since_date=_since_date,
            )
            st.session_state[_ext_key]                  = ext_df
            st.session_state[f"{_ext_key}_failed"]      = ext_failed
        st.rerun()
    except Exception as _e:
        st.error(f"API 연장 오류: {_e}")

# ── map_df 구성 ────────────────────────────────────────────────────
map_df     = static_data.copy()
date_label = ""

if data_mode == "2025년 예측" and sel_date and model_choice:
    day_preds = (
        preds_df[preds_df["date"] == pd.Timestamp(sel_date)]
        [["station_id", model_choice]]
        .rename(columns={model_choice: "proba"})
    )
    map_df     = map_df.merge(day_preds, on="station_id", how="left")
    map_df["map_score"] = map_df["proba"]
    date_label = f"{sel_date}  ·  {col_label(model_choice)}  ·  2025년 예측"

elif data_mode == "2026년 예측 (API)" and sel_date and model_choice:
    # 2026년 이후 데이터만 사용
    _ext_data = st.session_state.get(_ext_key)
    _ext_2026 = (
        _ext_data[_ext_data["date"] >= _2026_START]
        if (_ext_data is not None and not _ext_data.empty)
        else None
    )
    if _ext_2026 is not None and not _ext_2026.empty:
        day_ext = (
            _ext_2026[_ext_2026["date"] == pd.Timestamp(sel_date)]
            [["station_id", model_choice]]
            .rename(columns={model_choice: "proba"})
        )
        map_df     = map_df.merge(day_ext, on="station_id", how="left")
        map_df["map_score"] = map_df["proba"]
        date_label = f"{sel_date}  ·  {col_label(model_choice)}  ·  2026년 API"
    else:
        map_df["map_score"] = np.nan
else:
    map_df["map_score"] = np.nan

display = (
    map_df if selected_region == "전체"
    else map_df[map_df["region"] == selected_region]
)


# ── 색상 / 등급 함수 ───────────────────────────────────────────────
def risk_color(val):
    if pd.isna(val): return "#94a3b8"
    if val >= 0.70:  return "#ef4444"
    if val >= 0.50:  return "#f97316"
    if val >= 0.25:  return "#eab308"
    return "#22c55e"


def risk_label(val):
    if pd.isna(val): return "정보없음"
    if val >= 0.70:  return "매우높음"
    if val >= 0.50:  return "높음"
    if val >= 0.25:  return "보통"
    return "낮음"


# ── 지도 생성 ─────────────────────────────────────────────────────
if date_label:
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
    color = risk_color(val)
    label = risk_label(val)

    fmi_str    = (f"{row.get('ForestRiskScore', float('nan')):.2f}"
                  if not pd.isna(row.get("ForestRiskScore", float("nan"))) else "N/A")
    tmi_str    = (f"{row.get('TerrainRiskScore', float('nan')):.2f}"
                  if not pd.isna(row.get("TerrainRiskScore", float("nan"))) else "N/A")
    elev_str   = (f"{row.get('MeanElevation', float('nan')):.0f}m"
                  if not pd.isna(row.get("MeanElevation", float("nan"))) else "N/A")
    dom_forest = row.get("DominantForest", "N/A") or "N/A"

    if not pd.isna(val):
        score_line  = (f"<b>발생 확률:</b> "
                       f"<span style='color:{color}; font-weight:bold;'>"
                       f"{val*100:.1f}%</span><br>")
        tooltip_txt = f"{row['station_name']} | {val*100:.1f}% | {label}"
    else:
        score_line  = f"<b>발생 확률:</b> <span style='color:#94a3b8;'>데이터 없음</span><br>"
        tooltip_txt = f"{row['station_name']} | 데이터 없음"

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
  <span style="color:#22c55e; font-size:16px;">●</span> 낮음 (~25%)<br>
  <span style="color:#eab308; font-size:16px;">●</span> 보통 (25~50%)<br>
  <span style="color:#f97316; font-size:16px;">●</span> 높음 (50~70%)<br>
  <span style="color:#ef4444; font-size:16px;">●</span> 매우높음 (70%+)
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
    high = int((valid["map_score"] >= 0.70).sum()) if not valid.empty else 0
    st.metric("매우높음 (≥70%)", f"{high}개")
with c3:
    med = int(((valid["map_score"] >= 0.50) & (valid["map_score"] < 0.70)).sum()) if not valid.empty else 0
    st.metric("높음 (50~70%)", f"{med}개")
with c4:
    avg = valid["map_score"].mean() if not valid.empty else float("nan")
    avg_str = f"{avg*100:.1f}%" if pd.notna(avg) else "–"
    st.metric(
        "평균 발생 확률", avg_str,
        help="모델 점수 백분율 (실제 발생확률 아님 — 상대 위험 순위 비교용)",
    )

# ── Top 5 고위험 지점 ─────────────────────────────────────────────
st.subheader("⚠️ 고위험 지점 Top 5")
if not valid.empty and "proba" in valid.columns:
    top5 = valid.nlargest(5, "map_score")[
        ["station_name", "region", "proba", "ForestRiskScore", "TerrainRiskScore"]
    ].rename(columns={
        "station_name": "지점명", "region": "지역",
        "proba": "발생 확률", "ForestRiskScore": "FMI", "TerrainRiskScore": "TMI",
    })
    top5["발생 확률"] = (top5["발생 확률"] * 100).round(1).astype(str) + "%"
    top5["FMI"] = top5["FMI"].round(2)
    top5["TMI"] = top5["TMI"].round(2)
    st.dataframe(top5, use_container_width=True, hide_index=True)
else:
    st.info("날짜를 선택하거나 API 연장 데이터를 불러오면 고위험 지점이 표시됩니다.")

st.divider()

# ── 지역별 평균 + 등급 분포 ─────────────────────────────────────────
col_tbl, col_pie = st.columns([3, 2])

with col_tbl:
    st.subheader("📊 지역별 평균 발생 확률")
    if not valid.empty and "proba" in valid.columns:
        region_tbl = (
            valid.groupby("region")["proba"].mean().reset_index()
            .rename(columns={"region": "지역", "proba": "평균 발생 확률"})
            .sort_values("평균 발생 확률", ascending=False)
        )
        region_tbl["평균 발생 확률"] = (
            (region_tbl["평균 발생 확률"] * 100).round(1).astype(str) + "%"
        )
        st.dataframe(region_tbl, use_container_width=True, hide_index=True)
    else:
        st.info("데이터를 불러오면 지역별 평균이 표시됩니다.")

with col_pie:
    st.subheader("📈 위험 등급 분포")
    if not valid.empty:
        v_copy = valid.copy()
        v_copy["RiskLevel"] = pd.cut(
            v_copy["map_score"],
            bins=[0, 0.25, 0.50, 0.70, 1.01],
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

st.caption(
    "⚠️ 발생 확률은 모델 점수로 실제 발생확률이 아닌 상대적 위험 순위 비교에 활용하세요. "
    "2025년: v3a 모델 테스트 기간 / 2026년: API 기상 데이터 기반 실시간 확장 예측"
)
