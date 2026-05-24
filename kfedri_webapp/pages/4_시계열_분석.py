import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

st.set_page_config(page_title="시계열 분석", page_icon="📈", layout="wide")
st.title("📈 지점별 산불 위험도 시계열")

DATA_DIR = Path(__file__).parents[1] / "data"
PRED_PATH = DATA_DIR / "v3_predictions.csv"


# ── 예측 파일 존재 확인 ────────────────────────────────────────────
if not PRED_PATH.exists():
    st.warning(
        "**v3_predictions.csv** 파일이 없습니다.\n\n"
        "팀원에게 `step6.py` 실행 후 아래 파일을 `data/` 폴더에 추가해달라고 요청하세요.\n\n"
        "```\nD:\\K_FEDRI_v3\\v3_predictions.csv\n```"
    )
    st.stop()


@st.cache_data
def load_data():
    stations = pd.read_csv(DATA_DIR / "asos_stations.csv")
    preds = pd.read_csv(PRED_PATH, parse_dates=["date"])

    # 예측 확률 컬럼 자동 탐지 (LightGBM 우선, 없으면 XGBoost)
    proba_cols = [c for c in preds.columns if "proba" in c.lower()]
    lgbm_cols = [c for c in proba_cols if "light" in c.lower() or "lgb" in c.lower()]
    xgb_cols  = [c for c in proba_cols if "xgb" in c.lower() or "xgboost" in c.lower()]

    return stations, preds, lgbm_cols, xgb_cols, proba_cols


stations, preds, lgbm_cols, xgb_cols, proba_cols = load_data()

if not proba_cols:
    st.error("예측 확률 컬럼을 찾을 수 없습니다. CSV 컬럼명을 확인하세요.")
    st.dataframe(preds.head(3))
    st.stop()

# ── 사이드바 컨트롤 ───────────────────────────────────────────────
with st.sidebar:
    st.subheader("⚙️ 설정")

    # 시도 → 지점 선택
    regions = ["전체"] + sorted(stations["region"].dropna().unique().tolist())
    sel_region = st.selectbox("시도", regions)
    filtered_st = stations if sel_region == "전체" else stations[stations["region"] == sel_region]

    station_opts = {
        f"{r['station_name']} ({int(r['station_id'])})": int(r["station_id"])
        for _, r in filtered_st.iterrows()
    }
    sel_name = st.selectbox("ASOS 지점", list(station_opts.keys()))
    sel_id = station_opts[sel_name]

    st.divider()

    # 모델 선택
    model_choice = st.radio(
        "예측 모델",
        options=proba_cols,
        format_func=lambda c: (
            "LightGBM v3a" if "light" in c.lower() or "lgb" in c.lower()
            else "XGBoost v3a" if "xgb" in c.lower()
            else "LogisticRegression v3a" if "log" in c.lower()
            else c
        ),
    )

    st.divider()

    # 날짜 범위
    date_min = preds["date"].min().date()
    date_max = preds["date"].max().date()
    date_range = st.date_input(
        "날짜 범위",
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max,
    )

    st.divider()

    show_ma = st.checkbox("7일 이동평균 표시", value=True)
    show_fire = st.checkbox("실제 산불 발생 표시", value=True)
    show_avg = st.checkbox("전국 평균 비교", value=False)

# ── 데이터 필터 ───────────────────────────────────────────────────
station_data = preds[preds["station_id"] == sel_id].copy()

if len(date_range) == 2:
    start_d = pd.Timestamp(date_range[0])
    end_d   = pd.Timestamp(date_range[1])
    station_data = station_data[
        (station_data["date"] >= start_d) & (station_data["date"] <= end_d)
    ]

station_data = station_data.sort_values("date")

if station_data.empty:
    st.warning("해당 지점·기간 데이터가 없습니다.")
    st.stop()

# 이동평균
station_data["ma7"] = station_data[model_choice].rolling(7, center=True).mean()

# 실제 산불
fires = station_data[station_data["Y_ignition"] == 1]

# 전국 평균
if show_avg:
    avg_data = (
        preds[preds["date"].between(start_d, end_d)]
        .groupby("date")[model_choice]
        .mean()
        .reset_index()
        .rename(columns={model_choice: "national_avg"})
        .sort_values("date")
    )

# ── 요약 지표 ─────────────────────────────────────────────────────
stn_info = stations[stations["station_id"] == sel_id].iloc[0]
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("지점", stn_info["station_name"])
with c2:
    st.metric("분석 기간", f"{len(station_data)}일")
with c3:
    st.metric("실제 산불 발생", f"{len(fires)}건")
with c4:
    peak_date = station_data.loc[station_data[model_choice].idxmax(), "date"]
    st.metric("최고 위험일", peak_date.strftime("%Y-%m-%d"))

st.divider()

# ── 메인 시계열 차트 ──────────────────────────────────────────────
fig = make_subplots(
    rows=2, cols=1,
    row_heights=[0.75, 0.25],
    shared_xaxes=True,
    vertical_spacing=0.06,
    subplot_titles=("산불 위험도 예측 확률", "실제 산불 발생"),
)

# 예측 확률 (연한 선)
fig.add_trace(
    go.Scatter(
        x=station_data["date"],
        y=station_data[model_choice],
        mode="lines",
        name="예측 확률",
        line=dict(color="#fca5a5", width=1),
        opacity=0.6,
    ),
    row=1, col=1,
)

# 7일 이동평균
if show_ma:
    fig.add_trace(
        go.Scatter(
            x=station_data["date"],
            y=station_data["ma7"],
            mode="lines",
            name="7일 이동평균",
            line=dict(color="#ef4444", width=2.5),
        ),
        row=1, col=1,
    )

# 전국 평균
if show_avg and len(avg_data) > 0:
    avg_data["ma7_avg"] = avg_data["national_avg"].rolling(7, center=True).mean()
    fig.add_trace(
        go.Scatter(
            x=avg_data["date"],
            y=avg_data["ma7_avg"],
            mode="lines",
            name="전국 평균 (7일MA)",
            line=dict(color="#94a3b8", width=1.5, dash="dot"),
        ),
        row=1, col=1,
    )

# 위험 임계선 (0.5)
fig.add_hline(
    y=0.5, row=1, col=1,
    line=dict(color="#f97316", width=1, dash="dash"),
    annotation_text="임계값 0.5",
    annotation_position="right",
)

# 산불 발생 마커 (상단 차트)
if show_fire and len(fires) > 0:
    fig.add_trace(
        go.Scatter(
            x=fires["date"],
            y=fires[model_choice],
            mode="markers",
            name="실제 산불 발생",
            marker=dict(color="#7f1d1d", size=10, symbol="x", line=dict(width=2)),
        ),
        row=1, col=1,
    )

# 하단: 산불 발생 막대
fig.add_trace(
    go.Bar(
        x=station_data["date"],
        y=station_data["Y_ignition"],
        name="산불 발생 (0/1)",
        marker_color="#7f1d1d",
        showlegend=False,
    ),
    row=2, col=1,
)

fig.update_layout(
    height=520,
    hovermode="x unified",
    legend=dict(orientation="h", y=1.05, x=0),
    margin=dict(t=60, b=20, l=60, r=80),
    plot_bgcolor="#fafafa",
)
fig.update_yaxes(title_text="예측 확률", range=[0, 1], row=1, col=1)
fig.update_yaxes(title_text="산불", tickvals=[0, 1], row=2, col=1)
fig.update_xaxes(title_text="날짜", row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# ── 월별 위험도 히트맵 ────────────────────────────────────────────
st.subheader("📅 월별 평균 위험도")

station_data["year"]  = station_data["date"].dt.year
station_data["month"] = station_data["date"].dt.month

heatmap_df = (
    station_data.groupby(["year", "month"])[model_choice]
    .mean()
    .unstack("month")
    .fillna(0)
)

month_labels = ["1월", "2월", "3월", "4월", "5월", "6월",
                "7월", "8월", "9월", "10월", "11월", "12월"]

fig_hm = go.Figure(go.Heatmap(
    z=heatmap_df.values,
    x=[month_labels[m - 1] for m in heatmap_df.columns],
    y=[str(y) for y in heatmap_df.index],
    colorscale=[
        [0.0,  "#f0fdf4"],
        [0.25, "#86efac"],
        [0.5,  "#fde68a"],
        [0.75, "#f97316"],
        [1.0,  "#7f1d1d"],
    ],
    text=[[f"{v:.3f}" for v in row] for row in heatmap_df.values],
    texttemplate="%{text}",
    hovertemplate="<b>%{y}년 %{x}</b><br>평균 위험도: %{z:.3f}<extra></extra>",
    colorbar=dict(title="위험도"),
))
fig_hm.update_layout(height=200, margin=dict(t=10, b=10, l=60, r=80))
st.plotly_chart(fig_hm, use_container_width=True)

# ── 산불 발생일 상세 테이블 ───────────────────────────────────────
if len(fires) > 0:
    st.subheader(f"🔥 실제 산불 발생일 상세 ({len(fires)}건)")
    fire_tbl = fires[["date", model_choice]].copy()
    fire_tbl.columns = ["발생일", "예측 확률"]
    fire_tbl["발생일"] = fire_tbl["발생일"].dt.strftime("%Y-%m-%d")
    fire_tbl["예측 확률"] = fire_tbl["예측 확률"].round(4)
    fire_tbl["탐지 여부"] = fire_tbl["예측 확률"].apply(
        lambda v: "✅ 탐지 (≥0.5)" if v >= 0.5 else "❌ 미탐지 (<0.5)"
    )
    st.dataframe(fire_tbl, use_container_width=True, hide_index=True)
else:
    st.info("해당 기간 내 이 지점의 실제 산불 발생 기록이 없습니다.")
