import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import timedelta
from utils.style import apply_dark_theme, PLOTLY_BG, PLOTLY_PAPER_BG

st.set_page_config(page_title="발생 확률 추이", page_icon="📈", layout="wide")
apply_dark_theme()
st.title("📈 지점별 발생 확률 추이")

DATA_DIR = Path(__file__).parents[1] / "data"
PRED_PATH = DATA_DIR / "v3_predictions.csv"


# ── 예측 파일 존재 확인 ────────────────────────────────────────────
if not PRED_PATH.exists():
    st.warning(
        "**v3_predictions.csv** 파일이 없습니다.\n\n"
        "`step6.py` 실행 후 아래 파일을 `data/` 폴더에 추가하세요.\n\n"
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


stations, preds_base, lgbm_cols, xgb_cols, proba_cols = load_data()

if not proba_cols:
    st.error("예측 확률 컬럼을 찾을 수 없습니다. CSV 컬럼명을 확인하세요.")
    st.dataframe(preds_base.head(3))
    st.stop()

# ── 모델 파일 감지 (API 연장 가능 여부) ──────────────────────────
MODELS_DIR = Path(__file__).parents[1] / "models"
_avail_models = {}
for _stem in ["lgbm_v3a", "xgb_v3a"]:
    _p = MODELS_DIR / f"{_stem}.pkl"
    if _p.exists():
        _avail_models[_stem] = _p
CAN_EXTEND = (
    len(_avail_models) > 0
    and "kma" in st.secrets
    and "api_key" in st.secrets.get("kma", {})
)

# ── 사이드바 Part 1: 지점·모델·옵션 선택 ─────────────────────────
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
    def col_label(c):
        c_no = c.replace("_proba", "")
        if c_no.startswith("v2"):
            return f"v2  {c_no.split('_')[-1]}"
        elif c_no.startswith("v3"):
            return f"v3a {c_no.split('_')[-1]}"
        return c

    model_choice = st.radio(
        "예측 모델",
        options=proba_cols,
        format_func=col_label,
    )

    st.divider()

    show_avg  = st.checkbox("전국 평균 비교", value=False)

    # API 연장 옵션
    if CAN_EXTEND:
        st.divider()
        _last = preds_base["date"].max().date()
        extend_api = st.checkbox(
            "📡 API 연장 예측 포함",
            value=False,
            help=f"{_last + timedelta(days=1)} 이후를 실시간 API로 예측해 이어붙입니다.",
        )
    else:
        extend_api = False

# ── API 연장 처리 (사이드바 밖에서 실행, 이후 날짜범위 위젯에 반영) ──
preds      = preds_base.copy()
ext_banner = None   # 연장 결과 안내 메시지 보관

if extend_api and CAN_EXTEND:
    last_date  = preds["date"].max()
    since_date = last_date + timedelta(days=1)
    cache_key  = f"ext_preds_{since_date.date()}"

    if cache_key not in st.session_state:
        with st.spinner(
            f"API 연장 예측 중… "
            f"({since_date.date()} ~ 어제, {len(_avail_models)}개 모델)"
        ):
            try:
                from utils.api import run_extension_pipeline
                _imsang = pd.read_csv(DATA_DIR / "asos_imsangdo_features.csv")
                _dem    = pd.read_csv(DATA_DIR / "asos_dem_features.csv")
                _key    = st.secrets["kma"]["api_key"]

                ext_df, ext_failed = run_extension_pipeline(
                    api_key=_key,
                    station_ids=stations["station_id"].tolist(),
                    imsang_df=_imsang,
                    dem_df=_dem,
                    model_paths=_avail_models,
                    since_date=since_date,
                )
                st.session_state[cache_key] = ext_df
                st.session_state[f"{cache_key}_failed"] = ext_failed
            except Exception as _e:
                st.error(f"연장 예측 오류: {_e}")
                st.session_state[cache_key] = pd.DataFrame()

    _ext = st.session_state.get(cache_key, pd.DataFrame())
    if not _ext.empty:
        preds = (
            pd.concat([preds, _ext], ignore_index=True, sort=False)
            .drop_duplicates(subset=["station_id", "date"])
            .sort_values(["station_id", "date"])
            .reset_index(drop=True)
        )
        # 연장된 proba 컬럼 반영
        proba_cols = [c for c in preds.columns if "proba" in c.lower()]
        _failed_n  = len(st.session_state.get(f"{cache_key}_failed", []))
        ext_banner = (
            f"📡 API 연장: {since_date.date()} ~ {_ext['date'].max().date()} "
            f"({len(_ext['date'].unique())}일 추가"
            + (f", 실패 {_failed_n}개 지점" if _failed_n else "") + ")"
        )

# ── 사이드바 Part 2: 날짜 범위 (연장된 preds 기준) ───────────────
with st.sidebar:
    st.divider()
    date_min = preds["date"].min().date()
    date_max = preds["date"].max().date()
    date_range = st.date_input(
        "날짜 범위",
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max,
    )

if ext_banner:
    st.info(ext_banner)

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

# % 단위 컬럼 (차트 표시용)
station_data["proba_pct"] = station_data[model_choice] * 100

# 실제 산불 — 날짜 필터와 무관하게 해당 지점 전체 기록 사용
# (날짜 범위 밖 사건도 발생 전 확률 추이 분석에 포함)
all_stn = preds[preds["station_id"] == sel_id].sort_values("date").copy()
all_stn["proba_pct"] = all_stn[model_choice] * 100

if "Y_ignition" in all_stn.columns:
    fires = all_stn[all_stn["Y_ignition"] == 1].copy()
else:
    fires = pd.DataFrame()

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
    avg_data["national_avg_pct"] = avg_data["national_avg"] * 100

# ── 요약 지표 ─────────────────────────────────────────────────────
stn_info = stations[stations["station_id"] == sel_id].iloc[0]
_det_cnt = int((fires[model_choice] >= 0.5).sum()) if len(fires) > 0 else 0
_det_rate = f"{_det_cnt}/{len(fires)}건 ({_det_cnt/len(fires)*100:.0f}%)" if len(fires) > 0 else "해당없음"

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("지점", stn_info["station_name"])
with c2:
    st.metric("분석 기간", f"{len(station_data)}일")
with c3:
    st.metric("실제 산불 발생", f"{len(fires)}건")
with c4:
    st.metric("탐지율 (≥50%)", _det_rate,
              help="실제 산불 발생일 중 모델이 50% 이상을 예측한 비율입니다.")

st.divider()

# ── 산불 발생 전 발생 확률 추이 (메인 차트) ───────────────────────
st.subheader("🔥 산불 발생 전 발생 확률 추이")

if len(fires) == 0:
    if "Y_ignition" not in preds.columns:
        st.warning("예측 데이터에 `Y_ignition` 컬럼이 없습니다. 산불 발생 이력을 분석할 수 없습니다.")
    else:
        st.info(
            f"**{stn_info['station_name']}** 지점은 전체 데이터 기간 동안 산불 발생 기록이 없습니다.\n\n"
            "좌측 사이드바에서 다른 지점을 선택해보세요."
        )
else:
    N = st.slider("발생 전 분석 기간 (일)", min_value=3, max_value=14, value=7, step=1,
                  help="산불 발생일 기준 며칠 전까지 확률을 추적할지 설정합니다.")

    trajectories = []
    for i, fd in enumerate(fires["date"]):
        window = all_stn[
            (all_stn["date"] >= fd - pd.Timedelta(days=N)) &
            (all_stn["date"] <= fd)
        ].copy()
        if window.empty:
            continue
        window["days_before"] = (window["date"] - fd).dt.days
        window["fire_idx"]    = i
        trajectories.append(window[["days_before", "proba_pct", "fire_idx"]])

    if not trajectories:
        st.warning("선택한 날짜 범위 내에 분석 가능한 산불 사건이 없습니다.")
    else:
        traj_df  = pd.concat(trajectories, ignore_index=True)
        avg_traj = (
            traj_df.groupby("days_before")["proba_pct"]
            .mean().reset_index()
            .sort_values("days_before")
        )

        # X축 레이블: D-7, D-6 … 발생일
        avg_traj["label"] = avg_traj["days_before"].apply(
            lambda d: "발생일" if d == 0 else f"D{d}"
        )

        def bar_color(p):
            if p >= 70: return "#ef4444"
            if p >= 50: return "#f97316"
            if p >= 25: return "#eab308"
            return "#22c55e"

        fig_pre = go.Figure()

        # 개별 이벤트 선 (2건 이상일 때만)
        if len(fires) > 1:
            for i, grp in traj_df.groupby("fire_idx"):
                grp = grp.sort_values("days_before")
                fig_pre.add_trace(go.Scatter(
                    x=grp["days_before"],
                    y=grp["proba_pct"],
                    mode="lines",
                    line=dict(color="#fca5a5", width=1),
                    opacity=0.35,
                    showlegend=(i == 0),
                    name="개별 산불 사건",
                ))

        # 평균 막대
        fig_pre.add_trace(go.Bar(
            x=avg_traj["days_before"],
            y=avg_traj["proba_pct"],
            name="평균 발생 확률",
            marker_color=[bar_color(p) for p in avg_traj["proba_pct"]],
            text=[f"{p:.1f}%" for p in avg_traj["proba_pct"]],
            textposition="outside",
        ))

        # 50% 탐지 기준선
        fig_pre.add_hline(
            y=50, line_dash="dash", line_color="#94a3b8",
            annotation_text="탐지 기준 (50%)",
            annotation_position="top right",
        )

        y_max = max(avg_traj["proba_pct"].max() * 1.25, 60)
        fig_pre.update_layout(
            height=380,
            xaxis=dict(
                title="발생 전 일수",
                tickmode="array",
                tickvals=avg_traj["days_before"].tolist(),
                ticktext=avg_traj["label"].tolist(),
            ),
            yaxis=dict(title="발생 확률 (%)", range=[0, y_max]),
            legend=dict(orientation="h", y=1.08),
            margin=dict(t=40, b=40, l=60, r=80),
            bargap=0.25,
        )
        st.plotly_chart(fig_pre, use_container_width=True)

        # 인사이트 한 줄 요약
        d0   = avg_traj[avg_traj["days_before"] == 0]["proba_pct"]
        d_n  = avg_traj[avg_traj["days_before"] == avg_traj["days_before"].min()]["proba_pct"]
        det  = int((fires[model_choice] >= 0.5).sum())
        if len(d0) and len(d_n):
            rise = float(d0.iloc[0]) - float(d_n.iloc[0])
            st.caption(
                f"발생일 평균 확률 **{d0.iloc[0]:.1f}%** &nbsp;|&nbsp; "
                f"{N}일 전 대비 **{rise:+.1f}%p** &nbsp;|&nbsp; "
                f"탐지율(≥50%) **{det}/{len(fires)}건 "
                f"({det/len(fires)*100:.0f}%)**"
            )

with st.expander("⚠️ 발생 확률 수치 해석 주의"):
    st.caption(
        "표시된 발생 확률은 모델 점수(predict_proba)를 백분율로 변환한 값입니다. "
        "class_weight / scale_pos_weight 적용으로 수치 자체를 실제 발생확률로 해석하기 어렵습니다. "
        "지점 간 상대적 위험 순위 비교 또는 탐지 능력 평가에 활용하세요."
    )

st.divider()

# ── 월별 평균 발생 확률 히트맵 ────────────────────────────────────
st.subheader("📅 월별 평균 발생 확률")

station_data["year"]  = station_data["date"].dt.year
station_data["month"] = station_data["date"].dt.month

heatmap_df = (
    station_data.groupby(["year", "month"])[model_choice]
    .mean().unstack("month").fillna(0)
) * 100

month_labels = ["1월","2월","3월","4월","5월","6월",
                "7월","8월","9월","10월","11월","12월"]

fig_hm = go.Figure(go.Heatmap(
    z=heatmap_df.values,
    x=[month_labels[m - 1] for m in heatmap_df.columns],
    y=[str(y) for y in heatmap_df.index],
    zmin=0, zmax=100,
    colorscale=[
        [0.0, "#f0fdf4"], [0.25, "#86efac"],
        [0.5, "#fde68a"], [0.75, "#f97316"], [1.0, "#7f1d1d"],
    ],
    text=[[f"{v:.1f}%" for v in row] for row in heatmap_df.values],
    texttemplate="%{text}",
    hovertemplate="<b>%{y}년 %{x}</b><br>평균 발생 확률: %{z:.1f}%<extra></extra>",
    colorbar=dict(title="발생 확률 (%)"),
))
fig_hm.update_layout(height=200, margin=dict(t=10, b=10, l=60, r=80))
st.plotly_chart(fig_hm, use_container_width=True)

# ── 산불 발생일 상세 테이블 ───────────────────────────────────────
if len(fires) > 0:
    st.subheader(f"🔥 실제 산불 발생일 상세 ({len(fires)}건)")
    fire_tbl = fires[["date", model_choice]].copy()
    fire_tbl.columns = ["발생일", "발생 확률"]
    fire_tbl["발생일"]    = fire_tbl["발생일"].dt.strftime("%Y-%m-%d")
    fire_tbl["발생 확률"] = (fire_tbl["발생 확률"] * 100).round(1)
    fire_tbl["탐지 여부"] = fire_tbl["발생 확률"].apply(
        lambda v: "✅ 탐지 (≥50%)" if v >= 50 else "❌ 미탐지 (<50%)"
    )
    st.dataframe(fire_tbl, use_container_width=True, hide_index=True)
