import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from utils.style import apply_dark_theme
from utils.api import (fetch_all_stations, build_weather_features,
                       build_v3a_features, predict_today, load_model, V3A_FEATURE_COLS)

st.set_page_config(
    page_title="K-FEDRI | 산불 위험도 예측",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_dark_theme()

DATA_DIR   = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"
PRED_PATH  = DATA_DIR / "v3_predictions.csv"

# 실시간 예측에 사용할 모델 파일 (있는 것 모두 감지)
AVAILABLE_MODELS = {}
for _label, _stem in [("LightGBM v3a", "lgbm_v3a"), ("XGBoost v3a", "xgb_v3a")]:
    _p = MODELS_DIR / f"{_stem}.pkl"
    if _p.exists():
        AVAILABLE_MODELS[_label] = _p
HAS_MODEL = len(AVAILABLE_MODELS) > 0
HAS_APIKEY = "kma" in st.secrets and "api_key" in st.secrets.get("kma", {})


@st.cache_data
def load_stations():
    return pd.read_csv(DATA_DIR / "asos_stations.csv")


@st.cache_data
def load_preds():
    return pd.read_csv(PRED_PATH, parse_dates=["date"])


stations = load_stations()
has_preds = PRED_PATH.exists()
if has_preds:
    preds = load_preds()
    proba_cols = [c for c in preds.columns if "proba" in c.lower()]
else:
    preds, proba_cols = None, []


def col_label(c):
    c_no = c.replace("_proba", "")
    if c_no.startswith("v2"):
        return f"v2  {c_no.split('_')[-1]}"
    elif c_no.startswith("v3"):
        return f"v3a {c_no.split('_')[-1]}"
    return c


# ── 사이드바 ──────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("⚙️ 설정")

    if has_preds and proba_cols:
        model_choice = st.radio(
            "예측 모델",
            options=proba_cols,
            format_func=col_label,
        )
        st.divider()
        date_min = preds["date"].min().date()
        date_max = preds["date"].max().date()
        sel_date = st.date_input(
            "날짜 선택",
            value=date_max,
            min_value=date_min,
            max_value=date_max,
        )
        st.caption(f"데이터 범위: {date_min} ~ {date_max}")
    else:
        st.info("예측 데이터 없음\n\n`v3_predictions.csv`를 data/ 폴더에 추가하면\n실시간 발생 확률이 표시됩니다.")

# ── 헤더 ──────────────────────────────────────────────────────────
st.title("🔥 K-FEDRI 산불 위험도 예측 시스템")
st.markdown(
    "**K**orean **F**orest fir**E D**anger **R**ating **I**ndex v3a &nbsp;|&nbsp; "
    "기상청 ASOS 97개 지점 &nbsp;|&nbsp; 임상도·DEM 피처 통합 &nbsp;|&nbsp; "
    "LightGBM / XGBoost ML 모델"
)
st.divider()

# ── 예측 데이터가 있는 경우 ────────────────────────────────────────
if has_preds and proba_cols:
    day_df = (
        preds[preds["date"] == pd.Timestamp(sel_date)]
        .merge(stations[["station_id", "station_name", "region"]], on="station_id", how="left")
    )

    if day_df.empty:
        st.warning(f"{sel_date} 날짜 데이터가 없습니다.")
        st.stop()

    avg_prob   = day_df[model_choice].mean()
    high_cnt   = int((day_df[model_choice] >= 0.5).sum())
    top_row    = day_df.loc[day_df[model_choice].idxmax()]
    fire_cnt   = int(day_df["Y_ignition"].sum()) if "Y_ignition" in day_df.columns else 0

    # ── 핵심 지표 ─────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("선택 날짜", str(sel_date))
    with c2:
        st.metric("전국 평균 발생 확률", f"{avg_prob*100:.1f}%")
    with c3:
        st.metric("고위험 지점 (≥50%)", f"{high_cnt}개")
    with c4:
        st.metric("최고 위험 지점", top_row["station_name"])
    with c5:
        st.metric("최고 발생 확률", f"{top_row[model_choice]*100:.1f}%",
                  delta="🔥 산불 발생" if fire_cnt > 0 else None)

    st.divider()

    # ── 레이아웃: 테이블 + 지역별 차트 ─────────────────────────
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.subheader(f"📋 발생 확률 상위 지점 ({col_label(model_choice)})")
        top_tbl = (
            day_df[["station_name", "region", model_choice, "Y_ignition"]]
            .sort_values(model_choice, ascending=False)
            .head(15)
            .rename(columns={
                "station_name": "지점명",
                "region":       "지역",
                model_choice:   "발생 확률",
                "Y_ignition":   "실제 산불",
            })
        )
        top_tbl["발생 확률"] = (top_tbl["발생 확률"] * 100).round(1)
        top_tbl["실제 산불"] = top_tbl["실제 산불"].apply(
            lambda v: "🔥" if v == 1 else ""
        )

        def highlight_prob(s):
            styles = []
            for v in s:
                if v >= 70:
                    styles.append("color:#ef4444; font-weight:bold")
                elif v >= 50:
                    styles.append("color:#f97316; font-weight:bold")
                else:
                    styles.append("")
            return styles

        st.dataframe(
            top_tbl.style.apply(highlight_prob, subset=["발생 확률"]),
            use_container_width=True,
            hide_index=True,
        )

    with col_r:
        st.subheader("🗺️ 지역별 평균 발생 확률")
        region_avg = (
            day_df.groupby("region")[model_choice]
            .mean()
            .reset_index()
            .rename(columns={"region": "지역", model_choice: "평균 확률"})
            .sort_values("평균 확률", ascending=True)
        )
        region_avg["평균 확률"] = region_avg["평균 확률"] * 100
        fig_bar = px.bar(
            region_avg,
            x="평균 확률", y="지역",
            orientation="h",
            color="평균 확률",
            color_continuous_scale=["#22c55e", "#eab308", "#f97316", "#ef4444"],
            range_color=[0, 100],
            text="평균 확률",
        )
        fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_bar.update_layout(
            height=420,
            coloraxis_showscale=False,
            margin=dict(t=10, b=10, r=60),
            xaxis=dict(
                title="발생 확률 (%)",
                range=[0, max(region_avg["평균 확률"].max() * 1.2, 5)],
            ),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

# ── 예측 데이터 없을 때: 정적 지표 ───────────────────────────────
else:
    dem    = pd.read_csv(DATA_DIR / "asos_dem_features.csv")
    imsang = pd.read_csv(DATA_DIR / "asos_imsangdo_features.csv")
    merged = (
        stations[["station_id", "station_name", "region"]]
        .merge(dem[["station_id", "TerrainRiskScore"]], on="station_id", how="left")
        .merge(imsang[["station_id", "ForestRiskScore"]], on="station_id", how="left")
    )
    fmi = merged["ForestRiskScore"].fillna(0)
    tmi = merged["TerrainRiskScore"].fillna(0)
    fmi_norm = (fmi - fmi.min()) / (fmi.max() - fmi.min() + 1e-8)
    tmi_norm = (tmi - tmi.min()) / (tmi.max() - tmi.min() + 1e-8)
    merged["StaticRisk"] = fmi_norm * 0.65 + tmi_norm * 0.35

    high_risk = int((merged["StaticRisk"] >= 0.75).sum())
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("관측 지점", "97개")
    with c2: st.metric("최고 ROC-AUC", "0.8781", "XGBoost v3a")
    with c3: st.metric("Top5% Recall", "0.420",  "LightGBM v3a")
    with c4: st.metric("입력 피처",   "29개",    "기상+임상+지형")
    with c5: st.metric("고위험 지점", f"{high_risk}개", "정적 지형·임상 기준")
    st.divider()
    st.info("📂 `v3_predictions.csv`를 data/ 폴더에 추가하면 일별 예측 확률이 표시됩니다.")

# ── 실시간 예측 섹션 ──────────────────────────────────────────────
st.subheader("⚡ 실시간 예측")

if not HAS_MODEL:
    st.warning(
        "모델 파일이 없습니다. `models/lgbm_v3a.pkl` 또는 `models/xgb_v3a.pkl`을 추가하면 활성화됩니다.",
        icon="🔒",
    )
elif not HAS_APIKEY:
    st.warning(
        "API 키가 등록되지 않았습니다. `.streamlit/secrets.toml`에 아래 내용을 추가하세요.\n"
        "```toml\n[kma]\napi_key = \"YOUR_HUB_API_KEY\"\n```",
        icon="🔑",
    )
else:
    # ── 모델 선택 + 실행 버튼 ────────────────────────────────────
    ctrl_l, ctrl_r = st.columns([2, 1])
    with ctrl_l:
        if len(AVAILABLE_MODELS) > 1:
            rt_model_sel = st.radio(
                "실시간 예측 모델",
                list(AVAILABLE_MODELS.keys()),
                horizontal=True,
                help="API 데이터는 한 번만 호출하고, 선택한 모델로 즉시 전환합니다.",
            )
        else:
            rt_model_sel = list(AVAILABLE_MODELS.keys())[0]
            st.caption(f"모델: `{rt_model_sel}`  |  API: 기상청 API허브 ASOS 일자료")
    with ctrl_r:
        run_btn = st.button("🔄 오늘 전국 예측 실행", type="primary", use_container_width=True)

    # ── 실행 시: API 1회 호출 → 모든 모델 예측 ──────────────────
    if run_btn:
        try:
            imsang_df = pd.read_csv(DATA_DIR / "asos_imsangdo_features.csv")
            dem_df    = pd.read_csv(DATA_DIR / "asos_dem_features.csv")
            api_key   = st.secrets["kma"]["api_key"]
            stn_ids   = stations["station_id"].tolist()

            progress_bar = st.progress(0, text="API 호출 중…")

            def on_progress(done, total):
                progress_bar.progress(done / total,
                    text=f"API 호출 중… {done}/{total} 지점")

            with st.spinner(f"기상청 API 수집 → {len(AVAILABLE_MODELS)}개 모델 예측 중 (약 20초)"):
                # ① 기상 데이터 수집 (한 번만)
                raw_df, failed = fetch_all_stations(
                    stn_ids, api_key, progress_callback=on_progress,
                )
                weather_df  = build_weather_features(raw_df)
                features_df = build_v3a_features(weather_df, imsang_df, dem_df)

                # ② 모든 가용 모델로 예측
                all_results = {}
                for mname, mpath in AVAILABLE_MODELS.items():
                    mdl = load_model(mpath)
                    res = predict_today(features_df, mdl)
                    all_results[mname] = res

            progress_bar.empty()
            st.session_state["rt_all_results"] = all_results
            st.session_state["rt_failed"]      = failed

        except Exception as e:
            st.error(f"예측 실행 중 오류: {e}")

    # ── 결과 표시 (session_state 유지 → 모델 전환 시 재실행 불필요) ──
    if "rt_all_results" in st.session_state:
        all_results = st.session_state["rt_all_results"]
        failed      = st.session_state.get("rt_failed", [])

        # 선택 모델이 결과에 없으면 첫 번째로 fallback
        _cur = rt_model_sel if rt_model_sel in all_results else list(all_results.keys())[0]
        result_df = all_results[_cur].merge(
            stations[["station_id", "station_name", "region"]],
            on="station_id", how="left",
        )

        today    = result_df["date"].max().date()
        avg_p    = result_df["proba"].mean()
        high_cnt = int((result_df["proba"] >= 0.5).sum())
        top_row  = result_df.loc[result_df["proba"].idxmax()]

        st.success(f"✅ {today} 예측 완료 — {len(all_results)}개 모델 / {len(result_df)}개 지점")
        if failed:
            st.caption(f"API 실패 지점: {len(failed)}개 ({failed[:5]}…)")

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("예측 날짜",             str(today))
        r2.metric("전국 평균 발생 확률",   f"{avg_p*100:.1f}%")
        r3.metric("고위험 지점 (≥50%)",   f"{high_cnt}개")
        r4.metric("최고 위험 지점",
                  f"{top_row['station_name']} ({top_row['proba']*100:.1f}%)")

        # ── 모델 비교 요약표 (2개 이상일 때만) ───────────────────
        if len(all_results) > 1:
            st.divider()
            st.subheader("🤖 모델별 비교")
            comp_rows = []
            for mname, res in all_results.items():
                comp_rows.append({
                    "모델":              mname,
                    "전국 평균 확률":    f"{res['proba'].mean()*100:.1f}%",
                    "고위험 지점 수":    int((res["proba"] >= 0.5).sum()),
                    "최고 발생 확률":    f"{res['proba'].max()*100:.1f}%",
                    "현재 선택":        "✅" if mname == _cur else "",
                })
            st.dataframe(
                pd.DataFrame(comp_rows),
                use_container_width=True,
                hide_index=True,
            )

        st.divider()
        rt1, rt2 = st.columns([3, 2])

        with rt1:
            st.subheader(f"📋 발생 확률 상위 지점 ({_cur})")
            tbl = (
                result_df[["station_name", "region", "proba"]]
                .sort_values("proba", ascending=False)
                .head(15)
                .rename(columns={"station_name": "지점명",
                                 "region":       "지역",
                                 "proba":        "발생 확률"})
            )
            tbl["발생 확률"] = (tbl["발생 확률"] * 100).round(1)
            st.dataframe(tbl, use_container_width=True, hide_index=True)

        with rt2:
            st.subheader("🗺️ 지역별 평균 발생 확률")
            reg_avg = (
                result_df.groupby("region")["proba"]
                .mean().reset_index()
                .rename(columns={"region": "지역", "proba": "평균 확률"})
                .sort_values("평균 확률", ascending=True)
            )
            reg_avg["평균 확률"] = reg_avg["평균 확률"] * 100
            fig_rt = px.bar(
                reg_avg, x="평균 확률", y="지역",
                orientation="h", color="평균 확률",
                color_continuous_scale=["#22c55e", "#eab308", "#f97316", "#ef4444"],
                range_color=[0, 100], text="평균 확률",
            )
            fig_rt.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_rt.update_layout(
                height=400, coloraxis_showscale=False,
                margin=dict(t=10, b=10, r=60),
                xaxis=dict(
                    title="발생 확률 (%)",
                    range=[0, max(reg_avg["평균 확률"].max() * 1.2, 5)],
                ),
            )
            st.plotly_chart(fig_rt, use_container_width=True)

st.divider()

# ── 기능 안내 카드 ─────────────────────────────────────────────────
st.subheader("📌 주요 기능")
g1, g2, g3, g4 = st.columns(4)
with g1:
    st.info("**🗺️ 위험도 지도**\n\n"
            "97개 지점 위험도를 한국 지도 위에 색상으로 시각화합니다.\n"
            "마커 클릭 시 지점 상세 정보 확인 가능합니다.")
with g2:
    st.info("**📊 지점 분석**\n\n"
            "지점별 임상 구성, 지형 방위, FMI/TMI 지수를 분석합니다.\n"
            "전국 순위 및 레이더 차트를 제공합니다.")
with g3:
    st.info("**🤖 모델 성능**\n\n"
            "v2 vs v3a, LogReg·LightGBM·XGBoost 비교 결과와\n"
            "피처 중요도 분석 결과를 확인합니다.")
with g4:
    st.info("**📈 예측 확률 추이**\n\n"
            "지점별 날짜별 예측 확률 추이, 7일 이동평균,\n"
            "실제 산불 발생 이력을 시각화합니다.")

st.caption(
    "⚠️ 예측 확률은 2025년 Hold-out Test 결과입니다. "
    "기상 API 연동 및 모델 파일(.pkl) 등록 후 실시간 예측이 활성화됩니다."
)
