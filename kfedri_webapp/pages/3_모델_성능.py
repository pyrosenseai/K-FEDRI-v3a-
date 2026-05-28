import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from utils.style import apply_dark_theme

st.set_page_config(page_title="모델 성능", page_icon="🤖", layout="wide")
apply_dark_theme()
st.title("🤖 K-FEDRI v3a 모델 성능 분석")
st.markdown("v2 (기상 11개) vs v3a (기상+임상+지형 29개) | 2025년 Hold-out Test 결과")
st.divider()

DATA_DIR = Path(__file__).parents[1] / "data"

MODEL_MAP = {"LogReg": "LogisticRegression", "LightGBM": "LightGBM", "XGBoost": "XGBoost"}
SET_MAP   = {"v2_baseline": "v2 (기상 11개)", "v3a": "v3a (임상+지형 추가)"}


@st.cache_data
def load_data():
    results    = pd.read_csv(DATA_DIR / "v3_model_results.csv")
    importance = pd.read_csv(DATA_DIR / "v3_feature_importance.csv")

    results["피처셋"] = results["Feature_Set"].map(SET_MAP)
    results["모델"]   = results["Model"].map(MODEL_MAP)
    importance["모델"]   = importance["Model"].map(MODEL_MAP)
    importance["피처셋"] = importance["Feature_Set"].map(SET_MAP)
    return results, importance


results, importance = load_data()

# ── perf 테이블 ────────────────────────────────────────────────────
perf = results.rename(columns={
    "N_Features":         "피처 수",
    "ROC_AUC":            "ROC-AUC",
    "PR_AUC":             "PR-AUC",
    "TopK_Recall_5pct":   "Top5% Recall",
    "TopK_Recall_10pct":  "Top10% Recall",
    "F1_05":              "F1@0.5",
})[["피처셋", "모델", "피처 수", "ROC-AUC", "PR-AUC", "Top5% Recall", "Top10% Recall", "F1@0.5"]]

# ── topk 테이블 ────────────────────────────────────────────────────
total_fires = int(results["TP"].iloc[0] + results["FN"].iloc[0])   # 150

topk_rows = []
for _, r in results[results["Feature_Set"] == "v3a"].iterrows():
    topk_rows.append({
        "모델":      f"v3a + {r['모델']}",
        "Top5%":    f"{int(r['TopK_Detected_5pct'])}/{total_fires}",
        "Top10%":   f"{int(r['TopK_Detected_10pct'])}/{total_fires}",
        "Top20%":   f"{int(r['TopK_Detected_20pct'])}/{total_fires}",
        "Recall@5":  r["TopK_Recall_5pct"],
        "Recall@10": r["TopK_Recall_10pct"],
        "Recall@20": r["TopK_Recall_20pct"],
    })
topk = pd.DataFrame(topk_rows).sort_values("Recall@5", ascending=False)


def feat_category(feat: str) -> str:
    if feat.startswith("imsang_"): return "임상"
    if feat.startswith("dem_"):    return "지형"
    if feat in ("Season_sin", "Season_cos"): return "계절"
    return "기상"


def build_importance_top(model_name: str, fset: str = "v3a", top_n: int = 15):
    df = importance[(importance["모델"] == model_name) &
                    (importance["피처셋"] == SET_MAP[fset])].copy()
    df["Importance"] = df["Importance"].abs()
    total = df["Importance"].sum()
    df["중요도 (%)"] = (df["Importance"] / total * 100).round(2)
    df["카테고리"]   = df["Feature"].apply(feat_category)
    df["피처"]       = df["Feature"]
    return df.nlargest(top_n, "중요도 (%)")[["피처", "카테고리", "중요도 (%)"]].reset_index(drop=True)


def build_contribution():
    v3_imp = importance[importance["피처셋"] == "v3a (임상+지형 추가)"].copy()
    v3_imp["Importance"] = v3_imp["Importance"].abs()
    v3_imp["카테고리"] = v3_imp["Feature"].apply(feat_category).map({
        "기상": "기상 (v2)", "계절": "기상 (v2)",
        "임상": "임상 (임상도)", "지형": "지형 (DEM)"
    })
    rows = []
    for model, grp in v3_imp.groupby("모델"):
        total = grp["Importance"].sum()
        for cat, sg in grp.groupby("카테고리"):
            rows.append({"모델": model, "카테고리": cat,
                         "기여도 (%)": round(sg["Importance"].sum() / total * 100, 1)})
    return pd.DataFrame(rows)


# ── 탭 ───────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 성능 비교", "📈 Top-K 분석", "🔍 피처 중요도", "ℹ️ 모델 설명"])

# ── Tab 1: 성능 비교 ──────────────────────────────────────────────
with tab1:
    st.subheader("v2 vs v3a 성능 비교표")

    def highlight_max(s):
        return ["background-color:#431407; font-weight:bold"
                if v == s.max() else "" for v in s]

    st.dataframe(
        perf.style.apply(
            highlight_max, subset=["ROC-AUC", "PR-AUC", "Top5% Recall"]
        ).format({"ROC-AUC": "{:.4f}", "PR-AUC": "{:.4f}",
                  "Top5% Recall": "{:.3f}", "Top10% Recall": "{:.3f}", "F1@0.5": "{:.3f}"}),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    c1, c2 = st.columns(2)
    color_map = {
        "v2 (기상 11개)":        "#94a3b8",
        "v3a (임상+지형 추가)":  "#f97316",
    }

    with c1:
        fig = px.bar(perf, x="모델", y="ROC-AUC", color="피처셋",
                     barmode="group", title="ROC-AUC 비교",
                     color_discrete_map=color_map)
        fig.update_yaxes(range=[0.80, 0.895])
        fig.update_layout(legend_title_text="", height=320)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.bar(perf, x="모델", y="PR-AUC", color="피처셋",
                     barmode="group", title="PR-AUC 비교",
                     color_discrete_map=color_map)
        fig.update_layout(legend_title_text="", height=320)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("v3a 추가로 인한 성능 개선량")
    improve = []
    for model in ["LogisticRegression", "LightGBM", "XGBoost"]:
        v2 = perf[(perf["피처셋"] == "v2 (기상 11개)") & (perf["모델"] == model)].iloc[0]
        v3 = perf[(perf["피처셋"] == "v3a (임상+지형 추가)") & (perf["모델"] == model)].iloc[0]
        improve.append({
            "모델": model,
            "ROC-AUC 개선":       f"+{(v3['ROC-AUC']       - v2['ROC-AUC'])       * 100:.2f}%p",
            "PR-AUC 개선":        f"+{(v3['PR-AUC']        - v2['PR-AUC'])        * 100:.2f}%p",
            "Top5% Recall 개선":  f"+{(v3['Top5% Recall']  - v2['Top5% Recall'])  * 100:.1f}%p",
        })
    st.dataframe(pd.DataFrame(improve), use_container_width=True, hide_index=True)

# ── Tab 2: Top-K 분석 ─────────────────────────────────────────────
with tab2:
    st.subheader(f"Top-K Recall 분석 (Test set: 총 {total_fires}건 산불)")
    st.info("**Top-K Recall**: 상위 K% 고위험 예측 지점 중 실제 산불이 얼마나 포함되었는지")

    st.dataframe(
        topk.style.format({
            "Recall@5": "{:.3f}", "Recall@10": "{:.3f}", "Recall@20": "{:.3f}"
        }).highlight_max(subset=["Recall@5", "Recall@10", "Recall@20"],
                         props="background-color:#431407; font-weight:bold"),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    fig_topk = go.Figure()
    colors = {"v3a + LightGBM": "#f97316", "v3a + XGBoost": "#3b82f6",
              "v3a + LogisticRegression": "#94a3b8"}

    for _, r in topk.iterrows():
        fig_topk.add_trace(go.Scatter(
            x=[5, 10, 20],
            y=[r["Recall@5"], r["Recall@10"], r["Recall@20"]],
            mode="lines+markers", name=r["모델"],
            line=dict(color=colors.get(r["모델"], "#6b7280"), width=2),
            marker=dict(size=8),
        ))

    fig_topk.update_layout(
        title="Top-K% Recall 곡선 (v3a)",
        xaxis_title="상위 K%", yaxis_title="Recall",
        xaxis=dict(tickvals=[5, 10, 20], ticktext=["Top 5%", "Top 10%", "Top 20%"]),
        yaxis=dict(range=[0, 1]), height=350,
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig_topk, use_container_width=True)

    st.markdown(f"""
    **해석:**
    - **LightGBM**이 Top5% Recall {topk[topk['모델']=='v3a + LightGBM']['Recall@5'].values[0]:.3f}로 **조기 경보에 최적**
    - **XGBoost**는 ROC/PR AUC 기준 전반적 성능이 가장 우수
    - **LogisticRegression**은 해석 가능성이 높아 설명 필요 상황에 적합
    """)

# ── Tab 3: 피처 중요도 ────────────────────────────────────────────
with tab3:
    col_sel1, col_sel2 = st.columns([2, 2])
    with col_sel1:
        sel_model = st.selectbox("모델 선택", ["LightGBM", "XGBoost", "LogisticRegression"],
                                 key="imp_model")
    with col_sel2:
        top_n = st.slider("상위 피처 수", 5, 29, 15, key="imp_topn")

    imp_df = build_importance_top(sel_model, fset="v3a", top_n=top_n)

    cat_colors = {"기상": "#3b82f6", "임상": "#22c55e",
                  "지형": "#f97316", "계절": "#a855f7"}

    fig_imp = px.bar(
        imp_df.sort_values("중요도 (%)"),
        x="중요도 (%)", y="피처",
        color="카테고리", orientation="h",
        color_discrete_map=cat_colors,
        text="중요도 (%)",
        title=f"{sel_model} v3a 피처 중요도 Top {top_n} (정규화)",
    )
    fig_imp.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_imp.update_layout(height=max(350, top_n * 26),
                          margin=dict(t=40, r=70), legend_title_text="카테고리")
    st.plotly_chart(fig_imp, use_container_width=True)

    st.divider()
    st.subheader("모델별 피처 그룹 기여도 (v3a)")

    contrib_df = build_contribution()
    fig_contrib = px.bar(
        contrib_df, x="모델", y="기여도 (%)", color="카테고리",
        barmode="stack",
        color_discrete_map={
            "기상 (v2)": "#3b82f6",
            "임상 (임상도)": "#22c55e",
            "지형 (DEM)": "#f97316",
        },
    )
    fig_contrib.update_layout(height=320, legend_title_text="", yaxis_title="기여도 (%)")
    st.plotly_chart(fig_contrib, use_container_width=True)

    st.markdown("""
    **관찰:**
    - **XGBoost**: 기상·임상·지형을 가장 고르게 활용
    - **LightGBM**: DrynessRisk 중심 → 건조 지수에 강한 의존성
    - **LogisticRegression**: 지형 방위 비율이 선형 판별에 큰 역할
    """)

# ── Tab 4: 모델 설명 ──────────────────────────────────────────────
with tab4:
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.subheader("K-FEDRI v3a 수식")
        st.code(
            "P_ignition = f_ML(W, D, S, F, T, H)\n"
            "K-FEDRI_v3a = 100 × P_ignition",
            language=None,
        )
        st.markdown("""
| 그룹 | 변수 | 설명 |
|------|------|------|
| **W** 기상 | HeatRisk, WindRisk, SolarRisk, SnowRisk | 온도·바람·일사·적설 위험 |
| **D** 건조 | DrynessRisk, Dry7, Dry30, LongDryRisk | 단·중·장기 건조 위험 |
| **S** 계절 | Season_sin, Season_cos | 연 주기 계절성 인코딩 |
| **F** 임상 | ConiferRatio, FMI (ForestRiskScore) 등 | 침엽수 비율·임상 위험지수 |
| **T** 지형 | TMI (TerrainRiskScore), Slope, Aspect 등 | 남향·경사 위험지수 |
| **H** 인문 | WeekendRisk | 주말·공휴일 행락 활동 |

**학습 설정**
- Train: 2022-01-01 ~ 2024-12-31 (99,820행 | 양성 724건 | 양성률 0.725%)
- Test : 2025-01-01 ~ 2025-09-11 (24,530행 | 양성 150건 | 양성률 0.611%)
- 불균형 처리: class_weight='balanced' (LogReg) / scale_pos_weight (LightGBM·XGBoost)
        """)

        with st.expander("⚠️ 발생 확률 수치 해석 주의"):
            st.caption(
                "v3a 모델은 **class_weight='balanced'** (LogReg) 또는 "
                "**scale_pos_weight=N_neg/N_pos** (LightGBM·XGBoost) 설정을 사용합니다. "
                "이 경우 predict_proba 출력값은 실제 발생확률이 아닌 모델 점수입니다. "
                "수치 자체보다는 **지점 간 상대적 위험 순위 비교**에 활용하세요. "
                "(예: 이 지점의 위험도가 전국 상위 5%)"
            )

    with col_r:
        # 최고 성능 수치 동적으로 계산
        best_roc  = results.loc[results["ROC_AUC"].idxmax()]
        best_topk = results.loc[results["TopK_Recall_5pct"].idxmax()]

        st.subheader("모델 선택 가이드")
        st.success(
            f"**{best_roc['모델']} v3a**\n\n"
            f"ROC-AUC {best_roc['ROC_AUC']:.4f} · PR-AUC {best_roc['PR_AUC']:.4f}\n"
            "→ 종합 성능 최고 → **일반 위험도 서비스** 권장"
        )
        st.warning(
            f"**{best_topk['모델']} v3a**\n\n"
            f"Top5% Recall {best_topk['TopK_Recall_5pct']:.3f} "
            f"({int(best_topk['TopK_Detected_5pct'])}/{total_fires}건)\n"
            "→ 조기경보 최적 → **긴급 산불 경보** 권장"
        )
        st.info("**LogisticRegression v3a**\n\nROC-AUC "
                f"{results[results['모델']=='LogisticRegression']['ROC_AUC'].max():.4f}\n"
                "→ 계수 해석 가능 → **정책·보고서 설명** 권장")

        st.divider()
        st.subheader("🔌 실시간 예측 연동")
        st.markdown("""
        1. 훈련된 모델 파일(`.pkl`)을 `models/` 폴더에 저장
        2. KMA API 키를 `.streamlit/secrets.toml`에 등록
           ```toml
           [kma]
           api_key = "YOUR_HUB_API_KEY"
           ```
        3. **홈 화면** 하단 "⚡ 실시간 예측" 섹션에서 실행
        4. API 1회 호출 후 모든 가용 모델 예측 → 모델 전환 시 재호출 불필요
        """)
