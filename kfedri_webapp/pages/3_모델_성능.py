import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.style import apply_dark_theme

st.set_page_config(page_title="모델 성능", page_icon="🤖", layout="wide")
apply_dark_theme()
st.title("🤖 K-FEDRI v3a 모델 성능 분석")
st.markdown("v2 (기상 11개) vs v3a (기상+임상+지형 29개) | 2025년 Hold-out Test 결과")
st.divider()

# ── 보고서 기반 성능 데이터 ─────────────────────────────────────────
perf = pd.DataFrame([
    {"피처셋": "v2 (기상 11개)", "모델": "LogisticRegression", "피처 수": 11,
     "ROC-AUC": 0.8450, "PR-AUC": 0.0344, "Top5% Recall": 0.300, "Top10% Recall": None, "F1@0.5": 0.034},
    {"피처셋": "v2 (기상 11개)", "모델": "LightGBM",           "피처 수": 11,
     "ROC-AUC": 0.8279, "PR-AUC": 0.0319, "Top5% Recall": 0.293, "Top10% Recall": None, "F1@0.5": 0.045},
    {"피처셋": "v2 (기상 11개)", "모델": "XGBoost",            "피처 수": 11,
     "ROC-AUC": 0.8369, "PR-AUC": 0.0291, "Top5% Recall": 0.287, "Top10% Recall": None, "F1@0.5": 0.045},
    {"피처셋": "v3a (임상+지형 추가)", "모델": "LogisticRegression", "피처 수": 29,
     "ROC-AUC": 0.8667, "PR-AUC": 0.0386, "Top5% Recall": 0.360, "Top10% Recall": 0.553, "F1@0.5": 0.036},
    {"피처셋": "v3a (임상+지형 추가)", "모델": "LightGBM",           "피처 수": 29,
     "ROC-AUC": 0.8747, "PR-AUC": 0.0663, "Top5% Recall": 0.420, "Top10% Recall": 0.573, "F1@0.5": 0.065},
    {"피처셋": "v3a (임상+지형 추가)", "모델": "XGBoost",            "피처 수": 29,
     "ROC-AUC": 0.8781, "PR-AUC": 0.0686, "Top5% Recall": 0.393, "Top10% Recall": 0.593, "F1@0.5": 0.055},
])

topk = pd.DataFrame([
    {"모델": "v3a + LightGBM",           "Top5%": "63/150", "Top10%": "86/150",  "Top20%": "118/150",
     "Recall@5": 0.420, "Recall@10": 0.573, "Recall@20": 0.787},
    {"모델": "v3a + XGBoost",            "Top5%": "59/150", "Top10%": "89/150",  "Top20%": "117/150",
     "Recall@5": 0.393, "Recall@10": 0.593, "Recall@20": 0.780},
    {"모델": "v3a + LogisticRegression", "Top5%": "54/150", "Top10%": "83/150",  "Top20%": "116/150",
     "Recall@5": 0.360, "Recall@10": 0.553, "Recall@20": 0.773},
])

importance = pd.DataFrame([
    {"순위": 1,  "피처": "DrynessRisk",                   "카테고리": "기상", "중요도 (%)": 12.1},
    {"순위": 2,  "피처": "Dry7",                          "카테고리": "기상", "중요도 (%)": 5.2},
    {"순위": 3,  "피처": "imsang_ForestRiskScore (FMI)",  "카테고리": "임상", "중요도 (%)": 4.6},
    {"순위": 4,  "피처": "dem_FlatRatio",                 "카테고리": "지형", "중요도 (%)": 4.1},
    {"순위": 5,  "피처": "imsang_MixedRatio",             "카테고리": "임상", "중요도 (%)": 3.9},
    {"순위": 6,  "피처": "Dry30",                         "카테고리": "기상", "중요도 (%)": 3.8},
    {"순위": 7,  "피처": "Season_sin",                    "카테고리": "계절", "중요도 (%)": 3.8},
    {"순위": 8,  "피처": "dem_TerrainRiskScore (TMI)",    "카테고리": "지형", "중요도 (%)": 3.7},
    {"순위": 9,  "피처": "dem_WestRatio",                 "카테고리": "지형", "중요도 (%)": 3.6},
    {"순위": 10, "피처": "SnowRisk",                      "카테고리": "기상", "중요도 (%)": 3.5},
])

contribution = pd.DataFrame({
    "모델":        ["LogisticRegression", "LightGBM", "XGBoost"],
    "기상 (v2)":   [34.8, 80.7, 41.4],
    "임상 (임상도)": [13.6, 7.1, 22.9],
    "지형 (DEM)":  [51.7, 12.3, 35.7],
})

# ── 탭 ───────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 성능 비교", "📈 Top-K 분석", "🔍 피처 중요도", "ℹ️ 모델 설명"])

# ── Tab 1: 성능 비교 ──────────────────────────────────────────────
with tab1:
    st.subheader("v2 vs v3a 성능 비교표")
    disp_cols = ["피처셋", "모델", "피처 수", "ROC-AUC", "PR-AUC", "Top5% Recall", "F1@0.5"]

    def highlight_max(s):
        return ["background-color: #dcfce7; font-weight:bold"
                if v == s.max() else "" for v in s]

    st.dataframe(
        perf[disp_cols].style.apply(
            highlight_max, subset=["ROC-AUC", "PR-AUC", "Top5% Recall"]
        ).format({"ROC-AUC": "{:.4f}", "PR-AUC": "{:.4f}",
                  "Top5% Recall": "{:.3f}", "F1@0.5": "{:.3f}"}),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    c1, c2 = st.columns(2)

    color_map = {
        "v2 (기상 11개)": "#94a3b8",
        "v3a (임상+지형 추가)": "#f97316",
    }

    with c1:
        fig = px.bar(
            perf, x="모델", y="ROC-AUC", color="피처셋",
            barmode="group", title="ROC-AUC 비교",
            color_discrete_map=color_map,
        )
        fig.update_yaxes(range=[0.80, 0.895])
        fig.update_layout(legend_title_text="", height=320)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.bar(
            perf, x="모델", y="PR-AUC", color="피처셋",
            barmode="group", title="PR-AUC 비교",
            color_discrete_map=color_map,
        )
        fig.update_layout(legend_title_text="", height=320)
        st.plotly_chart(fig, use_container_width=True)

    # 개선량 요약
    st.subheader("v3a 추가로 인한 성능 개선량")
    improve = []
    for model in ["LogisticRegression", "LightGBM", "XGBoost"]:
        v2 = perf[(perf["피처셋"] == "v2 (기상 11개)") & (perf["모델"] == model)].iloc[0]
        v3 = perf[(perf["피처셋"] == "v3a (임상+지형 추가)") & (perf["모델"] == model)].iloc[0]
        improve.append({
            "모델": model,
            "ROC-AUC 개선": f"+{(v3['ROC-AUC'] - v2['ROC-AUC'])*100:.2f}%p",
            "PR-AUC 개선": f"+{(v3['PR-AUC'] - v2['PR-AUC'])*100:.2f}%p",
            "Top5% Recall 개선": f"+{(v3['Top5% Recall'] - v2['Top5% Recall'])*100:.1f}%p",
        })
    st.dataframe(pd.DataFrame(improve), use_container_width=True, hide_index=True)

# ── Tab 2: Top-K 분석 ─────────────────────────────────────────────
with tab2:
    st.subheader("Top-K Recall 분석 (Test set: 총 150건 산불)")
    st.info("**Top-K Recall**: 상위 K% 고위험 예측 지점 중 실제 산불이 얼마나 포함되었는지")

    st.dataframe(
        topk.style.format({
            "Recall@5": "{:.3f}", "Recall@10": "{:.3f}", "Recall@20": "{:.3f}"
        }).highlight_max(subset=["Recall@5", "Recall@10", "Recall@20"],
                         color="#dcfce7"),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    fig_topk = go.Figure()
    colors = {"v3a + LightGBM": "#f97316",
              "v3a + XGBoost": "#3b82f6",
              "v3a + LogisticRegression": "#94a3b8"}

    for _, r in topk.iterrows():
        fig_topk.add_trace(go.Scatter(
            x=[5, 10, 20],
            y=[r["Recall@5"], r["Recall@10"], r["Recall@20"]],
            mode="lines+markers",
            name=r["모델"],
            line=dict(color=colors.get(r["모델"], "#6b7280"), width=2),
            marker=dict(size=8),
        ))

    fig_topk.update_layout(
        title="Top-K% Recall 곡선 (v3a)",
        xaxis_title="상위 K%",
        yaxis_title="Recall",
        xaxis=dict(tickvals=[5, 10, 20], ticktext=["Top 5%", "Top 10%", "Top 20%"]),
        yaxis=dict(range=[0, 1]),
        height=350,
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig_topk, use_container_width=True)

    st.markdown("""
    **해석:**
    - **LightGBM**이 Top5% Recall 0.420으로 **조기 경보에 최적**
      → 전체의 5% 지점만 모니터링해도 실제 산불 150건 중 63건(42%) 포착
    - **XGBoost**는 ROC/PR AUC 기준 전반적 성능이 가장 우수
    - **LogisticRegression**은 해석 가능성이 높아 설명 필요 상황에 적합
    """)

# ── Tab 3: 피처 중요도 ────────────────────────────────────────────
with tab3:
    st.subheader("XGBoost v3a 피처 중요도 Top 10")

    cat_colors = {
        "기상": "#3b82f6",
        "임상": "#22c55e",
        "지형": "#f97316",
        "계절": "#a855f7",
    }

    fig_imp = px.bar(
        importance.sort_values("중요도 (%)"),
        x="중요도 (%)", y="피처",
        color="카테고리",
        orientation="h",
        color_discrete_map=cat_colors,
        text="중요도 (%)",
    )
    fig_imp.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_imp.update_layout(height=380, margin=dict(t=20, r=60),
                          legend_title_text="카테고리")
    st.plotly_chart(fig_imp, use_container_width=True)

    st.divider()
    st.subheader("모델별 피처 그룹 기여도")

    contrib_melt = contribution.melt(
        id_vars="모델", var_name="카테고리", value_name="기여도 (%)"
    )
    fig_contrib = px.bar(
        contrib_melt, x="모델", y="기여도 (%)", color="카테고리",
        barmode="stack",
        color_discrete_map={
            "기상 (v2)": "#3b82f6",
            "임상 (임상도)": "#22c55e",
            "지형 (DEM)": "#f97316",
        },
    )
    fig_contrib.update_layout(height=320, legend_title_text="",
                               yaxis_title="기여도 (%)")
    st.plotly_chart(fig_contrib, use_container_width=True)

    st.markdown("""
    **관찰:**
    - **XGBoost**: 기상 41.4% + 임상 22.9% + 지형 35.7% → v3 피처를 가장 고르게 활용
    - **LightGBM**: DrynessRisk 단일 42% 집중 → 건조 지수에 강한 의존성
    - **LogisticRegression**: 지형 51.7% → 지형 방위가 선형 판별에 큰 역할
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
        - Train: 2022-01-01 ~ 2024-12-31 (99,820행 | 양성률 0.725%)
        - Test : 2025-01-01 ~ 2025-09-11 (29,996행 | 양성률 0.500%)
        - 불균형 처리: class_weight='balanced' (LogReg) / scale_pos_weight (LightGBM·XGBoost)
        """)

    with col_r:
        st.subheader("모델 선택 가이드")
        st.success("**XGBoost v3a**\n\nROC-AUC 0.8781 · PR-AUC 0.0686\n"
                   "→ 종합 성능 최고 → **일반 위험도 서비스** 권장")
        st.warning("**LightGBM v3a**\n\nTop5% Recall 0.420 (63/150건)\n"
                   "→ 조기경보 최적 → **긴급 산불 경보** 권장")
        st.info("**LogisticRegression v3a**\n\nROC-AUC 0.8667\n"
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
        3. 일별 ASOS 기상 데이터 → 29개 피처 변환 → 모델 예측
        4. `pages/4_실시간_예측.py` 페이지 추가로 활성화
        """)
