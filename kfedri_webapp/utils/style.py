import streamlit as st
import plotly.io as pio

COLORS = {
    "bg":       "#0F172A",
    "bg_card":  "#1E293B",
    "border":   "#334155",
    "primary":  "#f97316",
    "danger":   "#ef4444",
    "warning":  "#eab308",
    "safe":     "#22c55e",
    "text":     "#E2E8F0",
    "muted":    "#94a3b8",
}

PLOTLY_TEMPLATE = "plotly_dark"
PLOTLY_BG       = "#000000"
PLOTLY_PAPER_BG = "#0F172A"

_CSS = """
<style>
/* 전체 배경 */
.stApp { background-color: #0F172A; }

/* 본문 텍스트 전체 */
.stApp, .stApp p, .stApp li,
.stApp label, .stApp span,
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li {
    color: #E2E8F0 !important;
}

/* caption */
.stApp .stCaption, div[data-testid="stCaptionContainer"] {
    color: #94a3b8 !important;
}

/* 헤더 */
h1 { color: #f97316 !important; }
h2, h3 { color: #fb923c !important; }
h4, h5, h6 { color: #E2E8F0 !important; }

/* 사이드바 배경 + 텍스트 */
section[data-testid="stSidebar"] {
    background-color: #1E293B;
    border-right: 1px solid #334155;
}
section[data-testid="stSidebar"] *,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span {
    color: #E2E8F0 !important;
}
/* 사이드바 네비게이션 메뉴 */
section[data-testid="stSidebar"] a,
section[data-testid="stSidebarNavItems"] span {
    color: #CBD5E1 !important;
}
section[data-testid="stSidebarNavItems"] li[aria-selected="true"] span {
    color: #f97316 !important;
    font-weight: 600;
}

/* 위젯 라벨 (selectbox, radio, checkbox, date_input 등) */
div[data-testid="stSelectbox"] label,
div[data-testid="stRadio"] label,
div[data-testid="stCheckbox"] label,
div[data-testid="stDateInput"] label,
div[data-testid="stSlider"] label,
div[data-testid="stTextInput"] label,
div[data-testid="stMultiSelect"] label {
    color: #E2E8F0 !important;
}

/* selectbox / date_input 입력창 */
div[data-baseweb="select"] div,
div[data-baseweb="input"] input {
    background-color: #0F172A !important;
    color: #E2E8F0 !important;
    border-color: #334155 !important;
}

/* 드롭다운 옵션 목록 */
ul[data-testid="stSelectboxVirtualDropdown"] li {
    background-color: #1E293B !important;
    color: #E2E8F0 !important;
}

/* 메트릭 카드 */
div[data-testid="metric-container"] {
    background-color: #1E293B;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 16px;
}
div[data-testid="stMetricLabel"],
div[data-testid="stMetricValue"],
div[data-testid="stMetricDelta"] {
    color: #E2E8F0 !important;
}

/* 구분선 */
hr { border-color: #334155 !important; }

/* 탭 */
button[data-baseweb="tab"] { color: #94a3b8 !important; }
button[data-baseweb="tab"][aria-selected="true"] {
    color: #f97316 !important;
    border-bottom-color: #f97316 !important;
}

/* info / warning / error 박스 */
div[data-testid="stInfo"] {
    background-color: #1E293B;
    border-left-color: #f97316;
    color: #E2E8F0 !important;
}
div[data-testid="stWarning"] {
    background-color: #1E293B;
    border-left-color: #eab308;
    color: #E2E8F0 !important;
}
div[data-testid="stError"] {
    background-color: #1E293B;
    border-left-color: #ef4444;
    color: #E2E8F0 !important;
}

/* subheader 아래 작은 텍스트 */
div[data-testid="stSubheader"] { color: #fb923c !important; }
</style>
"""


def apply_dark_theme():
    st.markdown(_CSS, unsafe_allow_html=True)
    pio.templates.default = PLOTLY_TEMPLATE


def risk_color(val: float) -> str:
    if pd_isna(val):
        return COLORS["muted"]
    if val >= 0.75:
        return COLORS["danger"]
    if val >= 0.50:
        return COLORS["primary"]
    if val >= 0.25:
        return COLORS["warning"]
    return COLORS["safe"]


def risk_label(val: float) -> str:
    if pd_isna(val):
        return "정보없음"
    if val >= 0.75:
        return "매우높음"
    if val >= 0.50:
        return "높음"
    if val >= 0.25:
        return "보통"
    return "낮음"


def pd_isna(val) -> bool:
    try:
        import math
        return math.isnan(float(val))
    except (TypeError, ValueError):
        return val is None
