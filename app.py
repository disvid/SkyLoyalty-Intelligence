import os
import sys
import textwrap
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    REPORTS_DIR, PLOTS_DIR, MODELS_DIR,
    HIGH_CHURN_RISK_THRESHOLD,
    MEDIUM_CHURN_RISK_THRESHOLD,
    CUSTOMER_ID_COL
)
from src.customer_360 import Customer360

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SkyLoyalty Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme state (light / dark) ──────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
LIGHT = st.session_state.theme == "light"

# ── Design System ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
@import url('https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@2.30.0/tabler-icons.min.css');

:root {
    --bg-primary:    #070B16;
    --bg-card:       rgba(20, 28, 46, 0.55);
    --bg-card-solid: #111827;
    --bg-card-hover: rgba(30, 41, 66, 0.75);
    --border:        rgba(80, 110, 160, 0.18);
    --border-strong: rgba(120, 150, 200, 0.32);
    --accent-blue:   #3B82F6;
    --accent-cyan:   #06B6D4;
    --accent-green:  #10B981;
    --accent-amber:  #F59E0B;
    --accent-red:    #EF4444;
    --accent-purple: #8B5CF6;
    --accent-pink:   #EC4899;
    --text-primary:  #F1F5F9;
    --text-muted:    #64748B;
    --text-dim:      #94A3B8;
    --glow:          0 8px 32px rgba(0, 0, 0, 0.45);
}

.stApp {
    background:
        radial-gradient(1200px 600px at 8% -10%, rgba(59,130,246,0.10), transparent 60%),
        radial-gradient(1000px 700px at 110% 0%, rgba(139,92,246,0.10), transparent 55%),
        radial-gradient(900px 600px at 50% 120%, rgba(6,182,212,0.08), transparent 60%),
        var(--bg-primary);
    background-attachment: fixed;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--text-primary) !important;
}

.main .block-container {
    padding: 3.2rem 2.2rem 3rem 2.2rem;
    max-width: 1480px;
}

[data-testid="stHeader"] {
    background: rgba(7, 11, 22, 0.0) !important;
    backdrop-filter: blur(6px);
}
[data-testid="stHeader"]::before { content: none !important; }
[data-testid="stToolbar"] { right: 1rem; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(11,16,28,0.96) 0%, rgba(15,22,38,0.92) 100%) !important;
    border-right: 1px solid var(--border);
    backdrop-filter: blur(18px);
}
[data-testid="stSidebar"] .block-container { padding: 1.2rem 1rem; }

.brand-wrap {
    display: flex; align-items: center; gap: 0.8rem;
    padding: 0.4rem 0.2rem 1.1rem 0.2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.1rem;
}
.brand-logo {
    width: 46px; height: 46px; border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem;
    background: linear-gradient(135deg, #3B82F6, #8B5CF6);
    box-shadow: 0 6px 18px rgba(59,130,246,0.45);
}
.brand-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.15rem; font-weight: 700; color: var(--text-primary);
    line-height: 1;
}
.brand-tag {
    font-size: 0.7rem; color: var(--text-muted);
    letter-spacing: 0.12em; text-transform: uppercase; margin-top: 0.25rem;
}

.side-stats {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px; padding: 1rem 1.1rem; margin-top: 1rem;
    backdrop-filter: blur(12px);
}
.side-stats-title {
    font-size: 0.65rem; font-weight: 600; letter-spacing: 0.16em;
    color: var(--text-muted); text-transform: uppercase; margin-bottom: 0.7rem;
}
.side-stat-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.35rem 0; font-size: 0.85rem;
}
.side-stat-row span:first-child { color: var(--text-dim); }
.side-stat-row span:last-child {
    font-family: 'Space Grotesk', sans-serif; font-weight: 600; color: var(--text-primary);
}
.side-footer {
    margin-top: 1.4rem; text-align: center;
    font-size: 0.68rem; color: var(--text-muted); letter-spacing: 0.04em;
}

.page-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.15rem; font-weight: 700;
    background: linear-gradient(135deg, #60A5FA, #22D3EE, #A78BFA);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem; line-height: 1.15;
}
.page-subtitle {
    font-size: 0.92rem; color: var(--text-muted); margin-bottom: 1.6rem;
}

.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(185px, 1fr));
    gap: 1rem; margin-bottom: 1.6rem;
}
.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.25rem 1.45rem;
    position: relative; overflow: hidden;
    backdrop-filter: blur(14px);
    box-shadow: var(--glow);
    transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    animation: kpiIn 0.5s ease both;
}
.kpi-card:hover {
    transform: translateY(-5px);
    border-color: var(--border-strong);
    box-shadow: 0 14px 40px rgba(0,0,0,0.55);
}
.kpi-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: var(--accent-color, #3B82F6);
    border-radius: 16px 16px 0 0;
}
.kpi-card::after {
    content: ''; position: absolute; top: -40%; right: -30%;
    width: 140px; height: 140px; border-radius: 50%;
    background: var(--accent-color, #3B82F6);
    opacity: 0.10; filter: blur(28px);
}
.kpi-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.05rem; font-weight: 700;
    color: var(--accent-color, #3B82F6);
    line-height: 1; margin-bottom: 0.45rem; position: relative;
}
.kpi-label {
    font-size: 0.72rem; font-weight: 500; color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 0.09em; position: relative;
}
.kpi-delta {
    font-size: 0.75rem; margin-top: 0.45rem; color: var(--text-dim); position: relative;
}
@keyframes kpiIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

.section-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.12rem; font-weight: 600; color: var(--text-primary);
    margin: 1.7rem 0 0.85rem 0; display: flex; align-items: center; gap: 0.55rem;
}
.section-header::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, var(--border-strong), transparent);
    margin-left: 0.55rem;
}

.badge {
    display: inline-block; padding: 0.22rem 0.75rem; border-radius: 20px;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.03em;
}
.badge-high   { background: rgba(239,68,68,0.15);  color: #EF4444; border: 1px solid rgba(239,68,68,0.3); }
.badge-medium { background: rgba(245,158,11,0.15); color: #F59E0B; border: 1px solid rgba(245,158,11,0.3);}
.badge-low    { background: rgba(16,185,129,0.15); color: #10B981; border: 1px solid rgba(16,185,129,0.3);}

.insight-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent-color, #3B82F6);
    border-radius: 0 14px 14px 0;
    padding: 1.05rem 1.25rem; margin-bottom: 0.8rem;
    backdrop-filter: blur(12px);
    transition: transform 0.2s ease, background 0.2s ease;
}
.insight-card:hover { transform: translateX(4px); background: var(--bg-card-hover); }
.insight-title { font-weight: 600; font-size: 0.92rem; color: var(--text-primary); }
.insight-body  { font-size: 0.83rem; color: var(--text-dim); margin-top: 0.25rem; line-height: 1.5; }

.result-pill {
    display: inline-flex; align-items: center; gap: 0.5rem;
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 30px; padding: 0.45rem 1.1rem; margin: 0.4rem 0 0.8rem 0;
    font-size: 0.85rem; color: var(--text-dim); backdrop-filter: blur(10px);
}
.result-pill b { color: var(--accent-blue); font-family: 'Space Grotesk', sans-serif; }

[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 14px; overflow: hidden;
    box-shadow: var(--glow);
}

[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px; padding: 0.9rem 1.1rem;
    backdrop-filter: blur(12px);
}
[data-testid="stMetricValue"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.7rem !important; font-weight: 700 !important;
}

.stButton > button {
    background: linear-gradient(135deg, #3B82F6, #06B6D4) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    font-weight: 600 !important; padding: 0.5rem 1.6rem !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
    box-shadow: 0 6px 18px rgba(59,130,246,0.35) !important;
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 10px 26px rgba(59,130,246,0.5) !important; }

.stSelectbox > div > div,
.stTextInput > div > div,
.stMultiSelect > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important; color: var(--text-primary) !important;
    backdrop-filter: blur(10px);
}

.stDownloadButton > button {
    background: rgba(59,130,246,0.1) !important; color: var(--accent-blue) !important;
    border: 1px solid rgba(59,130,246,0.3) !important; border-radius: 10px !important;
    font-weight: 500 !important; transition: background 0.2s !important;
}
.stDownloadButton > button:hover { background: rgba(59,130,246,0.22) !important; }

[data-testid="stSidebar"] .stRadio > div { gap: 0.25rem; }
[data-testid="stSidebar"] .stRadio label {
    background: transparent; border-radius: 10px; padding: 0.55rem 0.8rem;
    transition: background 0.15s, color 0.15s; color: var(--text-dim) !important;
    font-size: 0.92rem;
}
[data-testid="stSidebar"] .stRadio label:hover { background: rgba(59,130,246,0.10); }

.stTabs [data-baseweb="tab-list"] { gap: 0.4rem; }
.stTabs [data-baseweb="tab"] {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 10px 10px 0 0; color: var(--text-dim);
}
.stTabs [aria-selected="true"] {
    background: rgba(59,130,246,0.16) !important; color: var(--accent-blue) !important;
}

.streamlit-expanderHeader, [data-testid="stExpander"] summary {
    background: var(--bg-card) !important; border-radius: 10px !important;
}

hr { border-color: var(--border) !important; margin: 1rem 0 !important; }

.stAlert { border-radius: 12px !important; backdrop-filter: blur(8px); }

.js-plotly-plot { border-radius: 14px; width: 100% !important; }
[data-testid="stPlotlyChart"] { min-height: 240px; }

.app-footer {
    text-align: center; margin-top: 2.5rem; padding-top: 1.4rem;
    border-top: 1px solid var(--border);
    font-size: 0.78rem; color: var(--text-muted); letter-spacing: 0.03em;
}

::-webkit-scrollbar { width: 7px; height: 7px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 4px; }

@media (max-width: 900px) {
    [data-testid="column"] { width: 100% !important; flex: 1 1 100% !important; }
}
</style>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@2.30.0/tabler-icons.min.css">
""", unsafe_allow_html=True)

# ── Light theme override ────────────────────────────────────────────────────────
if LIGHT:
    st.markdown("""
    <style>
    :root {
        --bg-primary:    #F4F7FB;
        --bg-card:       rgba(255, 255, 255, 0.78);
        --bg-card-solid: #FFFFFF;
        --bg-card-hover: rgba(255, 255, 255, 0.95);
        --border:        rgba(30, 60, 110, 0.14);
        --border-strong: rgba(30, 60, 110, 0.28);
        --text-primary:  #0F1B2D;
        --text-muted:    #5A6B85;
        --text-dim:      #475569;
        --glow:          0 8px 30px rgba(30, 60, 110, 0.12);
    }
    .stApp {
        background:
            radial-gradient(1200px 600px at 8% -10%, rgba(59,130,246,0.10), transparent 60%),
            radial-gradient(1000px 700px at 110% 0%, rgba(139,92,246,0.08), transparent 55%),
            radial-gradient(900px 600px at 50% 120%, rgba(6,182,212,0.08), transparent 60%),
            var(--bg-primary) !important;
    }
    html, body, [class*="css"] { color: var(--text-primary) !important; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(255,255,255,0.96) 0%, rgba(244,247,251,0.94) 100%) !important;
    }
    [data-testid="stHeader"] { background: rgba(244, 247, 251, 0.0) !important; }
    </style>
    """, unsafe_allow_html=True)

# ── Theme-aware tokens (used by both dashboard + Customer 360) ────────────────
_grid  = "#D5DEEA" if LIGHT else "#1e2d45"
_fcol  = "#475569" if LIGHT else "#94A3B8"
_card  = "#FFFFFF" if LIGHT else "#111827"
_card_t= "rgba(255,255,255,0.85)" if LIGHT else "rgba(17,24,39,0.9)"
_border= "rgba(30,60,110,0.14)" if LIGHT else "#1e2d45"
_text  = "#0F1B2D" if LIGHT else "#F1F5F9"
_muted = "#5A6B85" if LIGHT else "#64748B"
_dim   = "#475569" if LIGHT else "#94A3B8"
_chartbg = "#FFFFFF" if LIGHT else "#0A0E1A"

PLOT_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color=_fcol, size=11),
    xaxis=dict(gridcolor=_grid, linecolor=_grid, zerolinecolor=_grid),
    yaxis=dict(gridcolor=_grid, linecolor=_grid, zerolinecolor=_grid),
    margin=dict(t=30, b=30, l=10, r=10),
)


def themed(**overrides):
    base = {
        k: (dict(v) if isinstance(v, dict) else v)
        for k, v in PLOT_THEME.items()
    }
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = {**base[k], **v}
        else:
            base[k] = v
    return base


COLORS = {
    "blue":   "#3B82F6",
    "cyan":   "#06B6D4",
    "green":  "#10B981",
    "amber":  "#F59E0B",
    "red":    "#EF4444",
    "purple": "#8B5CF6",
    "pink":   "#EC4899",
}
SEG_PALETTE = [
    "#3B82F6", "#06B6D4", "#10B981",
    "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899"
]

# ── Data helpers ───────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_csv(filename: str) -> pd.DataFrame:
    path = os.path.join(REPORTS_DIR, filename)
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_all_data():
    segments = load_csv("customer_segments.csv")
    scores   = load_csv("churn_scores.csv")
    recs     = load_csv("retention_recommendations.csv")
    metrics  = load_csv("model_metrics.csv")
    features = load_csv("features.csv")
    labels   = load_csv("churn_labels.csv")
    shap_imp = load_csv("shap_global_importance.csv")
    return segments, scores, recs, metrics, features, labels, shap_imp


@st.cache_resource(show_spinner=False)
def get_c360():
    return Customer360()


def risk_badge(p: float) -> str:
    if p >= HIGH_CHURN_RISK_THRESHOLD:
        return '🔴 High'
    elif p >= MEDIUM_CHURN_RISK_THRESHOLD:
        return '🟡 Medium'
    return '🟢 Low'


def risk_label(p: float) -> str:
    if p >= HIGH_CHURN_RISK_THRESHOLD:
        return "🔴 High Risk"
    elif p >= MEDIUM_CHURN_RISK_THRESHOLD:
        return "🟡 Medium Risk"
    return "🟢 Low Risk"


def kpi_card(value: str, label: str, color: str, delta: str = "") -> str:
    delta_html = f'<div class="kpi-delta">{delta}</div>' if delta else ""
    return (
        f'<div class="kpi-card" style="--accent-color:{color};">'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-label">{label}</div>'
        f'{delta_html}'
        f'</div>'
    )


def section_header(icon: str, title: str) -> None:
    st.markdown(
        f'<div class="section-header">{icon} {title}</div>',
        unsafe_allow_html=True
    )


def insight_card(title: str, body: str, color: str) -> str:
    return f"""
    <div class="insight-card" style="--accent-color:{color};">
        <div class="insight-title">{title}</div>
        <div class="insight-body">{body}</div>
    </div>
    """


def apply_plot_theme(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(height=height, **PLOT_THEME)
    return fig


def clean_html(markup: str) -> str:
    return textwrap.dedent(markup).strip()


def render_html(markup: str) -> None:
    st.markdown(clean_html(markup), unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# CUSTOMER 360 — helper builders (theme-aware, inline styles only)
# ════════════════════════════════════════════════════════════════════════════

def c360_gauge(value: float, color: str, title: str, suffix: str = "%") -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value, 1),
        number={"suffix": suffix,
                "font": {"size": 32, "color": _text, "family": "Space Grotesk"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": _muted,
                     "tickfont": {"color": _muted, "size": 10}},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": _chartbg,
            "borderwidth": 0,
            "steps": [
                {"range": [0,  30],  "color": "rgba(16,185,129,0.08)"},
                {"range": [30, 60],  "color": "rgba(245,158,11,0.08)"},
                {"range": [60, 80],  "color": "rgba(239,68,68,0.08)"},
                {"range": [80, 100], "color": "rgba(124,58,237,0.08)"},
            ],
            "threshold": {"line": {"color": color, "width": 3},
                          "thickness": 0.75, "value": value},
        },
        title={"text": title, "font": {"color": _muted, "size": 12}},
    ))
    fig.update_layout(height=240, **PLOT_THEME)
    return fig


def c360_section(icon: str, title: str) -> str:
    return clean_html(f"""
    <div style="display:flex; align-items:center; gap:8px;
                margin:2rem 0 0.8rem 0; font-family:'Space Grotesk',sans-serif;
                font-size:0.72rem; font-weight:600; letter-spacing:0.12em;
                text-transform:uppercase; color:{_muted};">
        <i class="ti {icon}" style="font-size:15px;"></i>
        {title}
        <div style="flex:1; height:1px; background:{_border}; margin-left:6px;"></div>
    </div>""")


def c360_kpi(value: str, label: str, color: str, sub: str = "") -> str:
    return clean_html(f"""
    <div style="background:{_card}; border:1px solid {_border}; border-radius:12px;
                padding:1rem 1.2rem; position:relative; overflow:hidden; flex:1; min-width:140px;">
        <div style="position:absolute; top:0; left:0; right:0; height:3px;
                    background:{color}; border-radius:12px 12px 0 0;"></div>
        <div style="font-family:'Space Grotesk',sans-serif; font-size:1.4rem;
                    font-weight:700; color:{color}; line-height:1.2;">{value}</div>
        <div style="font-size:0.68rem; text-transform:uppercase; letter-spacing:0.08em;
                    color:{_muted}; margin-top:3px;">{label}</div>
        {"<div style='font-size:0.75rem; color:" + _dim + "; margin-top:3px;'>" + sub + "</div>" if sub else ""}
    </div>""")


def c360_card(content: str, extra_style: str = "") -> str:
    return clean_html(f"""
    <div style="background:{_card_t}; border:1px solid {_border};
                border-radius:14px; padding:1.2rem 1.4rem; margin-bottom:0.8rem;
                {extra_style}">
        {content}
    </div>""")


def c360_badge(text: str, color: str) -> str:
    return clean_html(f"""
    <span style="display:inline-flex; align-items:center; gap:6px;
                 padding:0.35rem 1rem; border-radius:20px; font-size:0.82rem;
                 font-weight:600; background:{color}1A;
                 color:{color}; border:1px solid {color}40;">
        {text}
    </span>""")


def c360_progress(pct: float, color: str) -> str:
    pct = max(0, min(float(pct or 0), 100))
    return clean_html(f"""
    <div style="background:{_border}; border-radius:99px; height:8px; margin:6px 0;">
        <div style="width:{pct:.1f}%; height:8px; border-radius:99px;
                    background:{color};"></div>
    </div>""")


def c360_driver_card(factor: str, detail: str, impact: str,
                      direction: str, icon: str) -> str:
    ic = {"high": COLORS["red"], "medium": COLORS["amber"], "low": COLORS["green"]}.get(impact, COLORS["blue"])
    dc = COLORS["red"] if direction == "negative" else COLORS["green"]
    return clean_html(f"""
    <div style="background:{_card}; border:1px solid {_border}; border-radius:10px;
                padding:0.8rem 1rem; display:flex; align-items:flex-start;
                gap:10px; margin-bottom:0.6rem;">
        <div style="width:32px; height:32px; border-radius:8px; flex-shrink:0;
                    background:{ic}18; color:{ic}; display:flex;
                    align-items:center; justify-content:center; font-size:16px;">
            <i class="ti {icon}"></i>
        </div>
        <div style="flex:1;">
            <div style="display:flex; align-items:center; justify-content:space-between;">
                <div style="font-size:0.85rem; font-weight:600; color:{_text};">{factor}</div>
                <span style="font-size:0.68rem; font-weight:600; letter-spacing:0.06em;
                             color:{dc}; text-transform:uppercase;">{impact}</span>
            </div>
            <div style="font-size:0.78rem; color:{_dim}; margin-top:2px;">{detail}</div>
        </div>
    </div>""")


def c360_generate_csv_bytes(profile: dict, cid: str) -> bytes:
    ov, ch, sg, va, hs, na = (profile[k] for k in
        ["overview","churn","segment","value","health","next_action"])
    row = {
        "customer_id": cid,
        "loyalty_tier": ov["loyalty_tier"],
        "enrollment_type": ov["enrollment_type"],
        "education": ov["education"],
        "marital_status": ov["marital_status"],
        "province": ov["province"],
        "salary": ov["salary"],
        "clv": ov["clv"],
        "tenure_months": ov["tenure_months"],
        "months_inactive": ov["months_inactive"],
        "churn_probability": ch["probability"],
        "churn_probability_pct": ch["probability_pct"],
        "risk_category": ch["risk_category"],
        "segment": sg["name"],
        "segment_size": sg["count"],
        "segment_avg_churn_pct": round(sg["avg_churn"] * 100, 2),
        "segment_avg_clv": round(sg["avg_clv"], 2),
        "rfm_score": sg["rfm_score"],
        "future_value_score": va["score"],
        "value_category": va["category"],
        "health_score": hs["score"],
        "health_status": hs["status"],
        "recommended_action": na["action"],
        "channel": na["channel"],
        "timing": na["timing"],
        "est_retention_lift": na["est_lift"],
        "est_roi": na["est_roi"],
        "top_churn_drivers": "; ".join(d["factor"] for d in profile["drivers"]),
    }
    return pd.DataFrame([row]).to_csv(index=False).encode("utf-8")


def c360_generate_report_bytes(profile: dict, cid: str) -> tuple:
    """Returns (bytes, extension, mime_type)."""
    try:
        from fpdf import FPDF

        class PDF(FPDF):
            def header(self):
                self.set_font("Helvetica", "B", 14)
                self.set_text_color(30, 64, 175)  # Dark blue
                self.cell(0, 10, f"Customer Intelligence Report - {cid}", ln=True)
                self.set_draw_color(30, 45, 69)
                self.line(10, 20, 200, 20)
                self.ln(5)

            def footer(self):
                self.set_y(-15)
                self.set_font("Helvetica", "I", 8)
                self.set_text_color(100, 116, 139)
                self.cell(0, 10, f"SkyLoyalty Intelligence Platform | Page {self.page_no()}", align="C")

        pdf = PDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        def h2(text):
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(30, 64, 175)   # Dark blue for headings
            pdf.cell(0, 8, text, ln=True)
            pdf.set_text_color(30, 30, 30)    # Dark gray for body

        def row(label, value):
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(80, 80, 80)    # Medium gray for labels
            pdf.cell(60, 6, str(label), ln=False)
            pdf.set_text_color(30, 30, 30)    # Dark black for values
            safe_value = str(value).replace('—', '-').replace('–', '-').replace('•', '*')
            pdf.cell(0, 6, safe_value, ln=True)

        ov = profile.get("overview", {})
        ch = profile.get("churn", {})
        sg = profile.get("segment", {})
        va = profile.get("value", {})
        hs = profile.get("health", {})
        na = profile.get("next_action", {})

        h2("1. CUSTOMER OVERVIEW")
        pdf.ln(2)
        row("Loyalty Tier",    ov.get("loyalty_tier", ""))
        row("Enrollment Type", ov.get("enrollment_type", ""))
        row("Education",       ov.get("education", ""))
        row("Marital Status",  ov.get("marital_status", ""))
        row("Province",        ov.get("province", ""))
        row("Income",          f"${ov.get('salary', 0):,.0f}")
        row("CLV",             f"${ov.get('clv', 0):,.2f}")
        row("Tenure",          f"{ov.get('tenure_months', 0)} months")
        pdf.ln(4)

        h2("2. CHURN ANALYSIS")
        pdf.ln(2)
        row("Churn Probability", f"{ch.get('probability_pct', 0)}%")
        row("Risk Category",     ch.get("risk_category", ""))
        row("Months Inactive",   ch.get("months_inactive", ""))
        pdf.ln(4)

        h2("3. TOP CHURN DRIVERS")
        pdf.ln(2)
        for d in profile.get("drivers", []):
            row(d.get("factor", ""), d.get("detail", ""))
        pdf.ln(4)

        h2("4. CUSTOMER SEGMENT")
        pdf.ln(2)
        row("Segment",       sg.get("name", ""))
        row("Segment Size",  f"{sg.get('count', 0):,} customers")
        row("Avg Churn",     f"{sg.get('avg_churn', 0)*100:.1f}%")
        row("Avg CLV",       f"${sg.get('avg_clv', 0):,.0f}")
        row("RFM Score",     f"{sg.get('rfm_score', 0):.1f}")
        pdf.ln(4)

        h2("5. VALUE ASSESSMENT")
        pdf.ln(2)
        row("Future Value Score", f"{va.get('score', 0)}/100")
        row("Category",           va.get("category", ""))
        pdf.ln(4)

        h2("6. NEXT BEST ACTION")
        pdf.ln(2)
        row("Action",    na.get("action", ""))
        row("Channel",   na.get("channel", ""))
        row("Timing",    na.get("timing", ""))
        row("Est. Lift", f"{na.get('est_lift', 0)*100:.0f}%")
        row("Est. ROI",  f"${na.get('est_roi', 0):,.0f}")
        pdf.ln(4)

        h2("7. HEALTH SCORE")
        pdf.ln(2)
        row("Health Score", f"{hs.get('score', 0)}/100")
        row("Status",       hs.get("status", ""))

        # Reliable bytes output
        pdf_output = pdf.output(dest='S')
        if isinstance(pdf_output, str):
            pdf_bytes = pdf_output.encode("latin1", errors='replace')
        else:
            pdf_bytes = pdf_output

        return pdf_bytes, "pdf", "application/pdf"

    except Exception:
        # TXT Fallback
        lines = [
            "CUSTOMER INTELLIGENCE REPORT",
            f"Customer ID: {cid}",
            "=" * 60, "",
            "1. OVERVIEW",
            f"  Loyalty Tier:  {profile.get('overview',{}).get('loyalty_tier','')}",
            f"  CLV:           ${profile.get('overview',{}).get('clv',0):,.2f}",
            f"  Tenure:        {profile.get('overview',{}).get('tenure_months',0)} months",
            f"  Province:      {profile.get('overview',{}).get('province','')}", "",
            "2. CHURN",
            f"  Probability:   {profile.get('churn',{}).get('probability_pct',0)}%",
            f"  Risk Category: {profile.get('churn',{}).get('risk_category','')}", "",
            "3. SEGMENT",
            f"  Name:          {profile.get('segment',{}).get('name','')}",
            f"  RFM Score:     {profile.get('segment',{}).get('rfm_score',0)}", "",
            "4. VALUE",
            f"  Score:         {profile.get('value',{}).get('score',0)}/100",
            f"  Category:      {profile.get('value',{}).get('category','')}", "",
            "5. NEXT BEST ACTION",
            f"  Action:        {profile.get('next_action',{}).get('action','')}",
            f"  Channel:       {profile.get('next_action',{}).get('channel','')}",
            f"  Timing:        {profile.get('next_action',{}).get('timing','')}", "",
            "6. HEALTH",
            f"  Score:         {profile.get('health',{}).get('score',0)}/100",
            f"  Status:        {profile.get('health',{}).get('status','')}",
        ]
        return "\n".join(lines).encode("utf-8"), "txt", "text/plain"
# ════════════════════════════════════════════════════════════════════════════
# CUSTOMER 360 — main render function (theme-aware, fully integrated)
# ════════════════════════════════════════════════════════════════════════════

def render_customer_360():
    st.markdown('<div class="page-title">Customer Intelligence Center</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Complete 360° profile · Churn risk · Segment · Next best action · Health score</div>', unsafe_allow_html=True)

    with st.spinner("Loading customer data..."):
        c360 = get_c360()

    all_ids = c360.get_all_customer_ids()
    if not all_ids:
        st.error("No customer data found. Run `python run_pipeline.py` first.")
        return

    if "c360_customer_id" not in st.session_state:
        st.session_state.c360_customer_id = ""

    sc1, sc2 = st.columns([3, 1])
    with sc1:
        typed_id = st.text_input(
            "Customer ID",
            placeholder=f"e.g. {all_ids[0]}  ·  {len(all_ids):,} customers available",
            label_visibility="collapsed",
            key="c360_customer_search",
        )
    with sc2:
        use_dropdown = st.checkbox("Browse list", value=False, key="c360_browse_list")

    if use_dropdown:
        current_idx = all_ids.index(st.session_state.c360_customer_id) \
            if st.session_state.c360_customer_id in all_ids else 0
        customer_id = st.selectbox(
            "Select",
            all_ids,
            index=current_idx,
            label_visibility="collapsed",
            key="c360_customer_select",
        )
    else:
        customer_id = typed_id.strip() or st.session_state.c360_customer_id or None

    if customer_id:
        st.session_state.c360_customer_id = str(customer_id).strip()

    if not customer_id:
        render_html(f"""
        <div style="text-align:center; padding:4rem 2rem; color:{_muted}; font-size:1rem;">
            <div style="font-size:3rem; margin-bottom:1rem;">🔍</div>
            Enter a Customer ID above to generate their complete intelligence report.
        </div>
        """)
        return

    with st.spinner(f"Generating profile for {customer_id}..."):
        profile = c360.get_profile(customer_id)

    if profile is None:
        render_html(f"""
        <div style="text-align:center; padding:4rem 2rem; color:{_muted}; font-size:1rem;">
            <div style="font-size:3rem; margin-bottom:1rem; color:#EF4444;">✗</div>
            Customer <b style="color:{_text};">{customer_id}</b> not found.
            Check the ID and try again.
        </div>
        """)
        return

    ov      = profile["overview"]
    ch      = profile["churn"]
    sg      = profile["segment"]
    va      = profile["value"]
    na      = profile["next_action"]
    hs      = profile["health"]
    drivers = profile["drivers"]
    history = profile["flight_history"]

    # ── SECTION 1 — OVERVIEW ───────────────────────────────────────────────────
    st.markdown(c360_section("ti-user-circle", "Customer Overview"), unsafe_allow_html=True)

    tier_c = {"star": COLORS["amber"], "aurora": COLORS["blue"], "nova": COLORS["purple"]}.get(
        ov["loyalty_tier"].lower(), COLORS["blue"]
    )
    inact_c = COLORS["red"] if ov["months_inactive"] >= 6 else (
        COLORS["amber"] if ov["months_inactive"] >= 3 else COLORS["green"]
    )

    cols = st.columns(5)
    kpis = [
        (ov["loyalty_tier"],          "Loyalty Tier",     tier_c,      ""),
        (f"${ov['clv']:,.0f}",        "Lifetime Value",   COLORS["green"], ""),
        (f"${ov['salary']:,.0f}",     "Annual Income",    COLORS["cyan"],  ""),
        (f"{ov['tenure_months']}m",   "Programme Tenure", COLORS["purple"],""),
        (f"{ov['months_inactive']}m", "Months Inactive",  inact_c, "since last flight"),
    ]
    for col, (val, lbl, clr, sub) in zip(cols, kpis):
        with col:
            st.markdown(c360_kpi(val, lbl, clr, sub), unsafe_allow_html=True)
    components.html(
        c360_card(f"""
    <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(180px,1fr)); gap:1rem;">
        {"".join(f'''
        <div>
            <div style="font-size:0.68rem; text-transform:uppercase; letter-spacing:0.1em;
                        color:{_muted}; margin-bottom:4px;">{lbl}</div>
            <div style="font-weight:500; color:{_text};">{val}</div>
        </div>''' for lbl, val in [
            ("Customer ID",    customer_id),
            ("Enrollment",     ov["enrollment_type"]),
            ("Education",      ov["education"]),
            ("Marital Status", ov["marital_status"]),
            ("Gender",         ov["gender"]),
            ("Province",       ov["province"]),
        ])}
    </div>
    """),
        height=200,
        scrolling=False
    )
    # st.markdown(c360_card(f"""
    # <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(180px,1fr)); gap:1rem;">
    #     {"".join(f'''
    #     <div>
    #         <div style="font-size:0.68rem; text-transform:uppercase; letter-spacing:0.1em;
    #                     color:{_muted}; margin-bottom:4px;">{lbl}</div>
    #         <div style="font-weight:500; color:{_text};">{val}</div>
    #     </div>''' for lbl, val in [
    #         ("Customer ID",    customer_id),
    #         ("Enrollment",     ov["enrollment_type"]),
    #         ("Education",      ov["education"]),
    #         ("Marital Status", ov["marital_status"]),
    #         ("Gender",         ov["gender"]),
    #         ("Province",       ov["province"]),
    #     ])}
    # </div>
    # """), unsafe_allow_html=True)

    # ── SECTION 2 — CHURN ANALYSIS ─────────────────────────────────────────────
    st.markdown(c360_section("ti-alert-triangle", "Churn Analysis"), unsafe_allow_html=True)

    ch_c1, ch_c2 = st.columns([2, 3])
    with ch_c1:
        st.plotly_chart(
            c360_gauge(ch["probability_pct"], ch["risk_color"], "Churn Risk Score", "%"),
            use_container_width=True
        )

    with ch_c2:
        threshold_rows = "".join([
            f"""<div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
                <div style="width:8px; height:8px; border-radius:50%; background:{rc}; flex-shrink:0;"></div>
                <span style="font-size:0.78rem; color:{_dim};">{cat}</span>
                <span style="font-size:0.78rem; color:{_muted}; margin-left:auto;">{lo}-{hi}%</span>
            </div>"""
            for cat, lo, hi, rc in [
                ("Low Risk",      0,  30, "#10B981"),
                ("Medium Risk",   30, 60, "#F59E0B"),
                ("High Risk",     60, 80, "#EF4444"),
                ("Critical Risk", 80, 100,"#7C3AED"),
            ]
        ])
        components.html(
            c360_card(f"""
        <div style="margin-bottom:1rem;">
            {c360_badge(ch['risk_category'], ch['risk_color'])}
        </div>
        <div style="font-family:'Space Grotesk',sans-serif; font-size:2.5rem;
                    font-weight:700; color:{ch['risk_color']}; line-height:1;">
            {ch['probability_pct']}%
        </div>
        <div style="font-size:0.8rem; color:{_muted}; margin-top:4px; margin-bottom:1rem;">
            Churn probability
        </div>
        <div style="font-size:0.75rem; color:{_dim}; margin-bottom:4px;">Risk level</div>
        {c360_progress(ch['probability_pct'], ch['risk_color'])}
        <div style="display:flex; justify-content:space-between;
                    font-size:0.7rem; color:{_muted}; margin-top:4px;">
            <span>0% Low</span><span>30% Medium</span>
            <span>60% High</span><span>80% Critical</span>
        </div>
        <div style="margin-top:1.2rem; padding-top:1rem; border-top:1px solid {_border};">
            <div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em;
                        color:{_muted}; margin-bottom:8px;">Risk thresholds</div>
            {threshold_rows}
        </div>
        """, "margin-top:0.5rem;"),
            height=350,
            scrolling=False
        )
        # st.markdown(c360_card(f"""
        # <div style="margin-bottom:1rem;">
        #     {c360_badge(ch['risk_category'], ch['risk_color'])}
        # </div>
        # <div style="font-family:'Space Grotesk',sans-serif; font-size:2.5rem;
        #             font-weight:700; color:{ch['risk_color']}; line-height:1;">
        #     {ch['probability_pct']}%
        # </div>
        # <div style="font-size:0.8rem; color:{_muted}; margin-top:4px; margin-bottom:1rem;">
        #     Churn probability
        # </div>
        # <div style="font-size:0.75rem; color:{_dim}; margin-bottom:4px;">Risk level</div>
        # {c360_progress(ch['probability_pct'], ch['risk_color'])}
        # <div style="display:flex; justify-content:space-between;
        #             font-size:0.7rem; color:{_muted}; margin-top:4px;">
        #     <span>0% Low</span><span>30% Medium</span>
        #     <span>60% High</span><span>80% Critical</span>
        # </div>
        # <div style="margin-top:1.2rem; padding-top:1rem; border-top:1px solid {_border};">
        #     <div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em;
        #                 color:{_muted}; margin-bottom:8px;">Risk thresholds</div>
        #     {threshold_rows}
        # </div>
        # """, "margin-top:0.5rem;"), unsafe_allow_html=True)

    # ── SECTION 3 — TOP CHURN DRIVERS ──────────────────────────────────────────
    st.markdown(c360_section("ti-bulb", "Top Churn Drivers"), unsafe_allow_html=True)

    dr_c1, dr_c2 = st.columns(2)
    for i, d in enumerate(drivers):
        with (dr_c1 if i % 2 == 0 else dr_c2):
            st.markdown(
                c360_driver_card(d["factor"], d["detail"], d["impact"], d["direction"], d["icon"]),
                unsafe_allow_html=True
            )

    # ── SECTION 4 — CUSTOMER SEGMENT ───────────────────────────────────────────
    st.markdown(c360_section("ti-users-group", "Customer Segment"), unsafe_allow_html=True)

    seg_c1, seg_c2 = st.columns([3, 2])

    with seg_c1:
        components.html(
            c360_card(f"""
        <div style="margin-bottom:0.8rem;">
            <span style="display:inline-block; padding:0.3rem 1rem; border-radius:20px;
                         font-size:0.82rem; font-weight:600;
                         background:rgba(139,92,246,0.15); color:#8B5CF6;
                         border:1px solid rgba(139,92,246,0.3);">
                {sg['name']}
            </span>
        </div>
        <div style="font-size:0.85rem; color:{_dim}; line-height:1.6; margin-bottom:1rem;">
            {sg['description']}
        </div>
        <div style="padding-top:1rem; border-top:1px solid {_border};
                    display:grid; grid-template-columns:1fr 1fr 1fr; gap:1rem;">
            <div>
                <div style="font-size:0.68rem; text-transform:uppercase;
                            letter-spacing:0.08em; color:{_muted};">Segment Size</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-size:1.2rem;
                            font-weight:700; color:{_text}; margin-top:2px;">
                    {sg['count']:,}
                </div>
            </div>
            <div>
                <div style="font-size:0.68rem; text-transform:uppercase;
                            letter-spacing:0.08em; color:{_muted};">Avg Churn</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-size:1.2rem;
                            font-weight:700; color:#EF4444; margin-top:2px;">
                    {sg['avg_churn']*100:.1f}%
                </div>
            </div>
            <div>
                <div style="font-size:0.68rem; text-transform:uppercase;
                            letter-spacing:0.08em; color:{_muted};">Avg CLV</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-size:1.2rem;
                            font-weight:700; color:#10B981; margin-top:2px;">
                    ${sg['avg_clv']:,.0f}
                </div>
            </div>
        </div>
        """),
            height=260,
            scrolling=False
        )
        # st.markdown(c360_card(f"""
        # <div style="margin-bottom:0.8rem;">
        #     <span style="display:inline-block; padding:0.3rem 1rem; border-radius:20px;
        #                  font-size:0.82rem; font-weight:600;
        #                  background:rgba(139,92,246,0.15); color:#8B5CF6;
        #                  border:1px solid rgba(139,92,246,0.3);">
        #         {sg['name']}
        #     </span>
        # </div>
        # <div style="font-size:0.85rem; color:{_dim}; line-height:1.6; margin-bottom:1rem;">
        #     {sg['description']}
        # </div>
        # <div style="padding-top:1rem; border-top:1px solid {_border};
        #             display:grid; grid-template-columns:1fr 1fr 1fr; gap:1rem;">
        #     <div>
        #         <div style="font-size:0.68rem; text-transform:uppercase;
        #                     letter-spacing:0.08em; color:{_muted};">Segment Size</div>
        #         <div style="font-family:'Space Grotesk',sans-serif; font-size:1.2rem;
        #                     font-weight:700; color:{_text}; margin-top:2px;">
        #             {sg['count']:,}
        #         </div>
        #     </div>
        #     <div>
        #         <div style="font-size:0.68rem; text-transform:uppercase;
        #                     letter-spacing:0.08em; color:{_muted};">Avg Churn</div>
        #         <div style="font-family:'Space Grotesk',sans-serif; font-size:1.2rem;
        #                     font-weight:700; color:#EF4444; margin-top:2px;">
        #             {sg['avg_churn']*100:.1f}%
        #         </div>
        #     </div>
        #     <div>
        #         <div style="font-size:0.68rem; text-transform:uppercase;
        #                     letter-spacing:0.08em; color:{_muted};">Avg CLV</div>
        #         <div style="font-family:'Space Grotesk',sans-serif; font-size:1.2rem;
        #                     font-weight:700; color:#10B981; margin-top:2px;">
        #             ${sg['avg_clv']:,.0f}
        #         </div>
        #     </div>
        # </div>
        # """), unsafe_allow_html=True)

    with seg_c2:
        r_s = sg.get("r_score", 0)
        f_s = sg.get("f_score", 0)
        m_s = sg.get("m_score", 0)
        if any(s > 0 for s in [r_s, f_s, m_s]):
            cats = ["Recency", "Frequency", "Monetary"]
            vals = [r_s * 0.8, f_s * 0.8, m_s * 0.8, r_s * 0.8]
            fig_radar = go.Figure(go.Scatterpolar(
                r=vals, theta=cats + [cats[0]],
                fill="toself",
                fillcolor="rgba(59,130,246,0.12)",
                line=dict(color=COLORS["blue"], width=2),
                marker=dict(color=COLORS["blue"], size=6),
            ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(range=[0,5], tickcolor=_muted,
                                   gridcolor=_grid, linecolor=_grid,
                                   tickfont=dict(size=9, color=_muted)),
                    angularaxis=dict(tickcolor=_muted, linecolor=_grid,
                                    tickfont=dict(size=10, color=_dim)),
                ),
                showlegend=False, **PLOT_THEME, height=260,
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            components.html(
                c360_card(f'<div style="text-align:center; padding:2rem; color:{_muted};">RFM scores not available</div>'),
                height=180,
                scrolling=False
            )
            # st.markdown(c360_card(
            #     f'<div style="text-align:center; padding:2rem; color:{_muted};">RFM scores not available</div>'
            # ), unsafe_allow_html=True)

    # ── SECTION 5 — FLIGHT BEHAVIOUR ───────────────────────────────────────────
    st.markdown(c360_section("ti-plane", "Flight Behaviour"), unsafe_allow_html=True)

    if not history.empty and "year_month" in history.columns:
        fig_hist = make_subplots(
            rows=2, cols=2,
            subplot_titles=["Total Flights", "Distance (km)",
                            "Points Accumulated", "Points Redeemed"],
            vertical_spacing=0.18, horizontal_spacing=0.1,
        )
        x = history["year_month"]

        traces = [
            ("total_flights",      1, 1, COLORS["blue"],  "rgba(59,130,246,0.08)"),
            ("distance",           1, 2, COLORS["cyan"],  "rgba(6,182,212,0.08)"),
            ("points_accumulated", 2, 1, COLORS["green"], "rgba(16,185,129,0.08)"),
            ("points_redeemed",    2, 2, COLORS["amber"], "rgba(245,158,11,0.08)"),
        ]
        for col_name, row, col, color, fill in traces:
            y = history[col_name] if col_name in history.columns else [0]*len(x)
            fig_hist.add_trace(go.Scatter(
                x=x, y=y, mode="lines",
                line=dict(color=color, width=2),
                fill="tozeroy", fillcolor=fill,
                showlegend=False,
            ), row=row, col=col)

        for r in [1, 2]:
            for c_idx in [1, 2]:
                fig_hist.update_xaxes(
                    gridcolor=_grid, linecolor=_grid,
                    tickfont=dict(size=9, color=_muted), row=r, col=c_idx
                )
                fig_hist.update_yaxes(
                    gridcolor=_grid, linecolor=_grid,
                    tickfont=dict(size=9, color=_muted), row=r, col=c_idx
                )
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color=_fcol, size=10),
            margin=dict(t=40, b=20, l=10, r=10), height=420,
        )
        fig_hist.update_annotations(font=dict(size=11, color=_dim))
        st.plotly_chart(fig_hist, use_container_width=True)

        with st.expander("View monthly activity data"):
            disp = [c for c in ["year_month","year","month","total_flights",
                                 "distance","points_accumulated","points_redeemed",
                                 "dollar_cost_points_redeemed"] if c in history.columns]
            st.dataframe(
                history[disp].sort_values("year_month", ascending=False).head(36),
                use_container_width=True, height=300
            )
    else:
        st.info("No flight history available for this customer.")

    # ── SECTION 6 — VALUE ASSESSMENT ───────────────────────────────────────────
    st.markdown(c360_section("ti-star", "Value Assessment"), unsafe_allow_html=True)

    va_c1, va_c2 = st.columns([2, 3])
    with va_c1:
        st.plotly_chart(
            c360_gauge(va["score"], va["color"], "Future Value Score", ""),
            use_container_width=True
        )

    with va_c2:
        comp_bars = ""
        for comp, val in va["components"].items():
            bar_color = COLORS["red"] if val < 0 else COLORS["blue"]
            bar_pct   = min(abs(val) / 30 * 100, 100)
            comp_bars += f"""
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
                <div style="font-size:0.78rem; color:{_dim}; width:130px; flex-shrink:0;">
                    {comp}</div>
                <div style="background:{_border}; border-radius:99px; height:10px; flex:1;">
                    <div style="width:{bar_pct:.0f}%; height:10px; border-radius:99px;
                                background:{bar_color};"></div>
                </div>
                <div style="font-size:0.78rem; font-weight:600; color:{bar_color};
                            width:40px; text-align:right;">{val:+.1f}</div>
            </div>"""
        components.html(
            c360_card(f"""
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:1rem;">
            <div style="font-family:'Space Grotesk',sans-serif; font-size:2.5rem;
                        font-weight:700; color:{va['color']}; line-height:1;">{va['score']}</div>
            <div>
                <div style="font-weight:600; color:{va['color']}; font-size:1rem;">
                    {va['category']}</div>
                <div style="font-size:0.75rem; color:{_muted};">out of 100</div>
            </div>
        </div>
        <div style="font-size:0.82rem; color:{_dim}; line-height:1.6; margin-bottom:1rem;">
            {va['description']}</div>
        <div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em;
                    color:{_muted}; margin-bottom:8px;">Score components</div>
        {comp_bars}
        """),
            height=310,
            scrolling=False
        )
        # st.markdown(c360_card(f"""
        # <div style="display:flex; align-items:center; gap:12px; margin-bottom:1rem;">
        #     <div style="font-family:'Space Grotesk',sans-serif; font-size:2.5rem;
        #                 font-weight:700; color:{va['color']}; line-height:1;">{va['score']}</div>
        #     <div>
        #         <div style="font-weight:600; color:{va['color']}; font-size:1rem;">
        #             {va['category']}</div>
        #         <div style="font-size:0.75rem; color:{_muted};">out of 100</div>
        #     </div>
        # </div>
        # <div style="font-size:0.82rem; color:{_dim}; line-height:1.6; margin-bottom:1rem;">
        #     {va['description']}</div>
        # <div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em;
        #             color:{_muted}; margin-bottom:8px;">Score components</div>
        # {comp_bars}
        # """), unsafe_allow_html=True)

    # ── SECTION 7 — NEXT BEST ACTION ───────────────────────────────────────────
    st.markdown(c360_section("ti-rocket", "Next Best Action"), unsafe_allow_html=True)

    action_icons = {
        "Complimentary Lounge Pass":                            "ti-armchair",
        "Temporary Tier Status Upgrade (90 days)":              "ti-award",
        "Expiring Points Reminder + Bonus Redemption Offer":    "ti-clock",
        "Win-Back: 50% Bonus Miles on Next 2 Flights":          "ti-gift",
        "Earn 3x Miles for Next 30 Days":                       "ti-plane-tilt",
        "Seasonal Travel Offer (targeted destination discount)":"ti-map-pin",
        "High-Value Customer Dedicated Account Manager Outreach":"ti-headset",
        "No Immediate Action Required":                          "ti-check",
    }
    a_icon = action_icons.get(na["action"], "ti-bolt")

    nba_items = "".join([
        f"""<div style="flex:1; min-width:120px;">
            <div style="font-size:0.68rem; text-transform:uppercase; letter-spacing:0.1em;
                        color:{_muted}; margin-bottom:4px;">{lbl}</div>
            <div style="font-size:0.9rem; font-weight:500; color:{vc};">{val}</div>
        </div>"""
        for lbl, val, vc in [
            ("Channel",   na["channel"],                  _text),
            ("Timing",    na["timing"],                    _text),
            ("Goal",      na["goal"],                      _text),
            ("Est. Lift", f"+{na['est_lift']*100:.0f}%",  "#10B981"),
            ("Est. ROI",  f"${na['est_roi']:,.0f}",        "#10B981"),
        ]
    ])

    render_html(f"""
    <div style="background:linear-gradient(135deg,rgba(59,130,246,0.08),rgba(6,182,212,0.05));
                border:1px solid rgba(59,130,246,0.25); border-radius:14px;
                padding:1.4rem 1.6rem; margin-bottom:1rem;">
        <div style="display:flex; align-items:flex-start; gap:14px;">
            <div style="background:rgba(59,130,246,0.15); border-radius:10px;
                        padding:0.7rem; font-size:1.5rem; color:#3B82F6; flex-shrink:0;">
                <i class="ti {a_icon}"></i>
            </div>
            <div style="flex:1;">
                <div style="font-family:'Space Grotesk',sans-serif; font-size:1.1rem;
                            font-weight:700; color:{_text}; margin-bottom:4px;">
                    {na['action']}
                </div>
                <div style="font-size:0.82rem; color:{_dim}; line-height:1.6;">
                    <b style="color:{_muted};">WHY:</b> {na['why']}
                </div>
            </div>
        </div>
        <div style="display:flex; gap:1rem; flex-wrap:wrap; margin-top:1rem;">
            {nba_items}
        </div>
        <div style="margin-top:1rem; padding:0.7rem 1rem;
                    background:rgba(16,185,129,0.08);
                    border:1px solid rgba(16,185,129,0.2); border-radius:8px;
                    font-size:0.8rem; color:{_dim};">
            <b style="color:#10B981;">Expected Outcome:</b> {na['expected_outcome']}
        </div>
    </div>
    """)

    # ── SECTION 8 — CUSTOMER HEALTH SCORE ──────────────────────────────────────
    st.markdown(c360_section("ti-heart-rate-monitor", "Customer Health Score"), unsafe_allow_html=True)

    hs_c1, hs_c2 = st.columns([2, 3])
    with hs_c1:
        st.plotly_chart(
            c360_gauge(hs["score"], hs["color"], "Health Score", ""),
            use_container_width=True
        )
        render_html(f"""
        <div style="text-align:center; margin-top:-0.5rem;">
            <span style="background:{hs['color']}18; color:{hs['color']};
                         border:1px solid {hs['color']}40; border-radius:20px;
                         padding:0.3rem 1.2rem; font-size:0.9rem; font-weight:600;">
                <i class="ti {hs['icon']}"></i> {hs['status']}
            </span>
        </div>
        """)

    with hs_c2:
        comp_colors = {
            "Activity":      COLORS["blue"],
            "Engagement":    COLORS["cyan"],
            "Loyalty":       COLORS["purple"],
            "Churn Penalty": COLORS["red"],
        }
        health_bars = ""
        for comp, val in hs["components"].items():
            bc  = comp_colors.get(comp, COLORS["blue"])
            pct = min(abs(val) / 30 * 100, 100)
            health_bars += f"""
            <div style="margin-bottom:10px;">
                <div style="display:flex; justify-content:space-between;
                            font-size:0.78rem; color:{_dim}; margin-bottom:4px;">
                    <span>{comp}</span>
                    <span style="color:{bc}; font-weight:600;">{val:+.1f}</span>
                </div>
                {c360_progress(pct, bc)}
            </div>"""

        legend = "".join([
            f"""<div style="display:flex; align-items:center; gap:6px;
                            font-size:0.75rem; color:{_dim};">
                <div style="width:8px; height:8px; border-radius:50%;
                            background:{lc}; flex-shrink:0;"></div>
                {label}
            </div>"""
            for label, lc in [
                ("75-100 Healthy",   "#10B981"),
                ("55-74 Watchlist",  "#F59E0B"),
                ("35-54 At Risk",    "#F97316"),
                ("0-34 Critical",    "#EF4444"),
            ]
        ])
        components.html(
            c360_card(f"""
        <div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em;
                    color:{_muted}; margin-bottom:12px;">Score breakdown</div>
        {health_bars}
        <div style="margin-top:0.8rem; display:grid;
                    grid-template-columns:1fr 1fr; gap:6px;">
            {legend}
        </div>
        """),
            height=280,
            scrolling=False
        )
        # st.markdown(c360_card(f"""
        # <div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em;
        #             color:{_muted}; margin-bottom:12px;">Score breakdown</div>
        # {health_bars}
        # <div style="margin-top:0.8rem; display:grid;
        #             grid-template-columns:1fr 1fr; gap:6px;">
        #     {legend}
        # </div>
        # """), unsafe_allow_html=True)

    # ── SECTION 9 — EXPORT ──────────────────────────────────────────────────────
    st.markdown(c360_section("ti-download", "Export Customer Report"), unsafe_allow_html=True)

    components.html(
        c360_card(f'<div style="font-size:0.85rem; color:{_dim};">'
        'Download a complete snapshot of this customer\'s intelligence profile.'
        '</div>'),
        height=80,
        scrolling=False
    )
    # st.markdown(c360_card(
    #     f'<div style="font-size:0.85rem; color:{_dim};">'
    #     'Download a complete snapshot of this customer\'s intelligence profile.'
    #     '</div>'
    # ), unsafe_allow_html=True)

    ex_c1, ex_c2, _ = st.columns([1, 1, 3])

    with ex_c1:
        st.download_button(
            "📥  Download CSV",
            data=c360_generate_csv_bytes(profile, customer_id),
            file_name=f"customer_{customer_id}_report.csv",
            mime="text/csv",
            use_container_width=True
        )
    with ex_c2:
        pdf_data, ext, mime = c360_generate_report_bytes(profile, customer_id)
        st.download_button(
            "📄  Download Report",
            data=pdf_data,
            file_name=f"customer_{customer_id}_report.{ext}",
            mime=mime,
            use_container_width=True
        )

# ── Load data ──────────────────────────────────────────────────────────────────
with st.spinner("Loading loyalty intelligence…"):
    (segments, scores, recs, metrics,
     features, labels, shap_imp) = load_all_data()

master = pd.DataFrame()
if not scores.empty and CUSTOMER_ID_COL in scores.columns:
    master = scores.copy()
    if not labels.empty and CUSTOMER_ID_COL in labels.columns:
        master = master.merge(
            labels[[CUSTOMER_ID_COL, "churned", "months_inactive"]],
            on=CUSTOMER_ID_COL, how="left"
        )
    if not segments.empty and CUSTOMER_ID_COL in segments.columns:
        seg_cols = [CUSTOMER_ID_COL, "segment_name"] + [
            c for c in ["recency", "frequency", "monetary", "RFM_score"]
            if c in segments.columns
        ]
        master = master.merge(segments[seg_cols], on=CUSTOMER_ID_COL, how="left")


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="brand-wrap">
        <div class="brand-logo">✈️</div>
        <div>
            <div class="brand-name">SkyLoyalty</div>
            <div class="brand-tag">Intelligence Platform</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        options=[
            "🏠  Executive Overview",
            "🎯  Churn Prediction",
            "👥  Customer Segments",
            "💡  Retention Actions",
            "🔍  Explainability",
            "🧠  Customer 360",
        ],
        label_visibility="collapsed"
    )

    st.markdown('<div style="height:0.6rem;"></div>', unsafe_allow_html=True)

    def _toggle_theme():
        st.session_state.theme = "light" if st.session_state.get("theme_switch") else "dark"

    st.toggle(
        "☀️  Light mode" if not LIGHT else "🌙  Dark mode",
        value=LIGHT,
        key="theme_switch",
        on_change=_toggle_theme,
        help="Switch between light and dark appearance",
    )

    if not master.empty:
        total = len(master)
        churn_col_exists = "churn_probability" in master.columns
        n_high = int((master["churn_probability"] >= HIGH_CHURN_RISK_THRESHOLD).sum()) if churn_col_exists else 0
        n_segs = master['segment_name'].nunique() if 'segment_name' in master.columns else '—'

        st.markdown(f"""
        <div class="side-stats">
            <div class="side-stats-title">Live Stats</div>
            <div class="side-stat-row"><span>Members</span><span>{total:,}</span></div>
            <div class="side-stat-row"><span>High Risk</span><span style="color:#EF4444;">{n_high:,}</span></div>
            <div class="side-stat-row"><span>Segments</span><span>{n_segs}</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="side-footer">v1.0 · Airline Loyalty Analytics</div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

if "Overview" in page:

    st.markdown('<div class="page-title">Executive Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Airline Loyalty Programme · Customer Intelligence Dashboard</div>', unsafe_allow_html=True)

    total_customers = len(master) if not master.empty else 0
    churn_prob_col  = "churn_probability" in (master.columns if not master.empty else [])
    churned_col     = "churned" in (master.columns if not master.empty else [])

    n_high   = int((master["churn_probability"] >= HIGH_CHURN_RISK_THRESHOLD).sum())   if churn_prob_col else 0
    n_medium = int(((master["churn_probability"] >= MEDIUM_CHURN_RISK_THRESHOLD) & (master["churn_probability"] < HIGH_CHURN_RISK_THRESHOLD)).sum()) if churn_prob_col else 0
    n_low    = int((master["churn_probability"] < MEDIUM_CHURN_RISK_THRESHOLD).sum())  if churn_prob_col else 0
    churn_rate = float(master["churned"].mean() * 100) if churned_col else 0.0
    n_segs     = int(master["segment_name"].nunique()) if "segment_name" in (master.columns if not master.empty else []) else 0

    best_auc = 0.0
    if not metrics.empty and "roc_auc" in metrics.columns:
        best_auc = float(metrics["roc_auc"].max())

    kpis_html = f"""
    <div class="kpi-grid">
        {kpi_card(f"{total_customers:,}", "Total Members", COLORS['blue'])}
        {kpi_card(f"{churn_rate:.1f}%", "Actual Churn Rate", COLORS['red'])}
        {kpi_card(f"{n_high:,}", "High Risk Customers", COLORS['red'],
                  f"{n_high/max(total_customers,1)*100:.1f}% of base")}
        {kpi_card(f"{n_medium:,}", "Medium Risk", COLORS['amber'],
                  f"{n_medium/max(total_customers,1)*100:.1f}% of base")}
        {kpi_card(f"{n_low:,}", "Low Risk", COLORS['green'],
                  f"{n_low/max(total_customers,1)*100:.1f}% of base")}
        {kpi_card(f"{n_segs}", "Segments", COLORS['purple'])}
        {kpi_card(f"{best_auc:.3f}", "Best Model AUC", COLORS['cyan'])}
    </div>
    """
    st.markdown(kpis_html, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        section_header("📊", "Churn Risk Distribution")
        if churn_prob_col:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=master["churn_probability"],
                nbinsx=50,
                marker=dict(
                    color=master["churn_probability"].apply(
                        lambda p: COLORS['red'] if p >= HIGH_CHURN_RISK_THRESHOLD
                        else (COLORS['amber'] if p >= MEDIUM_CHURN_RISK_THRESHOLD
                              else COLORS['green'])
                    ),
                    opacity=0.8,
                    line=dict(width=0)
                ),
                name="Customers"
            ))
            fig.add_vline(x=HIGH_CHURN_RISK_THRESHOLD,
                          line_dash="dash", line_color=COLORS['red'], line_width=1.5,
                          annotation_text="High Risk", annotation_font_color=COLORS['red'])
            fig.add_vline(x=MEDIUM_CHURN_RISK_THRESHOLD,
                          line_dash="dash", line_color=COLORS['amber'], line_width=1.5,
                          annotation_text="Medium Risk", annotation_font_color=COLORS['amber'])
            fig.update_layout(
                xaxis_title="Churn Probability",
                yaxis_title="Customers",
                showlegend=False,
                **PLOT_THEME
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run the pipeline to generate churn scores.")

    with col2:
        section_header("🥧", "Segment Mix")
        if "segment_name" in master.columns:
            seg_counts = master["segment_name"].value_counts().reset_index()
            seg_counts.columns = ["segment", "count"]
            fig2 = go.Figure(go.Pie(
                labels=seg_counts["segment"],
                values=seg_counts["count"],
                hole=0.55,
                marker=dict(colors=SEG_PALETTE, line=dict(color=_chartbg, width=2)),
                textinfo="percent",
                textfont_size=11,
            ))
            fig2.update_layout(
                showlegend=True,
                legend=dict(
                    orientation="v", x=1.0, y=0.5,
                    font=dict(size=10, color=_dim),
                    bgcolor="rgba(0,0,0,0)"
                ),
                annotations=[dict(
                    text=f"{total_customers:,}<br>members",
                    x=0.5, y=0.5, font_size=14,
                    font_color=_text, showarrow=False
                )],
                **PLOT_THEME
            )
            st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        section_header("🏆", "Model Performance")
        if not metrics.empty:
            metric_cols = [c for c in ["accuracy", "precision", "recall", "f1", "roc_auc"]
                           if c in metrics.columns]
            if metric_cols and "model" in metrics.columns:
                fig3 = go.Figure()
                bar_colors = [COLORS['blue'], COLORS['cyan'], COLORS['green'],
                              COLORS['amber'], COLORS['purple']]
                for i, col_name in enumerate(metric_cols):
                    fig3.add_trace(go.Bar(
                        name=col_name.replace("_", " ").title(),
                        x=metrics["model"] if "model" in metrics.columns else metrics.index,
                        y=metrics[col_name],
                        marker_color=bar_colors[i % len(bar_colors)],
                        opacity=0.85,
                    ))
                fig3.update_layout(
                    barmode="group",
                    xaxis_title="", yaxis_title="Score",
                    yaxis_range=[0, 1.05],
                    legend=dict(font_size=10, bgcolor="rgba(0,0,0,0)"),
                    **PLOT_THEME
                )
                st.plotly_chart(fig3, use_container_width=True)

    with col4:
        section_header("⚠️", "Risk Tier Breakdown")
        if churn_prob_col:
            tiers = pd.Series({
                "🔴 High Risk":   n_high,
                "🟡 Medium Risk": n_medium,
                "🟢 Low Risk":    n_low,
            })
            fig4 = go.Figure(go.Bar(
                x=tiers.values,
                y=tiers.index,
                orientation="h",
                marker=dict(
                    color=[COLORS['red'], COLORS['amber'], COLORS['green']],
                    opacity=0.85,
                    line=dict(width=0),
                ),
                text=[f"{v:,}  ({v/max(total_customers,1)*100:.1f}%)"
                      for v in tiers.values],
                textposition="outside",
                textfont=dict(color=_dim, size=11),
            ))
            fig4.update_layout(
                xaxis_title="Customers",
                showlegend=False,
                **PLOT_THEME
            )
            st.plotly_chart(fig4, use_container_width=True)

    roc_path = os.path.join(PLOTS_DIR, "roc_curves.png")
    cm_path  = os.path.join(PLOTS_DIR, "confusion_matrices.png")

    if os.path.exists(roc_path) or os.path.exists(cm_path):
        section_header("📈", "Model Diagnostics")
        c1, c2 = st.columns(2)
        with c1:
            if os.path.exists(roc_path):
                st.image(roc_path, use_column_width=True)
        with c2:
            if os.path.exists(cm_path):
                st.image(cm_path, use_column_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CHURN PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

elif "Churn" in page:

    st.markdown('<div class="page-title">Churn Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Search customers · View risk scores · Analyse distributions</div>', unsafe_allow_html=True)

    if master.empty:
        st.warning("⚠️  No data found. Run `python run_pipeline.py` first.")
        st.stop()

    section_header("🔎", "Filters")
    fcol1, fcol2, fcol3, fcol4 = st.columns([2, 2, 2, 1])

    with fcol1:
        risk_filter = st.selectbox(
            "Risk Tier", ["All", "🔴 High Risk", "🟡 Medium Risk", "🟢 Low Risk"],
            label_visibility="visible"
        )
    with fcol2:
        seg_opts = ["All"] + sorted(
            master["segment_name"].dropna().unique().tolist()
        ) if "segment_name" in master.columns else ["All"]
        seg_filter = st.selectbox("Segment", seg_opts)
    with fcol3:
        search_id = st.text_input("Search Customer ID", placeholder="e.g. 100590")
    with fcol4:
        prob_range = st.slider(
            "Probability", 0.0, 1.0, (0.0, 1.0), step=0.05,
            label_visibility="visible"
        )

    df_view = master.copy()

    if "churn_probability" in df_view.columns:
        df_view = df_view[df_view["churn_probability"].between(*prob_range)]

        if risk_filter == "🔴 High Risk":
            df_view = df_view[df_view["churn_probability"] >= HIGH_CHURN_RISK_THRESHOLD]
        elif risk_filter == "🟡 Medium Risk":
            df_view = df_view[
                (df_view["churn_probability"] >= MEDIUM_CHURN_RISK_THRESHOLD) &
                (df_view["churn_probability"] < HIGH_CHURN_RISK_THRESHOLD)
            ]
        elif risk_filter == "🟢 Low Risk":
            df_view = df_view[df_view["churn_probability"] < MEDIUM_CHURN_RISK_THRESHOLD]

    if seg_filter != "All" and "segment_name" in df_view.columns:
        df_view = df_view[df_view["segment_name"] == seg_filter]

    if search_id.strip() and CUSTOMER_ID_COL in df_view.columns:
        df_view = df_view[df_view[CUSTOMER_ID_COL].astype(str).str.contains(search_id.strip(), case=False)]

    extra = f"of {len(master):,}" if len(df_view) < len(master) else ""
    st.markdown(f"""
    <div class="result-pill">Showing <b>{len(df_view):,}</b> customers {extra}</div>
    """, unsafe_allow_html=True)

    section_header("📋", "Customer Risk Table")

    display_cols = [CUSTOMER_ID_COL, "churn_probability", "segment_name",
                    "churned", "months_inactive"]
    display_cols = [c for c in display_cols if c in df_view.columns]
    view_df = df_view[display_cols].copy().head(1000)

    if "churn_probability" in view_df.columns:
        view_df["risk_tier"] = view_df["churn_probability"].apply(risk_label)
        view_df["churn_probability"] = view_df["churn_probability"].round(4)

    st.dataframe(
        view_df.style.background_gradient(
            subset=["churn_probability"] if "churn_probability" in view_df.columns else [],
            cmap="RdYlGn_r", vmin=0, vmax=1
        ),
        use_container_width=True,
        height=380
    )

    dl_col, _ = st.columns([1, 3])
    with dl_col:
        st.download_button(
            "📥 Export CSV",
            data=view_df.to_csv(index=False).encode("utf-8"),
            file_name="churn_predictions.csv",
            mime="text/csv"
        )

    section_header("📊", "Risk Analysis")
    ch1, ch2 = st.columns(2)

    with ch1:
        st.markdown("**Churn Probability by Segment**")
        if "segment_name" in df_view.columns and "churn_probability" in df_view.columns:
            fig5 = go.Figure()
            segs = sorted(df_view["segment_name"].dropna().unique())
            for i, seg in enumerate(segs):
                seg_data = df_view[df_view["segment_name"] == seg]["churn_probability"]
                fig5.add_trace(go.Box(
                    y=seg_data,
                    name=seg,
                    marker_color=SEG_PALETTE[i % len(SEG_PALETTE)],
                    boxmean=True,
                    line_width=1.5,
                ))
            fig5.update_layout(
                yaxis_title="Churn Probability",
                showlegend=False,
                **PLOT_THEME
            )
            st.plotly_chart(fig5, use_container_width=True)

    with ch2:
        st.markdown("**Churn Rate: Actual vs Predicted**")
        if not metrics.empty:
            m_disp = metrics.copy()
            if "model" not in m_disp.columns:
                m_disp = m_disp.reset_index()

            m_disp.columns = m_disp.columns.str.strip()
            fig6 = go.Figure()

            for i, row_m in m_disp.iterrows():
                model_name = str(row_m.get("model", f"Model {i}"))
                for metric, color in [("precision", COLORS['blue']),
                                      ("recall",    COLORS['cyan']),
                                      ("f1",        COLORS['green']),
                                      ("roc_auc",   COLORS['amber'])]:
                    if metric in row_m:
                        fig6.add_trace(go.Scatter(
                            x=[model_name], y=[row_m[metric]],
                            mode="markers",
                            marker=dict(color=color, size=14, symbol="diamond"),
                            name=metric.replace("_", " ").title(),
                            showlegend=(i == 0),
                        ))
            fig6.update_layout(
                yaxis_title="Score", yaxis_range=[0, 1.05],
                legend=dict(font_size=10, bgcolor="rgba(0,0,0,0)"),
                **PLOT_THEME
            )
            st.plotly_chart(fig6, use_container_width=True)

    if search_id.strip() and len(df_view) == 1:
        section_header("👤", f"Customer Profile: {search_id}")
        row = df_view.iloc[0]
        p   = float(row.get("churn_probability", 0))

        c_a, c_b, c_c, c_d = st.columns(4)
        with c_a:
            st.metric("Churn Probability", f"{p:.1%}")
        with c_b:
            st.metric("Risk Tier", risk_label(p))
        with c_c:
            st.metric("Segment", row.get("segment_name", "—"))
        with c_d:
            mi = row.get("months_inactive", "—")
            st.metric("Months Inactive", str(mi) if pd.notna(mi) else "—")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=p * 100,
            number={"suffix": "%", "font": {"size": 28, "color": _text}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": _muted,
                         "tickfont": {"color": _muted}},
                "bar":  {"color": COLORS['red'] if p > HIGH_CHURN_RISK_THRESHOLD
                         else (COLORS['amber'] if p > MEDIUM_CHURN_RISK_THRESHOLD
                               else COLORS['green'])},
                "bgcolor": _chartbg,
                "steps": [
                    {"range": [0, 40],   "color": "rgba(16,185,129,0.1)"},
                    {"range": [40, 70],  "color": "rgba(245,158,11,0.1)"},
                    {"range": [70, 100], "color": "rgba(239,68,68,0.1)"},
                ],
                "threshold": {
                    "line": {"color": _text, "width": 2},
                    "thickness": 0.75, "value": p * 100
                },
            },
            title={"text": "Churn Risk Score", "font": {"color": _dim, "size": 13}},
        ))
        fig_gauge.update_layout(height=280, **PLOT_THEME)
        _, gcol, _ = st.columns([1, 2, 1])
        with gcol:
            st.plotly_chart(fig_gauge, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — CUSTOMER SEGMENTS
# ══════════════════════════════════════════════════════════════════════════════

elif "Segment" in page:

    st.markdown('<div class="page-title">Customer Segments</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">RFM-based KMeans clustering · Explore behavioural personas</div>', unsafe_allow_html=True)

    if segments.empty:
        st.warning("⚠️  No segment data found. Run `python run_pipeline.py` first.")
        st.stop()

    seg_data = segments.copy()
    if not scores.empty and CUSTOMER_ID_COL in seg_data.columns:
        seg_data = seg_data.merge(
            scores[[CUSTOMER_ID_COL, "churn_probability"]], on=CUSTOMER_ID_COL, how="left"
        )

    all_segs = sorted(seg_data["segment_name"].dropna().unique().tolist())
    selected = st.multiselect(
        "Select Segments", all_segs, default=all_segs,
        help="Choose which segments to display"
    )
    filtered = seg_data[seg_data["segment_name"].isin(selected)] if selected else seg_data

    kpi_html = '<div class="kpi-grid">'
    for i, seg in enumerate(all_segs):
        n = int((seg_data["segment_name"] == seg).sum())
        pct = n / max(len(seg_data), 1) * 100
        avg_churn = float(
            seg_data.loc[seg_data["segment_name"] == seg, "churn_probability"].mean()
        ) if "churn_probability" in seg_data.columns else 0
        col = SEG_PALETTE[i % len(SEG_PALETTE)]
        kpi_html += kpi_card(
            f"{n:,}", seg, col,
            f"{pct:.0f}% · avg risk {avg_churn:.0%}"
        )
    kpi_html += "</div>"
    st.markdown(kpi_html, unsafe_allow_html=True)

    section_header("☀️", "Segment Composition")
    if "segment_name" in filtered.columns:
        comp = filtered["segment_name"].value_counts().reset_index()
        comp.columns = ["segment", "count"]
        fig_sb = px.treemap(
            comp, path=["segment"], values="count",
            color="count", color_continuous_scale="Blues",
        )
        fig_sb.update_traces(marker=dict(line=dict(color=_chartbg, width=2)))
        fig_sb.update_layout(**PLOT_THEME, height=320, coloraxis_showscale=False)
        st.plotly_chart(fig_sb, use_container_width=True)

    ch1, ch2 = st.columns(2)

    with ch1:
        section_header("💠", "Frequency vs Monetary Value")
        if "frequency" in filtered.columns and "monetary" in filtered.columns:
            plot_df = filtered.copy()
            q99f = plot_df["frequency"].quantile(0.99)
            q99m = plot_df["monetary"].quantile(0.99)
            plot_df["frequency"] = plot_df["frequency"].clip(upper=q99f)
            plot_df["monetary"]  = plot_df["monetary"].clip(upper=q99m)

            fig7 = px.scatter(
                plot_df,
                x="frequency", y="monetary",
                color="segment_name",
                color_discrete_sequence=SEG_PALETTE,
                opacity=0.55,
                labels={"frequency": "Total Flights", "monetary": "Points Accumulated"},
                hover_data=[CUSTOMER_ID_COL] if CUSTOMER_ID_COL in plot_df.columns else None,
            )
            fig7.update_traces(marker=dict(size=5))
            fig7.update_layout(**PLOT_THEME, height=400,
                               legend=dict(font_size=10, bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig7, use_container_width=True)

    with ch2:
        section_header("🕒", "Recency Distribution by Segment")
        if "recency" in filtered.columns:
            fig8 = go.Figure()
            for i, seg in enumerate(selected):
                d = filtered[filtered["segment_name"] == seg]["recency"]
                fig8.add_trace(go.Box(
                    y=d, name=seg,
                    marker_color=SEG_PALETTE[i % len(SEG_PALETTE)],
                    boxmean=True, line_width=1.5
                ))
            fig8.update_layout(
                yaxis_title="Months Since Last Flight",
                showlegend=False,
                **PLOT_THEME, height=400
            )
            st.plotly_chart(fig8, use_container_width=True)

    section_header("⚠️", "Average Churn Risk by Segment")
    if "churn_probability" in filtered.columns:
        risk_by_seg = (
            filtered.groupby("segment_name")["churn_probability"]
            .mean()
            .sort_values(ascending=True)
            .reset_index()
        )
        risk_by_seg.columns = ["segment", "avg_churn_prob"]
        fig9 = go.Figure(go.Bar(
            x=risk_by_seg["avg_churn_prob"],
            y=risk_by_seg["segment"],
            orientation="h",
            marker=dict(
                color=risk_by_seg["avg_churn_prob"],
                colorscale=[[0, "#10B981"], [0.5, "#F59E0B"], [1, "#EF4444"]],
                showscale=True,
                colorbar=dict(title="Risk", tickfont=dict(color=_dim)),
                line=dict(width=0)
            ),
            text=[f"{v:.1%}" for v in risk_by_seg["avg_churn_prob"]],
            textposition="outside",
            textfont=dict(color=_dim),
        ))
        fig9.update_layout(
            xaxis_title="Average Churn Probability",
            xaxis_range=[0, 1],
            **PLOT_THEME, height=350
        )
        st.plotly_chart(fig9, use_container_width=True)

    section_header("📋", "Segment Summary Statistics")
    sum_cols = ["segment_name", "recency", "frequency", "monetary",
                "R_score", "F_score", "M_score", "RFM_score", "churn_probability"]
    avail    = [c for c in sum_cols if c in filtered.columns]
    if avail:
        tbl = filtered[avail].groupby("segment_name").agg(["mean", "count"]).round(2)
        st.dataframe(tbl, use_container_width=True)

    radar_path = os.path.join(PLOTS_DIR, "segment_radar_charts.png")
    if os.path.exists(radar_path):
        section_header("🕸️", "RFM Radar Profiles")
        st.image(radar_path, use_column_width=True)

    dcol, _ = st.columns([1, 4])
    with dcol:
        st.download_button(
            "📥 Export Segments",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="segments.csv",
            mime="text/csv"
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — RETENTION RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════

elif "Retention" in page:

    st.markdown('<div class="page-title">Retention Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Personalised actions · Reduce churn · Maximise CLV</div>', unsafe_allow_html=True)

    if recs.empty:
        st.warning("⚠️  No recommendations found. Run `python run_pipeline.py` first.")
        st.stop()

    total_roi = float(recs["est_roi"].sum()) if "est_roi" in recs.columns else 0
    avg_roi   = float(recs["est_roi"].mean()) if "est_roi" in recs.columns else 0
    n_high    = int((recs["risk_tier"] == "High Risk").sum()) if "risk_tier" in recs.columns else 0
    avg_cost  = float(recs["cost_usd"].mean()) if "cost_usd" in recs.columns else 0

    st.markdown(f"""
    <div class="kpi-grid">
        {kpi_card(f"{len(recs):,}", "Total Recommendations", COLORS['blue'])}
        {kpi_card(f"${total_roi:,.0f}", "Estimated Total ROI", COLORS['green'])}
        {kpi_card(f"${avg_roi:.1f}", "Avg ROI per Customer", COLORS['cyan'])}
        {kpi_card(f"{n_high:,}", "High-Risk Interventions", COLORS['red'])}
        {kpi_card(f"${avg_cost:.1f}", "Avg Campaign Cost", COLORS['amber'])}
    </div>
    """, unsafe_allow_html=True)

    section_header("🔎", "Filters")
    fc1, fc2, fc3 = st.columns(3)

    with fc1:
        risk_opt = st.selectbox(
            "Risk Tier", ["All"] +
            sorted(recs["risk_tier"].dropna().unique().tolist())
            if "risk_tier" in recs.columns else ["All"]
        )
    with fc2:
        seg_opt = st.selectbox(
            "Segment", ["All"] +
            sorted(recs["segment_name"].dropna().unique().tolist())
            if "segment_name" in recs.columns else ["All"]
        )
    with fc3:
        chan_opt = st.selectbox(
            "Channel", ["All"] +
            sorted(recs["channel"].dropna().unique().tolist())
            if "channel" in recs.columns else ["All"]
        )

    df_r = recs.copy()
    if risk_opt != "All" and "risk_tier"    in df_r.columns: df_r = df_r[df_r["risk_tier"]    == risk_opt]
    if seg_opt  != "All" and "segment_name" in df_r.columns: df_r = df_r[df_r["segment_name"] == seg_opt]
    if chan_opt != "All" and "channel"      in df_r.columns: df_r = df_r[df_r["channel"]      == chan_opt]

    rc1, rc2 = st.columns(2)

    with rc1:
        section_header("📊", "Action Distribution")
        if "action" in df_r.columns:
            ac = df_r["action"].value_counts().reset_index()
            ac.columns = ["action", "count"]
            fig10 = go.Figure(go.Bar(
                x=ac["count"],
                y=ac["action"],
                orientation="h",
                marker=dict(
                    color=SEG_PALETTE[:len(ac)],
                    opacity=0.85, line=dict(width=0)
                ),
                text=ac["count"],
                textposition="outside",
                textfont=dict(color=_dim),
            ))
            fig10.update_layout(**themed(
                xaxis_title="Customers",
                yaxis=dict(categoryorder="total ascending"),
                showlegend=False,
                height=380,
            ))
            st.plotly_chart(fig10, use_container_width=True)

    with rc2:
        section_header("💰", "ROI by Risk Tier")
        if "est_roi" in df_r.columns and "risk_tier" in df_r.columns:
            roi_by_risk = df_r.groupby("risk_tier")["est_roi"].agg(["sum", "mean"]).reset_index()
            roi_by_risk.columns = ["risk_tier", "total_roi", "avg_roi"]

            fig11 = go.Figure()
            fig11.add_trace(go.Bar(
                name="Total ROI ($)",
                x=roi_by_risk["risk_tier"],
                y=roi_by_risk["total_roi"],
                marker_color=COLORS['green'],
                opacity=0.8,
                yaxis="y"
            ))
            fig11.add_trace(go.Scatter(
                name="Avg ROI ($)",
                x=roi_by_risk["risk_tier"],
                y=roi_by_risk["avg_roi"],
                mode="lines+markers",
                line=dict(color=COLORS['cyan'], width=2),
                marker=dict(size=8),
                yaxis="y2"
            ))
            fig11.update_layout(**themed(
                yaxis=dict(title="Total ROI ($)", color=COLORS['green']),
                yaxis2=dict(title="Avg ROI ($)", overlaying="y", side="right",
                            color=COLORS['cyan']),
                legend=dict(font_size=10, bgcolor="rgba(0,0,0,0)"),
                height=380,
            ))
            st.plotly_chart(fig11, use_container_width=True)

    section_header("📋", f"Recommendation Table  ({len(df_r):,} customers)")
    show = [CUSTOMER_ID_COL, "segment_name", "risk_tier", "churn_probability",
            "action", "channel", "timing", "cost_usd", "est_retention_lift", "est_roi"]
    show = [c for c in show if c in df_r.columns]

    st.dataframe(
        df_r[show].head(500).style.background_gradient(
            subset=["churn_probability"] if "churn_probability" in df_r.columns else [],
            cmap="RdYlGn_r", vmin=0, vmax=1
        ).background_gradient(
            subset=["est_roi"] if "est_roi" in df_r.columns else [],
            cmap="Greens", vmin=0
        ),
        use_container_width=True,
        height=420
    )

    dcol2, _ = st.columns([1, 4])
    with dcol2:
        st.download_button(
            "📥 Export Recommendations",
            data=df_r.to_csv(index=False).encode("utf-8"),
            file_name="retention_recommendations.csv",
            mime="text/csv"
        )

    section_header("📖", "Action Playbook")
    playbook = {
        "🛋️  Lounge Pass":       ("Complimentary lounge access", COLORS['blue']),
        "⬆️  Tier Boost":        ("90-day status upgrade to drive re-engagement", COLORS['purple']),
        "⏰  Expiring Points":   ("Remind hoarders before points expire", COLORS['amber']),
        "🏆  Win-Back Discount": ("50% bonus miles for dormant reactivation", COLORS['red']),
        "🌿  Seasonal Offer":    ("Targeted destination deal before peak season", COLORS['cyan']),
        "✈️  Mileage Bonus":     ("3x miles for 30 days to boost frequency", COLORS['green']),
        "🤝  CLV Protection":    ("Dedicated account manager for high-value customers", COLORS['pink']),
    }
    pb_cols = st.columns(4)
    for i, (title, (desc, color)) in enumerate(playbook.items()):
        with pb_cols[i % 4]:
            st.markdown(insight_card(title, desc, color), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════

elif "Explain" in page:

    st.markdown('<div class="page-title">Explainability Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">SHAP-based model explanations · Understand why customers churn</div>', unsafe_allow_html=True)

    section_header("🌍", "Global Churn Drivers (SHAP)")

    if not shap_imp.empty and "feature" in shap_imp.columns:
        top = shap_imp.head(20).copy()

        fig12 = go.Figure(go.Bar(
            x=top["mean_abs_shap"],
            y=top["feature"],
            orientation="h",
            marker=dict(
                color=top["mean_abs_shap"],
                colorscale=[[0, COLORS['blue']], [0.5, COLORS['amber']], [1, COLORS['red']]],
                showscale=True,
                colorbar=dict(title="SHAP", tickfont=dict(color=_dim)),
                line=dict(width=0)
            ),
            text=[f"{v:.4f}" for v in top["mean_abs_shap"]],
            textposition="outside",
            textfont=dict(color=_dim, size=10),
        ))
        fig12.update_layout(**themed(
            xaxis_title="Mean |SHAP Value|",
            yaxis=dict(categoryorder="total ascending"),
            height=600,
        ))
        st.plotly_chart(fig12, use_container_width=True)

        section_header("📋", "Full Feature Importance Table")
        st.dataframe(shap_imp, use_container_width=True, height=300)

    else:
        shap_path = os.path.join(PLOTS_DIR, "shap_global_importance.png")
        if os.path.exists(shap_path):
            st.image(shap_path, use_column_width=True)
        else:
            st.info("Run the pipeline with SHAP enabled to generate explanations.")

    beeswarm_path = os.path.join(PLOTS_DIR, "shap_beeswarm.png")
    if os.path.exists(beeswarm_path):
        section_header("🐝", "SHAP Beeswarm Summary")
        st.image(beeswarm_path, use_column_width=True)

    fi_rf  = os.path.join(PLOTS_DIR, "feature_importance_random_forest.png")
    fi_xgb = os.path.join(PLOTS_DIR, "feature_importance_xgboost.png")

    if os.path.exists(fi_rf) or os.path.exists(fi_xgb):
        section_header("🌲", "Model Feature Importance")
        mc1, mc2 = st.columns(2)
        with mc1:
            if os.path.exists(fi_rf):
                st.markdown("**Random Forest**")
                st.image(fi_rf, use_column_width=True)
        with mc2:
            if os.path.exists(fi_xgb):
                st.markdown("**XGBoost**")
                st.image(fi_xgb, use_column_width=True)

    section_header("💼", "Business Interpretation of Key Features")

    interpretations = [
        ("recency_months", COLORS['red'],
         "Months since last flight. The strongest churn signal — customers inactive for 3+ months need immediate intervention."),
        ("max_inactivity_streak", COLORS['amber'],
         "Longest consecutive run of zero-flight months. A streak >3 months predicts churn with high reliability."),
        ("alltime_total_flights_sum", COLORS['blue'],
         "Lifetime flight count. Low cumulative flights signals shallow programme engagement."),
        ("points_redeemed_ratio", COLORS['cyan'],
         "Redemption / accumulation ratio. Customers who earn but never redeem are weakly attached to the programme."),
        ("tenure_months", COLORS['green'],
         "Programme tenure. Newer members (<12 months) have significantly higher churn rates — critical onboarding period."),
        ("clv", COLORS['purple'],
         "Customer Lifetime Value. High-CLV customers justify premium retention investments (lounge passes, account managers)."),
        ("seasonal_concentration", COLORS['amber'],
         "Herfindahl index of quarterly flights. Highly seasonal travellers need targeted pre-season engagement."),
        ("dollar_redemption_rate", COLORS['cyan'],
         "Dollar value per redeemed point. Low rate suggests customers aren't getting value from redemptions."),
    ]

    ic_cols = st.columns(2)
    for i, (feat, color, interp) in enumerate(interpretations):
        with ic_cols[i % 2]:
            st.markdown(insight_card(feat, interp, color), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — CUSTOMER 360
# ══════════════════════════════════════════════════════════════════════════════

elif "360" in page:
    render_customer_360()


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    SkyLoyalty Intelligence Platform · Built with Streamlit + XGBoost + SHAP
    · Airline Loyalty Analytics v1.0
</div>
""", unsafe_allow_html=True)