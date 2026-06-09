"""
app.py
------
Streamlit dashboard — redesigned with a premium, futuristic SaaS UI.
All backend logic, variable names, function calls, dataframe operations,
ML predictions, file paths, imports and data connections are preserved.
Only the presentation layer (CSS / HTML / layout) has been upgraded.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    REPORTS_DIR, PLOTS_DIR, MODELS_DIR,
    HIGH_CHURN_RISK_THRESHOLD,
    MEDIUM_CHURN_RISK_THRESHOLD,
    CUSTOMER_ID_COL
)

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

/* ── Root variables ── */
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

/* ── Animated aurora background ── */
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

/* ── Make Streamlit's top header transparent so it doesn't cover titles ── */
[data-testid="stHeader"] {
    background: rgba(7, 11, 22, 0.0) !important;
    backdrop-filter: blur(6px);
}
[data-testid="stHeader"]::before { content: none !important; }
[data-testid="stToolbar"] { right: 1rem; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(11,16,28,0.96) 0%, rgba(15,22,38,0.92) 100%) !important;
    border-right: 1px solid var(--border);
    backdrop-filter: blur(18px);
}
[data-testid="stSidebar"] .block-container { padding: 1.2rem 1rem; }

/* ── Brand block ── */
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

/* ── Sidebar stats ── */
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

/* ── Page title ── */
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

/* ── KPI Cards ── */
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

/* ── Section headers ── */
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

/* ── Risk badges ── */
.badge {
    display: inline-block; padding: 0.22rem 0.75rem; border-radius: 20px;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.03em;
}
.badge-high   { background: rgba(239,68,68,0.15);  color: #EF4444; border: 1px solid rgba(239,68,68,0.3); }
.badge-medium { background: rgba(245,158,11,0.15); color: #F59E0B; border: 1px solid rgba(245,158,11,0.3);}
.badge-low    { background: rgba(16,185,129,0.15); color: #10B981; border: 1px solid rgba(16,185,129,0.3);}

/* ── Insight / persona cards ── */
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

/* ── Result count pill ── */
.result-pill {
    display: inline-flex; align-items: center; gap: 0.5rem;
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 30px; padding: 0.45rem 1.1rem; margin: 0.4rem 0 0.8rem 0;
    font-size: 0.85rem; color: var(--text-dim); backdrop-filter: blur(10px);
}
.result-pill b { color: var(--accent-blue); font-family: 'Space Grotesk', sans-serif; }

/* ── Dataframe styling ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 14px; overflow: hidden;
    box-shadow: var(--glow);
}

/* ── Metric overrides ── */
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

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #3B82F6, #06B6D4) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    font-weight: 600 !important; padding: 0.5rem 1.6rem !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
    box-shadow: 0 6px 18px rgba(59,130,246,0.35) !important;
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 10px 26px rgba(59,130,246,0.5) !important; }

/* ── Selectbox / inputs ── */
.stSelectbox > div > div,
.stTextInput > div > div,
.stMultiSelect > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important; color: var(--text-primary) !important;
    backdrop-filter: blur(10px);
}

/* ── Download button ── */
.stDownloadButton > button {
    background: rgba(59,130,246,0.1) !important; color: var(--accent-blue) !important;
    border: 1px solid rgba(59,130,246,0.3) !important; border-radius: 10px !important;
    font-weight: 500 !important; transition: background 0.2s !important;
}
.stDownloadButton > button:hover { background: rgba(59,130,246,0.22) !important; }

/* ── Radio nav (sidebar) ── */
[data-testid="stSidebar"] .stRadio > div { gap: 0.25rem; }
[data-testid="stSidebar"] .stRadio label {
    background: transparent; border-radius: 10px; padding: 0.55rem 0.8rem;
    transition: background 0.15s, color 0.15s; color: var(--text-dim) !important;
    font-size: 0.92rem;
}
[data-testid="stSidebar"] .stRadio label:hover { background: rgba(59,130,246,0.10); }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { gap: 0.4rem; }
.stTabs [data-baseweb="tab"] {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 10px 10px 0 0; color: var(--text-dim);
}
.stTabs [aria-selected="true"] {
    background: rgba(59,130,246,0.16) !important; color: var(--accent-blue) !important;
}

/* ── Expander ── */
.streamlit-expanderHeader, [data-testid="stExpander"] summary {
    background: var(--bg-card) !important; border-radius: 10px !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1rem 0 !important; }

/* ── Alert / info boxes ── */
.stAlert { border-radius: 12px !important; backdrop-filter: blur(8px); }

/* ── Plotly chart containers ── */
.js-plotly-plot { border-radius: 14px; }

/* ── Footer ── */
.app-footer {
    text-align: center; margin-top: 2.5rem; padding-top: 1.4rem;
    border-top: 1px solid var(--border);
    font-size: 0.78rem; color: var(--text-muted); letter-spacing: 0.03em;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 7px; height: 7px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Light theme override (injected only when light mode is active) ───────────────
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

# ── Plotly theme (adapts to light / dark) ───────────────────────────────────────
_grid  = "#D5DEEA" if LIGHT else "#1e2d45"
_fcol  = "#475569" if LIGHT else "#94A3B8"
PLOT_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color=_fcol, size=11),
    xaxis=dict(gridcolor=_grid, linecolor=_grid, zerolinecolor=_grid),
    yaxis=dict(gridcolor=_grid, linecolor=_grid, zerolinecolor=_grid),
    margin=dict(t=30, b=30, l=10, r=10),
)


def themed(**overrides):
    """Return PLOT_THEME merged with overrides (deep-merging axis dicts)
    so callers can safely pass their own xaxis/yaxis without colliding
    with the base theme keys."""
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
    # IMPORTANT: keep this HTML free of leading indentation. Streamlit's
    # Markdown renderer treats lines indented by 4+ spaces as a code block,
    # which makes the raw <div> tags appear on screen instead of rendering.
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


# ── Load data ──────────────────────────────────────────────────────────────────
with st.spinner("Loading loyalty intelligence…"):
    (segments, scores, recs, metrics,
     features, labels, shap_imp) = load_all_data()

# Build master for convenience
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



    # Live stats in sidebar
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

    # ── KPI Row ────────────────────────────────────────────────────────────────
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

    # ── Charts Row 1 ──────────────────────────────────────────────────────────
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
                marker=dict(colors=SEG_PALETTE, line=dict(color="#0A0E1A", width=2)),
                textinfo="percent",
                textfont_size=11,
            ))
            fig2.update_layout(
                showlegend=True,
                legend=dict(
                    orientation="v", x=1.0, y=0.5,
                    font=dict(size=10, color="#94A3B8"),
                    bgcolor="rgba(0,0,0,0)"
                ),
                annotations=[dict(
                    text=f"{total_customers:,}<br>members",
                    x=0.5, y=0.5, font_size=14,
                    font_color="#F1F5F9", showarrow=False
                )],
                **PLOT_THEME
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── Charts Row 2 ──────────────────────────────────────────────────────────
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
                textfont=dict(color="#94A3B8", size=11),
            ))
            fig4.update_layout(
                xaxis_title="Customers",
                showlegend=False,
                **PLOT_THEME
            )
            st.plotly_chart(fig4, use_container_width=True)

    # ── ROC Curve image ───────────────────────────────────────────────────────
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

    # ── Filter bar ────────────────────────────────────────────────────────────
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

    # ── Apply filters ─────────────────────────────────────────────────────────
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

    # ── Result count ──────────────────────────────────────────────────────────
    extra = f"of {len(master):,}" if len(df_view) < len(master) else ""
    st.markdown(f"""
    <div class="result-pill">Showing <b>{len(df_view):,}</b> customers {extra}</div>
    """, unsafe_allow_html=True)

    # ── Table ─────────────────────────────────────────────────────────────────
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

    # ── Distribution Charts ────────────────────────────────────────────────────
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

    # ── Individual customer deep-dive ─────────────────────────────────────────
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

        # Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=p * 100,
            number={"suffix": "%", "font": {"size": 28, "color": "#F1F5F9"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#64748B",
                         "tickfont": {"color": "#64748B"}},
                "bar":  {"color": COLORS['red'] if p > HIGH_CHURN_RISK_THRESHOLD
                         else (COLORS['amber'] if p > MEDIUM_CHURN_RISK_THRESHOLD
                               else COLORS['green'])},
                "bgcolor": "#111827",
                "steps": [
                    {"range": [0, 40],   "color": "rgba(16,185,129,0.1)"},
                    {"range": [40, 70],  "color": "rgba(245,158,11,0.1)"},
                    {"range": [70, 100], "color": "rgba(239,68,68,0.1)"},
                ],
                "threshold": {
                    "line": {"color": "#F1F5F9", "width": 2},
                    "thickness": 0.75, "value": p * 100
                },
            },
            title={"text": "Churn Risk Score", "font": {"color": "#94A3B8", "size": 13}},
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

    # ── Segment selector ──────────────────────────────────────────────────────
    all_segs = sorted(seg_data["segment_name"].dropna().unique().tolist())
    selected = st.multiselect(
        "Select Segments", all_segs, default=all_segs,
        help="Choose which segments to display"
    )
    filtered = seg_data[seg_data["segment_name"].isin(selected)] if selected else seg_data

    # ── Segment KPIs ──────────────────────────────────────────────────────────
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

    # ── Sunburst / treemap overview ────────────────────────────────────────────
    section_header("☀️", "Segment Composition")
    if "segment_name" in filtered.columns:
        comp = filtered["segment_name"].value_counts().reset_index()
        comp.columns = ["segment", "count"]
        fig_sb = px.treemap(
            comp, path=["segment"], values="count",
            color="count", color_continuous_scale="Blues",
        )
        fig_sb.update_traces(marker=dict(line=dict(color="#0A0E1A", width=2)))
        fig_sb.update_layout(**PLOT_THEME, height=320, coloraxis_showscale=False)
        st.plotly_chart(fig_sb, use_container_width=True)

    # ── Charts ────────────────────────────────────────────────────────────────
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

    # ── Churn risk per segment ─────────────────────────────────────────────────
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
                colorbar=dict(title="Risk", tickfont=dict(color="#94A3B8")),
                line=dict(width=0)
            ),
            text=[f"{v:.1%}" for v in risk_by_seg["avg_churn_prob"]],
            textposition="outside",
            textfont=dict(color="#94A3B8"),
        ))
        fig9.update_layout(
            xaxis_title="Average Churn Probability",
            xaxis_range=[0, 1],
            **PLOT_THEME, height=350
        )
        st.plotly_chart(fig9, use_container_width=True)

    # ── Summary table ─────────────────────────────────────────────────────────
    section_header("📋", "Segment Summary Statistics")
    sum_cols = ["segment_name", "recency", "frequency", "monetary",
                "R_score", "F_score", "M_score", "RFM_score", "churn_probability"]
    avail    = [c for c in sum_cols if c in filtered.columns]
    if avail:
        tbl = filtered[avail].groupby("segment_name").agg(["mean", "count"]).round(2)
        st.dataframe(tbl, use_container_width=True)

    # ── Radar charts ──────────────────────────────────────────────────────────
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

    # ── Summary KPIs ──────────────────────────────────────────────────────────
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

    # ── Filters ────────────────────────────────────────────────────────────────
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

    # ── Charts row ─────────────────────────────────────────────────────────────
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
                textfont=dict(color="#94A3B8"),
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

    # ── Recommendations table ─────────────────────────────────────────────────
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

    # ── Action playbook ───────────────────────────────────────────────────────
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

    # ── Global SHAP ────────────────────────────────────────────────────────────
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
                colorbar=dict(title="SHAP", tickfont=dict(color="#94A3B8")),
                line=dict(width=0)
            ),
            text=[f"{v:.4f}" for v in top["mean_abs_shap"]],
            textposition="outside",
            textfont=dict(color="#94A3B8", size=10),
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

    # ── SHAP Beeswarm ──────────────────────────────────────────────────────────
    beeswarm_path = os.path.join(PLOTS_DIR, "shap_beeswarm.png")
    if os.path.exists(beeswarm_path):
        section_header("🐝", "SHAP Beeswarm Summary")
        st.image(beeswarm_path, use_column_width=True)

    # ── Feature importance plots ───────────────────────────────────────────────
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

    # ── Business interpretation ────────────────────────────────────────────────
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
    # Import and render the Customer 360 page
    try:
        from pages.customer_360_page import render as render_360
        render_360()
    except Exception as e:
        st.error(f"Customer 360 failed to load: {e}")
        st.exception(e)
# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    SkyLoyalty Intelligence Platform · Built with Streamlit + XGBoost + SHAP
    · Airline Loyalty Analytics v1.0
</div>
""", unsafe_allow_html=True)
