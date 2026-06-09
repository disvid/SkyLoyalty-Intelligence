"""
pages/customer_360_page.py
--------------------------
Customer 360 / Customer Intelligence Center — Streamlit page.

Renders a complete intelligence report for any customer ID.
Sections:
  1. Customer Overview
  2. Churn Analysis
  3. Top Churn Drivers
  4. Customer Segment
  5. Flight Behaviour (interactive Plotly charts)
  6. Value Assessment
  7. Next Best Action
  8. Customer Health Score
  9. Export (CSV + PDF)
"""

import os
import io
import sys
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.customer_360 import Customer360
from config import CUSTOMER_ID_COL

# ── Design tokens (match app.py dark theme) ────────────────────────────────────
COLORS = {
    "blue":   "#3B82F6",
    "cyan":   "#06B6D4",
    "green":  "#10B981",
    "amber":  "#F59E0B",
    "red":    "#EF4444",
    "purple": "#8B5CF6",
    "pink":   "#EC4899",
    "orange": "#F97316",
}

PLOT_BG = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#94A3B8", size=11),
    xaxis=dict(gridcolor="#1e2d45", linecolor="#1e2d45", zerolinecolor="#1e2d45"),
    yaxis=dict(gridcolor="#1e2d45", linecolor="#1e2d45", zerolinecolor="#1e2d45"),
    margin=dict(t=30, b=30, l=10, r=10),
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
@import url('https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@2.30.0/tabler-icons.min.css');

:root {
    --bg:     #0A0E1A;
    --card:   #111827;
    --border: #1e2d45;
    --muted:  #64748B;
    --dim:    #94A3B8;
    --text:   #F1F5F9;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
.main .block-container { padding: 1.5rem 2rem; max-width: 1400px; }

/* ── Page title ── */
.c360-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.8rem; font-weight: 700;
    background: linear-gradient(135deg, #3B82F6, #06B6D4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.15rem;
}
.c360-sub { font-size: 0.85rem; color: var(--muted); margin-bottom: 1.5rem; }

/* ── Section label ── */
.sec-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--muted);
    display: flex; align-items: center; gap: 8px;
    margin: 1.8rem 0 0.8rem 0;
}
.sec-label::after {
    content: ''; flex: 1; height: 1px; background: var(--border);
}

/* ── Glass cards ── */
.glass-card {
    background: rgba(17,24,39,0.85);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.2s;
}
.glass-card:hover { border-color: #2d4a6e; }

/* ── KPI cards ── */
.kpi-row { display: flex; flex-wrap: wrap; gap: 0.8rem; margin-bottom: 1rem; }
.kpi-mini {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    flex: 1; min-width: 150px;
    position: relative; overflow: hidden;
}
.kpi-mini::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: var(--ac, #3B82F6); border-radius: 12px 12px 0 0;
}
.kpi-val  { font-family: 'Space Grotesk',sans-serif; font-size:1.4rem; font-weight:700; color: var(--ac, #3B82F6); }
.kpi-lbl  { font-size:0.68rem; text-transform:uppercase; letter-spacing:0.08em; color:var(--muted); margin-top:2px; }
.kpi-sub  { font-size:0.75rem; color:var(--dim); margin-top:4px; }

/* ── Risk badge ── */
.risk-badge {
    display:inline-flex; align-items:center; gap:6px;
    padding: 0.35rem 1rem; border-radius: 20px;
    font-size: 0.82rem; font-weight: 600; letter-spacing: 0.02em;
}

/* ── Driver cards ── */
.driver-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 0.8rem 1rem;
    display: flex; align-items: flex-start; gap: 10px;
    margin-bottom: 0.6rem;
}
.driver-icon {
    width: 32px; height: 32px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; flex-shrink: 0;
}
.driver-factor { font-size: 0.85rem; font-weight: 600; color: var(--text); }
.driver-detail { font-size: 0.78rem; color: var(--dim); margin-top: 2px; }

/* ── NBA card ── */
.nba-card {
    background: linear-gradient(135deg, rgba(59,130,246,0.08), rgba(6,182,212,0.05));
    border: 1px solid rgba(59,130,246,0.25);
    border-radius: 14px; padding: 1.4rem 1.6rem; margin-bottom: 1rem;
}
.nba-row { display:flex; gap: 1rem; flex-wrap: wrap; margin-top: 1rem; }
.nba-item { flex:1; min-width:130px; }
.nba-item-label {
    font-size: 0.68rem; text-transform:uppercase;
    letter-spacing: 0.1em; color: var(--muted); margin-bottom: 4px;
}
.nba-item-value { font-size: 0.9rem; font-weight: 500; color: var(--text); }

/* ── Health score ── */
.health-ring-wrap { text-align:center; padding: 1rem 0; }
.health-score-num {
    font-family: 'Space Grotesk',sans-serif;
    font-size: 3.5rem; font-weight: 700; line-height:1;
}
.health-status {
    font-size: 1rem; font-weight: 600; margin-top: 4px; letter-spacing: 0.05em;
}

/* ── Value bar ── */
.val-bar-wrap { background: #1e2d45; border-radius: 99px; height: 10px; margin: 8px 0; }
.val-bar { height: 10px; border-radius: 99px; }

/* ── Export buttons ── */
.export-row { display:flex; gap:0.8rem; flex-wrap:wrap; margin-top: 1rem; }

/* ── Segment chip ── */
.seg-chip {
    display:inline-block; padding: 0.3rem 1rem; border-radius: 20px;
    font-size: 0.82rem; font-weight: 600;
    background: rgba(139,92,246,0.15); color: #8B5CF6;
    border: 1px solid rgba(139,92,246,0.3);
}

/* ── Progress bar ── */
.prog-wrap { background:#1e2d45; border-radius:99px; height:8px; margin:6px 0; }
.prog-bar  { height:8px; border-radius:99px; transition: width 0.5s ease; }

/* ── Timeline dots ── */
.tl-dot {
    width:10px; height:10px; border-radius:50%;
    display:inline-block; margin-right:6px;
}

/* ── Not found ── */
.not-found {
    text-align:center; padding: 4rem 2rem;
    color: var(--muted); font-size: 1rem;
}

/* Streamlit overrides */
[data-testid="stSidebar"] { background: linear-gradient(180deg,#0D1321,#111827)!important; }
.stButton>button {
    background: linear-gradient(135deg,#3B82F6,#06B6D4)!important;
    color:#fff!important; border:none!important; border-radius:8px!important;
    font-weight:600!important; padding:0.45rem 1.2rem!important;
}
.stDownloadButton>button {
    background: rgba(59,130,246,0.1)!important; color:#3B82F6!important;
    border:1px solid rgba(59,130,246,0.3)!important; border-radius:8px!important;
    font-weight:500!important;
}
div[data-testid="stSelectbox"] > div > div {
    background: var(--card)!important; border:1px solid var(--border)!important;
    border-radius:8px!important; color:var(--text)!important;
}
</style>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@2.30.0/tabler-icons.min.css">
""", unsafe_allow_html=True)


# ── Cached data loader ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_c360():
    return Customer360()


# ── Helper renderers ───────────────────────────────────────────────────────────

def sec(icon: str, title: str):
    st.markdown(
        f'<div class="sec-label"><i class="ti {icon}" style="font-size:15px"></i> {title}</div>',
        unsafe_allow_html=True
    )


def kpi(value: str, label: str, color: str, sub: str = ""):
    st.markdown(f"""
    <div class="kpi-mini" style="--ac:{color}">
        <div class="kpi-val" style="color:{color}">{value}</div>
        <div class="kpi-lbl">{label}</div>
        {"<div class='kpi-sub'>" + sub + "</div>" if sub else ""}
    </div>""", unsafe_allow_html=True)


def risk_badge_html(cat: str, color: str) -> str:
    icons = {
        "Low Risk":      "ti-shield-check",
        "Medium Risk":   "ti-shield-half-filled",
        "High Risk":     "ti-shield-exclamation",
        "Critical Risk": "ti-skull",
    }
    icon = icons.get(cat, "ti-shield")
    return f"""
    <span class="risk-badge" style="background:rgba(0,0,0,0.3);
          color:{color}; border:1px solid {color}40;">
        <i class="ti {icon}"></i> {cat}
    </span>"""


def gauge_chart(value: float, color: str, title: str, suffix: str = "%") -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value, 1),
        number={"suffix": suffix, "font": {"size": 32, "color": "#F1F5F9",
                                            "family": "Space Grotesk"}},
        gauge={
            "axis":    {"range": [0, 100], "tickcolor": "#64748B",
                        "tickfont": {"color": "#64748B", "size": 10}},
            "bar":     {"color": color, "thickness": 0.25},
            "bgcolor": "#0A0E1A",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30],  "color": "rgba(16,185,129,0.08)"},
                {"range": [30, 60], "color": "rgba(245,158,11,0.08)"},
                {"range": [60, 80], "color": "rgba(239,68,68,0.08)"},
                {"range": [80, 100],"color": "rgba(124,58,237,0.08)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.75, "value": value
            },
        },
        title={"text": title, "font": {"color": "#64748B", "size": 12}},
    ))
    fig.update_layout(height=230, **PLOT_BG)
    return fig


# ── PDF generator (pure Python, no extra deps) ────────────────────────────────

def generate_pdf_bytes(profile: dict, cid: str) -> bytes:
    """
    Generate a simple plain-text 'PDF' as bytes using fpdf2 if available,
    falling back to a plain UTF-8 text report if not.
    """
    try:
        from fpdf import FPDF

        class PDF(FPDF):
            def header(self):
                self.set_font("Helvetica", "B", 14)
                self.set_text_color(59, 130, 246)
                self.cell(0, 10, f"Customer Intelligence Report — {cid}", ln=True)
                self.set_draw_color(30, 45, 69)
                self.line(10, 20, 200, 20)
                self.ln(5)

            def footer(self):
                self.set_y(-15)
                self.set_font("Helvetica", "I", 8)
                self.set_text_color(100, 116, 139)
                self.cell(0, 10, f"SkyLoyalty Intelligence Platform  |  Page {self.page_no()}", align="C")

        pdf = PDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        def h2(text):
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(59, 130, 246)
            pdf.cell(0, 8, text, ln=True)
            pdf.set_text_color(241, 245, 249)

        def row(label, value):
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(148, 163, 184)
            pdf.cell(60, 6, label, ln=False)
            pdf.set_text_color(241, 245, 249)
            pdf.cell(0, 6, str(value), ln=True)

        def spacer():
            pdf.ln(4)

        ov = profile["overview"]
        ch = profile["churn"]
        sg = profile["segment"]
        va = profile["value"]
        hs = profile["health"]
        na = profile["next_action"]

        h2("1. CUSTOMER OVERVIEW")
        spacer()
        row("Loyalty Tier",    ov["loyalty_tier"])
        row("Enrollment Type", ov["enrollment_type"])
        row("Education",       ov["education"])
        row("Marital Status",  ov["marital_status"])
        row("Province",        ov["province"])
        row("Income",          f"${ov['salary']:,.0f}")
        row("CLV",             f"${ov['clv']:,.2f}")
        row("Tenure",          f"{ov['tenure_months']} months")
        spacer()

        h2("2. CHURN ANALYSIS")
        spacer()
        row("Churn Probability", f"{ch['probability_pct']}%")
        row("Risk Category",     ch["risk_category"])
        row("Months Inactive",   ch["months_inactive"])
        spacer()

        h2("3. TOP CHURN DRIVERS")
        spacer()
        for d in profile["drivers"]:
            row(d["factor"], d["detail"])
        spacer()

        h2("4. CUSTOMER SEGMENT")
        spacer()
        row("Segment",       sg["name"])
        row("Segment Size",  f"{sg['count']:,} customers")
        row("Avg Seg Churn", f"{sg['avg_churn']*100:.1f}%")
        row("Avg Seg CLV",   f"${sg['avg_clv']:,.0f}")
        row("RFM Score",     f"{sg['rfm_score']:.1f}")
        spacer()

        h2("5. VALUE ASSESSMENT")
        spacer()
        row("Future Value Score", f"{va['score']}/100")
        row("Category",          va["category"])
        row("Description",       va["description"][:80] + "...")
        spacer()

        h2("6. NEXT BEST ACTION")
        spacer()
        row("Action",           na["action"])
        row("Channel",          na["channel"])
        row("Timing",           na["timing"])
        row("Est. Lift",        f"{na['est_lift']*100:.0f}%")
        row("Est. ROI",         f"${na['est_roi']:,.0f}")
        row("Why",              na["why"][:100] + "...")
        spacer()

        h2("7. HEALTH SCORE")
        spacer()
        row("Health Score",  f"{hs['score']}/100")
        row("Status",        hs["status"])

        return bytes(pdf.output())

    except ImportError:
        # Fallback: plain text report as bytes
        lines = [
            f"CUSTOMER INTELLIGENCE REPORT",
            f"Customer ID: {cid}",
            "=" * 50,
            "",
            "1. OVERVIEW",
            f"  Loyalty Tier:    {profile['overview']['loyalty_tier']}",
            f"  CLV:             ${profile['overview']['clv']:,.2f}",
            f"  Tenure:          {profile['overview']['tenure_months']} months",
            f"  Province:        {profile['overview']['province']}",
            "",
            "2. CHURN",
            f"  Probability:     {profile['churn']['probability_pct']}%",
            f"  Risk Category:   {profile['churn']['risk_category']}",
            "",
            "3. SEGMENT",
            f"  Name:            {profile['segment']['name']}",
            f"  RFM Score:       {profile['segment']['rfm_score']}",
            "",
            "4. VALUE SCORE",
            f"  Score:           {profile['value']['score']}/100",
            f"  Category:        {profile['value']['category']}",
            "",
            "5. NEXT BEST ACTION",
            f"  Action:          {profile['next_action']['action']}",
            f"  Channel:         {profile['next_action']['channel']}",
            f"  Timing:          {profile['next_action']['timing']}",
            "",
            "6. HEALTH SCORE",
            f"  Score:           {profile['health']['score']}/100",
            f"  Status:          {profile['health']['status']}",
        ]
        return "\n".join(lines).encode("utf-8")


def generate_csv_bytes(profile: dict, cid: str) -> bytes:
    """Flatten profile into a single-row CSV."""
    ov = profile["overview"]
    ch = profile["churn"]
    sg = profile["segment"]
    va = profile["value"]
    hs = profile["health"]
    na = profile["next_action"]
    drivers = "; ".join([d["factor"] for d in profile["drivers"]])

    row = {
        "customer_id":        cid,
        "loyalty_tier":       ov["loyalty_tier"],
        "enrollment_type":    ov["enrollment_type"],
        "education":          ov["education"],
        "marital_status":     ov["marital_status"],
        "province":           ov["province"],
        "salary":             ov["salary"],
        "clv":                ov["clv"],
        "tenure_months":      ov["tenure_months"],
        "months_inactive":    ov["months_inactive"],
        "churn_probability":  ch["probability"],
        "churn_probability_pct": ch["probability_pct"],
        "risk_category":      ch["risk_category"],
        "segment":            sg["name"],
        "segment_size":       sg["count"],
        "segment_avg_churn":  round(sg["avg_churn"] * 100, 2),
        "segment_avg_clv":    round(sg["avg_clv"], 2),
        "rfm_score":          sg["rfm_score"],
        "future_value_score": va["score"],
        "value_category":     va["category"],
        "health_score":       hs["score"],
        "health_status":      hs["status"],
        "recommended_action": na["action"],
        "channel":            na["channel"],
        "timing":             na["timing"],
        "est_retention_lift": na["est_lift"],
        "est_roi":            na["est_roi"],
        "top_churn_drivers":  drivers,
    }
    return pd.DataFrame([row]).to_csv(index=False).encode("utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN RENDER
# ══════════════════════════════════════════════════════════════════════════════

def render():
    """Main render function for Customer 360 page."""

    st.markdown('<div class="c360-title">Customer Intelligence Center</div>', unsafe_allow_html=True)
    st.markdown('<div class="c360-sub">Complete 360° profile · Churn risk · Segment · Next best action · Health score</div>', unsafe_allow_html=True)

    # ── Load backend ──────────────────────────────────────────────────────────
    with st.spinner("Loading customer data..."):
        c360 = get_c360()

    all_ids = c360.get_all_customer_ids()

    if not all_ids:
        st.error("No customer data found. Run `python run_pipeline.py` first.")
        return

    # ── Search bar ────────────────────────────────────────────────────────────
    sc1, sc2 = st.columns([3, 1])
    with sc1:
        typed_id = st.text_input(
            "Enter Customer ID",
            placeholder=f"e.g. {all_ids[0]} · {len(all_ids):,} customers available",
            label_visibility="collapsed"
        )
    with sc2:
        use_dropdown = st.checkbox("Browse list", value=False)

    if use_dropdown:
        selected_id = st.selectbox("Select Customer", all_ids, label_visibility="collapsed")
        customer_id = selected_id
    else:
        customer_id = typed_id.strip() if typed_id.strip() else None

    if not customer_id:
        st.markdown("""
        <div class="not-found">
            <i class="ti ti-search" style="font-size:3rem; color:#1e2d45; display:block; margin-bottom:1rem;"></i>
            Enter a Customer ID above to generate their complete intelligence report.
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Load profile ──────────────────────────────────────────────────────────
    with st.spinner(f"Generating intelligence report for customer {customer_id}..."):
        profile = c360.get_profile(customer_id)

    if profile is None:
        st.markdown(f"""
        <div class="not-found">
            <i class="ti ti-user-x" style="font-size:3rem; color:#EF4444; display:block; margin-bottom:1rem;"></i>
            Customer <b>{customer_id}</b> not found in the dataset.<br>
            <span style="font-size:0.85rem;">Check the ID and try again.</span>
        </div>
        """, unsafe_allow_html=True)
        return

    ov = profile["overview"]
    ch = profile["churn"]
    sg = profile["segment"]
    va = profile["value"]
    na = profile["next_action"]
    hs = profile["health"]
    drivers   = profile["drivers"]
    history   = profile["flight_history"]

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — CUSTOMER OVERVIEW
    # ══════════════════════════════════════════════════════════════════════════
    sec("ti-user-circle", "Customer Overview")

    tier_colors = {"star": COLORS["amber"], "aurora": COLORS["blue"], "nova": COLORS["purple"]}
    tier_c = tier_colors.get(ov["loyalty_tier"].lower(), COLORS["blue"])

    # Row 1: KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi(ov["loyalty_tier"], "Loyalty Tier", tier_c)
    with c2:
        kpi(f"${ov['clv']:,.0f}", "Customer Lifetime Value", COLORS["green"])
    with c3:
        kpi(f"${ov['salary']:,.0f}", "Annual Income", COLORS["cyan"])
    with c4:
        kpi(f"{ov['tenure_months']}m", "Programme Tenure", COLORS["purple"])
    with c5:
        kpi(
            str(ov["months_inactive"]) + "m",
            "Months Inactive",
            COLORS["red"] if ov["months_inactive"] >= 6 else COLORS["amber"]
            if ov["months_inactive"] >= 3 else COLORS["green"],
            "since last flight"
        )

    # Row 2: Profile details
    st.markdown(f"""
    <div class="glass-card">
        <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(180px,1fr)); gap:1rem;">
            <div>
                <div style="font-size:0.68rem; text-transform:uppercase; letter-spacing:0.1em; color:#64748B; margin-bottom:4px;">Customer ID</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-size:1.1rem; font-weight:600; color:#F1F5F9;">{customer_id}</div>
            </div>
            <div>
                <div style="font-size:0.68rem; text-transform:uppercase; letter-spacing:0.1em; color:#64748B; margin-bottom:4px;">Enrollment Type</div>
                <div style="font-weight:500; color:#F1F5F9;">{ov['enrollment_type']}</div>
            </div>
            <div>
                <div style="font-size:0.68rem; text-transform:uppercase; letter-spacing:0.1em; color:#64748B; margin-bottom:4px;">Education</div>
                <div style="font-weight:500; color:#F1F5F9;">{ov['education']}</div>
            </div>
            <div>
                <div style="font-size:0.68rem; text-transform:uppercase; letter-spacing:0.1em; color:#64748B; margin-bottom:4px;">Marital Status</div>
                <div style="font-weight:500; color:#F1F5F9;">{ov['marital_status']}</div>
            </div>
            <div>
                <div style="font-size:0.68rem; text-transform:uppercase; letter-spacing:0.1em; color:#64748B; margin-bottom:4px;">Gender</div>
                <div style="font-weight:500; color:#F1F5F9;">{ov['gender']}</div>
            </div>
            <div>
                <div style="font-size:0.68rem; text-transform:uppercase; letter-spacing:0.1em; color:#64748B; margin-bottom:4px;">Province</div>
                <div style="font-weight:500; color:#F1F5F9;">{ov['province']}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — CHURN ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    sec("ti-alert-triangle", "Churn Analysis")

    ch_col1, ch_col2 = st.columns([2, 3])

    with ch_col1:
        fig_gauge = gauge_chart(
            ch["probability_pct"],
            ch["risk_color"],
            "Churn Risk Score",
            "%"
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with ch_col2:
        st.markdown(f"""
        <div class="glass-card" style="margin-top:1rem;">
            <div style="margin-bottom:1rem;">
                {risk_badge_html(ch['risk_category'], ch['risk_color'])}
            </div>
            <div style="font-size:2.5rem; font-family:'Space Grotesk',sans-serif;
                        font-weight:700; color:{ch['risk_color']}; line-height:1;">
                {ch['probability_pct']}%
            </div>
            <div style="font-size:0.8rem; color:#64748B; margin-top:4px; margin-bottom:1rem;">
                Churn probability
            </div>

            <div style="font-size:0.75rem; color:#94A3B8; margin-bottom:4px;">Risk level</div>
            <div class="prog-wrap">
                <div class="prog-bar" style="width:{ch['probability_pct']}%;
                     background:{ch['risk_color']};"></div>
            </div>

            <div style="display:flex; justify-content:space-between;
                        font-size:0.7rem; color:#64748B; margin-top:4px;">
                <span>0% Low</span><span>30% Medium</span>
                <span>60% High</span><span>80% Critical</span>
            </div>

            <div style="margin-top:1.2rem; padding-top:1rem; border-top:1px solid #1e2d45;">
                <div style="font-size:0.72rem; text-transform:uppercase;
                            letter-spacing:0.08em; color:#64748B; margin-bottom:8px;">
                    Risk thresholds
                </div>
                {"".join([
                    f'''<div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
                        <div style="width:8px;height:8px;border-radius:50%;background:{c};"></div>
                        <span style="font-size:0.78rem; color:#94A3B8;">{cat}</span>
                        <span style="font-size:0.78rem; color:#64748B; margin-left:auto;">
                            {int(lo*100)}–{int(hi*100)}%
                        </span>
                    </div>'''
                    for cat, (lo, hi), c in [
                        ("Low Risk",      (0.00, 0.30), "#10B981"),
                        ("Medium Risk",   (0.30, 0.60), "#F59E0B"),
                        ("High Risk",     (0.60, 0.80), "#EF4444"),
                        ("Critical Risk", (0.80, 1.00), "#7C3AED"),
                    ]
                ])}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — TOP CHURN DRIVERS
    # ══════════════════════════════════════════════════════════════════════════
    sec("ti-bulb", "Top Churn Drivers")

    impact_colors = {
        "high":   COLORS["red"],
        "medium": COLORS["amber"],
        "low":    COLORS["green"],
    }
    dir_colors = {"negative": COLORS["red"], "positive": COLORS["green"]}

    dr_c1, dr_c2 = st.columns(2)
    for i, d in enumerate(drivers):
        col = dr_c1 if i % 2 == 0 else dr_c2
        ic  = impact_colors.get(d["impact"], COLORS["blue"])
        dc  = dir_colors.get(d["direction"], COLORS["blue"])
        with col:
            st.markdown(f"""
            <div class="driver-card">
                <div class="driver-icon" style="background:{ic}18; color:{ic};">
                    <i class="ti {d['icon']}"></i>
                </div>
                <div style="flex:1;">
                    <div style="display:flex; align-items:center; justify-content:space-between;">
                        <div class="driver-factor">{d['factor']}</div>
                        <span style="font-size:0.68rem; font-weight:600; letter-spacing:0.06em;
                                     color:{dc}; text-transform:uppercase;">
                            {d['impact']}
                        </span>
                    </div>
                    <div class="driver-detail">{d['detail']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — CUSTOMER SEGMENT
    # ══════════════════════════════════════════════════════════════════════════
    sec("ti-users-group", "Customer Segment")

    seg_c1, seg_c2 = st.columns([3, 2])

    with seg_c1:
        st.markdown(f"""
        <div class="glass-card">
            <div style="margin-bottom:0.8rem;">
                <span class="seg-chip">{sg['name']}</span>
            </div>
            <div style="font-size:0.85rem; color:#94A3B8; line-height:1.6;">
                {sg['description']}
            </div>
            <div style="margin-top:1rem; padding-top:1rem; border-top:1px solid #1e2d45;
                        display:grid; grid-template-columns:1fr 1fr 1fr; gap:1rem;">
                <div>
                    <div style="font-size:0.68rem; text-transform:uppercase;
                                letter-spacing:0.08em; color:#64748B;">Segment Size</div>
                    <div style="font-family:'Space Grotesk',sans-serif; font-size:1.2rem;
                                font-weight:700; color:#F1F5F9; margin-top:2px;">
                        {sg['count']:,}
                    </div>
                </div>
                <div>
                    <div style="font-size:0.68rem; text-transform:uppercase;
                                letter-spacing:0.08em; color:#64748B;">Avg Churn</div>
                    <div style="font-family:'Space Grotesk',sans-serif; font-size:1.2rem;
                                font-weight:700; color:#EF4444; margin-top:2px;">
                        {sg['avg_churn']*100:.1f}%
                    </div>
                </div>
                <div>
                    <div style="font-size:0.68rem; text-transform:uppercase;
                                letter-spacing:0.08em; color:#64748B;">Avg CLV</div>
                    <div style="font-family:'Space Grotesk',sans-serif; font-size:1.2rem;
                                font-weight:700; color:#10B981; margin-top:2px;">
                        ${sg['avg_clv']:,.0f}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with seg_c2:
        # RFM radar
        cats   = ["Recency", "Frequency", "Monetary"]
        scores = [sg.get("r_score",0), sg.get("f_score",0), sg.get("m_score",0)]
        if any(s > 0 for s in scores):
            angles = [0, 120, 240, 0]
            sx = [s * 0.8 for s in scores] + [scores[0] * 0.8]
            fig_radar = go.Figure(go.Scatterpolar(
                r=sx,
                theta=cats + [cats[0]],
                fill="toself",
                fillcolor="rgba(59,130,246,0.12)",
                line=dict(color=COLORS["blue"], width=2),
                marker=dict(color=COLORS["blue"], size=6),
            ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(range=[0, 5], tickcolor="#64748B",
                                   gridcolor="#1e2d45", linecolor="#1e2d45",
                                   tickfont=dict(size=9, color="#64748B")),
                    angularaxis=dict(tickcolor="#64748B", linecolor="#1e2d45",
                                    tickfont=dict(size=10, color="#94A3B8")),
                ),
                showlegend=False,
                **PLOT_BG,
                height=250,
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.markdown("""
            <div class="glass-card" style="text-align:center; padding:2rem; color:#64748B;">
                RFM scores not available
            </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5 — FLIGHT BEHAVIOUR
    # ══════════════════════════════════════════════════════════════════════════
    sec("ti-plane", "Flight Behaviour")

    if not history.empty and "year_month" in history.columns:
        # Build 4-panel chart
        fig_hist = make_subplots(
            rows=2, cols=2,
            subplot_titles=["Total Flights", "Distance (km)",
                            "Points Accumulated", "Points Redeemed"],
            vertical_spacing=0.18, horizontal_spacing=0.1,
        )

        x = history["year_month"]

        def add_trace(col_name, row, col, color, fill_color):
            y = history[col_name] if col_name in history.columns else [0] * len(x)
            fig_hist.add_trace(go.Scatter(
                x=x, y=y,
                mode="lines",
                line=dict(color=color, width=2),
                fill="tozeroy",
                fillcolor=fill_color,
                showlegend=False,
            ), row=row, col=col)

        add_trace("total_flights",               1, 1, COLORS["blue"],   "rgba(59,130,246,0.08)")
        add_trace("distance",                    1, 2, COLORS["cyan"],   "rgba(6,182,212,0.08)")
        add_trace("points_accumulated",          2, 1, COLORS["green"],  "rgba(16,185,129,0.08)")
        add_trace("points_redeemed",             2, 2, COLORS["amber"],  "rgba(245,158,11,0.08)")

        # Style subplots
        for r in [1, 2]:
            for c in [1, 2]:
                fig_hist.update_xaxes(
                    gridcolor="#1e2d45", linecolor="#1e2d45",
                    tickfont=dict(size=9, color="#64748B"), row=r, col=c
                )
                fig_hist.update_yaxes(
                    gridcolor="#1e2d45", linecolor="#1e2d45",
                    tickfont=dict(size=9, color="#64748B"), row=r, col=c
                )

        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#94A3B8", size=10),
            margin=dict(t=40, b=20, l=10, r=10),
            height=420,
        )
        fig_hist.update_annotations(font=dict(size=11, color="#94A3B8"))
        st.plotly_chart(fig_hist, use_container_width=True)

        # Monthly summary table
        with st.expander("View monthly activity data"):
            disp_cols = [c for c in ["year_month","year","month","total_flights",
                                      "distance","points_accumulated","points_redeemed",
                                      "dollar_cost_points_redeemed"] if c in history.columns]
            st.dataframe(
                history[disp_cols].sort_values("year_month", ascending=False).head(36),
                use_container_width=True, height=300
            )
    else:
        st.info("No flight history available for this customer.")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 6 — VALUE ASSESSMENT
    # ══════════════════════════════════════════════════════════════════════════
    sec("ti-star", "Value Assessment")

    va_c1, va_c2 = st.columns([2, 3])

    with va_c1:
        fig_val = gauge_chart(
            va["score"], va["color"],
            "Future Value Score", ""
        )
        st.plotly_chart(fig_val, use_container_width=True)

    with va_c2:
        st.markdown(f"""
        <div class="glass-card">
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:1rem;">
                <div style="font-family:'Space Grotesk',sans-serif; font-size:2.5rem;
                            font-weight:700; color:{va['color']}; line-height:1;">
                    {va['score']}
                </div>
                <div>
                    <div style="font-weight:600; color:{va['color']}; font-size:1rem;">
                        {va['category']}
                    </div>
                    <div style="font-size:0.75rem; color:#64748B;">out of 100</div>
                </div>
            </div>
            <div style="font-size:0.82rem; color:#94A3B8; line-height:1.6; margin-bottom:1rem;">
                {va['description']}
            </div>
            <div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em;
                        color:#64748B; margin-bottom:8px;">Score components</div>
        """, unsafe_allow_html=True)

        for comp, val in va["components"].items():
            bar_val = abs(val)
            bar_color = COLORS["red"] if val < 0 else COLORS["blue"]
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
                <div style="font-size:0.78rem; color:#94A3B8; width:130px; flex-shrink:0;">
                    {comp}
                </div>
                <div class="val-bar-wrap" style="flex:1;">
                    <div class="val-bar" style="width:{min(bar_val/30*100,100):.0f}%;
                         background:{bar_color};"></div>
                </div>
                <div style="font-size:0.78rem; font-weight:600;
                            color:{bar_color}; width:40px; text-align:right;">
                    {val:+.1f}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 7 — NEXT BEST ACTION
    # ══════════════════════════════════════════════════════════════════════════
    sec("ti-rocket", "Next Best Action")

    action_icons = {
        "Complimentary Lounge Pass":                          "ti-armchair",
        "Temporary Tier Status Upgrade (90 days)":            "ti-award",
        "Expiring Points Reminder + Bonus Redemption Offer":  "ti-clock",
        "Win-Back: 50% Bonus Miles on Next 2 Flights":        "ti-gift",
        "Earn 3x Miles for Next 30 Days":                     "ti-plane-tilt",
        "Seasonal Travel Offer (targeted destination discount)":"ti-map-pin",
        "High-Value Customer Dedicated Account Manager Outreach":"ti-headset",
        "No Immediate Action Required":                        "ti-check",
    }
    a_icon = action_icons.get(na["action"], "ti-bolt")

    st.markdown(f"""
    <div class="nba-card">
        <div style="display:flex; align-items:flex-start; gap:14px;">
            <div style="background:rgba(59,130,246,0.15); border-radius:10px;
                        padding:0.7rem; font-size:1.5rem; color:#3B82F6; flex-shrink:0;">
                <i class="ti {a_icon}"></i>
            </div>
            <div style="flex:1;">
                <div style="font-family:'Space Grotesk',sans-serif; font-size:1.1rem;
                            font-weight:700; color:#F1F5F9; margin-bottom:4px;">
                    {na['action']}
                </div>
                <div style="font-size:0.82rem; color:#94A3B8; line-height:1.6;">
                    <b style="color:#64748B;">WHY:</b> {na['why']}
                </div>
            </div>
        </div>

        <div class="nba-row">
            <div class="nba-item">
                <div class="nba-item-label"><i class="ti ti-device-mobile-message" style="font-size:12px;"></i> Channel</div>
                <div class="nba-item-value">{na['channel']}</div>
            </div>
            <div class="nba-item">
                <div class="nba-item-label"><i class="ti ti-clock" style="font-size:12px;"></i> Timing</div>
                <div class="nba-item-value">{na['timing']}</div>
            </div>
            <div class="nba-item">
                <div class="nba-item-label"><i class="ti ti-target" style="font-size:12px;"></i> Goal</div>
                <div class="nba-item-value">{na['goal']}</div>
            </div>
            <div class="nba-item">
                <div class="nba-item-label"><i class="ti ti-trending-up" style="font-size:12px;"></i> Est. Lift</div>
                <div class="nba-item-value" style="color:#10B981;">+{na['est_lift']*100:.0f}%</div>
            </div>
            <div class="nba-item">
                <div class="nba-item-label"><i class="ti ti-currency-dollar" style="font-size:12px;"></i> Est. ROI</div>
                <div class="nba-item-value" style="color:#10B981;">${na['est_roi']:,.0f}</div>
            </div>
        </div>

        <div style="margin-top:1rem; padding:0.7rem 1rem; background:rgba(16,185,129,0.08);
                    border:1px solid rgba(16,185,129,0.2); border-radius:8px;
                    font-size:0.8rem; color:#94A3B8;">
            <i class="ti ti-chart-bar" style="color:#10B981; margin-right:6px;"></i>
            <b style="color:#10B981;">Expected Outcome:</b> {na['expected_outcome']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 8 — CUSTOMER HEALTH SCORE
    # ══════════════════════════════════════════════════════════════════════════
    sec("ti-heart-rate-monitor", "Customer Health Score")

    hs_c1, hs_c2 = st.columns([2, 3])

    with hs_c1:
        fig_health = gauge_chart(hs["score"], hs["color"], "Health Score", "")
        st.plotly_chart(fig_health, use_container_width=True)

        st.markdown(f"""
        <div style="text-align:center; margin-top:-1rem;">
            <span style="background:{hs['color']}18; color:{hs['color']};
                         border:1px solid {hs['color']}40; border-radius:20px;
                         padding:0.3rem 1.2rem; font-size:0.9rem; font-weight:600;">
                <i class="ti {hs['icon']}"></i> {hs['status']}
            </span>
        </div>
        """, unsafe_allow_html=True)

    with hs_c2:
        st.markdown("""
        <div class="glass-card">
            <div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em;
                        color:#64748B; margin-bottom:12px;">Score breakdown</div>
        """, unsafe_allow_html=True)

        comp_colors = {
            "Activity":       COLORS["blue"],
            "Engagement":     COLORS["cyan"],
            "Loyalty":        COLORS["purple"],
            "Churn Penalty":  COLORS["red"],
        }
        for comp, val in hs["components"].items():
            bar_color = comp_colors.get(comp, COLORS["blue"])
            bar_pct   = min(abs(val) / 30 * 100, 100)
            st.markdown(f"""
            <div style="margin-bottom:10px;">
                <div style="display:flex; justify-content:space-between;
                            font-size:0.78rem; color:#94A3B8; margin-bottom:4px;">
                    <span>{comp}</span>
                    <span style="color:{bar_color}; font-weight:600;">{val:+.1f}</span>
                </div>
                <div class="prog-wrap">
                    <div class="prog-bar" style="width:{bar_pct:.0f}%;
                         background:{bar_color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Status legend
        st.markdown("""
        <div style="margin-top:0.8rem; display:grid; grid-template-columns:1fr 1fr; gap:6px;">
            <div style="display:flex;align-items:center;gap:6px;font-size:0.75rem;color:#94A3B8;">
                <div style="width:8px;height:8px;border-radius:50%;background:#10B981;flex-shrink:0;"></div>
                75–100 Healthy
            </div>
            <div style="display:flex;align-items:center;gap:6px;font-size:0.75rem;color:#94A3B8;">
                <div style="width:8px;height:8px;border-radius:50%;background:#F59E0B;flex-shrink:0;"></div>
                55–74 Watchlist
            </div>
            <div style="display:flex;align-items:center;gap:6px;font-size:0.75rem;color:#94A3B8;">
                <div style="width:8px;height:8px;border-radius:50%;background:#F97316;flex-shrink:0;"></div>
                35–54 At Risk
            </div>
            <div style="display:flex;align-items:center;gap:6px;font-size:0.75rem;color:#94A3B8;">
                <div style="width:8px;height:8px;border-radius:50%;background:#EF4444;flex-shrink:0;"></div>
                0–34 Critical
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 9 — EXPORT
    # ══════════════════════════════════════════════════════════════════════════
    sec("ti-download", "Export Customer Report")

    st.markdown("""
    <div class="glass-card">
        <div style="font-size:0.85rem; color:#94A3B8; margin-bottom:1rem;">
            Download a complete snapshot of this customer's intelligence profile.
        </div>
    </div>
    """, unsafe_allow_html=True)

    ex_c1, ex_c2, ex_c3 = st.columns([1, 1, 3])

    with ex_c1:
        csv_bytes = generate_csv_bytes(profile, customer_id)
        st.download_button(
            label="📥  Download CSV",
            data=csv_bytes,
            file_name=f"customer_{customer_id}_report.csv",
            mime="text/csv",
            use_container_width=True
        )

    with ex_c2:
        pdf_bytes = generate_pdf_bytes(profile, customer_id)
        pdf_ext   = "pdf" if pdf_bytes[:4] == b"%PDF" else "txt"
        pdf_mime  = "application/pdf" if pdf_ext == "pdf" else "text/plain"
        st.download_button(
            label="📄  Download Report",
            data=pdf_bytes,
            file_name=f"customer_{customer_id}_report.{pdf_ext}",
            mime=pdf_mime,
            use_container_width=True
        )


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    render()
else:
    render()