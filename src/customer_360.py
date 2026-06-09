"""
src/customer_360.py
-------------------
Backend logic for Customer 360 / Customer Intelligence Center.

Pulls data from all existing pipeline outputs:
  - features.csv          → engineered features per customer
  - churn_scores.csv      → churn probability
  - customer_segments.csv → segment assignment
  - retention_recommendations.csv → next best action
  - master_clean.csv      → raw monthly flight activity
  - churn_labels.csv      → actual churn label + months inactive

Also uses the saved best_churn_model.pkl for SHAP local explanations.
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
from typing import Optional, Dict, Any, List

warnings.filterwarnings("ignore")

from config import (
    CUSTOMER_ID_COL, REPORTS_DIR, MODELS_DIR, PLOTS_DIR,
    HIGH_CHURN_RISK_THRESHOLD, MEDIUM_CHURN_RISK_THRESHOLD
)
from src.utils import get_logger

logger = get_logger(__name__)


# ── Risk category thresholds ───────────────────────────────────────────────────
RISK_THRESHOLDS = {
    "Low Risk":      (0.00, 0.30),
    "Medium Risk":   (0.30, 0.60),
    "High Risk":     (0.60, 0.80),
    "Critical Risk": (0.80, 1.00),
}

RISK_COLORS = {
    "Low Risk":      "#10B981",
    "Medium Risk":   "#F59E0B",
    "High Risk":     "#EF4444",
    "Critical Risk": "#7C3AED",
}

# ── Segment descriptions ───────────────────────────────────────────────────────
SEGMENT_DESCRIPTIONS = {
    "High Value Loyalists": (
        "Top-tier customers with high flight frequency, strong point accumulation, "
        "and consistent redemption behaviour. They represent the programme's highest "
        "revenue contributors and brand advocates."
    ),
    "At Risk Premium": (
        "Previously high-value customers showing signs of disengagement. "
        "Flight frequency or point activity has declined recently. "
        "Immediate retention action can prevent high CLV loss."
    ),
    "Frequent Redeemers": (
        "Customers who actively earn and redeem points. Highly engaged with "
        "the rewards ecosystem. Respond well to bonus point campaigns and "
        "partner offers."
    ),
    "Dormant Members": (
        "Customers with extended inactivity. Have not flown or engaged "
        "with the programme in several months. Require win-back campaigns "
        "with strong incentives to re-activate."
    ),
    "Seasonal Travelers": (
        "Customers whose travel is concentrated in specific quarters. "
        "Often leisure travellers. Respond well to seasonal promotions "
        "and destination-specific offers."
    ),
    "Discount Flyers": (
        "Price-sensitive customers with moderate flight frequency. "
        "Primarily motivated by fare discounts and bonus mile promotions. "
        "Low average CLV but high volume potential."
    ),
}


class Customer360:
    """
    Assembles a complete 360-degree intelligence profile for a single customer.

    Usage
    -----
    c360 = Customer360()
    profile = c360.get_profile("100590")
    """

    def __init__(self):
        self.features    = self._load("features.csv")
        self.scores      = self._load("churn_scores.csv")
        self.segments    = self._load("customer_segments.csv")
        self.recs        = self._load("retention_recommendations.csv")
        self.master      = self._load("master_clean.csv")
        self.labels      = self._load("churn_labels.csv")
        self.model       = self._load_model()

        # Pre-compute segment stats for comparison
        self._segment_stats = self._compute_segment_stats()

    # ── Loaders ────────────────────────────────────────────────────────────────

    @staticmethod
    def _load(filename: str) -> pd.DataFrame:
        path = os.path.join(REPORTS_DIR, filename)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                logger.debug(f"Loaded {filename}: {df.shape}")
                return df
            except Exception as e:
                logger.warning(f"Could not load {filename}: {e}")
        return pd.DataFrame()

    @staticmethod
    def _load_model():
        path = os.path.join(MODELS_DIR, "best_churn_model.pkl")
        if os.path.exists(path):
            try:
                return joblib.load(path)
            except Exception as e:
                logger.warning(f"Could not load model: {e}")
        return None

    # ── Segment stats ──────────────────────────────────────────────────────────

    def _compute_segment_stats(self) -> Dict:
        """Pre-compute per-segment averages for comparison benchmarking."""
        stats = {}
        if self.segments.empty or self.scores.empty:
            return stats

        merged = self.segments.merge(
            self.scores[[CUSTOMER_ID_COL, "churn_probability"]],
            on=CUSTOMER_ID_COL, how="left"
        )
        if not self.features.empty and "clv" in self.features.columns:
            merged = merged.merge(
                self.features[[CUSTOMER_ID_COL, "clv"]],
                on=CUSTOMER_ID_COL, how="left"
            )

        for seg in merged["segment_name"].dropna().unique():
            sub = merged[merged["segment_name"] == seg]
            stats[seg] = {
                "count":        int(len(sub)),
                "avg_churn":    float(sub["churn_probability"].mean()) if "churn_probability" in sub else 0.0,
                "avg_clv":      float(sub["clv"].mean()) if "clv" in sub.columns else 0.0,
            }
        return stats

    # ── Main profile builder ───────────────────────────────────────────────────

    def get_profile(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """
        Build the complete 360 profile for a customer.

        Parameters
        ----------
        customer_id : str

        Returns
        -------
        dict with keys:
          overview, churn, drivers, segment, flight_history,
          value, next_action, health
        OR None if customer not found.
        """
        cid = str(customer_id).strip()

        # ── Verify customer exists ────────────────────────────────────────────
        if self.features.empty:
            logger.error("features.csv not loaded — run pipeline first")
            return None

        feat_row = self.features[
            self.features[CUSTOMER_ID_COL].astype(str) == cid
        ]
        if feat_row.empty:
            return None

        feat = feat_row.iloc[0].to_dict()

        # ── Pull from each dataset ────────────────────────────────────────────
        score_row = self._get_row(self.scores,    cid)
        seg_row   = self._get_row(self.segments,  cid)
        rec_row   = self._get_row(self.recs,      cid)
        label_row = self._get_row(self.labels,    cid)

        # ── Monthly history ───────────────────────────────────────────────────
        history = pd.DataFrame()
        if not self.master.empty:
            history = self.master[
                self.master[CUSTOMER_ID_COL].astype(str) == cid
            ].copy()
            if "year_month" in history.columns:
                history["year_month"] = pd.to_datetime(history["year_month"])
                history = history.sort_values("year_month")

        # ── Build each section ────────────────────────────────────────────────
        churn_prob = float(score_row.get("churn_probability", 0.5)) \
            if score_row else 0.5

        profile = {
            "customer_id": cid,
            "overview":    self._build_overview(feat, label_row),
            "churn":       self._build_churn(churn_prob, label_row),
            "drivers":     self._build_drivers(feat, churn_prob, history),
            "segment":     self._build_segment(seg_row),
            "flight_history": history,
            "value":       self._build_value(feat, churn_prob, seg_row),
            "next_action": self._build_next_action(rec_row, seg_row, feat, history, churn_prob),
            "health":      self._build_health(feat, churn_prob, history),
        }
        return profile

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _get_row(self, df: pd.DataFrame, cid: str) -> Optional[Dict]:
        if df.empty or CUSTOMER_ID_COL not in df.columns:
            return None
        row = df[df[CUSTOMER_ID_COL].astype(str) == cid]
        return row.iloc[0].to_dict() if not row.empty else None

    # ── Section builders ───────────────────────────────────────────────────────

    def _build_overview(self, feat: Dict, label_row: Optional[Dict]) -> Dict:
        """Section 1 — Customer Overview."""
        return {
            "loyalty_tier":    str(feat.get("loyalty_card", "—")),
            "enrollment_type": str(feat.get("enrollment_type", "—")),
            "education":       str(feat.get("education", "—")),
            "marital_status":  str(feat.get("marital_status", "—")),
            "gender":          str(feat.get("gender", "—")),
            "province":        str(feat.get("province", "—")),
            "salary":          float(feat.get("salary", 0) or 0),
            "clv":             float(feat.get("clv", 0) or 0),
            "tenure_months":   int(feat.get("tenure_months", 0) or 0),
            "is_cancelled":    bool(feat.get("is_cancelled", 0)),
            "months_inactive": int(label_row.get("months_inactive", 0))
                               if label_row else 0,
        }

    def _build_churn(self, churn_prob: float, label_row: Optional[Dict]) -> Dict:
        """Section 2 — Churn Analysis."""
        risk_cat = self._risk_category(churn_prob)
        return {
            "probability":     churn_prob,
            "probability_pct": round(churn_prob * 100, 1),
            "risk_category":   risk_cat,
            "risk_color":      RISK_COLORS[risk_cat],
            "is_churned":      int(label_row.get("churned", 0)) if label_row else 0,
            "months_inactive": int(label_row.get("months_inactive", 0)) if label_row else 0,
        }

    def _build_drivers(
        self,
        feat: Dict,
        churn_prob: float,
        history: pd.DataFrame
    ) -> List[Dict]:
        """
        Section 3 — Top Churn Drivers.

        Uses rule-based signals derived from features since SHAP
        requires the full model pipeline which may not always be available.
        Each driver has: factor, value, impact (high/medium/low), direction.
        """
        drivers = []

        # ── Inactivity ────────────────────────────────────────────────────────
        recency = float(feat.get("recency_months", 0) or 0)
        if recency >= 6:
            drivers.append({
                "factor":    "Extended inactivity",
                "detail":    f"{int(recency)} months since last flight",
                "impact":    "high",
                "direction": "negative",
                "icon":      "ti-clock",
            })
        elif recency >= 3:
            drivers.append({
                "factor":    "Recent inactivity",
                "detail":    f"{int(recency)} months since last flight",
                "impact":    "medium",
                "direction": "negative",
                "icon":      "ti-clock",
            })

        # ── Flight frequency trend ─────────────────────────────────────────────
        f3  = float(feat.get("3m_total_flights_sum", 0) or 0)
        f12 = float(feat.get("12m_total_flights_sum", 0) or 0) / 4
        if f12 > 0:
            pct_change = (f3 - f12) / max(f12, 1) * 100
            if pct_change < -30:
                drivers.append({
                    "factor":    "Declining flight frequency",
                    "detail":    f"Recent flights down {abs(pct_change):.0f}% vs historical avg",
                    "impact":    "high",
                    "direction": "negative",
                    "icon":      "ti-trending-down",
                })
            elif pct_change > 20:
                drivers.append({
                    "factor":    "Increasing flight activity",
                    "detail":    f"Recent flights up {pct_change:.0f}% vs historical avg",
                    "impact":    "medium",
                    "direction": "positive",
                    "icon":      "ti-trending-up",
                })

        # ── Points redemption ─────────────────────────────────────────────────
        pts_ratio = float(feat.get("points_redeemed_ratio", 0) or 0)
        if pts_ratio < 0.10:
            drivers.append({
                "factor":    "Very low point redemption",
                "detail":    f"Only {pts_ratio*100:.0f}% of earned points redeemed — weak programme engagement",
                "impact":    "high",
                "direction": "negative",
                "icon":      "ti-coins",
            })
        elif pts_ratio < 0.25:
            drivers.append({
                "factor":    "Low redemption activity",
                "detail":    f"{pts_ratio*100:.0f}% points redeemed — customer not extracting value",
                "impact":    "medium",
                "direction": "negative",
                "icon":      "ti-coins",
            })
        elif pts_ratio > 0.60:
            drivers.append({
                "factor":    "Active point redeemer",
                "detail":    f"{pts_ratio*100:.0f}% of points redeemed — highly engaged with rewards",
                "impact":    "medium",
                "direction": "positive",
                "icon":      "ti-gift",
            })

        # ── Tenure ────────────────────────────────────────────────────────────
        tenure = int(feat.get("tenure_months", 0) or 0)
        if tenure < 12:
            drivers.append({
                "factor":    "New member",
                "detail":    f"Only {tenure} months tenure — higher churn risk in first year",
                "impact":    "medium",
                "direction": "negative",
                "icon":      "ti-user-plus",
            })
        elif tenure > 48:
            drivers.append({
                "factor":    "Long-term loyalty",
                "detail":    f"{tenure} months tenure — strong programme attachment",
                "impact":    "medium",
                "direction": "positive",
                "icon":      "ti-award",
            })

        # ── Inactivity streak ─────────────────────────────────────────────────
        streak = int(feat.get("max_inactivity_streak", 0) or 0)
        if streak >= 4:
            drivers.append({
                "factor":    "Long inactivity streak",
                "detail":    f"Longest consecutive gap: {streak} months with no flights",
                "impact":    "high",
                "direction": "negative",
                "icon":      "ti-player-pause",
            })

        # ── Points accumulation trend ─────────────────────────────────────────
        pts3  = float(feat.get("3m_points_accumulated_sum", 0) or 0)
        pts12 = float(feat.get("12m_points_accumulated_sum", 0) or 0) / 4
        if pts12 > 0 and pts3 < pts12 * 0.5:
            drivers.append({
                "factor":    "Declining point accumulation",
                "detail":    f"Points earned in last 3 months well below quarterly average",
                "impact":    "medium",
                "direction": "negative",
                "icon":      "ti-chart-bar",
            })

        # ── Seasonal concentration ─────────────────────────────────────────────
        seasonal = float(feat.get("seasonal_concentration", 0) or 0)
        if seasonal > 0.7:
            drivers.append({
                "factor":    "Highly seasonal traveller",
                "detail":    f"Travel concentrated in specific quarters — vulnerable to off-season churn",
                "impact":    "medium",
                "direction": "negative",
                "icon":      "ti-calendar",
            })

        # If no drivers found, add a generic positive signal
        if not drivers:
            drivers.append({
                "factor":    "Stable engagement",
                "detail":    "No significant churn risk signals detected",
                "impact":    "low",
                "direction": "positive",
                "icon":      "ti-check",
            })

        # Sort: negative first, then by impact
        impact_order = {"high": 0, "medium": 1, "low": 2}
        drivers.sort(key=lambda x: (
            0 if x["direction"] == "negative" else 1,
            impact_order.get(x["impact"], 2)
        ))

        return drivers[:6]  # Top 6

    def _build_segment(self, seg_row: Optional[Dict]) -> Dict:
        """Section 4 — Customer Segment."""
        if not seg_row:
            return {
                "name": "Unknown", "description": "Segment data not available.",
                "count": 0, "avg_churn": 0.0, "avg_clv": 0.0,
                "rfm_score": 0.0, "r_score": 0.0, "f_score": 0.0, "m_score": 0.0,
            }
        name = str(seg_row.get("segment_name", "Unknown"))
        seg_stats = self._segment_stats.get(name, {})
        return {
            "name":        name,
            "description": SEGMENT_DESCRIPTIONS.get(name, "No description available."),
            "count":       seg_stats.get("count", 0),
            "avg_churn":   seg_stats.get("avg_churn", 0.0),
            "avg_clv":     seg_stats.get("avg_clv", 0.0),
            "rfm_score":   float(seg_row.get("RFM_score", 0) or 0),
            "r_score":     float(seg_row.get("R_score", 0) or 0),
            "f_score":     float(seg_row.get("F_score", 0) or 0),
            "m_score":     float(seg_row.get("M_score", 0) or 0),
            "recency":     float(seg_row.get("recency", 0) or 0),
            "frequency":   float(seg_row.get("frequency", 0) or 0),
            "monetary":    float(seg_row.get("monetary", 0) or 0),
        }

    def _build_value(
        self,
        feat: Dict,
        churn_prob: float,
        seg_row: Optional[Dict]
    ) -> Dict:
        """
        Section 6 — Future Value Score (0-100).

        Composite of:
          CLV score        (30%)
          Activity score   (25%)
          Tier score       (20%)
          Frequency score  (15%)
          Churn penalty    (10% deduction)
        """
        # CLV component (normalize to ~0-30)
        clv = float(feat.get("clv", 0) or 0)
        clv_score = min(clv / 10000 * 30, 30)

        # Activity (recent flights vs all-time avg)
        f3  = float(feat.get("3m_total_flights_sum", 0) or 0)
        avg = float(feat.get("avg_monthly_flights", 0) or 0)
        activity_score = min((f3 / 3) / max(avg, 0.1) * 25, 25)

        # Tier
        tier = str(feat.get("loyalty_card", "")).lower()
        tier_score = {"nova": 20, "aurora": 14, "star": 8}.get(tier, 8)

        # Frequency (lifetime)
        total_flights = float(feat.get("alltime_total_flights_sum", 0) or 0)
        freq_score = min(total_flights / 100 * 15, 15)

        # Churn penalty
        churn_penalty = churn_prob * 10

        raw = clv_score + activity_score + tier_score + freq_score - churn_penalty
        score = max(0, min(100, round(raw)))

        if score >= 70:
            category = "High Value"
            color    = "#10B981"
            desc     = (
                "This customer represents significant long-term revenue potential. "
                "Their CLV, activity, and tier justify premium retention investment."
            )
        elif score >= 40:
            category = "Medium Value"
            color    = "#F59E0B"
            desc     = (
                "Moderate future value. Targeted engagement can move this customer "
                "into the High Value tier. Focus on frequency and redemption activation."
            )
        else:
            category = "Low Value"
            color    = "#EF4444"
            desc     = (
                "Limited projected future value based on current trajectory. "
                "Cost-efficient retention actions are recommended over premium interventions."
            )

        return {
            "score":           score,
            "category":        category,
            "color":           color,
            "description":     desc,
            "components": {
                "CLV Score":      round(clv_score, 1),
                "Activity Score": round(activity_score, 1),
                "Tier Score":     round(tier_score, 1),
                "Frequency Score":round(freq_score, 1),
                "Churn Penalty":  round(-churn_penalty, 1),
            }
        }

    def _build_next_action(
        self,
        rec_row: Optional[Dict],
        seg_row: Optional[Dict],
        feat: Dict,
        history: pd.DataFrame,
        churn_prob: float
    ) -> Dict:
        """Section 7 — Next Best Action."""

        segment = str(seg_row.get("segment_name", "Unknown")) if seg_row else "Unknown"
        risk    = self._risk_category(churn_prob)

        # Use existing recommendation if available
        if rec_row:
            action  = str(rec_row.get("action",  "Personalised Retention Offer"))
            channel = str(rec_row.get("channel", "Email"))
            timing  = str(rec_row.get("timing",  "Within 7 days"))
            goal    = str(rec_row.get("goal",    "Improve retention"))
            lift    = float(rec_row.get("est_retention_lift", 0.10))
            cost    = float(rec_row.get("cost_usd", 20))
            roi     = float(rec_row.get("est_roi", 0))
        else:
            action = channel = timing = goal = "—"
            lift = cost = roi = 0.0

        # Build narrative WHY
        recency = float(feat.get("recency_months", 0) or 0)
        f3      = float(feat.get("3m_total_flights_sum", 0) or 0)
        f6      = float(feat.get("6m_total_flights_sum", 0) or 0)
        pts_r   = float(feat.get("points_redeemed_ratio", 0) or 0)

        why_parts = []
        if recency >= 3:
            why_parts.append(f"inactive for {int(recency)} months")
        if f6 > 0 and f3 < f6 / 2:
            pct = round((1 - f3 / (f6 / 2)) * 100)
            why_parts.append(f"flight activity dropped ~{pct}% in last 3 months")
        if pts_r < 0.15:
            why_parts.append(f"point redemption rate is very low ({pts_r*100:.0f}%)")

        why_narrative = (
            "Customer is a " + segment + ". " +
            (("Key signals: " + "; ".join(why_parts) + ".") if why_parts
             else "Proactive retention recommended based on segment profile.")
        )

        expected_outcome = (
            f"Est. {lift*100:.0f}% improvement in 90-day retention probability. "
            f"Estimated ROI: ${roi:,.0f}."
        ) if lift > 0 else "Monitoring recommended."

        return {
            "segment":          segment,
            "risk_category":    risk,
            "who":              f"{segment} · {risk}",
            "why":              why_narrative,
            "action":           action,
            "channel":          channel,
            "timing":           timing,
            "goal":             goal,
            "expected_outcome": expected_outcome,
            "est_lift":         lift,
            "est_cost":         cost,
            "est_roi":          roi,
        }

    def _build_health(
        self,
        feat: Dict,
        churn_prob: float,
        history: pd.DataFrame
    ) -> Dict:
        """
        Section 8 — Customer Health Score (0-100).

        Components:
          Activity score   (30%) — recent vs historical flight rate
          Engagement score (25%) — point earning + redemption ratio
          Loyalty score    (25%) — tenure + tier
          Churn penalty    (20%) — deducted based on churn probability
        """
        # Activity
        f3  = float(feat.get("3m_total_flights_sum", 0) or 0)
        avg = float(feat.get("avg_monthly_flights", 0.01) or 0.01)
        activity = min((f3 / 3) / avg * 30, 30)

        # Engagement (points)
        pts_acc = float(feat.get("alltime_points_accumulated_sum", 0) or 0)
        pts_red = float(feat.get("points_redeemed_ratio", 0) or 0)
        engage  = min(pts_acc / 50000 * 15 + pts_red * 10, 25)

        # Loyalty
        tenure = float(feat.get("tenure_months", 0) or 0)
        tier   = str(feat.get("loyalty_card", "")).lower()
        t_score = {"nova": 15, "aurora": 10, "star": 5}.get(tier, 5)
        ten_score = min(tenure / 60 * 10, 10)
        loyalty = t_score + ten_score

        # Churn penalty
        penalty = churn_prob * 20

        raw   = activity + engage + loyalty - penalty
        score = max(0, min(100, round(raw)))

        if score >= 75:
            status = "Healthy"
            color  = "#10B981"
            icon   = "ti-heart"
        elif score >= 55:
            status = "Watchlist"
            color  = "#F59E0B"
            icon   = "ti-eye"
        elif score >= 35:
            status = "At Risk"
            color  = "#F97316"
            icon   = "ti-alert-triangle"
        else:
            status = "Critical"
            color  = "#EF4444"
            icon   = "ti-alert-circle"

        return {
            "score":  score,
            "status": status,
            "color":  color,
            "icon":   icon,
            "components": {
                "Activity":   round(activity, 1),
                "Engagement": round(engage, 1),
                "Loyalty":    round(loyalty, 1),
                "Churn Penalty": round(-penalty, 1),
            }
        }

    # ── Static helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _risk_category(prob: float) -> str:
        if prob >= 0.80: return "Critical Risk"
        if prob >= 0.60: return "High Risk"
        if prob >= 0.30: return "Medium Risk"
        return "Low Risk"

    def get_all_customer_ids(self) -> List[str]:
        """Return sorted list of all available customer IDs."""
        if self.features.empty:
            return []
        return sorted(self.features[CUSTOMER_ID_COL].astype(str).unique().tolist())