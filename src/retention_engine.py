"""
src/retention_engine.py
-----------------------
Rule-based retention recommendation engine — KeyError fixed.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from config import (
    CUSTOMER_ID_COL,
    HIGH_CHURN_RISK_THRESHOLD,
    MEDIUM_CHURN_RISK_THRESHOLD,
    REPORTS_DIR
)
from src.utils import get_logger, timer

logger = get_logger(__name__)


ACTION_CATALOGUE: Dict[str, Dict] = {
    "lounge_pass": {
        "action":   "Complimentary Lounge Pass",
        "channel":  "Email + App Notification",
        "timing":   "Immediate",
        "goal":     "Reinforce premium status and emotional loyalty",
        "cost_usd": 40,
        "est_retention_lift": 0.18,
    },
    "tier_boost": {
        "action":   "Temporary Tier Status Upgrade (90 days)",
        "channel":  "Email + SMS",
        "timing":   "Within 48h",
        "goal":     "Incentivise continued flying to retain upgraded status",
        "cost_usd": 25,
        "est_retention_lift": 0.22,
    },
    "expiring_points": {
        "action":   "Expiring Points Reminder + Bonus Redemption Offer",
        "channel":  "Email",
        "timing":   "60 days before expiry",
        "goal":     "Drive redemption activity and re-engagement",
        "cost_usd": 5,
        "est_retention_lift": 0.12,
    },
    "seasonal_offer": {
        "action":   "Seasonal Travel Offer (targeted destination discount)",
        "channel":  "Email + App Push",
        "timing":   "4-6 weeks before typical travel season",
        "goal":     "Capture seasonal travel intent early",
        "cost_usd": 20,
        "est_retention_lift": 0.16,
    },
    "winback_discount": {
        "action":   "Win-Back: 50% Bonus Miles on Next 2 Flights",
        "channel":  "Email + SMS",
        "timing":   "After 3 months inactivity",
        "goal":     "Re-activate dormant customers before permanent churn",
        "cost_usd": 30,
        "est_retention_lift": 0.25,
    },
    "mileage_bonus": {
        "action":   "Earn 3x Miles for Next 30 Days",
        "channel":  "App Notification + Email",
        "timing":   "Immediate",
        "goal":     "Boost frequency among moderate flyers",
        "cost_usd": 12,
        "est_retention_lift": 0.10,
    },
    "partner_reward": {
        "action":   "Partner Hotel/Car Rental Bonus Miles Offer",
        "channel":  "Email",
        "timing":   "Post-flight follow-up",
        "goal":     "Increase lifetime value through ecosystem engagement",
        "cost_usd": 8,
        "est_retention_lift": 0.09,
    },
    "clv_protection": {
        "action":   "High-Value Customer Dedicated Account Manager Outreach",
        "channel":  "Phone + Email",
        "timing":   "Within 24h of risk detection",
        "goal":     "Protect highest-CLV customers with white-glove service",
        "cost_usd": 50,
        "est_retention_lift": 0.30,
    },
    "no_action": {
        "action":   "No Immediate Action Required",
        "channel":  "Standard Newsletter",
        "timing":   "Monthly cycle",
        "goal":     "Maintain awareness",
        "cost_usd": 1,
        "est_retention_lift": 0.02,
    },
}

SEGMENT_RULES: Dict[str, List[str]] = {
    "High Value Loyalists":  ["lounge_pass",     "clv_protection",  "tier_boost"],
    "At Risk Premium":       ["tier_boost",       "clv_protection",  "expiring_points"],
    "Frequent Redeemers":    ["expiring_points",  "mileage_bonus",   "partner_reward"],
    "Dormant Members":       ["winback_discount", "mileage_bonus",   "seasonal_offer"],
    "Seasonal Travelers":    ["seasonal_offer",   "mileage_bonus",   "expiring_points"],
    "Discount Flyers":       ["mileage_bonus",    "seasonal_offer",  "expiring_points"],
}

DEFAULT_RULES: List[str] = ["mileage_bonus", "seasonal_offer", "expiring_points"]


class RetentionEngine:
    """Rule-based personalised retention recommendation engine."""

    @timer
    def generate_recommendations(
        self,
        segment_df: pd.DataFrame,
        churn_proba: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Generate personalised retention recommendation per customer.

        FIX: churn_probability is now safely attached via parameter,
        not assumed to be in segment_df. This eliminates the KeyError.

        Parameters
        ----------
        segment_df  : pd.DataFrame — must have CUSTOMER_ID_COL + segment_name
        churn_proba : pd.Series (optional)
            Indexed by CUSTOMER_ID_COL, values 0–1.
            If None, defaults to 0.5 for all customers.

        Returns
        -------
        pd.DataFrame — one recommendation row per customer
        """
        logger.info("=== Generating Retention Recommendations ===")

        df = segment_df.copy()
        logger.info(f"  Input shape: {df.shape} | cols: {list(df.columns[:10])}")

        # ── SAFE churn probability attachment ─────────────────────────────────
        # Remove existing churn_probability column to avoid conflicts
        if "churn_probability" in df.columns:
            df.drop(columns=["churn_probability"], inplace=True)

        if churn_proba is not None:
            # churn_proba is a Series indexed by CUSTOMER_ID_COL
            churn_df = churn_proba.rename("churn_probability").reset_index()
            churn_df.columns = [CUSTOMER_ID_COL, "churn_probability"]

            df = df.merge(churn_df, on=CUSTOMER_ID_COL, how="left")
            logger.info(f"  Merged churn probabilities: {df['churn_probability'].notna().sum()} matched")
        else:
            df["churn_probability"] = 0.5
            logger.info("  No churn probabilities provided — using default 0.5")

        # Final safety: fill any remaining nulls
        df["churn_probability"] = pd.to_numeric(
            df["churn_probability"], errors="coerce"
        ).fillna(0.5).clip(0, 1)

        # ── Risk tier ─────────────────────────────────────────────────────────
        conditions = [
            df["churn_probability"] >= HIGH_CHURN_RISK_THRESHOLD,
            df["churn_probability"] >= MEDIUM_CHURN_RISK_THRESHOLD,
        ]
        choices = ["High Risk", "Medium Risk"]
        df["risk_tier"] = np.select(conditions, choices, default="Low Risk")

        # ── Ensure required columns exist with safe defaults ──────────────────
        if "segment_name" not in df.columns:
            df["segment_name"] = "Dormant Members"

        for col, default in [
            ("recency_months", 6.0),
            ("points_redeemed_ratio", 0.3),
            ("clv", 500.0),
            ("alltime_total_flights_sum", 10.0),
        ]:
            if col not in df.columns:
                df[col] = default
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

        # Alias: the engine uses 'recency' as the short name
        if "recency" not in df.columns:
            if "recency_months" in df.columns:
                df["recency"] = df["recency_months"]
            else:
                df["recency"] = 6.0

        # ── Apply recommendation rules ────────────────────────────────────────
        logger.info("  Applying recommendation rules...")
        recs = df.apply(self._select_action, axis=1)
        rec_df = pd.DataFrame(recs.tolist())

        # Build output columns
        keep_cols = [CUSTOMER_ID_COL, "segment_name", "churn_probability", "risk_tier"]
        for extra in ["recency_months", "alltime_total_flights_sum",
                      "months_inactive", "churned"]:
            if extra in df.columns:
                keep_cols.append(extra)

        result = pd.concat([df[keep_cols].reset_index(drop=True),
                             rec_df.reset_index(drop=True)], axis=1)

        # ── Estimated ROI ─────────────────────────────────────────────────────
        result["est_roi"] = (
            result["clv_proxy"].fillna(500)
            * result["est_retention_lift"]
            - result["cost_usd"]
        ).round(2)

        # ── Save ──────────────────────────────────────────────────────────────
        path = os.path.join(REPORTS_DIR, "retention_recommendations.csv")
        result.to_csv(path, index=False)
        logger.info(f"  Saved {len(result):,} recommendations → {path}")

        logger.info("  Action distribution:")
        for action, cnt in result["action"].value_counts().items():
            logger.info(f"    {action:<55} {cnt:>5}")

        return result

    def _select_action(self, row: pd.Series) -> Dict:
        """Select best retention action for a single customer."""
        segment   = str(row.get("segment_name", "Dormant Members"))
        risk      = str(row.get("risk_tier",    "Medium Risk"))
        recency   = float(row.get("recency",    6.0))
        pts_ratio = float(row.get("points_redeemed_ratio", 0.3))
        clv_val   = float(row.get("clv", 500.0))

        rules = SEGMENT_RULES.get(segment, DEFAULT_RULES)

        if recency > 9:
            action_key = "winback_discount"
        elif risk == "Low Risk":
            action_key = "no_action"
        elif clv_val > 5000 and risk == "High Risk":
            action_key = "clv_protection"
        elif pts_ratio < 0.15 and recency > 3:
            action_key = "expiring_points"
        elif risk == "High Risk":
            action_key = rules[1] if len(rules) > 1 else rules[0]
        else:
            action_key = rules[0]

        meta = ACTION_CATALOGUE[action_key].copy()
        meta["action_key"] = action_key
        meta["clv_proxy"]  = clv_val
        return meta