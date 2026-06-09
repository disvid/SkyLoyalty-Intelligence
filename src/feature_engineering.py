"""
src/feature_engineering.py
---------------------------
Feature engineering pipeline.

Based on ACTUAL columns available in this dataset:

Flight Activity (after cleaning):
  loyalty_number, year, month, total_flights, distance,
  points_accumulated, points_redeemed, dollar_cost_points_redeemed,
  year_month

Loyalty Profile (after cleaning):
  loyalty_number, country, province, city, postal_code,
  gender, education, salary, marital_status, loyalty_card,
  clv, enrollment_type, enrollment_year, enrollment_month,
  cancellation_year, cancellation_month

NOTE: 'flights_booked' and 'flights_with_companions' are NOT available.
      All features are derived from the confirmed columns only.
"""

import pandas as pd
import numpy as np
from typing import Optional

from config import CUSTOMER_ID_COL, DATE_COL
from src.utils import get_logger, timer

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Builds a customer-level feature matrix from the cleaned master DataFrame.

    Usage
    -----
    fe = FeatureEngineer()
    feature_df = fe.build_features(master_df)
    """

    # Only aggregate columns that actually exist in the dataset
    # No flights_booked, no flights_with_companions
    AGG_COLS = [
        "total_flights",
        "distance",
        "points_accumulated",
        "points_redeemed",
        "dollar_cost_points_redeemed",
    ]

    @timer
    def build_features(
        self,
        df: pd.DataFrame,
        snapshot_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Build full feature matrix — one row per customer.

        Parameters
        ----------
        df : pd.DataFrame  — clean master DataFrame
        snapshot_date : str or None
            If provided, use only data up to this date (e.g. '2017-06-30').
            Prevents data leakage during training.

        Returns
        -------
        pd.DataFrame — one row per customer, all engineered features
        """
        logger.info("=== Starting Feature Engineering ===")

        # ── Filter to observation window ──────────────────────────────────────
        if snapshot_date is not None:
            snap = pd.to_datetime(snapshot_date)
            df = df[df[DATE_COL] <= snap].copy()
            logger.info(f"  Snapshot: {snap.date()} | Filtered rows: {len(df):,}")

        # Only use AGG_COLS that actually exist in the dataframe
        self.AGG_COLS = [c for c in self.AGG_COLS if c in df.columns]
        logger.info(f"  Aggregation columns: {self.AGG_COLS}")

        # ── Build feature blocks ──────────────────────────────────────────────
        agg     = self._build_alltime_aggregates(df)
        roll_3m  = self._build_rolling_features(df, window_months=3,  suffix="_3m")
        roll_6m  = self._build_rolling_features(df, window_months=6,  suffix="_6m")
        roll_12m = self._build_rolling_features(df, window_months=12, suffix="_12m")
        recency  = self._build_recency(df, snapshot_date)
        streak   = self._build_inactivity_streak(df)
        seasonal = self._build_seasonal_features(df)
        profile  = self._build_profile_features(df)

        # ── Merge all feature blocks on CUSTOMER_ID_COL ───────────────────────
        features = (
            agg
            .join(roll_3m,  how="left")
            .join(roll_6m,  how="left")
            .join(roll_12m, how="left")
            .join(recency,  how="left")
            .join(streak,   how="left")
            .join(seasonal, how="left")
            .join(profile,  how="left")
        )

        # ── Derived ratio features ────────────────────────────────────────────
        features = self._build_ratio_features(features)

        # ── Fill any remaining nulls with 0 ──────────────────────────────────
        features = features.fillna(0)

        logger.info(
            f"Feature matrix: {features.shape[0]:,} customers "
            f"× {features.shape[1]} features"
        )
        return features.reset_index()

    # ── All-time Aggregates ────────────────────────────────────────────────────

    def _build_alltime_aggregates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        All-time totals, means, std, and max per customer.
        Only uses confirmed available columns.
        """
        grp = df.groupby(CUSTOMER_ID_COL)
        agg = grp[self.AGG_COLS].agg(["sum", "mean", "std", "max"])
        # Flatten MultiIndex → e.g. 'total_flights_sum'
        agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
        agg.columns = [f"alltime_{c}" for c in agg.columns]

        # Number of months with any flight activity
        agg["active_months"] = grp["total_flights"].apply(lambda x: (x > 0).sum())
        # Total months with any record (active or not)
        agg["total_months"]  = grp[DATE_COL].nunique()
        # Activity rate = proportion of months with flights
        agg["activity_rate"] = (
            agg["active_months"] / agg["total_months"].clip(lower=1)
        ).clip(0, 1)
        # Average monthly flights (only over active months)
        agg["avg_monthly_flights"] = (
            agg["alltime_total_flights_sum"]
            / agg["active_months"].clip(lower=1)
        )

        return agg

    # ── Rolling Window Features ────────────────────────────────────────────────

    def _build_rolling_features(
        self,
        df: pd.DataFrame,
        window_months: int,
        suffix: str
    ) -> pd.DataFrame:
        """
        Aggregate features over the last N months only.

        This captures recent behaviour trends (acceleration/deceleration)
        that are stronger churn signals than all-time averages.
        """
        max_date = df[DATE_COL].max()
        cutoff   = max_date - pd.DateOffset(months=window_months)
        window_df = df[df[DATE_COL] > cutoff].copy()

        if window_df.empty:
            logger.warning(f"  No data in rolling window {suffix}")
            return pd.DataFrame(index=df[CUSTOMER_ID_COL].unique())

        grp  = window_df.groupby(CUSTOMER_ID_COL)
        roll = grp[self.AGG_COLS].agg(["sum", "mean"])
        # Flatten: e.g. 'total_flights_sum' → '3m_total_flights_sum'
        pfx  = suffix.lstrip("_")
        roll.columns = [f"{pfx}_{col}_{stat}" for col, stat in roll.columns]

        # Active months in this window
        roll[f"active_months{suffix}"] = (
            grp["total_flights"].apply(lambda x: (x > 0).sum())
        )
        return roll

    # ── Recency ────────────────────────────────────────────────────────────────

    def _build_recency(
        self,
        df: pd.DataFrame,
        snapshot_date: Optional[str]
    ) -> pd.DataFrame:
        """
        Recency = months since last flight.

        This is one of the strongest churn predictors:
        a customer who flew 1 month ago is very different from
        one who flew 18 months ago.
        """
        active = df[df["total_flights"] > 0].copy()
        last_active = (
            active.groupby(CUSTOMER_ID_COL)[DATE_COL]
            .max()
            .rename("last_active_date")
        )

        snap = pd.to_datetime(snapshot_date) if snapshot_date else df[DATE_COL].max()

        recency_df = last_active.to_frame()
        recency_df["recency_months"] = (
            (snap.year  - recency_df["last_active_date"].dt.year)  * 12
            + (snap.month - recency_df["last_active_date"].dt.month)
        ).clip(lower=0)

        recency_df.drop(columns=["last_active_date"], inplace=True)
        return recency_df

    # ── Inactivity Streak ──────────────────────────────────────────────────────

    def _build_inactivity_streak(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Inactivity streak = longest consecutive run of months with zero flights.

        A streak > 3 months is a very strong leading churn indicator.
        This feature captures sustained disengagement patterns.
        """

        def longest_zero_run(series: pd.Series) -> int:
            """Find longest run of consecutive zeros in a series."""
            max_run = current_run = 0
            for val in series:
                if val == 0:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 0
            return max_run

        df_sorted = df.sort_values([CUSTOMER_ID_COL, DATE_COL])
        streak = (
            df_sorted.groupby(CUSTOMER_ID_COL)["total_flights"]
            .apply(longest_zero_run)
            .rename("max_inactivity_streak")
            .to_frame()
        )
        return streak

    # ── Seasonal Behaviour ─────────────────────────────────────────────────────

    def _build_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Measure seasonal travel concentration.

        Features:
          - pct_q1..pct_q4: share of total flights in each quarter
          - seasonal_concentration: Herfindahl index (1.0 = all flights in one quarter)
          - peak_quarter: quarter with most flights
          - yoy_flight_growth: YoY change in total flights (2017 vs 2016, etc.)
        """
        df2 = df.copy()
        df2["quarter"] = df2[DATE_COL].dt.quarter

        # Flights per quarter per customer
        q_flights = (
            df2.groupby([CUSTOMER_ID_COL, "quarter"])["total_flights"]
            .sum()
            .unstack(fill_value=0)
        )
        for q in [1, 2, 3, 4]:
            if q not in q_flights.columns:
                q_flights[q] = 0

        total = q_flights.sum(axis=1).clip(lower=1)
        pct = q_flights.div(total, axis=0)
        pct.columns = [f"pct_q{c}" for c in pct.columns]

        # Herfindahl concentration (higher = more seasonal traveller)
        pct["seasonal_concentration"] = (pct[[f"pct_q{i}" for i in [1,2,3,4]]] ** 2).sum(axis=1)
        # Peak quarter
        pct["peak_quarter"] = q_flights.idxmax(axis=1).astype(float)

        # YoY flight growth (last 2 years in data)
        df2["year"] = df2[DATE_COL].dt.year
        yearly = df2.groupby([CUSTOMER_ID_COL, "year"])["total_flights"].sum().unstack(fill_value=0)
        years  = sorted(yearly.columns)
        if len(years) >= 2:
            pct["yoy_flight_growth"] = (
                (yearly[years[-1]] - yearly[years[-2]])
                / (yearly[years[-2]].clip(lower=1))
            ).clip(-1, 5)
        else:
            pct["yoy_flight_growth"] = 0.0

        return pct

    # ── Profile Features ───────────────────────────────────────────────────────

    def _build_profile_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Static customer profile attributes from the loyalty history table.

        Confirmed columns available:
          loyalty_card, enrollment_type, enrollment_year, enrollment_month,
          cancellation_year, cancellation_month, clv, salary,
          gender, education, marital_status, country, province
        """
        profile_cols = [
            "loyalty_card", "enrollment_type",
            "enrollment_year", "enrollment_month",
            "cancellation_year", "cancellation_month",
            "clv", "salary",
            "gender", "education", "marital_status",
            "country", "province",
        ]
        available = [c for c in profile_cols if c in df.columns]

        # Take first (or last) value per customer — these are static
        profile = df.groupby(CUSTOMER_ID_COL)[available].first()

        # ── Tenure: months from enrollment to snapshot ────────────────────────
        if "enrollment_year" in profile.columns and "enrollment_month" in profile.columns:
            snap_year  = df[DATE_COL].max().year
            snap_month = df[DATE_COL].max().month
            profile["tenure_months"] = (
                (snap_year  - profile["enrollment_year"].clip(lower=2000)) * 12
                + (snap_month - profile["enrollment_month"].clip(lower=1, upper=12))
            ).clip(lower=0)

        # ── Flag: has the customer ever cancelled? ─────────────────────────────
        if "cancellation_year" in profile.columns:
            profile["is_cancelled"] = (profile["cancellation_year"] > 0).astype(int)

        return profile

    # ── Ratio Features ─────────────────────────────────────────────────────────

    def _build_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute meaningful business ratios.

        points_redeemed_ratio:
            redeemed / accumulated — 0 = hoarder (disengaged), 1 = active redeemer
            Customers who never redeem are weakly attached to the programme.

        distance_per_flight:
            avg km per flight — proxy for long-haul vs short-haul preference

        dollar_redemption_rate:
            dollar_cost_points_redeemed / points_redeemed
            How much $ value is extracted per redeemed point

        points_per_flight:
            avg points earned per flight — proxy for fare class / distance
        """
        df = df.copy()

        pts_acc = df.get("alltime_points_accumulated_sum", pd.Series(0, index=df.index))
        pts_red = df.get("alltime_points_redeemed_sum",    pd.Series(0, index=df.index))
        flights = df.get("alltime_total_flights_sum",       pd.Series(1, index=df.index)).clip(lower=1)
        dist    = df.get("alltime_distance_sum",            pd.Series(0, index=df.index))
        dollar  = df.get("alltime_dollar_cost_points_redeemed_sum", pd.Series(0, index=df.index))

        # Core ratio features
        df["points_redeemed_ratio"] = (
            pts_red / pts_acc.clip(lower=1)
        ).clip(0, 1)

        df["distance_per_flight"] = (
            dist / flights
        ).clip(lower=0)

        df["points_per_flight"] = (
            pts_acc / flights
        ).clip(lower=0)

        df["dollar_redemption_rate"] = (
            dollar / pts_red.clip(lower=1)
        ).clip(0, 10)

        # Points accumulation rate over recent vs all-time
        # Captures acceleration or deceleration of earning
        recent_pts = df.get("6m_points_accumulated_sum", pd.Series(0, index=df.index))
        df["recent_earning_ratio"] = (
            (recent_pts / 6) / ((pts_acc / df.get("total_months", pd.Series(1, index=df.index)).clip(lower=1)) + 1)
        ).clip(0, 10)

        return df