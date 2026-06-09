"""
src/clean_data.py
-----------------
Data cleaning pipeline.

Actual confirmed columns after standardisation:

Customer Flight Activity:
  loyalty_number, year, month, total_flights,
  distance, points_accumulated, points_redeemed,
  dollar_cost_points_redeemed

Customer Loyalty History:
  loyalty_number, country, province, city, postal_code,
  gender, education, salary, marital_status, loyalty_card,
  clv, enrollment_type, enrollment_year, enrollment_month,
  cancellation_year, cancellation_month

NOTE: 'flights_booked' and 'flights_with_companions' do NOT exist
in this dataset. All references have been removed.
"""

import pandas as pd
import numpy as np
from typing import Tuple

from config import CUSTOMER_ID_COL
from src.utils import get_logger, timer

logger = get_logger(__name__)


class DataCleaner:
    """
    Cleans and merges raw datasets into a single analysis-ready DataFrame.

    Usage
    -----
    cleaner = DataCleaner()
    master_df = cleaner.clean_and_merge(loyalty_df, flight_df)
    """

    # ── Confirmed numeric columns from actual CSVs ─────────────────────────────
    LOYALTY_NUMERIC_COLS = [
        "clv", "salary",
        "enrollment_year", "enrollment_month",
        "cancellation_year", "cancellation_month"
    ]

    # Only columns that actually exist in Customer Flight Activity.csv
    FLIGHT_NUMERIC_COLS = [
        "total_flights", "distance",
        "points_accumulated", "points_redeemed",
        "dollar_cost_points_redeemed"
    ]

    FLIGHT_KEY_COLS = [CUSTOMER_ID_COL, "year", "month"]

    # IQR cap multiplier — we cap rather than remove records
    IQR_MULTIPLIER = 3.0

    # ── Public API ─────────────────────────────────────────────────────────────

    @timer
    def clean_and_merge(
        self,
        loyalty: pd.DataFrame,
        flight: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Full cleaning pipeline.

        Steps
        -----
        1. Clean loyalty table
        2. Clean flight activity table
        3. Merge on loyalty_number
        4. Post-merge cleaning

        Parameters
        ----------
        loyalty : pd.DataFrame
        flight  : pd.DataFrame

        Returns
        -------
        pd.DataFrame — clean master DataFrame
        """
        logger.info("=== Starting Data Cleaning Pipeline ===")

        loyalty_clean = self._clean_loyalty(loyalty)
        flight_clean  = self._clean_flight(flight)
        master        = self._merge(loyalty_clean, flight_clean)
        master        = self._post_merge_cleaning(master)

        logger.info(f"Master DataFrame shape: {master.shape}")
        logger.info(f"Unique customers: {master[CUSTOMER_ID_COL].nunique():,}")
        logger.info(f"Date range: {master['year'].min()}–{master['year'].max()}")

        return master

    # ── Loyalty Cleaning ──────────────────────────────────────────────────────

    def _clean_loyalty(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean Customer Loyalty History table.

        Actual columns: loyalty_number, country, province, city, postal_code,
        gender, education, salary, marital_status, loyalty_card, clv,
        enrollment_type, enrollment_year, enrollment_month,
        cancellation_year, cancellation_month
        """
        logger.info(f"Cleaning loyalty table: {df.shape}")
        df = df.copy()

        # ── Drop duplicates ───────────────────────────────────────────────────
        before = len(df)
        df.drop_duplicates(inplace=True)
        logger.info(f"  Dropped {before - len(df)} full duplicate rows")

        # ── Ensure loyalty_number exists ──────────────────────────────────────
        if CUSTOMER_ID_COL not in df.columns:
            raise ValueError(
                f"'{CUSTOMER_ID_COL}' column not found in loyalty data.\n"
                f"Available columns: {list(df.columns)}"
            )

        df[CUSTOMER_ID_COL] = df[CUSTOMER_ID_COL].astype(str).str.strip()
        # Keep only first record per customer (should be unique already)
        df.drop_duplicates(subset=[CUSTOMER_ID_COL], keep="first", inplace=True)

        # ── Fix numeric columns ───────────────────────────────────────────────
        for col in self.LOYALTY_NUMERIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # ── CLV: clip negatives, impute nulls with median ─────────────────────
        if "clv" in df.columns:
            df["clv"] = df["clv"].clip(lower=0)
            df["clv"] = df["clv"].fillna(df["clv"].median())

        # ── Salary: impute by loyalty card tier ───────────────────────────────
        # Loyalty card tier (Star, Aurora, Nova) correlates with income level
        if "salary" in df.columns:
            df["salary"] = pd.to_numeric(df["salary"], errors="coerce")
            if "loyalty_card" in df.columns:
                df["salary"] = df.groupby("loyalty_card")["salary"].transform(
                    lambda x: x.fillna(x.median())
                )
            # Global median fallback
            df["salary"] = df["salary"].fillna(df["salary"].median())
            df["salary"] = df["salary"].clip(lower=0)

        # ── Categorical columns: fill nulls ───────────────────────────────────
        cat_cols = [
            "gender", "education", "marital_status", "loyalty_card",
            "enrollment_type", "country", "province", "city", "postal_code"
        ]
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown").astype(str).str.strip()

        # ── Cancellation year/month: 0 or NaN means still active ─────────────
        for col in ["cancellation_year", "cancellation_month"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        # ── Enrollment year/month: validate reasonable ranges ─────────────────
        if "enrollment_year" in df.columns:
            df["enrollment_year"] = df["enrollment_year"].clip(lower=2000, upper=2020).astype(int)
        if "enrollment_month" in df.columns:
            df["enrollment_month"] = df["enrollment_month"].clip(lower=1, upper=12).astype(int)

        logger.info(f"  Loyalty table clean: {df.shape}")
        return df

    # ── Flight Cleaning ────────────────────────────────────────────────────────

    def _clean_flight(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean Customer Flight Activity table.

        Actual columns: loyalty_number, year, month, total_flights,
        distance, points_accumulated, points_redeemed,
        dollar_cost_points_redeemed

        NOTE: No 'flights_booked' or 'flights_with_companions' in this dataset.
        """
        logger.info(f"Cleaning flight activity table: {df.shape}")
        df = df.copy()

        # ── Standardise ID ────────────────────────────────────────────────────
        df[CUSTOMER_ID_COL] = df[CUSTOMER_ID_COL].astype(str).str.strip()

        # ── Drop duplicates ───────────────────────────────────────────────────
        before = len(df)
        df.drop_duplicates(inplace=True)
        # One record per customer per year-month
        df.drop_duplicates(subset=self.FLIGHT_KEY_COLS, keep="first", inplace=True)
        logger.info(f"  Dropped {before - len(df)} duplicate flight rows")

        # ── Fix year and month ────────────────────────────────────────────────
        df["year"]  = pd.to_numeric(df["year"],  errors="coerce").astype("Int64")
        df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")

        # Keep only valid year/month records
        df = df[
            df["year"].between(2010, 2020) &
            df["month"].between(1, 12)
        ].copy()
        df["year"]  = df["year"].astype(int)
        df["month"] = df["month"].astype(int)

        # ── Fix numeric flight columns ─────────────────────────────────────────
        for col in self.FLIGHT_NUMERIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)

        # ── Cap outliers using IQR ─────────────────────────────────────────────
        for col in self.FLIGHT_NUMERIC_COLS:
            if col in df.columns:
                df[col] = self._cap_outliers_iqr(df[col], col)

        logger.info(f"  Flight activity clean: {df.shape}")
        return df

    # ── Merge ──────────────────────────────────────────────────────────────────

    def _merge(
        self,
        loyalty: pd.DataFrame,
        flight: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Left-join flight activity onto loyalty history on loyalty_number.

        All flight records are kept. Loyalty profile is attached where available.
        """
        logger.info("Merging loyalty + flight activity on loyalty_number...")
        master = flight.merge(loyalty, on=CUSTOMER_ID_COL, how="left")

        n_unmatched = master["clv"].isna().sum()
        if n_unmatched > 0:
            logger.warning(
                f"  {n_unmatched:,} flight rows could not be matched "
                f"to a loyalty profile — will be excluded from modelling"
            )

        logger.info(f"  Merged DataFrame: {master.shape}")
        return master

    # ── Post-merge cleaning ────────────────────────────────────────────────────

    def _post_merge_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Final cleaning after merge:
          - Drop rows where loyalty profile is missing
          - Create year_month datetime column
          - Sort chronologically per customer
        """
        df = df.copy()

        # Drop rows where loyalty profile couldn't be matched
        # (these have no CLV, no tier, no demographic info)
        before = len(df)
        df.dropna(subset=["clv"], inplace=True)
        logger.info(f"  Dropped {before - len(df):,} unmatched rows (no loyalty profile)")

        # Drop rows missing customer ID
        df.dropna(subset=[CUSTOMER_ID_COL], inplace=True)

        # ── Create datetime column for time-series operations ─────────────────
        df["year_month"] = pd.to_datetime(
            df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2),
            format="%Y-%m"
        )

        # ── Sort by customer and time ─────────────────────────────────────────
        df.sort_values([CUSTOMER_ID_COL, "year_month"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        logger.info(
            f"  Final master: {df.shape[0]:,} rows | "
            f"{df[CUSTOMER_ID_COL].nunique():,} unique customers | "
            f"Period: {df['year_month'].min().date()} → {df['year_month'].max().date()}"
        )
        return df

    # ── Utility ────────────────────────────────────────────────────────────────

    @staticmethod
    def _cap_outliers_iqr(
        series: pd.Series,
        name: str,
        multiplier: float = 3.0
    ) -> pd.Series:
        """
        Cap outliers at Q1 - k*IQR and Q3 + k*IQR (Winsorization).
        Caps rather than removes — preserves all customer records.
        """
        q1  = series.quantile(0.25)
        q3  = series.quantile(0.75)
        iqr = q3 - q1
        lower = max(0, q1 - multiplier * iqr)
        upper = q3 + multiplier * iqr
        capped = series.clip(lower=lower, upper=upper)
        n_capped = (series != capped).sum()
        if n_capped > 0:
            logger.debug(f"  Capped {n_capped} outliers in '{name}'")
        return capped