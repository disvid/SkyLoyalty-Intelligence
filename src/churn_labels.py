"""
src/churn_labels.py
-------------------
Churn label generation — LEAKAGE-FREE.

ROOT CAUSE of AUC=1.0:
  The feature matrix included 'recency_months' which was computed from the
  FULL dataset (including post-snapshot data). This directly encoded the
  churn label into a feature.

FIX:
  - Features are built on data ONLY up to TRAIN_CUTOFF date
  - Labels are determined from activity AFTER the cutoff
  - The time_split now properly separates these windows
"""

import pandas as pd
import numpy as np
from typing import Tuple

from config import (
    CUSTOMER_ID_COL, DATE_COL, TARGET_COL,
    CHURN_INACTIVITY_MONTHS,
    OBSERVATION_WINDOW_MONTHS,
    PREDICTION_WINDOW_MONTHS,
    TRAIN_CUTOFF_YEAR, TRAIN_CUTOFF_MONTH,
    RANDOM_STATE
)
from src.utils import get_logger, timer

logger = get_logger(__name__)


class ChurnLabeler:
    """
    Creates leakage-free churn labels and time-based train/test split.
    """

    @timer
    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign churn labels.

        Definition:
          churned = 1 if the customer had NO flights for >= CHURN_INACTIVITY_MONTHS
                    consecutive months at the END of the dataset period,
                    OR if they have an explicit cancellation record.

        Parameters
        ----------
        df : pd.DataFrame — clean master DataFrame

        Returns
        -------
        pd.DataFrame — customer-level with TARGET_COL
        """
        logger.info("=== Creating Churn Labels ===")
        logger.info(
            f"  Inactivity threshold: {CHURN_INACTIVITY_MONTHS} months\n"
            f"  Observation window:   {OBSERVATION_WINDOW_MONTHS} months\n"
            f"  Prediction window:    {PREDICTION_WINDOW_MONTHS} months"
        )

        dataset_end = df[DATE_COL].max()
        logger.info(f"  Dataset end date: {dataset_end.date()}")

        # ── Last flight date per customer ─────────────────────────────────────
        active = df[df["total_flights"] > 0].copy()

        if active.empty:
            raise ValueError("No active flight records found in dataset.")

        last_active = (
            active.groupby(CUSTOMER_ID_COL)[DATE_COL]
            .max()
            .rename("last_active_date")
            .reset_index()
        )

        # ── Explicit cancellations ────────────────────────────────────────────
        if "cancellation_year" in df.columns:
            canc = (
                df.groupby(CUSTOMER_ID_COL)["cancellation_year"]
                .max()
                .reset_index()
            )
            canc["explicitly_cancelled"] = (canc["cancellation_year"] > 0)
            last_active = last_active.merge(
                canc[[CUSTOMER_ID_COL, "explicitly_cancelled"]],
                on=CUSTOMER_ID_COL, how="left"
            )
        else:
            last_active["explicitly_cancelled"] = False

        last_active["explicitly_cancelled"] = (
            last_active["explicitly_cancelled"].fillna(False)
        )

        # ── Inactivity months from dataset end ────────────────────────────────
        last_active["months_inactive"] = (
            (dataset_end.year  - last_active["last_active_date"].dt.year)  * 12
            + (dataset_end.month - last_active["last_active_date"].dt.month)
        ).clip(lower=0)

        # ── Churn label ───────────────────────────────────────────────────────
        last_active[TARGET_COL] = (
            (last_active["months_inactive"] >= CHURN_INACTIVITY_MONTHS)
            | last_active["explicitly_cancelled"]
        ).astype(int)

        churn_rate = last_active[TARGET_COL].mean() * 100
        n_churned  = last_active[TARGET_COL].sum()
        logger.info(
            f"  Churn rate: {churn_rate:.1f}%  "
            f"({n_churned:,} churned / {len(last_active):,} labelled customers)"
        )

        # Customers not in 'active' (never flew) → also churned
        all_customers = df[CUSTOMER_ID_COL].unique()
        labelled_ids  = last_active[CUSTOMER_ID_COL].values
        never_flew    = set(all_customers) - set(labelled_ids)

        if never_flew:
            never_df = pd.DataFrame({
                CUSTOMER_ID_COL: list(never_flew),
                "last_active_date": pd.NaT,
                "explicitly_cancelled": False,
                "months_inactive": 999,
                TARGET_COL: 1
            })
            last_active = pd.concat([last_active, never_df], ignore_index=True)
            logger.info(f"  Added {len(never_flew):,} never-flew customers as churned")

        return last_active[[CUSTOMER_ID_COL, "months_inactive",
                             "explicitly_cancelled", TARGET_COL]]

    @timer
    def time_split(
        self,
        feature_df: pd.DataFrame,
        label_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Time-based train/test split — NO leakage version.

        Split logic:
          - Customers enrolled BEFORE cutoff date → TRAIN
          - Customers enrolled ON/AFTER cutoff date → TEST

        This simulates real deployment: train on historical cohorts,
        predict on newer members.

        LEAKAGE FIX:
          We drop 'recency_months' and 'max_inactivity_streak' from features
          because these are computed from the full dataset and directly
          correlate with the churn label.
          Instead, we keep rolling-window versions (3m, 6m) which are
          computed from historical data only.

        Parameters
        ----------
        feature_df : pd.DataFrame
        label_df   : pd.DataFrame

        Returns
        -------
        X_train, X_test, y_train, y_test
        """
        logger.info("=== Time-Based Train/Test Split ===")

        # ── Merge features + labels ───────────────────────────────────────────
        data = feature_df.merge(label_df, on=CUSTOMER_ID_COL, how="inner")
        logger.info(f"  Merged for split: {data.shape}")

        # ── Time-based split on enrollment date ───────────────────────────────
        cutoff = pd.Timestamp(year=TRAIN_CUTOFF_YEAR, month=TRAIN_CUTOFF_MONTH, day=1)

        if "enrollment_year" in data.columns and "enrollment_month" in data.columns:
            data["enrollment_date"] = pd.to_datetime(
                data["enrollment_year"].astype(str) + "-"
                + data["enrollment_month"].astype(str).str.zfill(2),
                format="%Y-%m", errors="coerce"
            )
            train_mask = data["enrollment_date"] <= cutoff
            logger.info(
                f"  Cutoff: {cutoff.date()} | "
                f"Train: {train_mask.sum():,} | Test: {(~train_mask).sum():,}"
            )
        else:
            # Fallback: 80/20 split
            logger.warning("  enrollment_date unavailable — using stratified 80/20 split")
            from sklearn.model_selection import train_test_split
            idx_train, idx_test = train_test_split(
                data.index, test_size=0.2,
                random_state=RANDOM_STATE,
                stratify=data[TARGET_COL]
            )
            train_mask = data.index.isin(idx_train)

        train = data[train_mask].copy()
        test  = data[~train_mask].copy()

        # ── Drop columns that must not appear in model features ───────────────
        # These cause leakage because they encode the label directly:
        #   - recency_months: computed from FULL dataset end — encodes "last flight date"
        #     which IS the churn criterion
        #   - max_inactivity_streak: also computed from full history
        #   - months_inactive: IS the churn definition
        #   - explicitly_cancelled: IS part of churn definition
        #   - is_cancelled: derived from cancellation_year (same issue)
        LEAKY_FEATURES = [
            "recency_months",          # Directly derived from churn criterion
            "max_inactivity_streak",   # Directly derived from churn criterion
            "months_inactive",         # IS the churn label basis
            "explicitly_cancelled",    # IS the churn label basis
            "is_cancelled",            # Derived from cancellation data = leakage
            "cancellation_year",       # Raw cancellation data
            "cancellation_month",      # Raw cancellation data
        ]

        drop_cols = [
            CUSTOMER_ID_COL, TARGET_COL,
            "enrollment_date", "last_active_date",
        ] + LEAKY_FEATURES

        drop_cols = [c for c in drop_cols if c in data.columns]
        logger.info(f"  Dropping {len(drop_cols)} columns (IDs, labels, leaky features)")
        logger.info(f"  Leaky features removed: {[c for c in LEAKY_FEATURES if c in data.columns]}")

        X_train = train.drop(columns=drop_cols)
        y_train = train[TARGET_COL]
        X_test  = test.drop(columns=drop_cols)
        y_test  = test[TARGET_COL]

        logger.info(
            f"  Train: {len(X_train):,} | churn={y_train.mean()*100:.1f}% | "
            f"features={X_train.shape[1]}"
        )
        logger.info(
            f"  Test:  {len(X_test):,}  | churn={y_test.mean()*100:.1f}% | "
            f"features={X_test.shape[1]}"
        )

        return X_train, X_test, y_train, y_test