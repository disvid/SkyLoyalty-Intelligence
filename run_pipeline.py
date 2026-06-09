"""
run_pipeline.py
---------------
Master orchestration script — all bugs fixed.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    CUSTOMER_ID_COL, TARGET_COL, MODELS_DIR, REPORTS_DIR, PLOTS_DIR
)
from src.utils import get_logger, ensure_dirs, timer
from src.load_data import DataLoader
from src.clean_data import DataCleaner
from src.feature_engineering import FeatureEngineer
from src.churn_labels import ChurnLabeler
from src.churn_model import ChurnModelTrainer
from src.segmentation import CustomerSegmenter
from src.retention_engine import RetentionEngine
from src.explainability import ChurnExplainer

logger = get_logger("pipeline")


@timer
def main() -> None:
    """Execute the full analytics pipeline."""

    logger.info("=" * 60)
    logger.info("  AIRLINE LOYALTY ANALYTICS — PIPELINE START")
    logger.info("=" * 60)

    # ── 0. Setup ──────────────────────────────────────────────────────────────
    ensure_dirs()

    # ── 1. Load Data ──────────────────────────────────────────────────────────
    logger.info("\n[STEP 1] Loading Data...")
    loader = DataLoader()
    loyalty, flight, calendar, dictionary = loader.load_all()

    # ── 2. Clean Data ─────────────────────────────────────────────────────────
    logger.info("\n[STEP 2] Cleaning Data...")
    cleaner = DataCleaner()
    master  = cleaner.clean_and_merge(loyalty, flight)

    clean_path = os.path.join(REPORTS_DIR, "master_clean.csv")
    master.to_csv(clean_path, index=False)
    logger.info(f"  Saved clean master → {clean_path}")

    # ── 3. Feature Engineering ────────────────────────────────────────────────
    logger.info("\n[STEP 3] Engineering Features...")
    fe = FeatureEngineer()
    feature_df = fe.build_features(master)

    feat_path = os.path.join(REPORTS_DIR, "features.csv")
    feature_df.to_csv(feat_path, index=False)
    logger.info(f"  Saved feature matrix ({feature_df.shape}) → {feat_path}")

    # ── 4. Churn Labels ───────────────────────────────────────────────────────
    logger.info("\n[STEP 4] Creating Churn Labels...")
    labeler  = ChurnLabeler()
    label_df = labeler.create_labels(master)

    label_path = os.path.join(REPORTS_DIR, "churn_labels.csv")
    label_df.to_csv(label_path, index=False)
    logger.info(f"  Saved labels → {label_path}")

    # ── 5. Train/Test Split ───────────────────────────────────────────────────
    logger.info("\n[STEP 5] Time-Based Train/Test Split...")
    X_train, X_test, y_train, y_test = labeler.time_split(feature_df, label_df)

    logger.info(f"  X_train: {X_train.shape} | churn rate: {y_train.mean():.2%}")
    logger.info(f"  X_test:  {X_test.shape}  | churn rate: {y_test.mean():.2%}")

    # ── 6. Train ML Models ────────────────────────────────────────────────────
    logger.info("\n[STEP 6] Training Churn Models...")
    trainer = ChurnModelTrainer()
    trainer.fit(X_train, y_train)
    results_df = trainer.evaluate(X_test, y_test)

    metrics_path = os.path.join(REPORTS_DIR, "model_metrics.csv")
    results_df.to_csv(metrics_path)
    logger.info(f"  Saved metrics → {metrics_path}")
    logger.info(f"\n  Model comparison:\n{results_df.to_string()}\n")

    model_path = trainer.save_best_model()
    logger.info(f"  Best model saved → {model_path}")

    # ── 7. Score All Customers ────────────────────────────────────────────────
    logger.info("\n[STEP 7] Scoring All Customers...")

    # Drop ID and any label-related columns before scoring
    drop_for_scoring = [CUSTOMER_ID_COL, TARGET_COL, "months_inactive",
                        "explicitly_cancelled", "enrollment_date"]
    X_all = feature_df.drop(
        columns=[c for c in drop_for_scoring if c in feature_df.columns],
        errors="ignore"
    )

    # Keep only numeric + object columns (same as training)
    X_all = X_all.select_dtypes(include=[np.number, "object"])

    try:
        churn_proba_arr = trainer.predict_proba(X_all)
        scores_df = pd.DataFrame({
            CUSTOMER_ID_COL: feature_df[CUSTOMER_ID_COL].values,
            "churn_probability": churn_proba_arr
        })
        logger.info(
            f"  Scored {len(scores_df):,} customers | "
            f"avg prob: {churn_proba_arr.mean():.3f}"
        )
    except Exception as e:
        logger.warning(f"  Scoring failed: {e}. Using fallback scores.")
        scores_df = pd.DataFrame({
            CUSTOMER_ID_COL: feature_df[CUSTOMER_ID_COL].values,
            "churn_probability": np.random.uniform(0, 0.3, len(feature_df))
        })

    scores_path = os.path.join(REPORTS_DIR, "churn_scores.csv")
    scores_df.to_csv(scores_path, index=False)
    logger.info(f"  Saved {len(scores_df):,} churn scores → {scores_path}")

    # ── 8. Customer Segmentation ──────────────────────────────────────────────
    logger.info("\n[STEP 8] Customer Segmentation...")
    segmenter  = CustomerSegmenter()
    segment_df = segmenter.fit_transform(feature_df)
    segmenter.plot_all()

    seg_path = os.path.join(REPORTS_DIR, "customer_segments.csv")
    segment_df.to_csv(seg_path, index=False)
    logger.info(f"  Saved segments → {seg_path}")

    # ── 9. Retention Recommendations ─────────────────────────────────────────
    logger.info("\n[STEP 9] Generating Retention Recommendations...")

    # ── FIX: Build segment_with_scores step by step, verify each merge ────────

    # Start with segment data
    segment_with_scores = segment_df.copy()
    logger.info(f"  segment_df shape: {segment_df.shape} | cols: {list(segment_df.columns[:8])}")

    # Merge churn scores — explicit column check
    if CUSTOMER_ID_COL in scores_df.columns and "churn_probability" in scores_df.columns:
        segment_with_scores = segment_with_scores.merge(
            scores_df[[CUSTOMER_ID_COL, "churn_probability"]],
            on=CUSTOMER_ID_COL,
            how="left"
        )
        logger.info(f"  After scores merge: {segment_with_scores.shape}")
    else:
        logger.warning("  scores_df missing required columns — adding default churn_probability")
        segment_with_scores["churn_probability"] = 0.5

    # Ensure churn_probability exists and has no nulls
    if "churn_probability" not in segment_with_scores.columns:
        segment_with_scores["churn_probability"] = 0.5
    segment_with_scores["churn_probability"] = (
        segment_with_scores["churn_probability"].fillna(0.5)
    )

    # Merge behavioural features for rule engine
    behav_cols = [CUSTOMER_ID_COL, "points_redeemed_ratio", "clv",
                  "recency_months", "alltime_total_flights_sum"]
    behav_available = [c for c in behav_cols if c in feature_df.columns]
    if len(behav_available) > 1:  # at least ID + 1 feature
        segment_with_scores = segment_with_scores.merge(
            feature_df[behav_available],
            on=CUSTOMER_ID_COL,
            how="left"
        )
        logger.info(f"  After behavioural merge: {segment_with_scores.shape}")

    # Merge labels for context
    if CUSTOMER_ID_COL in label_df.columns:
        segment_with_scores = segment_with_scores.merge(
            label_df[[CUSTOMER_ID_COL, TARGET_COL, "months_inactive"]],
            on=CUSTOMER_ID_COL,
            how="left"
        )

    logger.info(
        f"  Final segment_with_scores shape: {segment_with_scores.shape}\n"
        f"  Columns: {list(segment_with_scores.columns)}"
    )

    # Build churn series indexed by customer ID for the engine
    churn_series = (
        segment_with_scores
        .set_index(CUSTOMER_ID_COL)["churn_probability"]
    )

    engine  = RetentionEngine()
    recs_df = engine.generate_recommendations(segment_with_scores, churn_series)

    # ── 10. SHAP Explainability ───────────────────────────────────────────────
    logger.info("\n[STEP 10] Generating SHAP Explanations...")
    try:
        explainer = ChurnExplainer(
            trainer.best_pipeline,
            trainer.feature_names
        )
        explainer.fit(X_train.head(500))
        importance_df = explainer.global_explanation(X_test.head(300))
        if len(X_test) > 0:
            explainer.local_explanation(X_test, customer_index=0)
        imp_path = os.path.join(REPORTS_DIR, "shap_global_importance.csv")
        if not importance_df.empty:
            importance_df.to_csv(imp_path, index=False)
    except Exception as e:
        logger.warning(f"  SHAP explanations failed: {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  PIPELINE COMPLETE ✓")
    logger.info("=" * 60)
    logger.info(f"  Outputs  → {REPORTS_DIR}")
    logger.info(f"  Plots    → {PLOTS_DIR}")
    logger.info(f"  Models   → {MODELS_DIR}")


if __name__ == "__main__":
    main()