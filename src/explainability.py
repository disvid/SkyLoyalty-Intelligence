"""
src/explainability.py
---------------------
SHAP-based model explainability.

Produces:
  1. Global explanation — top churn drivers across all customers
  2. Local explanation  — why a specific customer is predicted to churn
  3. Summary bar plot
  4. Beeswarm / waterfall plots
  5. SHAP values CSV for dashboard use
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shap
import joblib
from typing import Optional

from config import PLOTS_DIR, REPORTS_DIR, MODELS_DIR
from src.utils import get_logger, timer, save_figure

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


class ChurnExplainer:
    """
    Wraps SHAP TreeExplainer / LinearExplainer to produce global and local
    explanations for the churn model.

    Usage
    -----
    explainer = ChurnExplainer(best_pipeline, feature_names)
    explainer.fit(X_train)
    explainer.global_explanation(X_test)
    explainer.local_explanation(X_test, customer_index=0)
    """

    def __init__(self, pipeline, feature_names: list):
        """
        Parameters
        ----------
        pipeline     : trained sklearn/imblearn Pipeline
        feature_names : list of feature names after preprocessing
        """
        self.pipeline      = pipeline
        self.feature_names = feature_names
        self.explainer     = None
        self.shap_values   = None
        self.X_transformed = None

    # ── Fit ────────────────────────────────────────────────────────────────────

    @timer
    def fit(self, X: pd.DataFrame, max_samples: int = 500) -> None:
        """
        Initialise SHAP explainer on a subsample of the data.

        We use TreeExplainer for tree-based models and
        LinearExplainer for Logistic Regression.

        Parameters
        ----------
        X          : pd.DataFrame  — training or background data
        max_samples : int  — max rows for SHAP background (performance)
        """
        logger.info("=== Initialising SHAP Explainer ===")

        # Extract the classifier from the pipeline
        classifier = self.pipeline.named_steps["classifier"]
        preprocessor = self.pipeline.named_steps["preprocessor"]

        # Transform data through preprocessing (exclude SMOTE step for explain)
        try:
            X_transformed = preprocessor.transform(X)
        except Exception as e:
            logger.error(f"Preprocessing transform failed: {e}")
            return

        # Subsample for speed
        n = min(max_samples, X_transformed.shape[0])
        idx = np.random.choice(X_transformed.shape[0], n, replace=False)
        X_sample = X_transformed[idx]

        self.X_transformed = X_transformed

        # Build SHAP explainer
        try:
            if hasattr(classifier, "feature_importances_"):
                # Tree-based model (RF, XGB)
                self.explainer = shap.TreeExplainer(classifier)
                logger.info("  Using TreeExplainer")
            else:
                # Linear model
                self.explainer = shap.LinearExplainer(classifier, X_sample)
                logger.info("  Using LinearExplainer")
        except Exception as e:
            logger.warning(f"  SHAP explainer init failed: {e}; using KernelExplainer")
            self.explainer = shap.KernelExplainer(
                classifier.predict_proba,
                shap.sample(X_sample, 50)
            )

    # ── Global Explanation ─────────────────────────────────────────────────────

    @timer
    def global_explanation(self, X: pd.DataFrame, max_samples: int = 300) -> pd.DataFrame:
        """
        Compute and plot global SHAP feature importances.

        Parameters
        ----------
        X          : pd.DataFrame
        max_samples : int

        Returns
        -------
        pd.DataFrame  — mean |SHAP| per feature, sorted descending
        """
        if self.explainer is None:
            logger.warning("Explainer not fitted. Call fit() first.")
            return pd.DataFrame()

        logger.info("  Computing global SHAP values...")
        preprocessor = self.pipeline.named_steps["preprocessor"]

        try:
            X_t = preprocessor.transform(X)
            n   = min(max_samples, X_t.shape[0])
            idx = np.random.choice(X_t.shape[0], n, replace=False)
            X_sample = X_t[idx]

            shap_vals = self.explainer.shap_values(X_sample)

            # For binary classification, shap_values returns list[2]; take class=1
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]

            self.shap_values = shap_vals

            # Feature names for transformed matrix
            n_features = shap_vals.shape[1]
            feat_names = (
                self.feature_names[:n_features]
                if self.feature_names and len(self.feature_names) >= n_features
                else [f"feature_{i}" for i in range(n_features)]
            )

            # Mean absolute SHAP
            mean_abs = np.abs(shap_vals).mean(axis=0)
            importance_df = pd.DataFrame({
                "feature": feat_names,
                "mean_abs_shap": mean_abs
            }).sort_values("mean_abs_shap", ascending=False)

            # Save
            imp_path = os.path.join(REPORTS_DIR, "shap_global_importance.csv")
            importance_df.to_csv(imp_path, index=False)

            # ── Summary bar plot ──────────────────────────────────────────────
            top_n = 20
            top_df = importance_df.head(top_n)
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.barh(range(top_n), top_df["mean_abs_shap"].values[::-1],
                    color="#FF5722", alpha=0.8)
            ax.set_yticks(range(top_n))
            ax.set_yticklabels(top_df["feature"].values[::-1], fontsize=9)
            ax.set_xlabel("Mean |SHAP Value|")
            ax.set_title(f"Top {top_n} Global Churn Drivers (SHAP)")
            ax.grid(axis="x", alpha=0.3)
            save_figure(fig, "shap_global_importance")
            logger.info("  Saved: shap_global_importance.png")

            # ── SHAP beeswarm summary plot ─────────────────────────────────
            try:
                fig2 = plt.figure(figsize=(12, 8))
                shap.summary_plot(
                    shap_vals[:, :top_n],
                    X_sample[:, :top_n],
                    feature_names=feat_names[:top_n],
                    show=False,
                    plot_size=None
                )
                save_figure(fig2, "shap_beeswarm")
                logger.info("  Saved: shap_beeswarm.png")
            except Exception as e:
                logger.warning(f"  Beeswarm plot failed: {e}")

            logger.info(f"  Top 5 churn drivers: {importance_df['feature'].head(5).tolist()}")
            return importance_df

        except Exception as e:
            logger.error(f"Global SHAP explanation failed: {e}")
            return pd.DataFrame()

    # ── Local Explanation ──────────────────────────────────────────────────────

    def local_explanation(
        self,
        X: pd.DataFrame,
        customer_index: int = 0,
        customer_id: Optional[str] = None
    ) -> dict:
        """
        Explain a single customer's churn prediction.

        Parameters
        ----------
        X              : pd.DataFrame  — feature matrix
        customer_index : int  — row index in X
        customer_id    : str (optional)  — for display purposes

        Returns
        -------
        dict  — top features pushing toward / against churn
        """
        if self.explainer is None:
            return {}

        try:
            preprocessor = self.pipeline.named_steps["preprocessor"]
            X_t = preprocessor.transform(X)
            row = X_t[customer_index:customer_index+1]

            sv = self.explainer.shap_values(row)
            if isinstance(sv, list):
                sv = sv[1]
            sv = sv[0]

            n_features = len(sv)
            feat_names = (
                self.feature_names[:n_features]
                if self.feature_names and len(self.feature_names) >= n_features
                else [f"feature_{i}" for i in range(n_features)]
            )

            explanation = pd.DataFrame({
                "feature": feat_names,
                "shap_value": sv,
                "abs_shap": np.abs(sv)
            }).sort_values("abs_shap", ascending=False).head(10)

            # Waterfall-style bar plot
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ["#F44336" if v > 0 else "#4CAF50"
                      for v in explanation["shap_value"]]
            ax.barh(range(len(explanation)), explanation["shap_value"].values[::-1],
                    color=colors[::-1], alpha=0.85)
            ax.set_yticks(range(len(explanation)))
            ax.set_yticklabels(explanation["feature"].values[::-1], fontsize=9)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("SHAP Value (positive = increases churn risk)")
            title = f"Customer {customer_id or customer_index} — Churn Explanation"
            ax.set_title(title)
            ax.grid(axis="x", alpha=0.3)
            save_figure(fig, f"shap_local_customer_{customer_id or customer_index}")

            result = {
                "customer_id": customer_id or customer_index,
                "top_churn_drivers": explanation[explanation["shap_value"] > 0]["feature"].tolist(),
                "top_retention_signals": explanation[explanation["shap_value"] < 0]["feature"].tolist(),
                "explanation_df": explanation,
            }
            return result

        except Exception as e:
            logger.error(f"Local explanation failed: {e}")
            return {}