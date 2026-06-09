"""
src/churn_model.py
------------------
Machine learning pipeline for churn prediction.

Models trained:
  1. Logistic Regression  — baseline, interpretable
  2. Random Forest        — ensemble, handles non-linearity
  3. XGBoost              — gradient boosting, typically best performer

Pipeline:
  preprocessing → SMOTE (train only) → model → evaluation

All models use a shared scikit-learn Pipeline for clean separation
of preprocessing and modelling steps.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import joblib

from typing import Dict, Tuple, Any
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

from config import (
    LOGISTIC_PARAMS, RF_PARAMS, XGB_PARAMS,
    RANDOM_STATE, MODELS_DIR, PLOTS_DIR
)
from src.utils import get_logger, timer, save_figure

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


class ChurnModelTrainer:
    """
    Trains, evaluates, and saves churn prediction models.

    Usage
    -----
    trainer = ChurnModelTrainer()
    trainer.fit(X_train, y_train)
    results = trainer.evaluate(X_test, y_test)
    trainer.save_best_model()
    """

    MODEL_DEFINITIONS = {
        "logistic_regression": LogisticRegression(**LOGISTIC_PARAMS),
        "random_forest":       RandomForestClassifier(**RF_PARAMS),
        "xgboost":             XGBClassifier(**XGB_PARAMS),
    }

    def __init__(self):
        self.pipelines: Dict[str, ImbPipeline] = {}
        self.results: Dict[str, Dict] = {}
        self.best_model_name: str = ""
        self.best_pipeline: Any = None
        self.preprocessor: ColumnTransformer = None
        self.feature_names: list = []

    # ── Preprocessing ──────────────────────────────────────────────────────────

    def _build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Build sklearn ColumnTransformer:
          - Numeric: median imputation + StandardScaler
          - Categorical: most-frequent imputation + OneHotEncoder

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        ColumnTransformer
        """
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

        logger.info(f"  Numeric features:  {len(num_cols)}")
        logger.info(f"  Categorical features: {len(cat_cols)}")

        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ])

        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        transformers = []
        if num_cols:
            transformers.append(("num", numeric_transformer, num_cols))
        if cat_cols:
            transformers.append(("cat", categorical_transformer, cat_cols))

        preprocessor = ColumnTransformer(transformers=transformers)
        return preprocessor

    def _get_feature_names(self, X: pd.DataFrame) -> list:
        """Extract final feature names after preprocessing."""
        try:
            num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
            cat_feature_names = (
                self.preprocessor.named_transformers_["cat"]["encoder"]
                .get_feature_names_out(cat_cols).tolist()
                if cat_cols else []
            )
            return num_cols + cat_feature_names
        except Exception:
            return []

    # ── Training ───────────────────────────────────────────────────────────────

    @timer
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train all three models using SMOTE for class imbalance.

        Parameters
        ----------
        X_train : pd.DataFrame
        y_train : pd.Series
        """
        logger.info("=== Training Churn Models ===")

        self.preprocessor = self._build_preprocessor(X_train)

        for name, model in self.MODEL_DEFINITIONS.items():
            logger.info(f"  Training {name}...")
            try:
                # imblearn Pipeline: preprocessor → SMOTE → model
                pipe = ImbPipeline([
                    ("preprocessor", self.preprocessor),
                    ("smote",        SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
                    ("classifier",   model),
                ])
                pipe.fit(X_train, y_train)
                self.pipelines[name] = pipe
                logger.info(f"    ✓ {name} trained")
            except Exception as e:
                logger.error(f"    ✗ {name} failed: {e}")

        self.feature_names = self._get_feature_names(X_train)

    # ── Evaluation ─────────────────────────────────────────────────────────────

    @timer
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> pd.DataFrame:
        """
        Evaluate all trained models and identify the best one.

        Parameters
        ----------
        X_test : pd.DataFrame
        y_test : pd.Series

        Returns
        -------
        pd.DataFrame  — metrics comparison table
        """
        logger.info("=== Evaluating Models ===")
        rows = []

        for name, pipe in self.pipelines.items():
            y_pred  = pipe.predict(X_test)
            y_proba = pipe.predict_proba(X_test)[:, 1]

            metrics = {
                "model":     name,
                "accuracy":  round(accuracy_score(y_test, y_pred),  4),
                "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
                "recall":    round(recall_score(y_test, y_pred, zero_division=0),    4),
                "f1":        round(f1_score(y_test, y_pred, zero_division=0),        4),
                "roc_auc":   round(roc_auc_score(y_test, y_proba),                   4),
            }
            self.results[name] = {**metrics, "y_pred": y_pred, "y_proba": y_proba}
            rows.append(metrics)

            logger.info(
                f"  {name:<25} | "
                f"AUC={metrics['roc_auc']:.4f} | "
                f"F1={metrics['f1']:.4f} | "
                f"Recall={metrics['recall']:.4f}"
            )

        results_df = pd.DataFrame(rows).set_index("model")

        # Best model = highest ROC-AUC
        self.best_model_name = results_df["roc_auc"].idxmax()
        self.best_pipeline   = self.pipelines[self.best_model_name]
        logger.info(f"  ★ Best model: {self.best_model_name}")

        # Generate plots
        self._plot_confusion_matrices(y_test)
        self._plot_roc_curves(y_test)
        self._plot_feature_importance()

        return results_df

    # ── Plots ──────────────────────────────────────────────────────────────────

    def _plot_confusion_matrices(self, y_test: pd.Series) -> None:
        """Plot confusion matrix for each model."""
        n = len(self.pipelines)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
        if n == 1:
            axes = [axes]

        for ax, (name, res) in zip(axes, self.results.items()):
            cm = confusion_matrix(y_test, res["y_pred"])
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Active", "Churned"],
                yticklabels=["Active", "Churned"],
                ax=ax
            )
            ax.set_title(f"{name}\nF1={res['f1']:.3f}", fontsize=12)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

        fig.suptitle("Confusion Matrices — Churn Prediction", fontsize=14, y=1.02)
        save_figure(fig, "confusion_matrices")
        logger.info("  Saved: confusion_matrices.png")

    def _plot_roc_curves(self, y_test: pd.Series) -> None:
        """Plot ROC curves for all models on one chart."""
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ["#2196F3", "#4CAF50", "#FF5722"]

        for (name, res), color in zip(self.results.items(), colors):
            fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
            ax.plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.3f})", color=color, lw=2)

        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves — Churn Prediction")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        save_figure(fig, "roc_curves")
        logger.info("  Saved: roc_curves.png")

    def _plot_feature_importance(self) -> None:
        """Plot feature importance for Random Forest and XGBoost."""
        for model_name in ["random_forest", "xgboost"]:
            if model_name not in self.pipelines:
                continue
            try:
                clf = self.pipelines[model_name].named_steps["classifier"]
                importances = clf.feature_importances_

                n = min(20, len(importances))
                names = self.feature_names[:len(importances)] if self.feature_names else [
                    f"f{i}" for i in range(len(importances))
                ]

                # Top N
                idx = np.argsort(importances)[::-1][:n]
                top_names = [names[i] if i < len(names) else f"f{i}" for i in idx]
                top_vals  = importances[idx]

                fig, ax = plt.subplots(figsize=(10, 7))
                bars = ax.barh(range(n), top_vals[::-1], color="#2196F3", alpha=0.8)
                ax.set_yticks(range(n))
                ax.set_yticklabels(top_names[::-1], fontsize=9)
                ax.set_xlabel("Feature Importance")
                ax.set_title(f"Top {n} Features — {model_name.replace('_', ' ').title()}")
                ax.grid(axis="x", alpha=0.3)
                save_figure(fig, f"feature_importance_{model_name}")
                logger.info(f"  Saved: feature_importance_{model_name}.png")
            except Exception as e:
                logger.warning(f"Could not plot feature importance for {model_name}: {e}")

    # ── Predict ────────────────────────────────────────────────────────────────

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return churn probabilities using the best model.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        np.ndarray — shape (n_samples,)  probability of churn
        """
        if self.best_pipeline is None:
            raise RuntimeError("Model not trained yet. Call fit() first.")
        return self.best_pipeline.predict_proba(X)[:, 1]

    # ── Save / Load ────────────────────────────────────────────────────────────

    def save_best_model(self) -> str:
        """
        Persist the best pipeline to disk using joblib.

        Returns
        -------
        str  — path where model was saved
        """
        path = os.path.join(MODELS_DIR, "best_churn_model.pkl")
        joblib.dump(self.best_pipeline, path)
        logger.info(f"  Saved best model ({self.best_model_name}) → {path}")

        # Also save metadata
        meta = {
            "best_model_name": self.best_model_name,
            "metrics": {k: v for k, v in self.results[self.best_model_name].items()
                        if k not in ("y_pred", "y_proba")},
            "feature_names": self.feature_names,
        }
        meta_path = os.path.join(MODELS_DIR, "model_metadata.pkl")
        joblib.dump(meta, meta_path)
        return path

    @staticmethod
    def load_model(path: str) -> Any:
        """Load a saved model from disk."""
        return joblib.load(path)