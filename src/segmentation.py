"""
src/segmentation.py
-------------------
Customer segmentation using:
  1. RFM analysis (Recency, Frequency, Monetary)
  2. KMeans clustering with elbow method to determine optimal K
  3. Interpretable segment naming based on cluster centroids

Segment examples:
  - High Value Loyalists
  - At Risk Premium
  - Discount Flyers
  - Dormant Members
  - Frequent Redeemers
  - Seasonal Travelers
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from typing import Tuple, Dict, List

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

from config import (
    CUSTOMER_ID_COL, N_CLUSTERS_MIN, N_CLUSTERS_MAX,
    N_CLUSTERS_FINAL, RANDOM_STATE, PLOTS_DIR, REPORTS_DIR
)
from src.utils import get_logger, timer, save_figure

logger = get_logger(__name__)


class CustomerSegmenter:
    """
    Performs RFM analysis and KMeans clustering on the customer feature matrix.

    Usage
    -----
    segmenter = CustomerSegmenter()
    segment_df = segmenter.fit_transform(feature_df)
    segmenter.plot_all()
    """

    # ── RFM columns (must exist in feature_df) ─────────────────────────────────
    RFM_MAP = {
        "recency":   "recency_months",            # Lower = better
        "frequency": "alltime_total_flights_sum", # Higher = better
        "monetary":  "alltime_points_accumulated_sum",  # Higher = better
    }

    # ── Segment name palette (assigned after clustering) ──────────────────────
    SEGMENT_NAMES = [
        "High Value Loyalists",
        "At Risk Premium",
        "Frequent Redeemers",
        "Dormant Members",
        "Seasonal Travelers",
        "Discount Flyers",
    ]

    def __init__(self, n_clusters: int = N_CLUSTERS_FINAL):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans: KMeans = None
        self.segment_df: pd.DataFrame = None
        self.rfm_df: pd.DataFrame = None
        self.cluster_names: Dict[int, str] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    @timer
    def fit_transform(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run full segmentation pipeline.

        Parameters
        ----------
        feature_df : pd.DataFrame  — output of FeatureEngineer.build_features()

        Returns
        -------
        pd.DataFrame  — with 'segment_id' and 'segment_name' columns
        """
        logger.info("=== Customer Segmentation ===")

        rfm = self._build_rfm(feature_df)
        self.rfm_df = rfm

        # Determine optimal K
        optimal_k = self._elbow_method(rfm)
        self.n_clusters = optimal_k

        # Fit KMeans
        labels = self._fit_kmeans(rfm)

        # Assign interpretable names
        self._assign_segment_names(rfm, labels)

        # Build output
        result = feature_df[[CUSTOMER_ID_COL]].copy() if CUSTOMER_ID_COL in feature_df.columns else feature_df.index.to_frame(name=CUSTOMER_ID_COL)
        result = result.reset_index(drop=True)
        result["segment_id"]   = labels
        result["segment_name"] = result["segment_id"].map(self.cluster_names)

        # Merge RFM scores back
        result = result.merge(rfm.reset_index(), on=CUSTOMER_ID_COL, how="left")

        self.segment_df = result
        logger.info("  Segment distribution:")
        for seg, cnt in result["segment_name"].value_counts().items():
            logger.info(f"    {seg:<30} {cnt:>5} ({cnt/len(result)*100:.1f}%)")

        return result

    # ── RFM ────────────────────────────────────────────────────────────────────

    def _build_rfm(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build RFM table from feature DataFrame.

        RFM Scores (1–5, 5 = best):
          R_score: 5 = most recent
          F_score: 5 = most frequent
          M_score: 5 = highest monetary
        """
        id_col = CUSTOMER_ID_COL if CUSTOMER_ID_COL in df.columns else None
        if id_col:
            rfm = df.set_index(id_col)[[
                c for c in self.RFM_MAP.values() if c in df.columns
            ]].copy()
        else:
            rfm = df[[
                c for c in self.RFM_MAP.values() if c in df.columns
            ]].copy()

        # Rename to R/F/M
        rename_map = {v: k for k, v in self.RFM_MAP.items() if v in rfm.columns}
        rfm.rename(columns=rename_map, inplace=True)

        # Fill missing columns with neutral values
        for col in ["recency", "frequency", "monetary"]:
            if col not in rfm.columns:
                rfm[col] = 0

        rfm.fillna(0, inplace=True)

        # Compute quintile scores (1–5)
        # rfm["R_score"] = pd.qcut(rfm["recency"],   5, labels=[5,4,3,2,1], duplicates="drop").astype(float)
        # rfm["F_score"] = pd.qcut(rfm["frequency"],  5, labels=[1,2,3,4,5], duplicates="drop").astype(float)
        # rfm["M_score"] = pd.qcut(rfm["monetary"],   5, labels=[1,2,3,4,5], duplicates="drop").astype(float)
        # Recency
        # Recency
        r_bins = pd.qcut(
            rfm["recency"],
            q=5,
            duplicates="drop"
        )

        rfm["R_score"] = pd.factorize(r_bins)[0] + 1
        rfm["R_score"] = 6 - rfm["R_score"]


        # Frequency
        f_bins = pd.qcut(
            rfm["frequency"],
            q=5,
            duplicates="drop"
        )

        rfm["F_score"] = pd.factorize(f_bins)[0] + 1


        # Monetary
        m_bins = pd.qcut(
            rfm["monetary"],
            q=5,
            duplicates="drop"
        )

        rfm["M_score"] = pd.factorize(m_bins)[0] + 1
        rfm["RFM_score"] = (
            rfm["R_score"] +
            rfm["F_score"] +
            rfm["M_score"]
        )
        return rfm
    # ── Elbow Method ───────────────────────────────────────────────────────────

    def _elbow_method(self, rfm: pd.DataFrame) -> int:
        """
        Determine optimal number of clusters using the elbow method.

        We compute distortion (mean squared distance to nearest centroid)
        for K = N_CLUSTERS_MIN to N_CLUSTERS_MAX, then find the elbow.

        Returns
        -------
        int — optimal K
        """
        feature_cols = ["recency", "frequency", "monetary"]
        X = rfm[feature_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X)

        distortions = []
        ks = range(N_CLUSTERS_MIN, N_CLUSTERS_MAX + 1)

        for k in ks:
            km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
            km.fit(X_scaled)
            distortion = sum(
                np.min(cdist(X_scaled, km.cluster_centers_, "euclidean"), axis=1)
            ) / X_scaled.shape[0]
            distortions.append(distortion)

        # Plot elbow
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(list(ks), distortions, "bo-", lw=2, markersize=8)
        ax.set_xlabel("Number of Clusters (K)")
        ax.set_ylabel("Average Distortion")
        ax.set_title("Elbow Method — Optimal Number of Segments")
        ax.grid(alpha=0.3)
        save_figure(fig, "elbow_method")

        # Elbow detection: find point of max curvature
        deltas = np.diff(distortions)
        second_deriv = np.diff(deltas)
        if len(second_deriv) > 0:
            elbow_idx = np.argmax(np.abs(second_deriv)) + 2
            optimal_k = list(ks)[elbow_idx]
        else:
            optimal_k = N_CLUSTERS_FINAL

        # Clamp to sane range
        optimal_k = max(N_CLUSTERS_MIN, min(optimal_k, N_CLUSTERS_MAX))
        logger.info(f"  Elbow method suggests K = {optimal_k}")
        return optimal_k

    # ── KMeans ─────────────────────────────────────────────────────────────────

    def _fit_kmeans(self, rfm: pd.DataFrame) -> np.ndarray:
        """
        Fit KMeans on RFM features.

        Returns
        -------
        np.ndarray — cluster labels (0-indexed)
        """
        feature_cols = ["recency", "frequency", "monetary"]
        X = rfm[feature_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X)

        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=RANDOM_STATE,
            n_init=20,
            max_iter=300,
        )
        labels = self.kmeans.fit_predict(X_scaled)
        logger.info(f"  KMeans fitted: {self.n_clusters} clusters")
        return labels

    # ── Segment Naming ─────────────────────────────────────────────────────────

    def _assign_segment_names(
        self,
        rfm: pd.DataFrame,
        labels: np.ndarray
    ) -> None:
        """
        Map cluster IDs to interpretable business names.

        Logic:
          1. Compute centroid of each cluster in RFM space
          2. Rank clusters by RFM_score
          3. Assign names from SEGMENT_NAMES list (highest RFM → first name)
        """
        rfm_with_labels = rfm.copy()
        rfm_with_labels["cluster_id"] = labels

        centroids = rfm_with_labels.groupby("cluster_id")["RFM_score"].mean()
        sorted_clusters = centroids.sort_values(ascending=False).index.tolist()

        # Pad names if we have more/fewer clusters than names
        names = self.SEGMENT_NAMES[:len(sorted_clusters)]
        while len(names) < len(sorted_clusters):
            names.append(f"Segment {len(names)+1}")

        self.cluster_names = {cluster_id: name
                               for cluster_id, name in zip(sorted_clusters, names)}
        logger.info(f"  Cluster → Segment mapping: {self.cluster_names}")

    # ── Plots ──────────────────────────────────────────────────────────────────

    def plot_all(self) -> None:
        """Generate all segmentation visualisations."""
        if self.segment_df is None:
            raise RuntimeError("Call fit_transform() first.")
        self._plot_rfm_scatter()
        self._plot_radar_chart()
        self._plot_segment_distribution()
        self._save_segment_summary()

    def _plot_rfm_scatter(self) -> None:
        """2D scatter of Frequency vs Monetary, coloured by segment."""
        df = self.segment_df
        fig, ax = plt.subplots(figsize=(10, 7))
        segments = df["segment_name"].unique()
        colors   = plt.cm.Set2(np.linspace(0, 1, len(segments)))

        for seg, color in zip(segments, colors):
            mask = df["segment_name"] == seg
            ax.scatter(
                df.loc[mask, "frequency"].clip(upper=df["frequency"].quantile(0.99)),
                df.loc[mask, "monetary"].clip(upper=df["monetary"].quantile(0.99)),
                label=seg, color=color, alpha=0.5, s=20
            )
        ax.set_xlabel("Frequency (Total Flights)")
        ax.set_ylabel("Monetary (Points Accumulated)")
        ax.set_title("Customer Segments — Frequency vs Monetary Value")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)
        save_figure(fig, "segment_rfm_scatter")

    def _plot_radar_chart(self) -> None:
        """Radar chart showing average RFM scores per segment."""
        df = self.segment_df
        metrics = ["R_score", "F_score", "M_score"]
        available = [m for m in metrics if m in df.columns]
        if len(available) < 2:
            return

        segments = df["segment_name"].unique()
        n_metrics = len(available)
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Close the radar

        fig, axes = plt.subplots(
            1, len(segments),
            figsize=(4 * len(segments), 4),
            subplot_kw={"polar": True}
        )
        if len(segments) == 1:
            axes = [axes]

        colors = plt.cm.Set2(np.linspace(0, 1, len(segments)))

        for ax, seg, color in zip(axes, segments, colors):
            vals = df[df["segment_name"] == seg][available].mean().tolist()
            vals += vals[:1]
            ax.plot(angles, vals, "o-", color=color, linewidth=2)
            ax.fill(angles, vals, alpha=0.25, color=color)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(["Recency", "Frequency", "Monetary"], size=8)
            ax.set_ylim(0, 5)
            ax.set_title(seg, size=9, pad=10)

        fig.suptitle("Segment Profiles — RFM Radar Charts", y=1.02)
        save_figure(fig, "segment_radar_charts")

    def _plot_segment_distribution(self) -> None:
        """Bar chart of segment sizes."""
        counts = self.segment_df["segment_name"].value_counts()
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(counts.index, counts.values,
                      color=plt.cm.Set2(np.linspace(0, 1, len(counts))))
        ax.set_ylabel("Number of Customers")
        ax.set_title("Customer Segment Distribution")
        plt.xticks(rotation=25, ha="right")
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    str(val), ha="center", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        save_figure(fig, "segment_distribution")

    def _save_segment_summary(self) -> None:
        """Save segment summary statistics to CSV."""
        df = self.segment_df
        summary_cols = ["segment_name", "recency", "frequency", "monetary",
                        "R_score", "F_score", "M_score", "RFM_score"]
        available = [c for c in summary_cols if c in df.columns]
        summary = df[available].groupby("segment_name").agg(
            ["mean", "median", "count"]
        ).round(2)
        path = os.path.join(REPORTS_DIR, "segment_summary.csv")
        summary.to_csv(path)
        logger.info(f"  Saved segment summary → {path}")