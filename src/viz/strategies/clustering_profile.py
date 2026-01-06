from __future__ import annotations

from typing import Any, Dict

import altair as alt
import pandas as pd
from scipy.cluster.vq import kmeans2, whiten

from src.services.qvt_metrics import compute_prefix_scores, prefix_label
from src.viz.base import IVisualizationStrategy
from src.viz.theme import apply_theme

# TODO: show what the clusters represent (e.g. average profiles) with % of socio demographic info
# TODO: number of clusters k should be automatically chosen
class ClusteringProfileStrategy(IVisualizationStrategy):
    """
    Segments respondents into clusters based on their QVT profile (Dimension scores).
    Uses K-Means clustering to identify "Typical Profiles" (e.g. "Satisfied", "Disengaged", "Overwhelmed").

    Visualizes the Centroids (average profile) of each cluster as a heatmap.
    """

    def generate(
        self,
        data: Dict[str, pd.DataFrame],
        config: Dict[str, Any],
        filters: Dict[str, Any],
        settings: Any,
    ) -> Dict[str, Any]:
        survey_df = data.get("survey")
        if survey_df is None:
            raise ValueError("Survey data required for clustering")

        # 1. Prepare Data
        df = survey_df.copy()
        # Apply filters
        for key, value in (filters or {}).items():
            if key in df.columns:
                df = df[df[key] == value]

        scores = compute_prefix_scores(df).dropna()

        if len(scores) < 20:
             raise ValueError("Not enough data for clustering (min 20).")

        # Features: All DIM_ columns
        feature_cols = [c for c in scores.columns if c.startswith("DIM_")]
        if not feature_cols:
            raise ValueError("No dimension scores available.")

        features = scores[feature_cols].values.astype(float)

        # 2. Clustering (K-Means)
        # Normalize (Whiten) is crucial for K-Means if variances differ.
        try:
            features_std = whiten(features)
        except Exception:
            # Fallback if whiten fails (e.g. zero variance)
            features_std = features

        # Number of clusters (k)
        # User can specify, or default to 3
        k = int(config.get("k", 3))

        # Run K-Means
        # minit='points' selects initial centroids from data points
        # check_finite=False to avoid errors with NaNs if any slipped through (though dropna used)
        try:
            centroids_std, labels = kmeans2(features_std, k, minit='points', check_finite=False)
        except Exception as e:
            raise ValueError(f"Clustering failed: {str(e)}")

        # Assign labels back to original data
        scores["cluster"] = labels
        scores["cluster_label"] = scores["cluster"].apply(lambda x: f"Groupe {x+1}")

        # Calculate centroids on ORIGINAL scale (1-5) for interpretability
        cluster_stats = scores.groupby("cluster_label")[feature_cols].mean().reset_index()

        # Calculate size of each cluster
        cluster_sizes = scores.groupby("cluster_label").size().reset_index(name="count")
        cluster_stats = cluster_stats.merge(cluster_sizes, on="cluster_label")

        # Rename cluster labels to include size (e.g. "Groupe 1 (n=45)")
        cluster_stats["label_with_n"] = cluster_stats.apply(lambda r: f"{r['cluster_label']} (n={r['count']})", axis=1)

        # 3. Format for Heatmap
        # Melt to long format: [Cluster, Dimension, Score]
        melted = cluster_stats.melt(
            id_vars=["cluster_label", "label_with_n", "count"],
            value_vars=feature_cols,
            var_name="dimension_col",
            value_name="mean_score"
        )

        melted["dimension"] = melted["dimension_col"].str.replace("DIM_", "")
        melted["dimension_label"] = melted["dimension"].apply(prefix_label)

        # 4. Visualization
        apply_theme()

        base = alt.Chart(melted).encode(
            x=alt.X("dimension_label:N", title="Dimension", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("label_with_n:N", title="Profil (Cluster)"),
        )

        heatmap = base.mark_rect().encode(
            color=alt.Color("mean_score:Q", scale=alt.Scale(domain=[1, 5], scheme="redyellowgreen"), title="Score Moyen"),
            tooltip=[
                alt.Tooltip("label_with_n", title="Groupe"),
                alt.Tooltip("dimension_label", title="Dimension"),
                alt.Tooltip("mean_score:Q", format=".2f", title="Score")
            ]
        )

        # Add text labels for score
        text = base.mark_text(color="black", size=10).encode(
            text=alt.Text("mean_score:Q", format=".1f"),
            color=alt.condition(
                (alt.datum.mean_score < 2.5) | (alt.datum.mean_score > 4.0), # Contrast adjustment
                alt.value("white"),
                alt.value("black")
            )
        )

        return (heatmap + text).properties(
            title="Profils Types (Clustering)"
        ).interactive().to_dict()
