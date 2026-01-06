from __future__ import annotations

from typing import Any, Dict, List, Optional

import altair as alt
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans, kmeans2, whiten

from src.services.qvt_metrics import compute_prefix_scores, prefix_label
from src.services.survey_utils import (
    DEMO_VALUE_MAPPING,
    add_age_band,
    add_seniority_band,
    available_demographics,
)
from src.viz.base import IVisualizationStrategy
from src.viz.theme import apply_theme

# TODO: improve UI/UX 
class ClusteringProfileStrategy(IVisualizationStrategy):
    """
    Segments respondents into clusters based on their QVT profile (Dimension scores).
    Automatically selects the optimal number of clusters k.
    Visualizes the average dimension profile (heatmap) and demographic composition (bars).
    """

    def generate(
        self,
        data: Dict[str, pd.DataFrame],
        config: Dict[str, Any],
        filters: Dict[str, Any],
        settings: Any,
    ) -> Dict[str, Any]:
        survey_df = data.get("survey")
        hr_df = data.get("hr")
        if survey_df is None or hr_df is None:
            raise ValueError("Survey and HR data required for clustering profile")

        # 1. Prepare Data
        # Merge survey and HR to have demographics and scores together
        df = add_age_band(survey_df.copy())
        df = add_seniority_band(df)
        if hr_df is not None and not hr_df.empty:
            # Assume shared index or ID column. If no common columns, we hope they are aligned.
            common = set(df.columns) & set(hr_df.columns)
            if "ID" in common:
                df = df.merge(hr_df, on="ID", how="left", suffixes=("", "_hr"))
            elif df.index.equals(hr_df.index):
                # If indices match perfectly, we can just join
                for col in hr_df.columns:
                    if col not in df.columns:
                        df[col] = hr_df[col]

        # Rename "Ancienne" to "Ancienneté" if present
        if "Ancienne" in df.columns:
            df = df.rename(columns={"Ancienne": "Ancienneté"})

        # Apply value mappings for demographics (1 -> Homme, etc.)
        for col, mapping in DEMO_VALUE_MAPPING.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(df[col])

        # Apply filters
        for key, value in (filters or {}).items():
            if key in df.columns:
                df = df[df[key] == value]

        scores_df = compute_prefix_scores(df)
        feature_cols = [c for c in scores_df.columns if c.startswith("DIM_")]
        if not feature_cols:
            raise ValueError("No dimension scores available for clustering.")

        # Combined dataset for clustering and demographics
        full_df = pd.concat([df, scores_df], axis=1)
        full_df = full_df.dropna(subset=feature_cols)

        if len(full_df) < 20:
            raise ValueError(f"Not enough data for clustering (min 20, got {len(full_df)})")

        features = full_df[feature_cols].values.astype(float)
        try:
            features_std = whiten(features)
        except Exception:
            features_std = features

        # 2. Automatic k Selection (Heuristic based on distortion reduction)
        k = self._select_best_k(features_std, config)

        # 3. Clustering Execution
        try:
            centroids_std, labels = kmeans2(features_std, k, minit="points", check_finite=False)
        except Exception as e:
            raise ValueError(f"Clustering failed: {str(e)}")

        full_df["cluster"] = labels
        full_df["cluster_label"] = full_df["cluster"].apply(lambda x: f"Profil {x+1}")

        # Cluster sizes
        cluster_sizes = full_df.groupby("cluster_label").size().reset_index(name="count")
        full_df = full_df.merge(cluster_sizes, on="cluster_label")
        full_df["label_with_n"] = full_df.apply(
            lambda r: f"{r['cluster_label']} (n={r['count']})", axis=1
        )

        # 4. Dimension Profile Data (Heatmap)
        profile_long = full_df.melt(
            id_vars=["label_with_n"],
            value_vars=feature_cols,
            var_name="dim_key",
            value_name="mean_score",
        )
        profile_long = profile_long.groupby(["label_with_n", "dim_key"])["mean_score"].mean().reset_index()
        profile_long["dimension"] = profile_long["dim_key"].str.replace("DIM_", "")
        profile_long["dimension_label"] = profile_long["dimension"].apply(prefix_label)

        # 5. Demographic Composition Data (Stacked Bars)
        demo_fields = config.get("demographic_fields")
        if not demo_fields:
            # Get all available demographics
            all_avail = available_demographics(full_df)
            # Filter out non-categorical/internal fields and raw numeric fields
            exclude = {"ID", "Age", "Ancienne", "Ancienneté"}
            demo_fields = [f for f in all_avail if f not in exclude]

        demo_charts = []
        for field in demo_fields:
            if field not in full_df.columns:
                continue

            # Compute distribution for this field per cluster
            counts = full_df.groupby(["label_with_n", field]).size().reset_index(name="sub_count")
            totals = full_df.groupby("label_with_n").size().reset_index(name="cluster_total")
            field_df = counts.merge(totals, on="label_with_n")
            field_df["percentage"] = field_df["sub_count"] / field_df["cluster_total"]

            # Ensure field values are strings for nominal encoding in Altair
            field_df[field] = field_df[field].astype(str)

            # Narrower stacked bar for each demographic to respect horizontal space
            d_chart = (
                alt.Chart(field_df)
                .mark_bar()
                .encode(
                    y=alt.Y("label_with_n:N", title=None, axis=alt.Axis(labels=False, ticks=False)),
                    x=alt.X(
                        "percentage:Q",
                        title=field,
                        axis=alt.Axis(format="%", ticks=False, labels=False, grid=False)
                    ),
                    color=alt.Color(
                        f"{field}:N",
                        title=field,
                        legend=alt.Legend(
                            orient="bottom",
                            columns=2,
                            titleFontSize=10,
                            labelFontSize=9,
                            symbolSize=40
                        )
                    ),
                    tooltip=[
                        alt.Tooltip("label_with_n", title="Profil"),
                        alt.Tooltip(f"{field}:N", title=field),
                        alt.Tooltip("percentage:Q", format=".1%", title="Proportion"),
                    ],
                )
                .properties(width=80, height=alt.Step(40))
            )
            demo_charts.append(d_chart)


        # 6. Visualization
        apply_theme()

        # Main Profile Heatmap
        base_heatmap = alt.Chart(profile_long).encode(
            x=alt.X("dimension_label:N", title="Dimensions QVT", axis=alt.Axis(labelAngle=-45, labelLimit=150)),
            y=alt.Y("label_with_n:N", title="Profils Types identifiés"),
        )

        heatmap = base_heatmap.mark_rect().encode(
            color=alt.Color(
                "mean_score:Q",
                scale=alt.Scale(domain=[1, 5], scheme="redyellowgreen"),
                title="Score Moyen",
                legend=alt.Legend(orient="left", title="Score")
            ),
            tooltip=[
                alt.Tooltip("label_with_n", title="Profil"),
                alt.Tooltip("dimension_label", title="Dimension"),
                alt.Tooltip("mean_score:Q", format=".2f", title="Score Moyen"),
            ],
        )

        text = base_heatmap.mark_text(size=10, fontWeight="bold").encode(
            text=alt.Text("mean_score:Q", format=".1f"),
            color=alt.condition(
                (alt.datum.mean_score < 2.2) | (alt.datum.mean_score > 4.2),
                alt.value("white"),
                alt.value("black"),
            ),
        )

        profile_chart = (heatmap + text).properties(
            width=300,
            height=alt.Step(40),
            title="Scores moyens par dimension"
        )

        if demo_charts:
            # Combine heatmap with all demographic bars.
            # hconcat ensures they share the Y axis (Profils) visually.
            final_chart = alt.hconcat(
                profile_chart,
                *demo_charts,
                spacing=5
            ).resolve_scale(y="shared")
        else:
            final_chart = profile_chart

        return (
            final_chart.properties(
                title={
                    "text": "Segmentation des profils types et composition démographique",
                    "subtitle": [f"Nombre de clusters (k={k}) déterminé automatiquement par analyse de variance."],
                    "anchor": "start",
                    "frame": "group"
                }
            )
            .to_dict()
        )

    def _select_best_k(self, features: np.ndarray, config: Dict[str, Any]) -> int:
        """Heuristic to select k between 2 and 6 based on distortion reduction."""
        if "k" in config:
            return int(config["k"])

        n_samples = features.shape[0]
        max_k = min(6, n_samples // 5)
        if max_k < 2:
            return 2

        distortions = []
        ks = range(1, max_k + 1)
        for k in ks:
            _, dist = kmeans(features, k)
            distortions.append(dist)

        if len(distortions) < 2:
            return 2

        # Use the "Elbow" logic: find k where the drop in distortion slows down significantly.
        # We look for the maximum 'curvature' or just the biggest drop in relative distortion.
        deltas = np.diff(distortions)
        # Relative improvement: (D_k-1 - D_k) / D_k-1
        rel_improvement = [-deltas[i] / distortions[i] for i in range(len(deltas))]

        # Default to 3 if we have enough data and no obvious elbow
        best_k = 3 if max_k >= 3 else 2

        # If k=2 provides more than 25% improvement over k=1, keep looking
        # If k=3 provides another 15% improvement over k=2, take it.
        if rel_improvement[0] > 0.2:
            best_k = 2
            if len(rel_improvement) > 1 and rel_improvement[1] > 0.12:
                best_k = 3
                if len(rel_improvement) > 2 and rel_improvement[2] > 0.08:
                    best_k = 4

        return best_k
