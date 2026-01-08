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


class ClusteringProfileStrategy(IVisualizationStrategy):
    """
    Segments respondents into clusters based on their QVT profile (Dimension scores).
    Automatically selects the optimal number of clusters k.
    Visualizes the average dimension profile (line chart) and demographic composition.
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
        df = add_age_band(survey_df.copy())
        df = add_seniority_band(df)
        if hr_df is not None and not hr_df.empty:
            common = set(df.columns) & set(hr_df.columns)
            if "ID" in common:
                to_drop = [c for c in common if c != "ID"]
                hr_clean = hr_df.drop(columns=to_drop)
                df = df.merge(hr_clean, on="ID", how="left")
            elif df.index.equals(hr_df.index):
                for col in hr_df.columns:
                    if col not in df.columns:
                        df[col] = hr_df[col]

        if "Ancienne" in df.columns:
            df = df.rename(columns={"Ancienne": "Ancienneté"})
            
        df = df.loc[:, ~df.columns.duplicated()]

        for col, mapping in DEMO_VALUE_MAPPING.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(df[col])

        for key, value in (filters or {}).items():
            if key in df.columns:
                df = df[df[key] == value]

        scores_df = compute_prefix_scores(df)
        feature_cols = [c for c in scores_df.columns if c.startswith("DIM_")]
        if not feature_cols:
            raise ValueError("No dimension scores available for clustering.")

        to_concat = scores_df[[c for c in scores_df.columns if c not in df.columns]]
        full_df = pd.concat([df, to_concat], axis=1)
        full_df = full_df.dropna(subset=feature_cols)

        if len(full_df) < 20:
            raise ValueError(f"Not enough data for clustering (min 20, got {len(full_df)})")

        features = full_df[feature_cols].values.astype(float)
        try:
            features_std = whiten(features)
        except Exception:
            features_std = features

        # 2. Automatic k Selection
        k = self._select_best_k(features_std, config)

        # 3. Clustering Execution
        try:
            centroids_std, labels = kmeans2(features_std, k, minit="points", check_finite=False)
        except Exception as e:
            raise ValueError(f"Clustering failed: {str(e)}")

        full_df["cluster"] = labels
        full_df["cluster_label"] = full_df["cluster"].apply(lambda x: f"Segment {x+1}")

        cluster_sizes = full_df.groupby("cluster_label").size().reset_index(name="count")
        full_df = full_df.merge(cluster_sizes, on="cluster_label")
        full_df["label_with_n"] = full_df.apply(
            lambda r: f"{r['cluster_label']} (n={r['count']})", axis=1
        )

        # 4. Dimension Profile Data
        profile_long = full_df.melt(
            id_vars=["label_with_n", "cluster_label", "count"],
            value_vars=feature_cols,
            var_name="dim_key",
            value_name="mean_score",
        )
        profile_long = profile_long.groupby(["label_with_n", "cluster_label", "count", "dim_key"], observed=False)["mean_score"].mean().reset_index()
        profile_long["dimension"] = profile_long["dim_key"].str.replace("DIM_", "")
        profile_long["dimension_label"] = profile_long["dimension"].apply(prefix_label)

        # 5. Demographic Composition
        demo_fields = config.get("demographic_fields")
        if not demo_fields:
            all_avail = available_demographics(full_df)
            exclude = {"ID", "Age", "Ancienne", "Ancienneté"}
            # Include more demographics to "represent everything"
            demo_fields = [f for f in all_avail if f not in exclude][:6]

        demo_data_list = []
        for field in demo_fields:
            if field not in full_df.columns: continue
            counts = full_df.groupby(["cluster_label", field], observed=False).size().reset_index(name="n")
            totals = full_df.groupby("cluster_label", observed=False).size().reset_index(name="total")
            merged = counts.merge(totals, on="cluster_label")
            merged["percentage"] = merged["n"] / merged["total"]
            merged["variable"] = field
            merged = merged.rename(columns={field: "value"})
            demo_data_list.append(merged[["cluster_label", "variable", "value", "percentage", "n"]])
        
        demo_df = pd.concat(demo_data_list) if demo_data_list else pd.DataFrame()

        # 6. Visualization
        apply_theme()
        
        segment_colors = alt.Scale(
            domain=sorted(full_df["cluster_label"].unique()),
            range=["#4F46E5", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"]
        )

        # --- CHART 1: Profile Signature (Line) ---
        profile_line = alt.Chart(profile_long).mark_line(
            strokeWidth=3, 
            interpolate="monotone",
            opacity=0.85
        ).encode(
            x=alt.X("dimension_label:N", title=None, axis=alt.Axis(labelAngle=-45, labelLimit=150, labelFontSize=10)),
            y=alt.Y("mean_score:Q", title="Score Moyen", scale=alt.Scale(domain=[1, 5]), axis=alt.Axis(grid=True, gridDash=[2,2])),
            color=alt.Color("cluster_label:N", scale=segment_colors, legend=alt.Legend(title="Segments", orient="top-left", offset=10)),
            tooltip=[
                alt.Tooltip("cluster_label", title="Segment"),
                alt.Tooltip("dimension_label", title="Dimension"),
                alt.Tooltip("mean_score:Q", format=".2f", title="Moyenne"),
            ]
        )

        profile_points = profile_line.mark_point(size=80, filled=True, opacity=1).encode(opacity=alt.value(1))

        profile_view = (profile_line + profile_points).properties(
            width=550,
            height=320,
            title=alt.TitleParams(
                text="Signature de Réponse par Segment",
                subtitle="Profil psychométrique moyen des groupes de collaborateurs identifiés.",
                fontSize=14, anchor="start", color="#1E293B"
            )
        )

        # --- CHART 2: Segment Size (Reduced gaps) ---
        size_chart = alt.Chart(cluster_sizes).mark_bar(cornerRadius=4, size=24).encode(
            y=alt.Y("cluster_label:N", title=None, axis=alt.Axis(labelFontSize=10)),
            x=alt.X("count:Q", title="Nombre de répondants", axis=alt.Axis(grid=True, gridDash=[2,2])),
            color=alt.Color("cluster_label:N", scale=segment_colors, legend=None),
            tooltip=[alt.Tooltip("cluster_label", title="Segment"), alt.Tooltip("count:Q", title="Effectif")]
        ).properties(
            width=180, 
            height=alt.Step(45), # Tight gaps proportional to bar size
            title=alt.TitleParams(text="Répartition Numérique", fontSize=12, anchor="start", color="#475569")
        )

        # --- CHART 3: Demographic Composition (Stacked Bars) ---
        demo_bars = alt.Chart(demo_df).mark_bar(cornerRadius=2).encode(
            y=alt.Y("cluster_label:N", title=None, axis=alt.Axis(labelFontSize=10)),
            x=alt.X("percentage:Q", title=None, axis=alt.Axis(format="%", grid=False, labels=False)),
            color=alt.Color(
                "value:N", 
                title=None, 
                scale=alt.Scale(scheme="tableau20"),
                legend=alt.Legend(orient="bottom", columns=2, labelFontSize=9, symbolSize=40, padding=10)
            ),
            tooltip=[
                alt.Tooltip("cluster_label", title="Segment"),
                alt.Tooltip("variable", title="Critère"),
                alt.Tooltip("value", title="Catégorie"),
                alt.Tooltip("percentage:Q", format=".1%", title="Proportion"),
                alt.Tooltip("n:Q", title="Effectif"),
            ]
        ).properties(width=180, height=alt.Step(30))

        facet_demo = demo_bars.facet(
            facet=alt.Facet("variable:N", title=None, header=alt.Header(labelFontSize=11, labelFontWeight="bold", labelColor="#4F46E5")),
            columns=3,
            spacing=30
        ).resolve_scale(color="independent")

        # Layout Assembly
        top_row = alt.hconcat(profile_view, size_chart, spacing=50).resolve_scale(color="shared")
        
        final_chart = alt.vconcat(
            top_row,
            facet_demo,
            spacing=60
        ).properties(
            title={
                "text": "Analyse Structurelle des Segments Collaborateurs",
                "subtitle": [
                    f"Segmentation automatique regroupant n={len(full_df)} collaborateurs en {k} profils types.",
                    "Le haut du rapport définit le comportement de réponse ; le bas révèle l'identité sociodémographique de chaque groupe."
                ],
                "anchor": "start",
                "fontSize": 18,
                "fontWeight": 700,
                "color": "#1E293B",
                "subtitleColor": "#64748B",
                "subtitleFontSize": 12,
                "dy": -25
            }
        ).configure_view(stroke=None).configure_concat(spacing=60)

        return final_chart.to_dict()

    def _select_best_k(self, features: np.ndarray, config: Dict[str, Any]) -> int:
        if "k" in config: return int(config["k"])
        n_samples = features.shape[0]
        max_k = min(6, n_samples // 5)
        if max_k < 2: return 2
        distortions = []
        for k in range(1, max_k + 1):
            _, dist = kmeans(features, k)
            distortions.append(dist)
        if len(distortions) < 2: return 2
        deltas = np.diff(distortions)
        rel_improvement = [-deltas[i] / distortions[i] for i in range(len(deltas))]
        best_k = 3 if max_k >= 3 else 2
        if rel_improvement[0] > 0.2:
            best_k = 2
            if len(rel_improvement) > 1 and rel_improvement[1] > 0.12:
                best_k = 3
                if len(rel_improvement) > 2 and rel_improvement[2] > 0.08:
                    best_k = 4
        return best_k