from typing import Any, Dict, List

import altair as alt
import pandas as pd
import scipy.cluster.hierarchy as sch

from src.config.observability import log_event
from src.services.qvt_metrics import compute_prefix_scores, prefix_label
from src.viz.base import IVisualizationStrategy
from src.viz.theme import apply_theme


class CorrelationMatrixStrategy(IVisualizationStrategy):
    """Heatmap correlation matrix for aggregated QVT dimensions with hierarchical clustering."""

    def generate(
        self, data: Dict[str, pd.DataFrame], config: Dict[str, Any], filters: Dict[str, Any], settings: Any
    ) -> Dict[str, Any]:
        hr_df = data["hr"].copy()
        survey_df = data.get("survey")

        # Apply filters to both datasets
        for key, value in (filters or {}).items():
            if key in hr_df.columns:
                hr_df = hr_df[hr_df[key] == value]
            if survey_df is not None and key in survey_df.columns:
                survey_df = survey_df[survey_df.index.isin(hr_df.index)]

        # 1. Aggregate Likert items into dimensions if survey data is present
        if survey_df is not None:
            try:
                # This computes DIM_EPUI, DIM_COM, etc.
                scores_df = compute_prefix_scores(survey_df)
                # Map internal keys to labels for the chart
                labels = {
                    col: prefix_label(col.replace("DIM_", ""))
                    for col in scores_df.columns
                }
                numeric = scores_df.rename(columns=labels)
            except ValueError:
                # No Likert items found, fallback to numeric HR fields
                numeric = hr_df.select_dtypes(include="number")
        else:
            numeric = hr_df.select_dtypes(include="number")

        # If specific fields requested, restrict to those (matching labels or column names)
        requested: List[str] = config.get("numeric_fields") or []
        if requested:
            numeric_cols = [c for c in requested if c in numeric.columns]
            numeric = numeric[numeric_cols]

        if numeric.empty or len(numeric.columns) < 2:
             raise ValueError("Insufficient numeric data (dimensions or fields) for correlation matrix")

        numeric = numeric.apply(pd.to_numeric, errors="coerce").dropna()
        if numeric.empty:
            raise ValueError("Empty dataset after cleaning for correlation")

        # Compute correlation
        corr = numeric.corr()
        all_cols = corr.columns.tolist()

        # 2. Perform hierarchical clustering to order the labels
        if len(all_cols) > 1:
            d = sch.distance.pdist(corr)
            L = sch.linkage(d, method="ward")
            ind = sch.leaves_list(L)
            ordered_labels = [all_cols[i] for i in ind]
        else:
            ordered_labels = all_cols

        corr_reset = corr.stack().reset_index()
        corr_reset.columns = ["metric_x", "metric_y", "correlation"]

        apply_theme()

        base = alt.Chart(corr_reset).encode(
            x=alt.X("metric_x:N", sort=ordered_labels, title=None),
            y=alt.Y("metric_y:N", sort=ordered_labels, title=None)
        )

        heatmap = base.mark_rect().encode(
            color=alt.Color(
                "correlation:Q",
                scale=alt.Scale(scheme="blueorange", domain=[-1, 1]),
                legend=alt.Legend(title="Corrélation")
            ),
            tooltip=["metric_x", "metric_y", alt.Tooltip("correlation:Q", format=".2f")]
        )

        text = base.mark_text().encode(
            text=alt.Text("correlation:Q", format=".2f"),
            color=alt.condition(
                "abs(datum.correlation) > 0.5",
                alt.value("white"),
                alt.value("black")
            )
        )

        return (heatmap + text).properties(
            title="Matrice de corrélation des dimensions",
            width=alt.Step(45),
            height=alt.Step(45)
        ).interactive().to_dict()
