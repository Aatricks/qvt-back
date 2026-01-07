from typing import Any, Dict, List

import altair as alt
import pandas as pd
import scipy.cluster.hierarchy as sch

from src.config.observability import log_event
from src.services.qvt_metrics import compute_prefix_scores, prefix_label
from src.services.survey_utils import DEMO_VALUE_MAPPING
from src.viz.base import IVisualizationStrategy
from src.viz.theme import apply_theme


class CorrelationMatrixStrategy(IVisualizationStrategy):
    """Heatmap correlation matrix for aggregated QVT dimensions with hierarchical clustering."""

    def generate(
        self, data: Dict[str, pd.DataFrame], config: Dict[str, Any], filters: Dict[str, Any], settings: Any
    ) -> Dict[str, Any]:
        hr_df = data["hr"].copy()
        survey_df = data.get("survey")

        # Apply value mappings for demographics (1 -> Homme, etc.)
        for df_target in [hr_df, survey_df]:
            if df_target is not None:
                for col, mapping in DEMO_VALUE_MAPPING.items():
                    if col in df_target.columns:
                        df_target[col] = df_target[col].map(mapping).fillna(df_target[col])

        # Apply filters to both datasets
        for key, value in (filters or {}).items():
            if key in hr_df.columns:
                hr_df = hr_df[hr_df[key] == value]
            if survey_df is not None and key in survey_df.columns:
                survey_df = survey_df[survey_df.index.isin(hr_df.index)]

        facet_field: Optional[str] = config.get("facet_field")
        if facet_field and survey_df is not None and facet_field not in survey_df.columns:
            # Try to get it from HR data if linked by index
            if facet_field in hr_df.columns:
                 survey_df[facet_field] = hr_df[facet_field]
            else:
                 raise ValueError(f"Facet field '{facet_field}' not found in dataset")

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
                if facet_field:
                    numeric[facet_field] = survey_df[facet_field].values
            except ValueError:
                # No Likert items found, fallback to numeric HR fields
                numeric = hr_df.select_dtypes(include="number")
                if facet_field:
                    numeric[facet_field] = hr_df[facet_field]
        else:
            numeric = hr_df.select_dtypes(include="number")
            if facet_field:
                numeric[facet_field] = hr_df[facet_field]

        # If specific fields requested, restrict to those (matching labels or column names)
        requested: List[str] = config.get("numeric_fields") or []
        if requested:
            numeric_cols = [c for c in requested if c in numeric.columns]
            if facet_field: numeric_cols.append(facet_field)
            numeric = numeric[numeric_cols]

        if numeric.empty or (len(numeric.columns) - (1 if facet_field else 0)) < 2:
             raise ValueError("Insufficient numeric data (dimensions or fields) for correlation matrix")

        if facet_field:
            all_corr = []
            for val, group in numeric.groupby(facet_field):
                group_numeric = group.drop(columns=[facet_field]).apply(pd.to_numeric, errors="coerce").dropna()
                if len(group_numeric) < 2: continue
                corr = group_numeric.corr().stack().reset_index()
                corr.columns = ["metric_x", "metric_y", "correlation"]
                corr[facet_field] = val
                all_corr.append(corr)
            if not all_corr:
                raise ValueError(f"Insufficient data in groups for faceted correlation")
            corr_reset = pd.concat(all_corr)
            all_cols = [c for c in numeric.columns if c != facet_field]
        else:
            numeric_clean = numeric.apply(pd.to_numeric, errors="coerce").dropna()
            corr = numeric_clean.corr()
            all_cols = corr.columns.tolist()
            corr_reset = corr.stack().reset_index()
            corr_reset.columns = ["metric_x", "metric_y", "correlation"]

        # 2. Hierarchical clustering for ordering (on global correlation)
        if not facet_field and len(all_cols) > 1:
            d = sch.distance.pdist(corr)
            L = sch.linkage(d, method="ward")
            ind = sch.leaves_list(L)
            ordered_labels = [all_cols[i] for i in ind]
        else:
            ordered_labels = all_cols

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

        chart = (heatmap + text).properties(
            width=alt.Step(45),
            height=alt.Step(45)
        )

        if facet_field:
            chart = chart.facet(
                column=alt.Column(f"{facet_field}:N", title=facet_field)
            ).properties(
                title=f"Matrice de corrélation par {facet_field}"
            )
        else:
            chart = chart.properties(title="Matrice de corrélation des dimensions")

        return chart.interactive().to_dict()
