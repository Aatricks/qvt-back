from __future__ import annotations

from typing import Any, Dict, Optional

import altair as alt
import numpy as np
import pandas as pd

from src.services.qvt_metrics import compute_prefix_scores, prefix_label
from src.services.survey_utils import DEMO_VALUE_MAPPING, add_age_band
from src.viz.base import IVisualizationStrategy
from src.viz.theme import apply_theme


class DimensionCIBarsStrategy(IVisualizationStrategy):
    """
    Visualizes dimension mean scores with Standard Deviation as error bars to show dispersion.
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
            raise ValueError("Survey data required for dimension dispersion bars")

        df = add_age_band(survey_df.copy())
        
        # Apply value mappings for demographics (1 -> Homme, etc.)
        for col, mapping in DEMO_VALUE_MAPPING.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(df[col])

        # Apply simple equality filters
        for key, value in (filters or {}).items():
            if key in df.columns:
                df = df[df[key] == value]

        if df.empty:
            raise ValueError("Empty dataset after filtering")

        segment_field: Optional[str] = config.get("segment_field")
        if segment_field and segment_field not in df.columns:
            raise ValueError(f"Segment field '{segment_field}' not found in dataset")

        facet_field: Optional[str] = config.get("facet_field")
        if facet_field and facet_field not in df.columns:
            raise ValueError(f"Facet field '{facet_field}' not found in dataset")

        # Compute respondent-level dimension scores (DIM_<PREFIX> columns)
        dim_scores = compute_prefix_scores(df)

        # Build a long table
        id_vars = []
        if segment_field: id_vars.append(segment_field)
        if facet_field: id_vars.append(facet_field)

        long_df = dim_scores.reset_index(drop=True)
        for f in id_vars:
            long_df[f] = df[f].reset_index(drop=True)

        long_df = long_df.melt(
            id_vars=id_vars if id_vars else None,
            var_name="dimension_key",
            value_name="score",
        )

        long_df["score"] = pd.to_numeric(long_df["score"], errors="coerce")
        long_df = long_df.dropna(subset=["score"])

        if long_df.empty:
            raise ValueError("No usable Likert data for dispersion computation")

        # DIM_<PREFIX> -> PREFIX
        long_df["dimension_prefix"] = (
            long_df["dimension_key"].astype(str).str.replace("DIM_", "", regex=False)
        )
        long_df["dimension_label"] = long_df["dimension_prefix"].map(prefix_label)

        # Optionally limit segments to most frequent values
        max_segments = int(config.get("max_segments", 6))
        for field in [segment_field, facet_field]:
            if field:
                counts = (
                    long_df[[field]]
                    .dropna()
                    .value_counts()
                    .reset_index(name="n")
                    .sort_values("n", ascending=False)
                )
                keep = counts[field].head(max_segments).tolist()
                long_df = long_df[long_df[field].isin(keep)]

        if long_df.empty:
            raise ValueError("No usable data after segment/facet limiting")

        group_fields = ["dimension_label"] + id_vars

        agg = (
            long_df.groupby(group_fields)["score"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(columns={"mean": "mean_score", "std": "std_score", "count": "n"})
        )

        # Use Standard Deviation for error bars instead of CI
        agg["lower"] = agg["mean_score"] - agg["std_score"]
        agg["upper"] = agg["mean_score"] + agg["std_score"]

        # Clamp to Likert domain for display
        likert_domain = config.get("likert_domain", [1, 5])
        try:
            lo, hi = float(likert_domain[0]), float(likert_domain[1])
        except Exception:
            lo, hi = 1.0, 5.0
        agg["lower"] = agg["lower"].clip(lower=lo, upper=hi)
        agg["upper"] = agg["upper"].clip(lower=lo, upper=hi)

        min_n = int(config.get("min_n", 30))
        agg["low_n"] = agg["n"] < min_n

        # For sorting: overall mean per dimension (regardless of segment/facet)
        overall = (
            long_df.groupby("dimension_label")["score"].mean().reset_index(name="overall_mean")
        )
        agg = agg.merge(overall, on="dimension_label", how="left")

        apply_theme()

        x = alt.X(
            "mean_score:Q",
            title="Score moyen (1-5)",
            scale=alt.Scale(domain=[lo, hi]),
        )
        y = alt.Y(
            "dimension_label:N",
            title="Dimension QVCT",
            sort=alt.SortField(field="overall_mean", order="descending"),
            axis=alt.Axis(labelLimit=260, labelPadding=8),
        )

        tooltip = [
            alt.Tooltip("dimension_label:N", title="Dimension"),
            alt.Tooltip("mean_score:Q", title="Moyenne", format=".2f"),
            alt.Tooltip("std_score:Q", title="Écart-type", format=".2f"),
            alt.Tooltip("lower:Q", title="Moyenne - 1 SD", format=".2f"),
            alt.Tooltip("upper:Q", title="Moyenne + 1 SD", format=".2f"),
            alt.Tooltip("n:Q", title="Répondants"),
        ]

        if segment_field:
            tooltip.insert(1, alt.Tooltip(f"{segment_field}:N", title=segment_field))
        if facet_field:
            tooltip.insert(1, alt.Tooltip(f"{facet_field}:N", title=facet_field))

        base = alt.Chart(agg)

        # Use Grouped Bars when segment_field is present
        if segment_field:
            # Grouped bar chart: offset Y by segment
            bars = base.mark_bar().encode(
                y=y,
                yOffset=alt.YOffset(f"{segment_field}:N"),
                x=x,
                color=alt.Color(f"{segment_field}:N", title=segment_field),
                tooltip=tooltip,
            )
            
            eb = base.mark_errorbar().encode(
                y=y,
                yOffset=alt.YOffset(f"{segment_field}:N"),
                x=alt.X("lower:Q", scale=alt.Scale(domain=[lo, hi])),
                x2="upper:Q",
                color=alt.value("black"), # Better contrast on colored bars
                tooltip=tooltip,
            )
            
            chart = (bars + eb).properties(height={"step": 30}) # Reduced step for thinner bars
        else:
            bars = base.mark_bar().encode(
                y=y,
                x=x,
                x2=alt.datum(lo),
                color=alt.Color(
                    "mean_score:Q",
                    scale=alt.Scale(scheme="blues"),
                    legend=None,
                ),
                tooltip=tooltip,
            )
            eb = base.mark_errorbar().encode(
                y=y,
                x=alt.X("lower:Q", scale=alt.Scale(domain=[lo, hi])),
                x2="upper:Q",
                tooltip=tooltip,
            )
            chart = alt.layer(bars, eb).properties(height={"step": 22})

        if facet_field:
            chart = chart.facet(
                column=alt.Column(f"{facet_field}:N", title=facet_field)
            ).properties(
                title=f"Scores par dimension (moyenne et écart-type) par {facet_field}"
            )
        else:
            chart = chart.properties(
                title="Scores par dimension (moyenne et écart-type)",
                padding={"left": 120, "right": 40},
                width="container",
            )

        return chart.interactive().to_dict()
