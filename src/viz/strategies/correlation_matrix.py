from typing import Any, Dict, List

import altair as alt
import pandas as pd

from src.viz.base import IVisualizationStrategy
from src.services.survey_utils import add_age_band, detect_likert_columns
from src.viz.theme import apply_theme
from src.config.observability import log_event


class CorrelationMatrixStrategy(IVisualizationStrategy):
    """Heatmap correlation matrix for numeric HR/QVT metrics."""

    def generate(
        self, data: Dict[str, pd.DataFrame], config: Dict[str, Any], filters: Dict[str, Any], settings: Any
    ) -> Dict[str, Any]:
        hr_df = data["hr"].copy()
        for key, value in (filters or {}).items():
            if key in hr_df.columns:
                hr_df = hr_df[hr_df[key] == value]

        # If the user provided numeric_fields in config, keep only those present in the dataset.
        # If none remain, fall back to auto-detected numeric columns.
        requested: List[str] = config.get("numeric_fields") or []
        if requested:
            numeric_cols = [c for c in requested if c in hr_df.columns]
            missing = [c for c in requested if c not in hr_df.columns]
            if missing:
                # Log which requested fields were ignored so we can troubleshoot datasets
                try:
                    log_event("correlation_matrix.ignored_fields", requested=requested, missing=missing)
                except Exception:
                    pass
        else:
            numeric_cols = hr_df.select_dtypes(include="number").columns.tolist()

        if not numeric_cols:
            raise ValueError("No numeric columns available for correlation matrix after filtering requested fields")

        numeric = hr_df[numeric_cols].apply(pd.to_numeric, errors="coerce").dropna()
        corr = numeric.corr()
        corr_reset = corr.stack().reset_index()
        corr_reset.columns = ["metric_x", "metric_y", "correlation"]

        apply_theme()
        
        base = alt.Chart(corr_reset).encode(
            x="metric_x:N",
            y="metric_y:N"
        )
        
        heatmap = base.mark_rect().encode(
            color=alt.Color("correlation:Q", scale=alt.Scale(scheme="blueorange")),
            tooltip=["metric_x", "metric_y", alt.Tooltip("correlation:Q", format=".2f")]
        )
        
        text = base.mark_text().encode(
            text=alt.Text("correlation:Q", format=".2f"),
            color=alt.condition(
                alt.datum.correlation > 0.5,
                alt.value("white"),
                alt.value("black")
            )
        )
        
        return (heatmap + text).properties(
            width=alt.Step(40),
            height=alt.Step(40)
        ).interactive().to_dict()