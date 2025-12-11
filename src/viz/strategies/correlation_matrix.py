from typing import Any, Dict, List

import altair as alt
import pandas as pd

from src.viz.base import IVisualizationStrategy
from src.viz.theme import apply_theme


class CorrelationMatrixStrategy(IVisualizationStrategy):
    """Heatmap correlation matrix for numeric HR/QVT metrics."""

    def generate(
        self, data: Dict[str, pd.DataFrame], config: Dict[str, Any], filters: Dict[str, Any], settings: Any
    ) -> Dict[str, Any]:
        hr_df = data["hr"].copy()
        for key, value in (filters or {}).items():
            if key in hr_df.columns:
                hr_df = hr_df[hr_df[key] == value]

        numeric_cols: List[str] = config.get("numeric_fields")
        if not numeric_cols:
            numeric_cols = hr_df.select_dtypes(include="number").columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric columns available for correlation matrix")
        numeric = hr_df[numeric_cols].apply(pd.to_numeric, errors="coerce").dropna()
        corr = numeric.corr()
        corr_reset = corr.stack().reset_index()
        corr_reset.columns = ["metric_x", "metric_y", "correlation"]

        apply_theme()
        chart = (
            alt.Chart(corr_reset)
            .mark_rect()
            .encode(
                x="metric_x:N",
                y="metric_y:N",
                color=alt.Color("correlation:Q", scale=alt.Scale(scheme="blueorange")),
                tooltip=["metric_x", "metric_y", alt.Tooltip("correlation:Q", format=".2f")],
            )
        )
        return chart.to_dict()
