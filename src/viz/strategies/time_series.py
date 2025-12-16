from typing import Any, Dict

import altair as alt
import pandas as pd

from src.viz.base import IVisualizationStrategy
from src.viz.theme import apply_theme


class TimeSeriesStrategy(IVisualizationStrategy):
    """Line chart of HR indicator over time.

    Required columns: year (or period), numeric metric (default: absenteeism_rate).
    """

    def generate(
        self, data: Dict[str, pd.DataFrame], config: Dict[str, Any], filters: Dict[str, Any], settings: Any
    ) -> Dict[str, Any]:
        hr_df = data["hr"]
        metric = config.get("measure_field")
        if not metric or metric not in hr_df.columns:
            numeric_cols = hr_df.select_dtypes(include="number").columns.tolist()
            if not numeric_cols:
                raise ValueError("No numeric metric available for time series")
            metric = numeric_cols[0]
        time_field = config.get("time_field")

        # Heuristic: prefer an explicit time column if present.
        if not time_field:
            preferred = [
                "year",
                "annee",
                "année",
                "date",
                "period",
                "periode",
                "période",
                "month",
                "mois",
                "id",
            ]
            lower_to_actual = {str(c).strip().lower(): c for c in hr_df.columns}
            for key in preferred:
                if key in lower_to_actual:
                    time_field = lower_to_actual[key]
                    break

        if not time_field or time_field not in hr_df.columns:
            time_field = hr_df.columns[0]

        df = hr_df.copy()
        for key, value in (filters or {}).items():
            if key in df.columns:
                df = df[df[key] == value]

        apply_theme()
        chart = (
            alt.Chart(df)
            .mark_line(point=True)
            .encode(
                 x=alt.X(f"{time_field}:O", title="Période"),
                 y=alt.Y(f"{metric}:Q", title=metric.replace("_", " ").title()),
                 tooltip=[time_field, metric],
            )
        )
        return chart.to_dict()
