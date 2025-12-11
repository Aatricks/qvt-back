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
        metric = config.get("measure_field", "absenteeism_rate")
        time_field = config.get("time_field", "year")

        df = hr_df.copy()
        for key, value in (filters or {}).items():
            if key in df.columns:
                df = df[df[key] == value]

        apply_theme()
        chart = (
            alt.Chart(df)
            .mark_line(point=True)
            .encode(
                x=alt.X(f"{time_field}:O", title="Period"),
                y=alt.Y(f"{metric}:Q", title=metric.replace("_", " ").title()),
                tooltip=[time_field, metric],
            )
        )
        return chart.to_dict()
