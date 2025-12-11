from typing import Any, Dict

import altair as alt
import pandas as pd

from src.viz.base import IVisualizationStrategy
from src.viz.theme import apply_theme


class ExampleNewChartStrategy(IVisualizationStrategy):
    """Example extensibility strategy for contributors.

    Required columns: year, absenteeism_rate.
    Config options: measure_field (default absenteeism_rate), time_field (default year).
    """

    def generate(
        self, data: Dict[str, pd.DataFrame], config: Dict[str, Any], filters: Dict[str, Any], settings: Any
    ) -> Dict[str, Any]:
        hr_df = data["hr"].copy()
        metric = config.get("measure_field", "absenteeism_rate")
        time_field = config.get("time_field", "year")

        for key, value in (filters or {}).items():
            if key in hr_df.columns:
                hr_df = hr_df[hr_df[key] == value]

        apply_theme()
        chart = (
            alt.Chart(hr_df)
            .mark_bar()
            .encode(
                x=alt.X(f"{time_field}:O", title="Period"),
                y=alt.Y(f"{metric}:Q", title=metric.replace("_", " ").title()),
                tooltip=[time_field, metric],
            )
        )
        return chart.to_dict()
