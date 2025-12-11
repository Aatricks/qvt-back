from typing import Any, Dict

import altair as alt
import pandas as pd

from src.viz.base import IVisualizationStrategy
from src.viz.theme import apply_theme


class LikertDistributionStrategy(IVisualizationStrategy):
    """Diverging bar distribution of Likert responses.

    Required columns: question_label, response_value (1-5). Optional: filters on metadata.
    """

    def generate(
        self, data: Dict[str, pd.DataFrame], config: Dict[str, Any], filters: Dict[str, Any], settings: Any
    ) -> Dict[str, Any]:
        survey_df = data.get("survey")
        if survey_df is None:
            raise ValueError("Survey data required for likert distribution")

        df = survey_df.copy()
        for key, value in (filters or {}).items():
            if key in df.columns:
                df = df[df[key] == value]

        apply_theme()
        df["response_value"] = pd.to_numeric(df["response_value"], errors="coerce")
        df = df.dropna(subset=["response_value"])
        df["response_value"] = df["response_value"].astype(int)

        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("count():Q", stack="normalize", title="Share"),
                y=alt.Y("question_label:N", title="Question"),
                color=alt.Color("response_value:O", legend=alt.Legend(title="Response 1-5")),
                tooltip=["question_label", "response_value", "count()"],
            )
        )
        return chart.to_dict()
