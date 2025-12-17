from typing import Any, Dict, List

import altair as alt
import pandas as pd

from src.services.survey_utils import classify_distribution, detect_likert_columns, to_likert_long
from src.viz.base import IVisualizationStrategy
from src.viz.theme import apply_theme


class DistributionAnomaliesStrategy(IVisualizationStrategy):
    """Detects skewed or uniform Likert distributions."""

    def generate(
        self, data: Dict[str, pd.DataFrame], config: Dict[str, Any], filters: Dict[str, Any], settings: Any
    ) -> Dict[str, Any]:
        survey_df = data.get("survey")
        if survey_df is None:
            raise ValueError("Survey data required for distribution anomalies")

        df = survey_df.copy()
        for key, value in (filters or {}).items():
            if key in df.columns:
                df = df[df[key] == value]

        likert_cols = detect_likert_columns(df)
        if not likert_cols:
            raise ValueError("No Likert columns detected")
        long_df = to_likert_long(df, likert_cols)
        long_df["response_value"] = pd.to_numeric(long_df["response_value"], errors="coerce")
        long_df = long_df.dropna(subset=["response_value"])

        records: List[Dict[str, Any]] = []
        for question, group in long_df.groupby("question_label"):
            classification = classify_distribution(group["response_value"])
            if classification == "insufficient_data":
                continue
            records.append(
                {
                    "question_label": question,
                    "skewness": group["response_value"].skew(),
                    "mean": group["response_value"].mean(),
                    "std": group["response_value"].std(),
                    "classification": classification,
                }
            )

        if not records:
            raise ValueError("No analyzable distributions found")

        chart_df = pd.DataFrame(records)
        apply_theme()
        chart = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                y=alt.Y("question_label:N", sort="-x", title="Question"),
                x=alt.X("skewness:Q", title="Asymétrie (skew)"),
                color=alt.Color("classification:N", title="Profil"),
                tooltip=[
                    "question_label",
                    alt.Tooltip("skewness:Q", format=".2f"),
                    alt.Tooltip("mean:Q", format=".2f"),
                    alt.Tooltip("std:Q", format=".2f"),
                    "classification",
                ],
            )
            .properties(width="container")
        )
        return chart.to_dict()
