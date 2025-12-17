from typing import Any, Dict

import altair as alt
import pandas as pd

from src.services.survey_utils import (
    LIKERT_PREFIX_LABELS,
    add_age_band,
    detect_likert_columns,
    to_likert_long,
)
from src.viz.base import IVisualizationStrategy
from src.viz.theme import apply_theme


class PracticeDimensionSummaryStrategy(IVisualizationStrategy):
    """Average score per QVT practice prefix, optionally segmented by a demographic field."""

    def generate(
        self, data: Dict[str, pd.DataFrame], config: Dict[str, Any], filters: Dict[str, Any], settings: Any
    ) -> Dict[str, Any]:
        survey_df = data.get("survey")
        if survey_df is None:
            raise ValueError("Survey data required for practice summary")

        df = add_age_band(survey_df.copy())
        for key, value in (filters or {}).items():
            if key in df.columns:
                df = df[df[key] == value]

        likert_cols = detect_likert_columns(df)
        if not likert_cols:
            raise ValueError("No Likert columns detected for practice summary")

        long_df = to_likert_long(df, likert_cols)
        long_df["response_value"] = pd.to_numeric(long_df["response_value"], errors="coerce")
        long_df = long_df.dropna(subset=["response_value"])
        long_df["dimension_label"] = (
            long_df["dimension_prefix"].map(LIKERT_PREFIX_LABELS).fillna(long_df["dimension_prefix"])
        )

        segment_field = config.get("segment_field")
        if segment_field and segment_field not in long_df.columns:
            raise ValueError(f"Segment field '{segment_field}' not found in dataset")

        group_fields = ["dimension_label"]
        if segment_field:
            group_fields.append(segment_field)

        agg = (
            long_df.groupby(group_fields)["response_value"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "mean_score", "count": "responses"})
        )

        apply_theme()
        x_encoding = alt.X(
            "mean_score:Q",
            title="Score moyen (1-5)",
            scale=alt.Scale(domain=[0, 5]),
        )
        y_encoding = alt.Y(
            "dimension_label:N",
            sort="-x",
            title="Dimension QVT",
            axis=alt.Axis(labelLimit=260, labelPadding=8),
        )

        if segment_field:
            chart = (
                alt.Chart(agg)
                .mark_bar()
                .encode(
                    y=y_encoding,
                    x=x_encoding,
                    x2=alt.value(1),
                    color=alt.Color(f"{segment_field}:N", title=segment_field),
                    tooltip=[
                        "dimension_label",
                        segment_field,
                        alt.Tooltip("mean_score:Q", format=".2f"),
                        "responses",
                    ],
                )
            ).properties(height={"step": 22}, padding={"left": 120}).interactive()
        else:
            chart = (
                alt.Chart(agg)
                .mark_bar()
                .encode(
                    y=y_encoding,
                    x=x_encoding,
                    x2=alt.value(1),
                    color=alt.Color("mean_score:Q", scale=alt.Scale(scheme="blues"), legend=None),
                    tooltip=["dimension_label", alt.Tooltip("mean_score:Q", format=".2f"), "responses"],
                )
            ).properties(height={"step": 22}, padding={"left": 120}).interactive()
        return chart.to_dict()
