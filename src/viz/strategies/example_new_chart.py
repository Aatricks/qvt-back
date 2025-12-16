from __future__ import annotations

from typing import Any, Dict, List

import altair as alt
import pandas as pd

from src.services.survey_utils import LIKERT_PREFIX_LABELS, add_age_band, detect_likert_columns, to_likert_long
from src.viz.base import IVisualizationStrategy
from src.viz.theme import apply_theme


class ExampleNewChartStrategy(IVisualizationStrategy):
    """High-level QVT overview: mean score per dimension + qualitative status.

    This is intentionally opinionated for QVT/QVCT use:
    - Converts the wide POV export into a long Likert table.
    - Aggregates by dimension prefix (COM, RECO, ENG, EPUI, ...).
    - Classifies each dimension into a simple decision-aid status.

    Config (optional):
      - warn_threshold (float): below => "Alerte" (default 2.5)
      - good_threshold (float): above => "Point fort" (default 3.5)
      - min_responses (int): minimum responses required per dimension (default 10)
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
            raise ValueError("Survey data required for example_new_chart")

        df = add_age_band(survey_df.copy())
        for key, value in (filters or {}).items():
            if key in df.columns:
                df = df[df[key] == value]

        likert_cols = detect_likert_columns(df)
        if not likert_cols:
            # Long-format survey could be supported later; for now this chart targets the POV export.
            raise ValueError("No Likert columns detected (expected wide POV export with PGC/COM/ENG/EPUI...)" )

        long_df = to_likert_long(df, likert_cols)
        long_df["response_value"] = pd.to_numeric(long_df["response_value"], errors="coerce")
        long_df = long_df.dropna(subset=["response_value"])

        if long_df.empty:
            raise ValueError("No usable Likert responses after cleaning")

        long_df["dimension_label"] = (
            long_df["dimension_prefix"].map(LIKERT_PREFIX_LABELS).fillna(long_df["dimension_prefix"])
        )

        agg = (
            long_df.groupby(["dimension_prefix", "dimension_label"])["response_value"]
            .agg(["mean", "count", "std"])
            .reset_index()
            .rename(columns={"mean": "mean_score", "count": "responses", "std": "std_dev"})
        )

        min_responses = int(config.get("min_responses", 10))
        agg = agg[agg["responses"] >= min_responses]
        if agg.empty:
            raise ValueError("Not enough responses per dimension to build a stable overview")

        warn_threshold = float(config.get("warn_threshold", 2.5))
        good_threshold = float(config.get("good_threshold", 3.5))

        def status(mean_score: float) -> str:
            if mean_score < warn_threshold:
                return "Alerte"
            if mean_score >= good_threshold:
                return "Point fort"
            return "Vigilance"

        agg["status"] = agg["mean_score"].apply(status)

        # Keep dimensions in a useful order: worst first, so decision-makers see problems immediately.
        agg = agg.sort_values("mean_score", ascending=True)

        apply_theme()

        chart = (
            alt.Chart(agg)
            .mark_bar()
            .encode(
                y=alt.Y(
                    "dimension_label:N",
                    sort=None,
                    title="Dimension QVT",
                    axis=alt.Axis(labelLimit=260, labelPadding=8),
                ),
                x=alt.X("mean_score:Q", title="Score moyen (1-5)", scale=alt.Scale(domain=[0, 5])),
                x2=alt.value(1),
                color=alt.Color(
                    "status:N",
                    title="Statut",
                    scale=alt.Scale(domain=["Alerte", "Vigilance", "Point fort"], range=["#ef4444", "#f59e0b", "#10b981"]),
                ),
                tooltip=[
                    "dimension_label:N",
                    "status:N",
                    alt.Tooltip("mean_score:Q", title="Moyenne", format=".2f"),
                    alt.Tooltip("std_dev:Q", title="Écart-type", format=".2f"),
                    alt.Tooltip("responses:Q", title="Réponses"),
                ],
            )
        )

        # Add reference lines for the thresholds to make the decision logic explicit.
        rules: List[alt.Chart] = [
            alt.Chart(pd.DataFrame({"x": [warn_threshold]}))
            .mark_rule(color="#ef4444", strokeDash=[4, 4])
            .encode(x="x:Q"),
            alt.Chart(pd.DataFrame({"x": [good_threshold]}))
            .mark_rule(color="#10b981", strokeDash=[4, 4])
            .encode(x="x:Q"),
        ]

        return alt.layer(chart, *rules).properties(height={"step": 22}, padding={"left": 120}).to_dict()
