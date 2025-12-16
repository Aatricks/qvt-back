from __future__ import annotations

from typing import Any, Dict, Optional

import altair as alt
import numpy as np
import pandas as pd
from scipy import stats

from src.services.qvt_metrics import compute_prefix_scores, prefix_label
from src.services.survey_utils import add_age_band
from src.viz.base import IVisualizationStrategy
from src.viz.theme import apply_theme


class DimensionCIBarsStrategy(IVisualizationStrategy):
    """Mean score per QVT dimension with confidence intervals.

    This chart is meant as a "statistically responsible" companion to `dimension_summary`.

    Key idea: compute a per-respondent score for each dimension (mean across the dimension's
    items), then aggregate across respondents to get mean and 95% CI.

    Config:
      - segment_field (str, optional): demographic field to compare segments (e.g. "Sexe").
      - min_n (int, optional): minimum respondents required per (dimension, segment) (default 30).
      - max_segments (int, optional): maximum number of segment categories to keep (default 6).
      - alpha (float, optional): CI alpha (default 0.05 for 95% CI).
      - likert_domain (list|tuple, optional): bounds for x-axis (default [1, 5]).

    Filters:
      - applied as equality filters on existing columns.
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
            raise ValueError("Survey data required for dimension CI bars")

        df = add_age_band(survey_df.copy())

        # Apply simple equality filters
        for key, value in (filters or {}).items():
            if key in df.columns:
                df = df[df[key] == value]

        if df.empty:
            raise ValueError("Empty dataset after filtering")

        segment_field: Optional[str] = config.get("segment_field")
        if segment_field and segment_field not in df.columns:
            raise ValueError(f"Segment field '{segment_field}' not found in dataset")

        # Compute respondent-level dimension scores (DIM_<PREFIX> columns)
        dim_scores = compute_prefix_scores(df)

        # Build a long table
        long_df = dim_scores.reset_index(drop=True)
        if segment_field:
            long_df[segment_field] = df[segment_field].reset_index(drop=True)

        long_df = long_df.melt(
            id_vars=[segment_field] if segment_field else None,
            var_name="dimension_key",
            value_name="score",
        )

        long_df["score"] = pd.to_numeric(long_df["score"], errors="coerce")
        long_df = long_df.dropna(subset=["score"])

        if long_df.empty:
            raise ValueError("No usable Likert data for CI computation")

        # DIM_<PREFIX> -> PREFIX
        long_df["dimension_prefix"] = (
            long_df["dimension_key"].astype(str).str.replace("DIM_", "", regex=False)
        )
        long_df["dimension_label"] = long_df["dimension_prefix"].map(prefix_label)

        # Optionally limit segments to most frequent values
        max_segments = int(config.get("max_segments", 6))
        if segment_field:
            counts = (
                long_df[[segment_field]]
                .dropna()
                .value_counts()
                .reset_index(name="n")
                .sort_values("n", ascending=False)
            )
            keep = counts[segment_field].head(max_segments).tolist()
            long_df = long_df[long_df[segment_field].isin(keep)]

        if long_df.empty:
            raise ValueError("No usable data after segment limiting")

        group_fields = ["dimension_label"] + ([segment_field] if segment_field else [])

        agg = (
            long_df.groupby(group_fields)["score"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(columns={"mean": "mean_score", "std": "std_score", "count": "n"})
        )

        # CI computation (t-based; falls back gracefully for n < 2)
        alpha = float(config.get("alpha", 0.05))
        agg["se"] = agg["std_score"] / np.sqrt(agg["n"].clip(lower=1))

        def _tcrit(n: int) -> float:
            if n is None or n < 2:
                return float("nan")
            return float(stats.t.ppf(1 - alpha / 2.0, df=n - 1))

        agg["t_crit"] = agg["n"].apply(_tcrit)
        agg["ci"] = agg["t_crit"] * agg["se"]

        agg["lower"] = agg["mean_score"] - agg["ci"]
        agg["upper"] = agg["mean_score"] + agg["ci"]

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

        # For sorting: overall mean per dimension (regardless of segment)
        overall = (
            long_df.groupby("dimension_label")["score"].mean().reset_index(name="overall_mean")
        )
        agg = agg.merge(overall, on="dimension_label", how="left")

        apply_theme()

        x = alt.X(
            "mean_score:Q",
            title="Score moyen (1-5)",
            scale=alt.Scale(domain=[0, hi]),
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
            alt.Tooltip("lower:Q", title="Borne basse", format=".2f"),
            alt.Tooltip("upper:Q", title="Borne haute", format=".2f"),
            alt.Tooltip("n:Q", title="Répondants"),
        ]

        if segment_field:
            tooltip.insert(1, alt.Tooltip(f"{segment_field}:N", title=segment_field))

        base = alt.Chart(agg)

        # Error bars (CI)
        if segment_field:
            eb = base.mark_errorbar().encode(
                y=y,
                x=alt.X("lower:Q", scale=alt.Scale(domain=[0, hi])),
                x2="upper:Q",
                color=alt.Color(f"{segment_field}:N", title=segment_field),
                tooltip=tooltip,
            )
            pts = base.mark_point(filled=True, size=70).encode(
                y=y,
                x=x,
                color=alt.Color(f"{segment_field}:N", title=segment_field),
                shape=alt.Shape("low_n:N", title=f"n < {min_n}", legend=alt.Legend(orient="bottom")),
                tooltip=tooltip,
            )
            chart = (eb + pts).properties(height={"step": 22})
        else:
            bars = base.mark_bar().encode(
                y=y,
                x=x,
                x2=alt.value(lo),
                color=alt.Color(
                    "mean_score:Q",
                    scale=alt.Scale(scheme="blues"),
                    legend=None,
                ),
                tooltip=tooltip,
            )
            eb = base.mark_errorbar().encode(
                y=y,
                x=alt.X("lower:Q", scale=alt.Scale(domain=[0, hi])),
                x2="upper:Q",
                tooltip=tooltip,
            )
            chart = alt.layer(bars, eb).properties(height={"step": 22})

        chart = chart.properties(
            title="Scores par dimension (moyenne et intervalle de confiance)",
            padding={"left": 120},
        )

        return chart.to_dict()
