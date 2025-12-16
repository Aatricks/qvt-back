from typing import Any, Dict, List

import altair as alt
import pandas as pd
from scipy.stats import f_oneway

from src.services.survey_utils import add_age_band, available_demographics, detect_likert_columns, to_likert_long
from src.viz.base import IVisualizationStrategy
from src.viz.theme import apply_theme


class AnovaSignificanceStrategy(IVisualizationStrategy):
    """Finds socio-demographic splits with significant mean differences (ANOVA)."""

    def generate(
        self, data: Dict[str, pd.DataFrame], config: Dict[str, Any], filters: Dict[str, Any], settings: Any
    ) -> Dict[str, Any]:
        survey_df = data.get("survey")
        if survey_df is None:
            raise ValueError("Survey data required for ANOVA")

        df = add_age_band(survey_df.copy())
        for key, value in (filters or {}).items():
            if key in df.columns:
                df = df[df[key] == value]

        likert_cols = detect_likert_columns(df)
        if not likert_cols:
            raise ValueError("No Likert columns detected")

        demographics = available_demographics(df)
        if "AgeClasse" in df.columns:
            demographics.append("AgeClasse")

        long_df = to_likert_long(df, likert_cols)
        long_df["response_value"] = pd.to_numeric(long_df["response_value"], errors="coerce")
        long_df = long_df.dropna(subset=["response_value"])

        significant_combos: List[Dict[str, Any]] = []
        for question, q_df in long_df.groupby("question_label"):
            for demo in demographics:
                if demo not in q_df.columns:
                    continue
                split = q_df.dropna(subset=[demo])
                groups = [group["response_value"].values for _, group in split.groupby(demo, observed=False)]
                groups = [g for g in groups if len(g) >= 2]
                if len(groups) < 2:
                    continue
                f_stat, p_value = f_oneway(*groups)
                if pd.isna(f_stat) or pd.isna(p_value):
                    continue
                significant_combos.append(
                    {
                        "question_label": question,
                        "group_variable": demo,
                        "p_value": p_value,
                        "f_stat": f_stat,
                    }
                )

        if not significant_combos:
            raise ValueError("No significant differences detected")

        top_n = config.get("top_n", 5)
        top = sorted(significant_combos, key=lambda r: r["p_value"])[:top_n]
        plot_rows: List[Dict[str, Any]] = []
        for combo in top:
            subset = long_df[long_df["question_label"] == combo["question_label"]].dropna(
                subset=[combo["group_variable"]]
            )
            for group_value, group_df in subset.groupby(combo["group_variable"], observed=False):
                plot_rows.append(
                    {
                        "question_label": combo["question_label"],
                        "group_variable": combo["group_variable"],
                        "group_value": group_value,
                        "mean_response": group_df["response_value"].mean(),
                        "p_value": combo["p_value"],
                        "f_stat": combo["f_stat"],
                    }
                )

        chart_df = pd.DataFrame(plot_rows)
        apply_theme()
        chart = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X("group_value:N", title="Groupe"),
                y=alt.Y("mean_response:Q", title="Moyenne de réponse"),
                color=alt.Color("group_variable:N", title="Variable"),
                column=alt.Column("question_label:N", title="Question", spacing=8),
                tooltip=[
                    "question_label",
                    "group_variable",
                    "group_value",
                    alt.Tooltip("mean_response:Q", format=".2f"),
                    alt.Tooltip("p_value:Q", format=".3f"),
                    alt.Tooltip("f_stat:Q", format=".2f"),
                ],
            )
        )
        return chart.to_dict()
