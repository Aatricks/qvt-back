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

        top_n = int(config.get("top_n", 5))
        columns = int(config.get("columns", 3))
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

        # Build small multiples without a single long row of facets.
        # We concatenate per-question charts and wrap them into a grid.
        charts: List[alt.Chart] = []
        for q in chart_df["question_label"].dropna().unique().tolist():
            sub = chart_df[chart_df["question_label"] == q].copy()
            if sub.empty:
                continue
            gv = str(sub["group_variable"].iloc[0]) if "group_variable" in sub.columns else "Groupe"
            pv = float(sub["p_value"].iloc[0]) if "p_value" in sub.columns and not sub["p_value"].empty else float("nan")
            title = f"{q} — {gv} (p={pv:.3g})" if pv == pv else f"{q} — {gv}"

            c = (
                alt.Chart(sub, title=title)
                .mark_bar()
                .encode(
                    x=alt.X("group_value:N", title="Groupe", axis=alt.Axis(labelAngle=0, labelLimit=80)),
                    y=alt.Y("mean_response:Q", title="Moyenne", scale=alt.Scale(domain=[0, 5])),
                    color=alt.Color("group_value:N", title="Groupe", legend=None),
                    tooltip=[
                        alt.Tooltip("question_label:N", title="Question"),
                        alt.Tooltip("group_variable:N", title="Variable"),
                        alt.Tooltip("group_value:N", title="Groupe"),
                        alt.Tooltip("mean_response:Q", title="Moyenne", format=".2f"),
                        alt.Tooltip("p_value:Q", title="p-value", format=".3f"),
                        alt.Tooltip("f_stat:Q", title="F", format=".2f"),
                    ],
                )
                .properties(width=200, height=150) # Set explicit width/height
            )
            charts.append(c)

        if not charts:
            raise ValueError("No significant differences detected")

        # Interactive concatenation
        return alt.concat(*charts, columns=columns).resolve_scale(color='independent').to_dict()