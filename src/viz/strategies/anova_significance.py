from typing import Any, Dict, List

import altair as alt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import f_oneway

from src.services.survey_utils import (
    DEMO_VALUE_MAPPING,
    add_age_band,
    add_seniority_band,
    available_demographics,
    detect_likert_columns,
    to_likert_long,
)
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
        df = add_seniority_band(df)

        # Apply value mappings for demographics (1 -> Homme, etc.) early
        for col, mapping in DEMO_VALUE_MAPPING.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(df[col])

        # Apply simple equality filters
        for key, value in (filters or {}).items():
            if key in df.columns:
                df = df[df[key] == value]

        likert_cols = detect_likert_columns(df)
        if not likert_cols:
            raise ValueError("No Likert columns detected")

        # Exclude raw numeric fields, focus on categories/bands
        exclude = {"ID", "Age", "Ancienne", "Ancienneté"}
        demographics = [d for d in available_demographics(df) if d not in exclude]

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
                
                # Eta-squared effect size: SS_between / SS_total
                all_data = np.concatenate(groups)
                grand_mean = np.mean(all_data)
                ss_total = np.sum((all_data - grand_mean)**2)
                ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
                eta_sq = ss_between / ss_total if ss_total > 0 else 0

                significant_combos.append(
                    {
                        "question_label": question,
                        "group_variable": demo,
                        "p_value": p_value,
                        "f_stat": f_stat,
                        "eta_squared": eta_sq
                    }
                )

        if not significant_combos:
            raise ValueError("No significant differences detected")

        # Limit to top-N most significant results
        top_n = int(config.get("top_n", 6))
        columns = int(config.get("columns", 2))
        top = sorted(significant_combos, key=lambda r: r["p_value"])[:top_n]
        
        plot_rows: List[Dict[str, Any]] = []
        alpha = 0.05
        for combo in top:
            subset = long_df[long_df["question_label"] == combo["question_label"]].dropna(
                subset=[combo["group_variable"]]
            )
            for group_value, group_df in subset.groupby(combo["group_variable"], observed=False):
                vals = group_df["response_value"]
                n = len(vals)
                mean = vals.mean()
                std = vals.std()
                # 95% Confidence Interval using t-distribution
                ci = 0
                if n > 1:
                    ci = stats.t.ppf(1 - alpha/2, n-1) * (std / np.sqrt(n))

                plot_rows.append(
                    {
                        "question_label": combo["question_label"],
                        "group_variable": combo["group_variable"],
                        "group_value": group_value,
                        "mean": mean,
                        "lower": max(1, mean - ci),
                        "upper": min(5, mean + ci),
                        "n": n,
                        "p_value": combo["p_value"],
                        "f_stat": combo["f_stat"],
                        "eta_sq": combo["eta_squared"]
                    }
                )

        chart_df = pd.DataFrame(plot_rows)
        apply_theme()

        charts: List[alt.Chart] = []
        for q in chart_df["question_label"].unique():
            sub = chart_df[chart_df["question_label"] == q]
            gv = sub["group_variable"].iloc[0]
            pv = sub["p_value"].iloc[0]
            title = f"{q} (split: {gv}, p={pv:.3g})"

            base = alt.Chart(sub, title=title).encode(
                x=alt.X("group_value:N", title=None, axis=alt.Axis(labelAngle=-45, labelLimit=100))
            )

            bars = base.mark_bar(opacity=0.7).encode(
                y=alt.Y("mean:Q", title="Moyenne (1-5)", scale=alt.Scale(domain=[1, 5])),
                y2=alt.datum(1), # Anchor bars to the Likert scale minimum (1)
                color=alt.Color("group_value:N", legend=None),
                tooltip=[
                    alt.Tooltip("group_value:N", title="Groupe"),
                    alt.Tooltip("mean:Q", title="Moyenne", format=".2f"),
                    alt.Tooltip("lower:Q", title="CI Bas", format=".2f"),
                    alt.Tooltip("upper:Q", title="CI Haut", format=".2f"),
                    alt.Tooltip("n:Q", title="N"),
                    alt.Tooltip("p_value:Q", title="ANOVA p", format=".3f"),
                    alt.Tooltip("eta_sq:Q", title="Effet (η²)", format=".2f"),
                ]
            )

            error = base.mark_errorbar().encode(
                y=alt.Y("lower:Q", title=""),
                y2="upper:Q"
            )

            charts.append((bars + error).properties(width=250, height=180))

        if not charts:
            raise ValueError("No visualizable significant differences")

        return alt.concat(*charts, columns=columns).resolve_scale(color='independent').to_dict()
