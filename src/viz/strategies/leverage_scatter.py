from __future__ import annotations

from typing import Any, Dict, List, Literal

import altair as alt
import pandas as pd

from src.services.qvt_metrics import compute_prefix_scores, prefix_label
from src.services.survey_utils import add_age_band
from src.viz.base import IVisualizationStrategy
from src.viz.theme import apply_theme


class LeverageScatterStrategy(IVisualizationStrategy):
    """Scatter of dimensions: mean score vs leverage on an outcome (EPUI or ENG).

    This complements `action_priority_index` by making trade-offs visible:
      - X: mean score (1–5)
      - Y: leverage (strength of association with outcome)
      - Color/size: priority index proxy

    Config:
      - outcome ("EPUI"|"ENG")        : default EPUI
      - method ("pearson"|"spearman") : default spearman
      - min_n (int)                    : default 30

    Filters: equality on available columns.
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
            raise ValueError("Survey data required for leverage scatter")

        df = add_age_band(survey_df.copy())
        for key, value in (filters or {}).items():
            if key in df.columns:
                df = df[df[key] == value]

        if df.empty:
            raise ValueError("Dataset vide après filtrage pour le scatter de leviers")

        outcome: Literal["EPUI", "ENG"] = (config.get("outcome") or "EPUI").upper()
        if outcome not in {"EPUI", "ENG"}:
            raise ValueError("outcome must be 'EPUI' or 'ENG'")

        method: Literal["pearson", "spearman"] = (config.get("method") or "spearman").lower()
        if method not in {"pearson", "spearman"}:
            raise ValueError("method must be 'pearson' or 'spearman'")

        min_n = int(config.get("min_n", 30))

        scores = compute_prefix_scores(df)
        outcome_col = f"DIM_{outcome}"
        if outcome_col not in scores.columns:
            raise ValueError(f"Outcome '{outcome}' not available (missing {outcome_col})")

        joined = scores.dropna(subset=[outcome_col]).copy()
        if joined.shape[0] < min_n:
            raise ValueError("Pas assez de répondants pour estimer les leviers")

        rows: List[Dict[str, Any]] = []
        y = joined[outcome_col]

        for col in joined.columns:
            if not col.startswith("DIM_"):
                continue
            prefix = col.replace("DIM_", "", 1)
            if prefix == outcome:
                continue

            x = joined[col]
            pair = pd.concat([x, y], axis=1).dropna()
            if pair.shape[0] < min_n:
                continue

            corr = pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method)
            if pd.isna(corr):
                continue

            mean_score = float(pair.iloc[:, 0].mean())
            gap_to_5 = float(5.0 - mean_score)

            if outcome == "EPUI":
                leverage = float(max(0.0, -corr))
            else:
                leverage = float(max(0.0, corr))

            priority = gap_to_5 * leverage

            rows.append(
                {
                    "dimension_prefix": prefix,
                    "dimension_label": prefix_label(prefix),
                    "mean_score": mean_score,
                    "gap_to_5": gap_to_5,
                    "corr_with_outcome": float(corr),
                    "leverage": leverage,
                    "priority_index": float(priority),
                    "n": int(pair.shape[0]),
                }
            )

        if not rows:
            raise ValueError("Aucune dimension exploitable pour le scatter")

        out = pd.DataFrame(rows)

        apply_theme()

        title = "Carte des leviers vs Épuisement" if outcome == "EPUI" else "Carte des leviers vs Engagement"

        base = alt.Chart(out)

        points = (
            base.mark_circle(opacity=0.85)
            .encode(
                x=alt.X("mean_score:Q", title="Score moyen (1-5)", scale=alt.Scale(domain=[1, 5])),
                y=alt.Y("leverage:Q", title="Levier (association)", scale=alt.Scale(domain=[0, 1])),
                size=alt.Size("priority_index:Q", title="Priorité", scale=alt.Scale(range=[50, 900])),
                color=alt.Color(
                    "priority_index:Q",
                    title="Priorité",
                    scale=alt.Scale(scheme="redyellowgreen"),
                ),
                tooltip=[
                    "dimension_label:N",
                    alt.Tooltip("mean_score:Q", title="Score moyen", format=".2f"),
                    alt.Tooltip("gap_to_5:Q", title="Écart à 5", format=".2f"),
                    alt.Tooltip("corr_with_outcome:Q", title=f"Corr. avec {outcome}", format=".2f"),
                    alt.Tooltip("leverage:Q", title="Levier", format=".2f"),
                    alt.Tooltip("priority_index:Q", title="Priorité", format=".3f"),
                    alt.Tooltip("n:Q", title="N"),
                ],
            )
            .properties(title=title)
        )

        labels = (
            base.mark_text(dx=8, dy=-8, fontSize=11, color="#111827")
            .encode(
                x="mean_score:Q",
                y="leverage:Q",
                text="dimension_prefix:N",
            )
        )

        return alt.layer(points, labels).interactive().to_dict()
