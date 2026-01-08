from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

import altair as alt
import pandas as pd

from src.services.qvt_metrics import compute_prefix_scores, prefix_label
from src.services.survey_utils import add_age_band, available_demographics
from src.viz.base import IVisualizationStrategy
from src.viz.theme import apply_theme


class ActionPriorityIndexStrategy(IVisualizationStrategy):
    """Decision-aid ranking: which QVT dimensions are the best action priorities.

    Idea:
      - For each dimension prefix (COM, RECO, ...), compute a per-respondent dimension score.
      - Compute an outcome score (EPUI or ENG) per respondent.
      - Compute a *leverage* proxy: correlation between dimension score and outcome.
      - Combine with the *gap* to 5 to form a priority index.

    Defaults are chosen for QVT:
      - outcome = "EPUI" (burnout/strain). We prioritize levers that are low *and* associated with lower EPUI.
      - leverage = max(0, -corr(dim, EPUI))  (higher dimension => lower EPUI)
      - priority_index = gap_to_5 * leverage

    Config:
      - outcome ("EPUI"|"ENG")            : outcome to link to (default EPUI)
      - method ("pearson"|"spearman")     : correlation method (default spearman for robustness)
      - min_n (int)                        : minimum respondents required to compute correlations (default 30)
      - segment_field (str, optional)      : compute ranking within a demographic segment (e.g. "Sexe")
      - top_n (int)                        : limit displayed dimensions (default 12)

    Filters:
      - applied as equality on available columns.

    Notes:
      - This is a heuristic “priority” metric, not causal proof.
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
            raise ValueError("Survey data required for action priority index")

        df = add_age_band(survey_df.copy())
        for key, value in (filters or {}).items():
            if key in df.columns:
                df = df[df[key] == value]

        if df.empty:
            raise ValueError("Dataset vide après filtrage pour l'indice de priorité")

        outcome: Literal["EPUI", "ENG"] = (config.get("outcome") or "EPUI").upper()
        if outcome not in {"EPUI", "ENG"}:
            raise ValueError("outcome must be 'EPUI' or 'ENG'")

        method: Literal["pearson", "spearman"] = (config.get("method") or "spearman").lower()
        if method not in {"pearson", "spearman"}:
            raise ValueError("method must be 'pearson' or 'spearman'")

        min_n = int(config.get("min_n", 30))
        top_n = int(config.get("top_n", 12))

        # Compute per-respondent scores per dimension prefix.
        dim_scores = compute_prefix_scores(df)

        # Outcome score (per respondent)
        outcome_col = f"DIM_{outcome}"
        if outcome_col not in dim_scores.columns:
            raise ValueError(f"Outcome '{outcome}' not available (missing {outcome_col} based on item prefixes)")

        # Optional segmentation
        segment_field: Optional[str] = config.get("segment_field")
        if segment_field and segment_field not in df.columns:
            raise ValueError(f"segment_field '{segment_field}' not found in dataset")

        if segment_field:
            seg_series = df[segment_field]
        else:
            seg_series = pd.Series(["Organisation"] * len(df), index=df.index, name="segment")
            segment_field = "segment"

        # Build long table for correlation computations
        joined = pd.concat([dim_scores, seg_series.rename(segment_field)], axis=1)
        joined = joined.dropna(subset=[outcome_col, segment_field])

        if joined.empty or joined.shape[0] < min_n:
            raise ValueError("Pas assez de répondants pour calculer des priorités robustes")

        rows: List[Dict[str, Any]] = []
        for seg_value, seg_df in joined.groupby(segment_field):
            if seg_df.shape[0] < min_n:
                continue

            y = seg_df[outcome_col]
            for col in seg_df.columns:
                if not col.startswith("DIM_"):
                    continue
                prefix = col.replace("DIM_", "", 1)
                if prefix in {outcome}:
                    continue

                x = seg_df[col]
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
                    # For ENG, positive corr means higher dimension => higher engagement.
                    leverage = float(max(0.0, corr))

                priority_index = gap_to_5 * leverage

                rows.append(
                    {
                        "segment": seg_value,
                        "dimension_prefix": prefix,
                        "dimension_label": prefix_label(prefix),
                        "mean_score": mean_score,
                        "gap_to_5": gap_to_5,
                        "corr_with_outcome": float(corr),
                        "leverage": leverage,
                        "priority_index": float(priority_index),
                        "n": int(pair.shape[0]),
                    }
                )

        if not rows:
            raise ValueError("Aucune dimension ne répond aux critères (min_n) pour calculer l'indice")

        out = pd.DataFrame(rows)

        # Keep top N per segment
        out = out.sort_values(["segment", "priority_index"], ascending=[True, False])
        out = out.groupby("segment").head(top_n).copy()

        # Sort for chart readability: highest priority first
        out["dimension_order"] = out.groupby("segment")["priority_index"].rank(method="first", ascending=False)

        apply_theme()

        title = (
            "Leviers de prévention de l'épuisement" if outcome == "EPUI" else "Leviers de promotion de l'engagement"
        )

        chart = (
            alt.Chart(out)
            .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
            .encode(
                y=alt.Y(
                    "dimension_label:N",
                    sort="-x",
                    title=None,
                    axis=alt.Axis(labelLimit=280, labelPadding=12, labelFontSize=10),
                ),
                x=alt.X(
                    "priority_index:Q",
                    title="Indice de Priorité (Heuristique)",
                    scale=alt.Scale(zero=True),
                    axis=alt.Axis(grid=True, gridDash=[2,2], titleFontSize=11)
                ),
                color=alt.Color(
                    "segment:N", 
                    title="Segment",
                    legend=alt.Legend(orient="bottom", titleFontSize=10, labelFontSize=9)
                ) if out["segment"].nunique() > 1 else alt.value("#4F46E5"),
                tooltip=[
                    alt.Tooltip("dimension_label:N", title="Dimension"),
                    alt.Tooltip("mean_score:Q", title="Score moyen", format=".2f"),
                    alt.Tooltip("gap_to_5:Q", title="Marge d'amélioration", format=".2f"),
                    alt.Tooltip("corr_with_outcome:Q", title=f"Impact sur {outcome}", format=".2f"),
                    alt.Tooltip("priority_index:Q", title="Priorité relative", format=".3f"),
                    alt.Tooltip("n:Q", title="Effectif"),
                ],
            )
            .properties(title=alt.TitleParams(text=title, anchor="start", fontSize=14))
            .configure_view(stroke=None)
        )

        return chart.to_dict()
