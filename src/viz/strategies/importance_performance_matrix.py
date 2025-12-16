from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

import altair as alt
import pandas as pd

from src.services.qvt_metrics import compute_prefix_scores, prefix_label
from src.services.survey_utils import add_age_band, detect_likert_columns
from src.viz.base import IVisualizationStrategy
from src.viz.theme import apply_theme


class ImportancePerformanceMatrixStrategy(IVisualizationStrategy):
    """Importance–Performance (driver) matrix for QVT/QVCT survey dimensions.

    Purpose:
      - Help decision-making by crossing:
        X = performance (mean score 1–5)
        Y = importance (association with an outcome: ENG or EPUI)

    Importance is computed as a correlation-based *impact proxy*:
      - outcome=ENG  -> importance = max(0, corr(dim, ENG))
      - outcome=EPUI -> importance = max(0, -corr(dim, EPUI))   (higher dim -> lower exhaustion)

    Config:
      - outcome ("EPUI"|"ENG")        : default EPUI
      - method ("pearson"|"spearman") : default spearman
      - min_n (int)                    : minimum respondents for correlations (default 30)
      - segment_field (str, optional)  : compute matrix per segment and expose a dropdown filter

    Filters:
      - applied as equality on available columns.

    Notes:
      - This is a prioritization heuristic, not causal proof.
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
            raise ValueError("Survey data required for importance-performance matrix")

        df = add_age_band(survey_df.copy())
        for key, value in (filters or {}).items():
            if key in df.columns:
                df = df[df[key] == value]

        if df.empty:
            raise ValueError("Dataset vide après filtrage pour la matrice importance-performance")

        # Require wide Likert columns.
        if not detect_likert_columns(df):
            raise ValueError("No Likert columns detected (wide survey format required)")

        outcome: Literal["EPUI", "ENG"] = (config.get("outcome") or "EPUI").upper()
        if outcome not in {"EPUI", "ENG"}:
            raise ValueError("outcome must be 'EPUI' or 'ENG'")

        method: Literal["pearson", "spearman"] = (config.get("method") or "spearman").lower()
        if method not in {"pearson", "spearman"}:
            raise ValueError("method must be 'pearson' or 'spearman'")

        min_n = int(config.get("min_n", 5))

        segment_field: Optional[str] = config.get("segment_field")
        if segment_field and segment_field not in df.columns:
            raise ValueError(f"segment_field '{segment_field}' not found in dataset")

        # Per-respondent mean score per dimension prefix.
        scores = compute_prefix_scores(df)
        outcome_col = f"DIM_{outcome}"
        if outcome_col not in scores.columns:
            raise ValueError(f"Outcome '{outcome}' not available (missing {outcome_col} based on item prefixes)")

        if segment_field:
            seg = df[segment_field].astype(str)
        else:
            seg = pd.Series(["Organisation"] * len(df), index=df.index, name="segment")
            segment_field = "segment"

        joined = pd.concat([scores, seg.rename(segment_field)], axis=1)
        joined = joined.dropna(subset=[outcome_col, segment_field])
        if joined.shape[0] < min_n:
            raise ValueError("Pas assez de répondants pour estimer une matrice robuste")

        rows: List[Dict[str, Any]] = []
        for seg_value, seg_df in joined.groupby(segment_field):
            if seg_df.shape[0] < min_n:
                continue

            y = seg_df[outcome_col]
            for col in seg_df.columns:
                if not col.startswith("DIM_"):
                    continue
                prefix = col.replace("DIM_", "", 1)
                if prefix == outcome:
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
                    importance = float(max(0.0, -corr))
                else:
                    importance = float(max(0.0, corr))

                priority = gap_to_5 * importance

                rows.append(
                    {
                        "segment": seg_value,
                        "dimension_prefix": prefix,
                        "dimension_label": prefix_label(prefix),
                        "mean_score": mean_score,
                        "gap_to_5": gap_to_5,
                        "corr_with_outcome": float(corr),
                        "importance": importance,
                        "priority_index": float(priority),
                        "n": int(pair.shape[0]),
                    }
                )

        if not rows:
            raise ValueError("Aucune dimension ne répond aux critères (min_n) pour la matrice")

        out = pd.DataFrame(rows)

        # Segment-specific medians to define quadrants.
        cuts = (
            out.groupby("segment")
            .agg(x_cut=("mean_score", "median"), y_cut=("importance", "median"))
            .reset_index()
        )
        out = out.merge(cuts, on="segment", how="left")

        def quadrant(row: pd.Series) -> str:
            hi_imp = row["importance"] >= row["y_cut"]
            hi_perf = row["mean_score"] >= row["x_cut"]
            if hi_imp and not hi_perf:
                return "À prioriser"
            if hi_imp and hi_perf:
                return "À maintenir"
            if (not hi_imp) and hi_perf:
                return "Sur-investi"
            return "Secondaire"

        out["quadrant"] = out.apply(quadrant, axis=1)

        # Quadrant labels positions.
        label_rows: List[Dict[str, Any]] = []
        for _, r in cuts.iterrows():
            x_cut = float(r["x_cut"])
            y_cut = float(r["y_cut"])
            left_x = (1.0 + x_cut) / 2.0
            right_x = (x_cut + 5.0) / 2.0
            bottom_y = (0.0 + y_cut) / 2.0
            top_y = (y_cut + 1.0) / 2.0
            seg_value = r["segment"]

            label_rows.extend(
                [
                    {"segment": seg_value, "x": left_x, "y": top_y, "label": "À prioriser"},
                    {"segment": seg_value, "x": right_x, "y": top_y, "label": "À maintenir"},
                    {"segment": seg_value, "x": right_x, "y": bottom_y, "label": "Sur-investi"},
                    {"segment": seg_value, "x": left_x, "y": bottom_y, "label": "Secondaire"},
                ]
            )

        labels_df = pd.DataFrame(label_rows)

        apply_theme()

        title = (
            "Matrice importance–performance (impact vs Épuisement)" if outcome == "EPUI" else "Matrice importance–performance (impact vs Engagement)"
        )

        # Segment selector (optional)
        seg_param = None
        if out["segment"].nunique() > 1:
            seg_vals = sorted([str(v) for v in out["segment"].dropna().unique()])
            seg_param = alt.param(
                name="segment",
                value=seg_vals[0],
                bind=alt.binding_select(options=seg_vals, name=f"{segment_field}: "),
            )

        base = alt.Chart(out)
        cuts_chart = alt.Chart(cuts)
        quad_labels = alt.Chart(labels_df)

        if seg_param is not None:
            base = base.add_params(seg_param).transform_filter(alt.datum.segment == seg_param)
            cuts_chart = cuts_chart.add_params(seg_param).transform_filter(alt.datum.segment == seg_param)
            quad_labels = quad_labels.add_params(seg_param).transform_filter(alt.datum.segment == seg_param)

        color_scale = alt.Scale(
            domain=["À prioriser", "À maintenir", "Sur-investi", "Secondaire"],
            range=["#DC2626", "#16A34A", "#F59E0B", "#9CA3AF"],
        )

        points = (
            base.mark_circle(opacity=0.9)
            .encode(
                x=alt.X("mean_score:Q", title="Performance (score moyen 1–5)", scale=alt.Scale(domain=[1, 5])),
                y=alt.Y("importance:Q", title="Importance (impact)", scale=alt.Scale(domain=[0, 1])),
                size=alt.Size("priority_index:Q", title="Priorité", scale=alt.Scale(range=[60, 1200])),
                color=alt.Color("quadrant:N", title="Catégorie", scale=color_scale),
                tooltip=[
                    alt.Tooltip("dimension_label:N", title="Dimension"),
                    alt.Tooltip("mean_score:Q", title="Score moyen", format=".2f"),
                    alt.Tooltip("importance:Q", title="Importance", format=".2f"),
                    alt.Tooltip("corr_with_outcome:Q", title=f"Corr. avec {outcome}", format=".2f"),
                    alt.Tooltip("priority_index:Q", title="Priorité", format=".3f"),
                    alt.Tooltip("n:Q", title="N"),
                    alt.Tooltip("segment:N", title=segment_field),
                ],
            )
            .properties(title=title)
        )

        dim_labels = base.mark_text(dx=8, dy=-8, fontSize=11, color="#111827").encode(
            x="mean_score:Q",
            y="importance:Q",
            text="dimension_prefix:N",
        )

        v_rule = cuts_chart.mark_rule(color="#111827", strokeDash=[4, 4]).encode(x="x_cut:Q")
        h_rule = cuts_chart.mark_rule(color="#111827", strokeDash=[4, 4]).encode(y="y_cut:Q")

        q_text = quad_labels.mark_text(opacity=0.35, fontSize=16, fontWeight="bold").encode(
            x="x:Q",
            y="y:Q",
            text="label:N",
        )

        return alt.layer(points, dim_labels, v_rule, h_rule, q_text).to_dict()
