from __future__ import annotations

from typing import Any, Dict, List, Optional

import altair as alt
import pandas as pd

from src.services.survey_utils import (
    DEMO_VALUE_MAPPING,
    add_age_band,
    detect_likert_columns,
    to_likert_long,
)
from src.viz.base import IVisualizationStrategy
from src.viz.theme import apply_theme

# TODO: charte graphique sur les couleurs
class LikertDistributionStrategy(IVisualizationStrategy):
    """Diverging stacked bar distribution of Likert responses.

    Supports survey data provided either as:
        - wide format (one row per respondent + Likert item columns like PGC2, EPUI1, ...), or
        - long format with columns: question_label, response_value.

    Config:
        - top_n (int, ignored): formerly limited questions, now all questions are provided for hierarchical exploration.
        - focus ("lowest"|"highest"): which questions to keep by net agreement (default "lowest")
        - sort ("net_agreement"|"mean"): ordering metric within the kept set (default "net_agreement")
        - segment_field (str, optional): include segment in aggregation and expose an interactive dropdown
        - interactive_dimension (bool): dropdown filter on dimension prefix (default True)

    Filters:
        - applied as equality on available columns.

    Notes:
                    - Uses Vega-Lite `stack='center'` so the neutral category (3) naturally sits centered
                        (single segment), with 1–2 on the left and 4–5 on the right.
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
            raise ValueError("Survey data required for likert distribution")

        df = survey_df.copy()
        df = df.loc[:, ~df.columns.duplicated()]
        
        # Apply value mappings for demographics (1 -> Homme, etc.)
        for col, mapping in DEMO_VALUE_MAPPING.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(df[col])

        for key, value in (filters or {}).items():
            if key in df.columns:
                df = df[df[key] == value]

        if df.empty:
            raise ValueError("Dataset vide après filtrage pour la distribution Likert")

        # top_n is now ignored to ensure all questions are available in filtered mode
        focus = str(config.get("focus") or "lowest").lower()
        if focus not in {"lowest", "highest"}:
            raise ValueError("focus must be 'lowest' or 'highest'")

        sort_metric = str(config.get("sort") or "net_agreement").lower()
        if sort_metric not in {"net_agreement", "mean"}:
            raise ValueError("sort must be 'net_agreement' or 'mean'")

        segment_field: Optional[str] = config.get("segment_field")
        if segment_field and segment_field not in df.columns:
            raise ValueError(f"segment_field '{segment_field}' not found in dataset")

        facet_field: Optional[str] = config.get("facet_field")
        if facet_field and facet_field not in df.columns:
            raise ValueError(f"facet_field '{facet_field}' not found in dataset")

        likert_cols = detect_likert_columns(df)
        if "question_label" not in df.columns or "response_value" not in df.columns:
            if not likert_cols:
                raise ValueError("No Likert columns detected for distribution")
            
            id_vars = []
            if segment_field: id_vars.append(segment_field)
            if facet_field: id_vars.append(facet_field)
            
            df = to_likert_long(
                df, likert_cols, extra_id_vars=id_vars if id_vars else None
            )
        else:
            # Long format may not contain dimension_prefix.
            if "dimension_prefix" not in df.columns:
                df = df.copy()
                df["dimension_prefix"] = (
                    df["question_label"].astype(str).str.extract(r"^([A-Za-z]+)")[0]
                )

        apply_theme()

        df["response_value"] = pd.to_numeric(df["response_value"], errors="coerce")
        df = df.dropna(subset=["response_value", "question_label"])
        df["response_value"] = df["response_value"].astype(int)
        df = df[df["response_value"].between(1, 5)]

        # --- Compute distribution for QUESTIONS ---
        group_cols: List[str] = ["question_label", "dimension_prefix"]
        if segment_field:
            group_cols.append(segment_field)
        if facet_field:
            group_cols.append(facet_field)

        def get_dist(target_df, group_vars):
            counts = (
                target_df.groupby(group_vars + ["response_value"], dropna=False)
                .size()
                .rename("count")
                .reset_index()
            )
            totals = counts.groupby(group_vars)["count"].transform("sum")
            counts["total"] = totals
            counts["share"] = counts["count"] / counts["total"].where(counts["total"] != 0, 1)

            wide = counts.pivot_table(
                index=group_vars,
                columns="response_value",
                values="count",
                aggfunc="sum",
                fill_value=0,
            ).reset_index()
            for k in [1, 2, 3, 4, 5]:
                if k not in wide.columns:
                    wide[k] = 0
            wide["total"] = wide[[1, 2, 3, 4, 5]].sum(axis=1)
            denom = wide["total"].where(wide["total"] != 0, 1)
            wide["mean"] = (wide[1] * 1 + wide[2] * 2 + wide[3] * 3 + wide[4] * 4 + wide[5] * 5) / denom
            wide["net_agreement"] = ((wide[4] + wide[5]) - (wide[1] + wide[2])) / denom

            return counts.merge(
                wide[group_vars + ["mean", "net_agreement"]], on=group_vars, how="left"
            )

        q_counts = get_dist(df, group_cols)
        q_counts["is_category"] = 0
        q_counts["display_label"] = q_counts["question_label"]

        # --- Compute distribution for CATEGORIES (Dimension prefixes) ---
        cat_group_cols = ["dimension_prefix"]
        if segment_field:
            cat_group_cols.append(segment_field)
        if facet_field:
            cat_group_cols.append(facet_field)
        
        cat_counts = get_dist(df, cat_group_cols)
        cat_counts["is_category"] = 1
        cat_counts["display_label"] = cat_counts["dimension_prefix"]
        cat_counts["question_label"] = "Category Summary"

        plot_df = pd.concat([cat_counts, q_counts], ignore_index=True)
        
        # We need a sort value that works for both categories and questions
        plot_df["sort_value"] = plot_df[sort_metric]

        dims = sorted([d for d in plot_df["dimension_prefix"].dropna().unique() if str(d).strip()])
        interactive_dimension = bool(config.get("interactive_dimension", True))
        dim_param = None
        if interactive_dimension and dims:
            dim_param = alt.param(
                name="dim_select",
                value="All",
                bind=alt.binding_select(options=["All", *dims], name="Dimension: "),
            )

        seg_param = None
        if segment_field:
            seg_vals = [str(v) for v in plot_df[segment_field].dropna().unique()]
            seg_vals = sorted(seg_vals)
            seg_param = alt.param(
                name="segment",
                value="All",
                bind=alt.binding_select(options=["All", *seg_vals], name=f"{segment_field}: "),
            )

        # Ensure constant field exists for faceting (avoid transform_calculate inside facet)
        base = alt.Chart(plot_df)
        
        # NOTE: Filter is applied at the TOP LEVEL (final_chart) to ensure 'dim_select' visibility.
        # We use integer check for is_category (1=True, 0=False) for robustness.

        color_scale = alt.Scale(
            domain=[1, 2, 3, 4, 5],
            range=["#B91C1C", "#FCA5A5", "#D1D5DB", "#93C5FD", "#1D4ED8"],
        )

        chart = (
            base.mark_bar()
            .encode(
                y=alt.Y(
                    "display_label:N",
                    sort=alt.SortField(
                        "sort_value", order="ascending" if focus == "lowest" else "descending"
                    ),
                    title="Catégorie / Question",
                    axis=alt.Axis(labelLimit=350, labelPadding=8),
                ),
                x=alt.X(
                    "share:Q",
                    stack="normalize",
                    axis=alt.Axis(
                        title="Répartition des réponses",
                        format="%",
                    ),
                ),
                color=alt.Color(
                    "response_value:O",
                    title="Réponse (1–5)",
                    sort=[1, 2, 3, 4, 5],
                    scale=color_scale,
                    legend=alt.Legend(title="Réponse (1–5)"),
                ),
                tooltip=[
                    alt.Tooltip("display_label:N", title="Label"),
                    alt.Tooltip("dimension_prefix:N", title="Dimension"),
                    alt.Tooltip("is_category:N", title="Est une catégorie"),
                    alt.Tooltip("response_value:O", title="Réponse"),
                    alt.Tooltip("count:Q", title="N (segment)", format=".0f"),
                    alt.Tooltip("share:Q", title="Part", format=".1%"),
                    alt.Tooltip("mean:Q", title="Moyenne", format=".2f"),
                    alt.Tooltip("net_agreement:Q", title="Net agreement", format=".1%"),
                ],
            )
        )

        if facet_field:
            final_chart = chart.facet(
                column=alt.Column(f"{facet_field}:N", title=facet_field)
            ).resolve_scale(
                y="independent"
            ).properties(
                title=f"Distribution Likert par {facet_field}"
            )
        else:
            # Flattened view: No dummy facet. This ensures params and filters are in the same scope.
            final_chart = chart.properties(
                title="Distribution des réponses (Likert)"
            )

        # Apply filters to the TOP-LEVEL chart
        if dim_param is not None:
            final_chart = final_chart.transform_filter(
                ((dim_param == "All") & (alt.datum.is_category == 1)) |
                ((dim_param != "All") & (alt.datum.dimension_prefix == dim_param) & (alt.datum.is_category == 0))
            )
        else:
            # Fallback
            final_chart = final_chart.transform_filter(alt.datum.is_category == 1)

        if seg_param is not None and segment_field is not None:
            final_chart = final_chart.transform_filter(
                (seg_param == "All") | (alt.datum[segment_field] == seg_param)
            )

        # IMPORTANT: Interactive params must be added to the TOP-LEVEL chart
        if dim_param is not None:
            final_chart = final_chart.add_params(dim_param)
        if seg_param is not None:
            final_chart = final_chart.add_params(seg_param)

        return final_chart.to_dict()
