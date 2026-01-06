from __future__ import annotations

from typing import Any, Dict, List, Optional

import altair as alt
import pandas as pd

from src.services.survey_utils import add_age_band, detect_likert_columns, to_likert_long
from src.viz.base import IVisualizationStrategy
from src.viz.theme import apply_theme


class LikertDistributionStrategy(IVisualizationStrategy):
    """Diverging stacked bar distribution of Likert responses.

    Supports survey data provided either as:
        - wide format (one row per respondent + Likert item columns like PGC2, EPUI1, ...), or
        - long format with columns: question_label, response_value.

    Config:
        - top_n (int): limit number of questions shown (default 25)
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

        df = add_age_band(survey_df.copy())
        for key, value in (filters or {}).items():
            if key in df.columns:
                df = df[df[key] == value]

        if df.empty:
            raise ValueError("Dataset vide après filtrage pour la distribution Likert")

        top_n = int(config.get("top_n", 25))
        focus = str(config.get("focus") or "lowest").lower()
        if focus not in {"lowest", "highest"}:
            raise ValueError("focus must be 'lowest' or 'highest'")

        sort_metric = str(config.get("sort") or "net_agreement").lower()
        if sort_metric not in {"net_agreement", "mean"}:
            raise ValueError("sort must be 'net_agreement' or 'mean'")

        segment_field: Optional[str] = config.get("segment_field")
        if segment_field and segment_field not in df.columns:
            raise ValueError(f"segment_field '{segment_field}' not found in dataset")

        likert_cols = detect_likert_columns(df)
        if "question_label" not in df.columns or "response_value" not in df.columns:
            if not likert_cols:
                raise ValueError("No Likert columns detected for distribution")
            df = to_likert_long(
                df, likert_cols, extra_id_vars=[segment_field] if segment_field else None
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

        # --- Compute ranking stats per question (and optional segment) ---
        group_cols: List[str] = ["question_label"]
        if segment_field:
            group_cols.append(segment_field)

        counts = (
            df.groupby(group_cols + ["response_value"], dropna=False)
            .size()
            .rename("count")
            .reset_index()
        )
        totals = counts.groupby(group_cols)["count"].transform("sum")
        counts["total"] = totals
        counts["share"] = counts["count"] / counts["total"].where(counts["total"] != 0, 1)

        # Compute net agreement and mean score.
        wide = counts.pivot_table(
            index=group_cols,
            columns="response_value",
            values="count",
            aggfunc="sum",
            fill_value=0,
        ).reset_index()
        for k in [1, 2, 3, 4, 5]:
            if k not in wide.columns:
                wide[k] = 0
        wide["total"] = wide[[1, 2, 3, 4, 5]].sum(axis=1)
        # Avoid divide-by-zero.
        denom = wide["total"].where(wide["total"] != 0, 1)
        wide["mean"] = (wide[1] * 1 + wide[2] * 2 + wide[3] * 3 + wide[4] * 4 + wide[5] * 5) / denom
        wide["net_agreement"] = ((wide[4] + wide[5]) - (wide[1] + wide[2])) / denom

        counts = counts.merge(
            wide[group_cols + ["mean", "net_agreement"]], on=group_cols, how="left"
        )

        # Keep top-N questions by net agreement (decision-aid: surface weakest/strongest first).
        metric_series = counts.drop_duplicates(subset=group_cols)[
            group_cols + ["net_agreement"]
        ].copy()
        if focus == "lowest":
            metric_series = metric_series.sort_values("net_agreement", ascending=True)
        else:
            metric_series = metric_series.sort_values("net_agreement", ascending=False)
        keep_keys = metric_series.head(top_n)[group_cols]
        counts = counts.merge(keep_keys.assign(_keep=True), on=group_cols, how="left")
        # Avoid pandas FutureWarning about fillna() downcasting on object dtype:
        # after the merge, _keep is either True or NaN.
        counts["_keep"] = counts["_keep"].eq(True)
        counts = counts[counts["_keep"]].drop(columns=["_keep"])

        # Ensure dimension_prefix exists (for interactive filtering).
        if "dimension_prefix" not in df.columns:
            df = df.copy()
            df["dimension_prefix"] = "NA"
        dim_map = df[["question_label", "dimension_prefix"]].drop_duplicates()
        counts = counts.merge(dim_map, on="question_label", how="left")

        # --- Standard 100% Stacked Bar ---
        # No negative shares, standard normalization.
        plot_df = counts.copy()
        
        # Ordering: by chosen metric within the kept set
        plot_df["sort_value"] = plot_df[sort_metric]

        dims = sorted([d for d in plot_df["dimension_prefix"].dropna().unique() if str(d).strip()])
        interactive_dimension = bool(config.get("interactive_dimension", True))
        dim_param = None
        if interactive_dimension and dims:
            dim_param = alt.param(
                name="dimension",
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

        base = alt.Chart(plot_df)
        if dim_param is not None:
            base = base.add_params(dim_param).transform_filter(
                (dim_param == "All") | (alt.datum.dimension_prefix == dim_param)
            )
        if seg_param is not None and segment_field is not None:
            base = base.add_params(seg_param).transform_filter(
                (seg_param == "All") | (getattr(alt.datum, segment_field) == seg_param)
            )

        color_scale = alt.Scale(
            domain=[1, 2, 3, 4, 5],
            range=["#B91C1C", "#FCA5A5", "#D1D5DB", "#93C5FD", "#1D4ED8"],
        )

        chart = (
            base.mark_bar()
            .encode(
                y=alt.Y(
                    "question_label:N",
                    sort=alt.SortField(
                        "sort_value", order="ascending" if focus == "lowest" else "descending"
                    ),
                    title="Question",
                    axis=alt.Axis(labelLimit=260, labelPadding=8),
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
                    alt.Tooltip("question_label:N", title="Question"),
                    alt.Tooltip("dimension_prefix:N", title="Dimension"),
                    alt.Tooltip("response_value:O", title="Réponse"),
                    alt.Tooltip("count:Q", title="N (segment)", format=".0f"),
                    alt.Tooltip("share:Q", title="Part", format=".1%"),
                    alt.Tooltip("mean:Q", title="Moyenne", format=".2f"),
                    alt.Tooltip("net_agreement:Q", title="Net agreement", format=".1%"),
                ],
            )
            .properties(title="Distribution des réponses (Likert)", padding={"left": 120}, width="container")
            .interactive()
        )

        return chart.to_dict()
