from typing import Any, Dict

import altair as alt
import pandas as pd

from src.services.survey_utils import (
    LIKERT_PREFIX_LABELS,
    add_age_band,
    available_demographics,
    detect_likert_columns,
    to_likert_long,
)
from src.viz.base import IVisualizationStrategy
from src.viz.theme import apply_theme


class LikertItemHeatmapStrategy(IVisualizationStrategy):
    """
    Heatmap des items Likert par groupe, avec moyenne ou % favorable.

    Config attendue :
      - group_field (str, optionnel)     : colonne de segmentation (ex. "Encadre", "Sexe", "Secteur", "TailleOr", "AgeClasse").
                                           Défaut : première variable sociodémographique disponible.
      - stat (str, optionnel)            : "mean" (défaut) ou "percent_favorable".
      - favorable_threshold (int, opt.)  : borne d'inclusion pour favorable (défaut 4, donc réponses >=4 comptées).
      - likert_domain (list|tuple, opt.) : bornes de l'échelle Likert, défaut [1, 5].
      - top_n (int, optionnel)           : limiter aux top N items par variance (priorité aux items les plus discriminants).

    Filtres : appliqués en amont (égalité sur colonnes présentes).
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
            raise ValueError("Survey data required for Likert item heatmap")

        df = add_age_band(survey_df.copy())

        # Appliquer filtres simples (égalité)
        for key, value in (filters or {}).items():
            if key in df.columns:
                df = df[df[key] == value]

        if df.empty:
            raise ValueError("Dataset vide après filtrage pour la heatmap d'items")

        likert_cols = detect_likert_columns(df)
        if not likert_cols:
            raise ValueError("No Likert columns detected for item heatmap")

        long_df = to_likert_long(df, likert_cols)
        long_df["response_value"] = pd.to_numeric(long_df["response_value"], errors="coerce")
        long_df = long_df.dropna(subset=["response_value"])
        long_df["dimension_label"] = (
            long_df["dimension_prefix"].map(LIKERT_PREFIX_LABELS).fillna(long_df["dimension_prefix"])
        )

        if long_df.empty:
            raise ValueError("Aucune donnée exploitable après conversion longue pour la heatmap d'items")

        # Group field selection
        group_field = config.get("group_field")
        if not group_field:
            demos = available_demographics(long_df)
            if not demos:
                raise ValueError("Aucun champ de segmentation disponible pour la heatmap d'items")
            group_field = demos[0]
        if group_field not in long_df.columns:
            raise ValueError(f"Segment field '{group_field}' not found in dataset")

        # Optionnel : réduire aux items les plus variés
        top_n = config.get("top_n")
        if top_n:
            top_n = int(top_n)
            variances = (
                long_df.groupby("question_label")["response_value"]
                .var()
                .reset_index()
                .sort_values("response_value", ascending=False)
            )
            selected_items = variances.head(top_n)["question_label"].tolist()
            long_df = long_df[long_df["question_label"].isin(selected_items)]

        stat = config.get("stat", "mean")
        favorable_threshold = config.get("favorable_threshold", 4)
        likert_domain = config.get("likert_domain", [1, 5])

        if stat == "percent_favorable":
            agg = (
                long_df.assign(fav=long_df["response_value"] >= favorable_threshold)
                .groupby([group_field, "question_label", "dimension_label"])
                .agg(
                    score=("fav", "mean"),
                    responses=("fav", "count"),
                )
                .reset_index()
            )
            score_title = f"% favorable (≥{favorable_threshold})"
            color_scale = alt.Scale(domain=[0, 1], scheme="blues")
            format_str = ".0%"
        elif stat == "mean":
            agg = (
                long_df.groupby([group_field, "question_label", "dimension_label"])["response_value"]
                .agg(["mean", "count"])
                .reset_index()
                .rename(columns={"mean": "score", "count": "responses"})
            )
            score_title = "Score moyen (1-5)"
            color_scale = alt.Scale(domain=likert_domain, scheme="blues")
            format_str = ".2f"
        else:
            raise ValueError("stat must be 'mean' or 'percent_favorable'")

        agg = agg.dropna(subset=["score"])
        if agg.empty:
            raise ValueError("Aucune donnée agrégée disponible pour la heatmap d'items")

        apply_theme()
        chart = (
            alt.Chart(agg)
            .mark_rect()
            .encode(
                x=alt.X(f"{group_field}:N", title=group_field),
                y=alt.Y("question_label:N", sort="x", title="Item"),
                color=alt.Color("score:Q", title=score_title, scale=color_scale),
                tooltip=[
                    "question_label",
                    group_field,
                    alt.Tooltip("score:Q", title=score_title, format=format_str),
                    "responses:Q",
                    "dimension_label:N",
                ],
            )
        )

        return chart.to_dict()
