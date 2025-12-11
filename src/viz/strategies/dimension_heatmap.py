from typing import Any, Dict, List

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


class DimensionHeatmapStrategy(IVisualizationStrategy):
    """
    Heatmap des scores moyens (ou médians) par dimension QVT et par groupe.

    Config attendue :
      - group_field (str) : colonne de segmentation (ex : "Encadre", "Sexe", "Secteur", "TailleOr", "AgeClasse").
      - stat (str, optionnel) : "mean" (défaut) ou "median".
      - likert_domain (list|tuple, optionnel) : bornes de l'échelle, par défaut [1, 5].
    Filtres : appliqués avant calcul (égalité sur colonnes présentes).
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
            raise ValueError("Survey data required for dimension heatmap")

        df = add_age_band(survey_df.copy())

        # Appliquer filtres simples
        for key, value in (filters or {}).items():
            if key in df.columns:
                df = df[df[key] == value]

        if df.empty:
            raise ValueError("Dataset vide après filtrage pour la dimension heatmap")

        likert_cols = detect_likert_columns(df)
        if not likert_cols:
            raise ValueError("No Likert columns detected for dimension heatmap")

        long_df = to_likert_long(df, likert_cols)
        long_df["response_value"] = pd.to_numeric(long_df["response_value"], errors="coerce")
        long_df = long_df.dropna(subset=["response_value"])
        long_df["dimension_label"] = (
            long_df["dimension_prefix"].map(LIKERT_PREFIX_LABELS).fillna(long_df["dimension_prefix"])
        )

        group_field = config.get("group_field")
        if not group_field:
            demos = available_demographics(long_df)
            if not demos:
                raise ValueError("Aucun champ de segmentation disponible pour la heatmap")
            group_field = demos[0]

        if group_field not in long_df.columns:
            raise ValueError(f"Segment field '{group_field}' not found in dataset")

        stat = config.get("stat", "mean")
        if stat not in {"mean", "median"}:
            raise ValueError("stat must be 'mean' or 'median'")

        agg = (
            long_df.groupby(["dimension_label", group_field])["response_value"]
            .agg(stat)
            .reset_index()
            .rename(columns={"response_value": "score"})
        )

        # Nettoyage des catégories manquantes
        agg = agg.dropna(subset=["dimension_label", group_field, "score"])
        if agg.empty:
            raise ValueError("Aucune donnée agrégée disponible pour la heatmap")

        domain = config.get("likert_domain", [1, 5])

        apply_theme()
        chart = (
            alt.Chart(agg)
            .mark_rect()
            .encode(
                x=alt.X(f"{group_field}:N", title=group_field),
                y=alt.Y("dimension_label:N", sort="-x", title="Dimension QVT"),
                color=alt.Color(
                    "score:Q",
                    title=f"Score ({stat})",
                    scale=alt.Scale(domain=domain, scheme="blues"),
                ),
                tooltip=[
                    "dimension_label",
                    group_field,
                    alt.Tooltip("score:Q", format=".2f", title=f"Score ({stat})"),
                ],
            )
        )
        return chart.to_dict()
