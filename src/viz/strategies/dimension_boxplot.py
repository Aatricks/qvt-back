from typing import Any, Dict, List, Optional

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


class DimensionBoxplotStrategy(IVisualizationStrategy):
    """
    Boxplots des distributions Likert par dimension, segmentées par une variable de groupe.

    Config attendue :
      - group_field (str, optionnel) : colonne de segmentation (ex. "Encadre", "Sexe", "Secteur", "TailleOr", "AgeClasse").
        Par défaut : première variable sociodémographique disponible.
      - dimensions (list[str], optionnel) : filtrer sur une liste de préfixes de dimensions (ex. ["COM", "RECO"]).
      - likert_domain (list|tuple, optionnel) : bornes de l'échelle, par défaut [1, 5].
      - show_outliers (bool, optionnel) : afficher les points atypiques (défaut True).
      - min_per_group (int, optionnel) : nombre min de réponses par groupe/dimension pour apparaître (défaut 3).
      - facet_columns (int, optionnel) : nb de colonnes dans la grille de facettes (défaut 3).

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
            raise ValueError("Survey data required for dimension boxplot")

        df = add_age_band(survey_df.copy())

        # Appliquer filtres simples (égalité)
        for key, value in (filters or {}).items():
            if key in df.columns:
                df = df[df[key] == value]

        if df.empty:
            raise ValueError("Dataset vide après filtrage pour les boxplots de dimension")

        likert_cols = detect_likert_columns(df)
        if not likert_cols:
            raise ValueError("No Likert columns detected for dimension boxplot")

        long_df = to_likert_long(df, likert_cols)
        long_df["response_value"] = pd.to_numeric(long_df["response_value"], errors="coerce")
        long_df = long_df.dropna(subset=["response_value"])
        long_df["dimension_label"] = (
            long_df["dimension_prefix"].map(LIKERT_PREFIX_LABELS).fillna(long_df["dimension_prefix"])
        )

        # Filtrer sur certaines dimensions si demandé
        dimensions: Optional[List[str]] = config.get("dimensions")
        if dimensions:
            dims_upper = {d.upper() for d in dimensions}
            long_df = long_df[long_df["dimension_prefix"].str.upper().isin(dims_upper)]

        if long_df.empty:
            raise ValueError("Aucune donnée exploitable après filtrage des dimensions")

        # Choix du group_field
        group_field = config.get("group_field")
        if not group_field:
            demos = available_demographics(long_df)
            if not demos:
                raise ValueError("Aucun champ de segmentation disponible pour les boxplots")
            group_field = demos[0]

        if group_field not in long_df.columns:
            raise ValueError(f"Segment field '{group_field}' not found in dataset")

        # Filtrer par taille minimale de groupe/dimension
        min_per_group = int(config.get("min_per_group", 3))
        counts = (
            long_df.groupby([group_field, "dimension_label"])["response_value"]
            .count()
            .reset_index(name="n")
        )
        valid_pairs = counts[counts["n"] >= min_per_group][[group_field, "dimension_label"]]
        long_df = long_df.merge(valid_pairs, on=[group_field, "dimension_label"], how="inner")

        if long_df.empty:
            raise ValueError("Aucun groupe/dimension n'atteint le seuil minimal de réponses")

        domain = config.get("likert_domain", [1, 5])
        show_outliers = bool(config.get("show_outliers", True))

        apply_theme()
        chart = (
            alt.Chart(long_df)
            .mark_boxplot(outliers=show_outliers)
            .encode(
                x=alt.X("response_value:Q", title="Réponse (1-5)", scale=alt.Scale(domain=domain)),
                y=alt.Y("dimension_label:N", title="Dimension QVT"),
                color=alt.Color(f"{group_field}:N", title=group_field),
                tooltip=[
                    "dimension_label",
                    group_field,
                    alt.Tooltip("response_value:Q", format=".2f", title="Valeur"),
                ],
            )
        )

        return chart.to_dict()
