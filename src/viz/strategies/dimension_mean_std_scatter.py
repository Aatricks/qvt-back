from typing import Any, Dict, Optional

import altair as alt
import pandas as pd

from src.services.survey_utils import (
    DEMO_VALUE_MAPPING,
    LIKERT_PREFIX_LABELS,
    add_age_band,
    detect_likert_columns,
    to_likert_long,
)
from src.viz.base import IVisualizationStrategy
from src.viz.theme import apply_theme


class DimensionMeanStdScatterStrategy(IVisualizationStrategy):
    """
    Nuage de points (bulle) par dimension : moyenne vs écart-type pour détecter la polarisation.

    Config attendue :
      - likert_domain (list|tuple, optionnel) : bornes de l'échelle Likert, défaut [1, 5].
      - min_responses (int, optionnel)       : nombre minimal de réponses par dimension (défaut 5).
      - size_field (str, optionnel)          : champ pour dimensionner les points ; par défaut, l'effectif (n).
      - max_size (int, optionnel)            : taille max des bulles (défaut 800).
      - color_scheme (str, optionnel)        : palette Vega (défaut "blues").
      - show_labels (bool, optionnel)        : afficher les labels de dimension sur le scatter (défaut False).

    Filtres : appliqués par égalité sur colonnes présentes dans le DataFrame.
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
            raise ValueError("Survey data required for dimension mean/std scatter")

        df = add_age_band(survey_df.copy())
        
        # Apply value mappings for demographics (1 -> Homme, etc.)
        for col, mapping in DEMO_VALUE_MAPPING.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(df[col])

        # Appliquer filtres simples (égalité)
        for key, value in (filters or {}).items():
            if key in df.columns:
                df = df[df[key] == value]

        if df.empty:
            raise ValueError("Dataset vide après filtrage pour le scatter mean/std")

        likert_cols = detect_likert_columns(df)
        if not likert_cols:
            raise ValueError("No Likert columns detected for dimension mean/std scatter")

        long_df = to_likert_long(df, likert_cols)
        long_df["response_value"] = pd.to_numeric(long_df["response_value"], errors="coerce")
        long_df = long_df.dropna(subset=["response_value"])
        long_df["dimension_label"] = (
            long_df["dimension_prefix"].map(LIKERT_PREFIX_LABELS).fillna(long_df["dimension_prefix"])
        )

        if long_df.empty:
            raise ValueError("Aucune donnée exploitable après conversion longue")

        min_responses = int(config.get("min_responses", 5))
        segment_field: Optional[str] = config.get("segment_field")
        if segment_field and segment_field not in df.columns:
            raise ValueError(f"Segment field '{segment_field}' not found in dataset")

        group_cols = ["dimension_label"]
        if segment_field:
            group_cols.append(segment_field)

        agg = (
            long_df.groupby(group_cols)["response_value"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(columns={"mean": "mean_score", "std": "std_dev", "count": "responses"})
        )

        # Filtrer les dimensions trop petites
        agg = agg[agg["responses"] >= min_responses]
        agg = agg.dropna(subset=["mean_score", "std_dev"])

        if agg.empty:
            raise ValueError("Aucune dimension n'atteint le seuil minimal de réponses")

        # Taille des bulles
        size_field: Optional[str] = config.get("size_field")
        if size_field and size_field in agg.columns:
            agg["size"] = pd.to_numeric(agg[size_field], errors="coerce")
        else:
            agg["size"] = agg["responses"]

        agg = agg.dropna(subset=["size"])
        if agg.empty:
            raise ValueError("Aucune donnée exploitable après calcul des tailles")

        likert_domain = config.get("likert_domain", [1, 5])
        color_scheme = config.get("color_scheme", "blues")
        max_size = int(config.get("max_size", 800))
        show_labels = bool(config.get("show_labels", False))

        apply_theme()

        base = alt.Chart(agg)

        color_encoding = alt.Color("mean_score:Q", title="Score moyen", scale=alt.Scale(scheme=color_scheme))
        if segment_field:
            color_encoding = alt.Color(f"{segment_field}:N", title=segment_field)

        tooltip = [
            "dimension_label",
            alt.Tooltip("mean_score:Q", format=".2f", title="Moyenne"),
            alt.Tooltip("std_dev:Q", format=".2f", title="Écart-type"),
            alt.Tooltip("responses:Q", title="Réponses"),
        ]
        if segment_field:
            tooltip.insert(1, alt.Tooltip(f"{segment_field}:N", title=segment_field))

        points = (
            base.mark_circle(opacity=0.8)
            .encode(
                x=alt.X("mean_score:Q", title="Score moyen (1-5)", scale=alt.Scale(domain=likert_domain)),
                y=alt.Y("std_dev:Q", title="Écart-type (dispersion)", scale=alt.Scale(zero=True)),
                size=alt.Size("size:Q", title="Effectif", scale=alt.Scale(range=[50, max_size])),
                color=color_encoding,
                tooltip=tooltip,
            )
        )

        layers = [points]

        if show_labels:
            labels = (
                base.mark_text(dx=8, dy=-8, fontSize=11, color="#111827")
                .encode(
                    x="mean_score:Q",
                    y="std_dev:Q",
                    text="dimension_label",
                )
            )
            layers.append(labels)

        chart = alt.layer(*layers)

        return chart.interactive().to_dict()
