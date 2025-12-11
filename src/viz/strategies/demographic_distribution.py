from typing import Any, Dict

import altair as alt
import pandas as pd

from src.services.survey_utils import add_age_band, available_demographics
from src.viz.base import IVisualizationStrategy
from src.viz.theme import apply_theme


class DemographicDistributionStrategy(IVisualizationStrategy):
    """
    Univariate distribution d'une variable sociodémographique (histogramme pour numérique, barres pour catégoriel).

    Config attendue:
      - field (str): colonne cible (ex: "Age", "Sexe", "Contrat", "Temps", "Encadre", "Secteur", "TailleOr").
      - bin_size (int, optionnel): taille des bins pour les numériques (par ex. 5 ans).
      - max_bins (int, optionnel): nombre max de bins (fallback si bin_size absent).
      - normalize (bool, défaut False): normaliser la hauteur des barres en pourcentage.
      - sort (str|None): "alpha", "count" ou None pour le tri des catégories.
    Filtres: appliqués avant le calcul (par égalité sur colonnes présentes).
    """

    def generate(
        self,
        data: Dict[str, pd.DataFrame],
        config: Dict[str, Any],
        filters: Dict[str, Any],
        settings: Any,
    ) -> Dict[str, Any]:
        hr_df = add_age_band(data["hr"].copy())

        # Appliquer les filtres simples (égalité)
        for key, value in (filters or {}).items():
            if key in hr_df.columns:
                hr_df = hr_df[hr_df[key] == value]

        if hr_df.empty:
            raise ValueError("Dataset vide après filtrage pour la distribution démographique")

        field = config.get("field")
        if not field:
            # Choix par défaut : Age si présent, sinon première variable socio détectée
            if "Age" in hr_df.columns:
                field = "Age"
            else:
                demos = available_demographics(hr_df)
                if not demos:
                    raise ValueError("Aucune colonne sociodémographique détectée")
                field = demos[0]

        if field not in hr_df.columns:
            raise ValueError(f"Colonne '{field}' non trouvée dans le dataset")

        series = hr_df[field]

        is_numeric = pd.api.types.is_numeric_dtype(series) or field.lower() == "age"
        normalize = bool(config.get("normalize", False))

        apply_theme()

        if is_numeric:
            # Convertir en numérique pour histogramme
            numeric_series = pd.to_numeric(series, errors="coerce").dropna()
            if numeric_series.empty:
                raise ValueError(f"Aucune valeur numérique exploitable pour '{field}'")

            bin_size = config.get("bin_size")
            max_bins = config.get("max_bins", 10)

            bin_params: Dict[str, Any]
            if bin_size:
                bin_params = {"step": bin_size}
            else:
                bin_params = {"maxbins": max_bins}

            df_plot = pd.DataFrame({field: numeric_series})

            chart = (
                alt.Chart(df_plot)
                .mark_bar()
                .encode(
                    x=alt.X(f"{field}:Q", bin=bin_params, title=field),
                    y=alt.Y(
                        "count()",
                        stack=None,
                        title="Pourcentage" if normalize else "Effectif",
                        axis=alt.Axis(format="%" if normalize else None),
                        scale=alt.Scale(domain=[0, 1]) if normalize else alt.Undefined,
                    )
                    if normalize
                    else "count()",
                    tooltip=[alt.Tooltip(f"{field}:Q", bin=bin_params), alt.Tooltip("count()", title="Effectif")],
                )
            )

            if normalize:
                chart = chart.transform_joinaggregate(total="count()").transform_calculate(
                    pct="datum.count / datum.total"
                ).encode(
                    y=alt.Y(
                        "pct:Q",
                        title="Pourcentage",
                        axis=alt.Axis(format="%"),
                    )
                )
        else:
            # Catégoriel : barres horizontales
            cat_series = series.dropna().astype(str)
            if cat_series.empty:
                raise ValueError(f"Aucune valeur catégorielle exploitable pour '{field}'")
            df_plot = pd.DataFrame({field: cat_series})

            sort_param = config.get("sort")
            if sort_param == "alpha":
                sort = "ascending"
            elif sort_param == "count":
                sort = "-y"
            else:
                sort = None

            y_enc = alt.Y(f"{field}:N", sort=sort, title=field)
            x_enc = (
                alt.X(
                    "count()",
                    stack=None,
                    title="Pourcentage" if normalize else "Effectif",
                    axis=alt.Axis(format="%" if normalize else None),
                    scale=alt.Scale(domain=[0, 1]) if normalize else alt.Undefined,
                )
                if normalize
                else alt.X("count()", title="Effectif")
            )

            chart = (
                alt.Chart(df_plot)
                .mark_bar()
                .encode(
                    y=y_enc,
                    x=x_enc,
                    tooltip=[field, alt.Tooltip("count()", title="Effectif")],
                )
            )

            if normalize:
                chart = chart.transform_joinaggregate(total="count()").transform_calculate(
                    pct="datum.count / datum.total"
                ).encode(
                    x=alt.X("pct:Q", title="Pourcentage", axis=alt.Axis(format="%"))
                )

        return chart.to_dict()
