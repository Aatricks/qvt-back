from typing import Any, Dict

import altair as alt
import pandas as pd

from src.services.survey_utils import add_age_band
from src.viz.base import IVisualizationStrategy
from src.viz.theme import apply_theme


class ScatterRegressionStrategy(IVisualizationStrategy):
    """
    Nuage de points pratique/indicateur avec option de régression.

    Config attendue :
      - x_field (str) : champ numérique/pratique (obligatoire si aucune détection auto possible).
      - y_field (str) : champ numérique/indicateur (idem).
      - color_field (str, optionnel) : champ catégoriel pour colorer les points (ex. Encadre, Sexe, Secteur).
      - regression (bool, défaut True) : affiche la droite/LOESS de régression.
      - method (str, défaut "linear") : méthode de régression ("linear", "poly", "loess").
      - order (int, optionnel) : ordre du polynôme si method="poly".
      - ci (bool, défaut True) : afficher un intervalle de confiance sur la régression (erreur-band).
      - opacity (float, défaut 0.6) : opacité des points.
    Filtres : appliqués par égalité sur colonnes présentes.
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
            raise ValueError("Survey data required for scatter regression")

        df = add_age_band(survey_df.copy())

        # Appliquer filtres simples (égalité)
        for key, value in (filters or {}).items():
            if key in df.columns:
                df = df[df[key] == value]

        if df.empty:
            raise ValueError("Dataset vide après filtrage pour scatter regression")

        x_field = config.get("x_field")
        y_field = config.get("y_field")

        # Fallback auto sur deux colonnes numériques si non spécifié
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if not x_field or x_field not in df.columns:
            if len(numeric_cols) >= 1:
                x_field = numeric_cols[0]
            else:
                raise ValueError("Aucune colonne numérique disponible pour x_field")
        if not y_field or y_field not in df.columns:
            remaining = [c for c in numeric_cols if c != x_field]
            if remaining:
                y_field = remaining[0]
            elif numeric_cols:
                y_field = numeric_cols[-1]
            else:
                raise ValueError("Aucune colonne numérique disponible pour y_field")

        # Conversion numérique et nettoyage
        plot_df = df[[x_field, y_field]].copy()
        color_field = config.get("color_field")
        if color_field:
            if color_field not in df.columns:
                raise ValueError(f"color_field '{color_field}' non trouvé dans le dataset")
            plot_df[color_field] = df[color_field]

        plot_df[x_field] = pd.to_numeric(plot_df[x_field], errors="coerce")
        plot_df[y_field] = pd.to_numeric(plot_df[y_field], errors="coerce")
        plot_df = plot_df.dropna(subset=[x_field, y_field])
        if plot_df.empty:
            raise ValueError("Aucune donnée numérique exploitable pour le scatter regression")

        apply_theme()
        opacity = float(config.get("opacity", 0.6))

        x_enc = alt.X(f"{x_field}:Q", title=x_field)
        y_enc = alt.Y(f"{y_field}:Q", title=y_field)

        color_enc = alt.Color(f"{color_field}:N", title=color_field) if color_field else alt.value("#3B82F6")

        base = (
            alt.Chart(plot_df)
            .mark_circle(size=70, opacity=opacity)
            .encode(
                x=x_enc,
                y=y_enc,
                color=color_enc,
                tooltip=[x_field, y_field] + ([color_field] if color_field else []),
            )
        )

        layers = [base]

        if bool(config.get("regression", True)):
            method = config.get("method", "linear")
            order = config.get("order")
            ci = bool(config.get("ci", False))

            regression_kwargs: Dict[str, Any] = {"method": method}
            if order is not None:
                regression_kwargs["order"] = order
            if ci:
                regression_kwargs["ci"] = True

            reg_line = (
                alt.Chart(plot_df)
                .transform_regression(x_field, y_field, **regression_kwargs)
                .mark_line(color="#ef4444")
                .encode(x=x_enc, y=y_enc)
            )
            layers.append(reg_line)

            if ci:
                band = (
                    alt.Chart(plot_df)
                    .transform_regression(x_field, y_field, **regression_kwargs)
                    .mark_errorband(color="#ef4444", opacity=0.2)
                    .encode(x=x_enc, y=y_enc)
                )
                layers.append(band)

        chart = alt.layer(*layers)

        return chart.to_dict()
