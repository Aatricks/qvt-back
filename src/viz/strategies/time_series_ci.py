from typing import Any, Dict, Optional

import altair as alt
import pandas as pd

from src.viz.base import IVisualizationStrategy
from src.viz.theme import apply_theme


class TimeSeriesCIStreamingStrategy(IVisualizationStrategy):
    """
    Courbe temporelle avec intervalle de confiance (approx. normale) sur une métrique numérique.

    Config attendue :
      - measure_field (str, optionnel) : colonne numérique à tracer. Défaut : première colonne numérique trouvée.
      - time_field   (str, optionnel) : colonne temporelle/ordinale. Défaut : "ID" si présent, sinon première colonne.
      - group_field  (str, optionnel) : segmentation pour multi-lignes (ex. "Secteur", "Encadre").
      - ci_z         (float, opt.)    : quantile normal (défaut 1.96 pour IC95%).
      - min_count    (int, opt.)      : n minimum pour calculer un IC (défaut 2). Sinon IC masqué.
    Filtres : appliqués par égalité sur colonnes présentes.
    """

    def generate(
        self,
        data: Dict[str, pd.DataFrame],
        config: Dict[str, Any],
        filters: Dict[str, Any],
        settings: Any,
    ) -> Dict[str, Any]:
        hr_df = data["hr"].copy()

        # Filtres simples
        for key, value in (filters or {}).items():
            if key in hr_df.columns:
                hr_df = hr_df[hr_df[key] == value]

        if hr_df.empty:
            raise ValueError("Dataset vide après filtrage pour la série temporelle CI")

        metric = config.get("measure_field")
        if not metric or metric not in hr_df.columns:
            numeric_cols = hr_df.select_dtypes(include="number").columns.tolist()
            if not numeric_cols:
                raise ValueError("Aucune colonne numérique disponible pour la série temporelle CI")
            metric = numeric_cols[0]

        time_field = config.get("time_field") or ("ID" if "ID" in hr_df.columns else None)
        if not time_field or time_field not in hr_df.columns:
            time_field = hr_df.columns[0]

        group_field: Optional[str] = config.get("group_field")
        if group_field and group_field not in hr_df.columns:
            raise ValueError(f"Segment field '{group_field}' introuvable dans le dataset")

        # Nettoyage numérique
        hr_df[metric] = pd.to_numeric(hr_df[metric], errors="coerce")
        hr_df = hr_df.dropna(subset=[metric, time_field])

        if hr_df.empty:
            raise ValueError("Aucune donnée exploitable après nettoyage pour la série temporelle CI")

        ci_z = float(config.get("ci_z", 1.96))
        min_count = int(config.get("min_count", 2))

        group_fields = [time_field]
        if group_field:
            group_fields.append(group_field)

        agg = (
            hr_df.groupby(group_fields)[metric]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(columns={"mean": "metric_mean", "std": "metric_std", "count": "n"})
        )

        # Calcul IC
        agg["sem"] = agg["metric_std"] / agg["n"].pow(0.5)
        agg.loc[agg["n"] < min_count, ["sem"]] = pd.NA
        agg["lower_ci"] = agg["metric_mean"] - ci_z * agg["sem"]
        agg["upper_ci"] = agg["metric_mean"] + ci_z * agg["sem"]

        # Si sem manquant, on met les bornes à NaN pour éviter l'affichage du band
        agg.loc[agg["sem"].isna(), ["lower_ci", "upper_ci"]] = pd.NA

        apply_theme()

        x_enc = alt.X(f"{time_field}:O", title="Période")
        y_enc = alt.Y("metric_mean:Q", title=metric.replace("_", " ").title())

        base = alt.Chart(agg)

        if group_field:
            color_enc = alt.Color(f"{group_field}:N", title=group_field)
        else:
            color_enc = alt.value("#2563EB")

        band = (
            base.mark_area(opacity=0.2)
            .encode(
                x=x_enc,
                y=alt.Y("lower_ci:Q", title=""),
                y2="upper_ci:Q",
                color=color_enc if group_field else alt.value("#93C5FD"),
                tooltip=[
                    time_field,
                    *( [group_field] if group_field else [] ),
                    alt.Tooltip("metric_mean:Q", title="Moyenne", format=".2f"),
                    alt.Tooltip("n:Q", title="Effectif"),
                    alt.Tooltip("lower_ci:Q", title="IC basse", format=".2f"),
                    alt.Tooltip("upper_ci:Q", title="IC haute", format=".2f"),
                ],
            )
        )

        line = (
            base.mark_line(point=True)
            .encode(
                x=x_enc,
                y=y_enc,
                color=color_enc,
                tooltip=[
                    time_field,
                    *( [group_field] if group_field else [] ),
                    alt.Tooltip("metric_mean:Q", title="Moyenne", format=".2f"),
                    alt.Tooltip("n:Q", title="Effectif"),
                ],
            )
        )

        chart = alt.layer(band, line)

        return chart.to_dict()
