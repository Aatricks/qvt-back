from typing import Any, Dict

import altair as alt
import pandas as pd

from src.services.survey_utils import add_age_band, available_demographics
from src.viz.base import IVisualizationStrategy
from src.viz.theme import apply_theme


class EngEpuiQuadrantsStrategy(IVisualizationStrategy):
    """
    Positionnement des groupes sur ENG (Y) vs EPUI (X) avec quadrants de risque.

    Config attendue :
      - x_field (str, optionnel) : colonne pour l'axe X (défaut: "EPUI").
      - y_field (str, optionnel) : colonne pour l'axe Y (défaut: "ENG").
      - group_field (str, optionnel) : colonne de segmentation (ex. "Encadre", "Sexe", "Secteur", "TailleOr", "AgeClasse").
                                      Défaut : première variable sociodémographique disponible.
      - x_threshold (float, optionnel) : seuil vertical pour EPUI. Défaut : médiane globale de x_field.
      - y_threshold (float, optionnel) : seuil horizontal pour ENG. Défaut : médiane globale de y_field.
      - size_field (str, optionnel) : champ numérique pour dimensionner les points. Défaut : effectif du groupe (n).
      - max_size (int, optionnel) : taille max des points (défaut: 400).
      - show_labels (bool, optionnel) : afficher les étiquettes de groupe (défaut False).

    Filtres : appliqués en amont par égalité sur colonnes présentes.
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
            raise ValueError("Survey data required for ENG/EPU quadrants")

        df = add_age_band(survey_df.copy())

        # Appliquer filtres simples (égalité)
        for key, value in (filters or {}).items():
            if key in df.columns:
                df = df[df[key] == value]

        if df.empty:
            raise ValueError("Dataset vide après filtrage pour les quadrants ENG/EPUI")

        # Compute EPUI and ENG means if not present
        if "EPUI" not in df.columns:
            epui_cols = [col for col in df.columns if col.startswith("EPUI")]
            if epui_cols:
                df["EPUI"] = df[epui_cols].mean(axis=1)
            else:
                raise ValueError("No EPUI columns found to compute mean")
        if "ENG" not in df.columns:
            eng_cols = [col for col in df.columns if col.startswith("ENG")]
            if eng_cols:
                df["ENG"] = df[eng_cols].mean(axis=1)
            else:
                raise ValueError("No ENG columns found to compute mean")

        x_field = config.get("x_field", "EPUI")
        y_field = config.get("y_field", "ENG")

        if x_field not in df.columns:
            raise ValueError(f"Colonne '{x_field}' introuvable pour l'axe X")
        if y_field not in df.columns:
            raise ValueError(f"Colonne '{y_field}' introuvable pour l'axe Y")

        # Choix du group_field
        group_field = config.get("group_field")
        if not group_field:
            demos = available_demographics(df)
            if not demos:
                raise ValueError("Aucun champ de segmentation disponible pour les quadrants ENG/EPUI")
            group_field = demos[0]
        if group_field not in df.columns:
            raise ValueError(f"Segment field '{group_field}' non trouvé dans le dataset")

        # Nettoyage numérique
        df[x_field] = pd.to_numeric(df[x_field], errors="coerce")
        df[y_field] = pd.to_numeric(df[y_field], errors="coerce")
        df = df.dropna(subset=[x_field, y_field, group_field])

        if df.empty:
            raise ValueError("Aucune donnée exploitable pour les quadrants ENG/EPUI après nettoyage")

        # Seuils (médianes globales si non fournis)
        x_threshold = config.get("x_threshold")
        if x_threshold is None:
            x_threshold = float(df[x_field].median())
        y_threshold = config.get("y_threshold")
        if y_threshold is None:
            y_threshold = float(df[y_field].median())

        # Agrégation par groupe
        agg = (
            df.groupby(group_field)[[x_field, y_field]]
            .agg("mean")
            .reset_index()
            .rename(columns={x_field: "x_mean", y_field: "y_mean"})
        )
        agg["n"] = df[group_field].value_counts().reindex(agg[group_field]).fillna(0).astype(int).values

        if agg.empty:
            raise ValueError("Aucune agrégation disponible pour les quadrants ENG/EPUI")

        # Taille des points
        size_field = config.get("size_field")
        if size_field and size_field in agg.columns:
            agg["size"] = pd.to_numeric(agg[size_field], errors="coerce")
        else:
            agg["size"] = agg["n"]

        agg = agg.dropna(subset=["x_mean", "y_mean", "size"])
        if agg.empty:
            raise ValueError("Données vides après calcul des tailles et moyennes")

        # Classification des quadrants
        def quadrant(row: pd.Series) -> str:
            if row["x_mean"] >= x_threshold and row["y_mean"] >= y_threshold:
                return "Épuisement élevé / Engagement élevé"
            if row["x_mean"] >= x_threshold and row["y_mean"] < y_threshold:
                return "Épuisement élevé / Engagement faible"
            if row["x_mean"] < x_threshold and row["y_mean"] >= y_threshold:
                return "Épuisement faible / Engagement élevé"
            return "Épuisement faible / Engagement faible"

        agg["quadrant"] = agg.apply(quadrant, axis=1)

        apply_theme()

        base = alt.Chart(agg)

        points = (
            base.mark_circle(opacity=0.75)
            .encode(
                x=alt.X("x_mean:Q", title=x_field),
                y=alt.Y("y_mean:Q", title=y_field),
                color=alt.Color("quadrant:N", title="Quadrant"),
                size=alt.Size("size:Q", title="Effectif", scale=alt.Scale(range=[50, config.get("max_size", 400)])),
                tooltip=[
                    group_field,
                    alt.Tooltip("x_mean:Q", format=".2f", title=f"{x_field} (moy.)"),
                    alt.Tooltip("y_mean:Q", format=".2f", title=f"{y_field} (moy.)"),
                    alt.Tooltip("n:Q", title="Effectif"),
                    "quadrant",
                ],
            )
        )

        layers = [points]

        # Règles de quadrant
        v_rule = base.mark_rule(color="#9ca3af", strokeDash=[4, 4]).encode(x=alt.datum(x_threshold))
        h_rule = base.mark_rule(color="#9ca3af", strokeDash=[4, 4]).encode(y=alt.datum(y_threshold))
        layers.extend([v_rule, h_rule])

        # Labels optionnels
        if bool(config.get("show_labels", False)):
            labels = (
                base.mark_text(dx=8, dy=-8, fontSize=11, color="#111827")
                .encode(
                    x="x_mean:Q",
                    y="y_mean:Q",
                    text=group_field,
                )
            )
            layers.append(labels)

        chart = alt.layer(*layers)

        return chart.to_dict()
