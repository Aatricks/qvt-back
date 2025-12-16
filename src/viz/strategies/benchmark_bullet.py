from typing import Any, Dict, Optional

import altair as alt
import pandas as pd

from src.viz.base import IVisualizationStrategy
from src.viz.theme import apply_theme
from src.config.observability import log_event


class BenchmarkBulletStrategy(IVisualizationStrategy):
    """
    Bullet chart pour comparer un score organisation vs benchmark et cible.

    Config attendue :
      - metric_field (str)   : champ numérique représentant le score de l'organisation (obligatoire).
      - benchmark_field (str, optionnel) : champ numérique pour le benchmark (référence externe).
      - target_field (str, optionnel)    : champ numérique pour l'objectif/cible interne.
      - group_field (str, optionnel)     : segmentation (ex. Secteur, Encadre). Si fourni, un bullet par groupe.
      - normalize (bool, optionnel)      : si True, divise les valeurs par 100 pour afficher en pourcentage (défaut False).
      - scale_domain (list, optionnel)   : borne min/max pour l'axe X, sinon auto.
      - max_groups (int, optionnel)      : limiter le nombre de groupes (défaut 50).

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

        # Appliquer filtres simples (égalité)
        for key, value in (filters or {}).items():
            if key in hr_df.columns:
                hr_df = hr_df[hr_df[key] == value]

        if hr_df.empty:
            raise ValueError("Dataset vide après filtrage pour le bullet chart")

        # Determine metric field: prefer user-provided, otherwise try to auto-detect a sensible numeric column
        metric_field: Optional[str] = config.get("metric_field")
        if (not metric_field) or (metric_field not in hr_df.columns):
            # Heuristic detection from numeric columns
            numeric_cols = hr_df.select_dtypes(include="number").columns.tolist()
            candidates = []
            for token in ["absentee", "absence", "rate", "score", "metric"]:
                for c in numeric_cols:
                    if token in c.lower():
                        candidates.append(c)
            # Fallback to any numeric column
            if not candidates and numeric_cols:
                candidates = numeric_cols
            if candidates:
                metric_field = candidates[0]
            else:
                raise ValueError("metric_field est requis et doit exister dans le dataset")

        # Determine benchmark field: if provided, validate; otherwise try to find a different numeric column
        benchmark_field: Optional[str] = config.get("benchmark_field")
        if benchmark_field and benchmark_field not in hr_df.columns:
            raise ValueError(f"Colonne '{benchmark_field}' introuvable dans le dataset")
        if not benchmark_field:
            numeric_cols = hr_df.select_dtypes(include="number").columns.tolist()
            possible = [c for c in numeric_cols if c != metric_field]
            bf = None
            for token in ["turnover", "benchmark", "target", "rate"]:
                for c in possible:
                    if token in c.lower():
                        bf = c
                        break
                if bf:
                    break
            if not bf and possible:
                bf = possible[0]
            benchmark_field = bf

        target_field: Optional[str] = config.get("target_field")
        group_field: Optional[str] = config.get("group_field")

        # Log the selected fields for observability (harmless if logging fails)
        try:
            log_event(
                "benchmark_bullet.selected_fields",
                metric_field=metric_field,
                benchmark_field=benchmark_field,
                target_field=target_field,
            )
        except Exception:
            pass

        # Vérification des champs optionnels
        for fld in [benchmark_field, target_field]:
            if fld and fld not in hr_df.columns:
                raise ValueError(f"Colonne '{fld}' introuvable dans le dataset")

        numeric_cols = [metric_field] + [f for f in [benchmark_field, target_field] if f]
        hr_df[numeric_cols] = hr_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        hr_df = hr_df.dropna(subset=[metric_field])

        if group_field and group_field not in hr_df.columns:
            raise ValueError(f"group_field '{group_field}' introuvable dans le dataset")

        # Agrégation (moyenne par groupe ou globale)
        if group_field:
            agg = (
                hr_df.groupby(group_field)[numeric_cols]
                .mean()
                .reset_index()
                .rename(columns={metric_field: "metric"})
            )
        else:
            agg = (
                hr_df[numeric_cols]
                .mean()
                .to_frame()
                .T.assign(Group="Organisation")
                .rename(columns={metric_field: "metric", "Group": "group"})
            )
            group_field = "group"

        if agg.empty:
            raise ValueError("Aucune donnée agrégée disponible pour le bullet chart")

        # Limiter le nombre de groupes
        max_groups = int(config.get("max_groups", 50))
        agg = agg.head(max_groups)

        # Normalisation optionnelle (ex: valeurs 0-100 en proportion)
        normalize = bool(config.get("normalize", False))
        if normalize:
            for fld in ["metric", benchmark_field, target_field]:
                if fld and fld in agg.columns:
                    agg[fld] = agg[fld] / 100.0

        scale_domain = config.get("scale_domain")
        if scale_domain and len(scale_domain) != 2:
            raise ValueError("scale_domain doit être une liste [min, max]")

        apply_theme()

        # Bar principale (performance)
        base = alt.Chart(agg)

        bars = (
            base.mark_bar(height=20, color="#3B82F6")
            .encode(
                x=alt.X(
                    "metric:Q",
                    title="Score" + (" (%)" if normalize else ""),
                    scale=alt.Scale(domain=scale_domain) if scale_domain else alt.Undefined,
                ),
                y=alt.Y(f"{group_field}:N", title=group_field or "Groupe", sort="-x"),
                tooltip=[
                    f"{group_field}:N",
                    alt.Tooltip("metric:Q", title="Score", format=".2f"),
                ],
            )
        )

        layers = [bars]

        # Benchmark (ligne)
        if benchmark_field:
            benchmark_rule = (
                base.mark_rule(color="#10B981", strokeWidth=2)
                .encode(
                    x=alt.X(
                        f"{benchmark_field}:Q",
                        scale=alt.Scale(domain=scale_domain) if scale_domain else alt.Undefined,
                    ),
                    y=alt.Y(f"{group_field}:N", sort="-x"),
                    tooltip=[
                        f"{group_field}:N",
                        alt.Tooltip(f"{benchmark_field}:Q", title="Benchmark", format=".2f"),
                    ],
                )
            )
            layers.append(benchmark_rule)

        # Cible (ligne pointillée)
        if target_field:
            target_rule = (
                base.mark_rule(color="#F59E0B", strokeDash=[4, 4], strokeWidth=2)
                .encode(
                    x=alt.X(
                        f"{target_field}:Q",
                        scale=alt.Scale(domain=scale_domain) if scale_domain else alt.Undefined,
                    ),
                    y=alt.Y(f"{group_field}:N", sort="-x"),
                    tooltip=[
                        f"{group_field}:N",
                        alt.Tooltip(f"{target_field}:Q", title="Cible", format=".2f"),
                    ],
                )
            )
            layers.append(target_rule)

        chart = alt.layer(*layers)

        return chart.to_dict()
