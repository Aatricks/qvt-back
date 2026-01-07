from typing import Any, Dict, List

import altair as alt
import pandas as pd

from src.services.survey_utils import (
    DEMO_VALUE_MAPPING,
    add_age_band,
    add_seniority_band,
    available_demographics,
)
from src.viz.base import IVisualizationStrategy
from src.viz.theme import apply_theme


class DemographicDistributionStrategy(IVisualizationStrategy):
    """
    Univariate distribution of socio-demographic variables.
    If no 'field' is provided, generates an overview dashboard of key indicators.
    """

    def generate(
        self,
        data: Dict[str, pd.DataFrame],
        config: Dict[str, Any],
        filters: Dict[str, Any],
        settings: Any,
    ) -> Dict[str, Any]:
        hr_df = data["hr"].copy()
        
        # Ensure column names are unique (final safety check)
        hr_df = hr_df.loc[:, ~hr_df.columns.duplicated()]

        # Apply value mappings for demographics (1 -> Homme, etc.)
        for col, mapping in DEMO_VALUE_MAPPING.items():
            if col in hr_df.columns:
                hr_df[col] = hr_df[col].map(mapping).fillna(hr_df[col])

        # Apply simple equality filters
        for key, value in (filters or {}).items():
            if key in hr_df.columns:
                hr_df = hr_df[hr_df[key] == value]

        if hr_df.empty:
            raise ValueError("Empty dataset after filtering for demographics")

        apply_theme()
        field = config.get("field")
        segment_field = config.get("segment_field")
        facet_field = config.get("facet_field")

        # Robustness: ignore segment/facet if not found in data
        if segment_field and segment_field not in hr_df.columns:
            segment_field = None
        if facet_field and facet_field not in hr_df.columns:
            facet_field = None
            
        # Update config for downstream use in _make_single_chart
        config = {**config, "segment_field": segment_field, "facet_field": facet_field}

        if not field:
            # Multi-indicators mode: overview dashboard
            preferred = ["AgeClasse", "Sexe", "Contrat", "Secteur", "AnciennetéClasse", "Encadre"]
            available = available_demographics(hr_df)
            
            # Identify redundant fields to skip in overview (they are already used for slicing/filtering)
            to_skip = {segment_field, facet_field}
            if filters:
                to_skip.update(filters.keys())
            
            to_plot = [
                f for f in preferred 
                if (f in hr_df.columns or f in available) and f not in to_skip
            ]
            
            # Limit to top 6 for layout sanity
            to_plot = to_plot[:6]

            if not to_plot:
                # Fallback: if absolutely nothing is found, try any column
                to_plot = [
                    c for c in hr_df.columns 
                    if c not in {"ID", "Age", "Ancienne", "Ancienneté"} and c not in to_skip
                ][:4]

            if not to_plot:
                raise ValueError("No demographic indicators detected for overview")

            charts: List[alt.Chart] = []
            for f in to_plot:
                if f in hr_df.columns:
                    charts.append(self._make_single_chart(hr_df, f, config))

            if not charts:
                 raise ValueError("No visualizable demographic indicators found")

            # Arrange in 2 or 3 columns
            cols = 3 if len(charts) > 3 else 2
            final = alt.concat(*charts, columns=cols).properties(
                title="Aperçu de la composition de l'effectif"
            )
            
            if facet_field:
                final = final.facet(column=alt.Column(f"{facet_field}:N", title=facet_field))

            return final.to_dict()

        # Single indicator mode
        if field not in hr_df.columns:
            return alt.Chart().mark_text(text=f"Champ '{field}' absent du jeu de données").properties(width=400, height=200).to_dict()

        chart = self._make_single_chart(hr_df, field, config)
        
        if facet_field:
            chart = chart.facet(column=alt.Column(f"{facet_field}:N", title=facet_field))

        return chart.interactive().to_dict()

    def _make_single_chart(self, df: pd.DataFrame, field: str, config: Dict[str, Any]) -> alt.Chart:
        """Internal helper to create a single bar chart or histogram."""
        normalize = bool(config.get("normalize", True)) # Default to % for overview
        segment_field = config.get("segment_field")
        series = df[field]
        
        # Binned fields should be treated as Nominal/Ordinal, not Quantitative
        is_binned = "Classe" in field or "tranche" in field.lower()
        is_numeric = pd.api.types.is_numeric_dtype(series) and not is_binned
        
        cols = [field]
        if segment_field and segment_field != field:
            cols.append(segment_field)
        subset = df[cols].dropna().copy()
        
        if subset.empty:
            return alt.Chart().mark_text().properties(title=f"{field} (no data)")

        color = alt.value("#2563EB")
        if segment_field:
            color = alt.Color(f"{segment_field}:N", title=segment_field)

        tooltip = [field, "count()"]
        if segment_field: tooltip.insert(0, segment_field)

        encoding = {
            "color": color,
            "tooltip": tooltip
        }
        
        # Grouped bars (xOffset) if comparison is active
        if segment_field:
            encoding["xOffset"] = alt.XOffset(f"{segment_field}:N", padding=0.1)

        if is_numeric:
            bin_size = config.get("bin_size")
            bin_params = {"step": bin_size} if bin_size else {"maxbins": 10}
            
            base = alt.Chart(subset).mark_bar(opacity=0.8).encode(
                x=alt.X(f"{field}:Q", bin=bin_params, title=field, scale=alt.Scale(paddingInner=0.2)),
                **encoding
            )
        else:
            sort = config.get("sort")
            if sort == "alpha":
                s = "ascending"
            elif sort == "count":
                s = "-y"
            else:
                s = None

            base = alt.Chart(subset).mark_bar(opacity=0.8).encode(
                x=alt.X(
                    f"{field}:N", 
                    sort=s, 
                    title=None, 
                    axis=alt.Axis(labelAngle=-45, labelLimit=100),
                    scale=alt.Scale(paddingInner=0.4)
                ),
                **encoding
            )

        if normalize:
            # Transform to percentages using window transform (safer for concat)
            group_by = [segment_field] if segment_field else []
            if is_numeric:
                chart = base.transform_window(
                    total="count()",
                    frame=[None, None],
                    groupby=group_by
                ).transform_calculate(
                    pct="datum.count / datum.total"
                ).encode(
                    y=alt.Y("pct:Q", title=None, axis=alt.Axis(format="%"))
                )
            else:
                chart = base.transform_window(
                    total="count()",
                    frame=[None, None],
                    groupby=group_by
                ).transform_calculate(
                    pct="1 / datum.total"
                ).encode(
                    y=alt.Y("sum(pct):Q", title=None, axis=alt.Axis(format="%"))
                )
        else:
            chart = base.encode(y=alt.Y("count()", title=None))

        # Dynamic width to prevent overlap in grouped bars, but conservative to keep it compact
        n_segments = subset[segment_field].nunique() if segment_field else 1
        step_width = max(20, n_segments * 12)

        return chart.properties(width={"step": step_width}, height=150, title=field)
