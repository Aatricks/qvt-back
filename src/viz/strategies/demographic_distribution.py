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
        hr_df = add_age_band(hr_df)
        hr_df = add_seniority_band(hr_df)

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

        if not field:
            # Multi-indicators mode: overview dashboard
            preferred = ["AgeClasse", "Sexe", "Contrat", "Secteur", "AnciennetéClasse", "Encadre"]
            available = available_demographics(hr_df)
            to_plot = [f for f in preferred if f in hr_df.columns or f in available]
            # Limit to top 6 for layout sanity
            to_plot = to_plot[:6]

            if not to_plot:
                raise ValueError("No demographic indicators detected for overview")

            charts: List[alt.Chart] = []
            for f in to_plot:
                charts.append(self._make_single_chart(hr_df, f, config))

            # Arrange in 2 or 3 columns
            cols = 3 if len(charts) > 3 else 2
            final = alt.concat(*charts, columns=cols).properties(
                title="Aperçu de la composition de l'effectif"
            )
            return final.to_dict()

        # Single indicator mode
        if field not in hr_df.columns:
            raise ValueError(f"Column '{field}' not found in dataset")

        return self._make_single_chart(hr_df, field, config).interactive().to_dict()

    def _make_single_chart(self, df: pd.DataFrame, field: str, config: Dict[str, Any]) -> alt.Chart:
        """Internal helper to create a single bar chart or histogram."""
        normalize = bool(config.get("normalize", True)) # Default to % for overview
        series = df[field]
        
        # Binned fields should be treated as Nominal/Ordinal, not Quantitative
        is_binned = "Classe" in field or "tranche" in field.lower()
        is_numeric = pd.api.types.is_numeric_dtype(series) and not is_binned
        
        subset = df[[field]].dropna().copy()
        if subset.empty:
            return alt.Chart().mark_text().properties(title=f"{field} (no data)")

        if is_numeric:
            bin_size = config.get("bin_size")
            bin_params = {"step": bin_size} if bin_size else {"maxbins": 10}
            
            base = alt.Chart(subset).mark_bar(opacity=0.8, color="#2563EB").encode(
                x=alt.X(f"{field}:Q", bin=bin_params, title=field),
                tooltip=[alt.Tooltip(f"{field}:Q", bin=bin_params, title="Tranche"), "count()"]
            )
        else:
            sort = config.get("sort")
            if sort == "alpha":
                s = "ascending"
            elif sort == "count":
                s = "-y"
            else:
                s = None

            base = alt.Chart(subset).mark_bar(opacity=0.8, color="#2563EB").encode(
                x=alt.X(f"{field}:N", sort=s, title=None, axis=alt.Axis(labelAngle=-45, labelLimit=100)),
                tooltip=[field, "count()"]
            )

        if normalize:
            # Transform to percentages using window transform (safer for concat)
            if is_numeric:
                chart = base.transform_window(
                    total="count()",
                    frame=[None, None]
                ).transform_calculate(
                    pct="datum.count / datum.total"
                ).encode(
                    y=alt.Y("pct:Q", title=None, axis=alt.Axis(format="%"))
                )
            else:
                chart = base.transform_window(
                    total="count()",
                    frame=[None, None]
                ).transform_calculate(
                    pct="1 / datum.total"
                ).encode(
                    y=alt.Y("sum(pct):Q", title=None, axis=alt.Axis(format="%"))
                )
        else:
            chart = base.encode(y=alt.Y("count()", title=None))

        return chart.properties(width=200, height=150, title=field)
