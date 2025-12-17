from __future__ import annotations

from typing import Any, Dict, List, Literal

import altair as alt
import numpy as np
import pandas as pd

from src.services.qvt_metrics import compute_prefix_scores, prefix_label
from src.viz.base import IVisualizationStrategy
from src.viz.theme import apply_theme

# Known outcome dimensions that should generally not be used as predictors for other outcomes
KNOWN_OUTCOMES = {"ENG", "EPUI", "CSE"}

class PredictiveSimulationStrategy(IVisualizationStrategy):
    """
    Simulates the impact of QVT practices on a target outcome (e.g. Engagement)
    using a linear regression model (Ordinary Least Squares).

    Visualizes the regression coefficients (impact weights) to identify which levers
    have the strongest potential effect on the target.
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
            raise ValueError("Survey data required for predictive simulation")

        # 1. Prepare Data
        df = survey_df.copy()
        # Apply filters
        for key, value in (filters or {}).items():
            if key in df.columns:
                df = df[df[key] == value]

        scores = compute_prefix_scores(df)

        # 2. Identify Target and Features
        target_name = (config.get("target") or "ENG").upper()
        target_col = f"DIM_{target_name}"

        if target_col not in scores.columns:
            # Fallback if specific target not found, try generic EPUI or ENG if available
            if "DIM_ENG" in scores.columns:
                target_name = "ENG"
                target_col = "DIM_ENG"
            elif "DIM_EPUI" in scores.columns:
                target_name = "EPUI"
                target_col = "DIM_EPUI"
            else:
                raise ValueError(f"Target '{target_name}' not found in scores and no fallback available.")

        # Features: All dimensions starting with DIM_ except the target
        # AND exclude other known outcomes to avoid confounding (e.g. predicting Engagement with Burnout)
        # unless the user specifically asks for it? For now, we exclude known outcomes from predictors.
        
        feature_cols = []
        for col in scores.columns:
            if not col.startswith("DIM_"):
                continue
            if col == target_col:
                continue
            
            prefix = col.replace("DIM_", "")
            if prefix in KNOWN_OUTCOMES:
                continue
            
            feature_cols.append(col)

        if not feature_cols:
            raise ValueError("No feature dimensions available for prediction.")

        # Prepare X and y
        modeling_df = scores[[target_col] + feature_cols].dropna()

        if len(modeling_df) < 10:
             raise ValueError("Not enough complete responses for regression (min 10).")

        y = modeling_df[target_col].values
        X = modeling_df[feature_cols].values
        
        # Add intercept
        X_with_bias = np.c_[np.ones(X.shape[0]), X]

        # 3. Train Model (OLS)
        # coeffs = (X^T X)^-1 X^T y
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X_with_bias, y, rcond=None)
        except np.linalg.LinAlgError:
             raise ValueError("Singular matrix - cannot solve regression. Check for collinearity.")

        # coeffs[0] is intercept
        feature_coeffs = coeffs[1:]

        # 4. Format Results
        results = []
        for col, coeff in zip(feature_cols, feature_coeffs):
            prefix = col.replace("DIM_", "")
            results.append({
                "dimension": prefix,
                "label": prefix_label(prefix),
                "impact": float(coeff),  # The slope
                "abs_impact": abs(float(coeff))
            })

        results_df = pd.DataFrame(results)
        
        if results_df.empty:
            raise ValueError("Regression produced no results.")

        # Sort by absolute impact to show most important first
        results_df = results_df.sort_values("impact", ascending=False)

        # 5. Build Visualization
        apply_theme()
        
        results_df["sign"] = results_df["impact"].apply(lambda x: "Positif" if x > 0 else "Négatif")

        target_label = prefix_label(target_name)
        
        chart = alt.Chart(results_df).mark_bar().encode(
            x=alt.X("impact:Q", title=f"Impact estimé sur {target_label} (Coeff)"),
            y=alt.Y("label:N", sort="-x", title="Levier (Pratique)", axis=alt.Axis(labelLimit=200)),
            color=alt.Color("sign:N", scale=alt.Scale(domain=["Positif", "Négatif"], range=["#10b981", "#ef4444"]), title="Effet"),
            tooltip=[
                alt.Tooltip("label", title="Dimension"), 
                alt.Tooltip("impact:Q", format=".3f", title="Impact"),
            ]
        ).properties(
            title=f"Drivers de : {target_label}"
        ).interactive()

        return chart.to_dict()
