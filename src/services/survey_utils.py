from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

# Socio-demographic columns expected in the POV survey export
SOCIO_COLUMNS: List[str] = [
    "ID",
    "Sexe",
    "Age",
    "Contrat",
    "Temps",
    "Encadre",
    "Ancienne",
    "Secteur",
    "TailleOr",
]

# Map of Likert prefixes to human-readable labels
LIKERT_PREFIX_LABELS: Dict[str, str] = {
    "POV": "Pratiques organisationnelles vertueuses",
    "PGC": "Pratiques de gestion de carrière",
    "CSA": "Pratiques de santé et de sécurité",
    "EVPVP": "Pratiques de conciliation vie privée / personnelle",
    "RECO": "Pratiques de reconnaissance",
    "COM": "Pratiques de communication",
    "DL": "Pratiques de dialogue social",
    "PPD": "Participation à la prise de décision",
    "JUST": "Pratiques de justice organisationnelle",
    "PI": "Pratiques d'inclusion",
    "PD": "Pratiques de développement durable",
    "ENG": "Engagement au travail",
    "EPUI": "Épuisement émotionnel",
    "CSE": "Conditions de santé et sécurité",
}


def _normalize_column_name(col: str) -> str:
    return str(col).strip()


def _extract_prefix(column: str) -> str:
    upper = _normalize_column_name(column).upper()
    for prefix in LIKERT_PREFIX_LABELS.keys():
        if upper.startswith(prefix):
            return prefix
    return upper.rstrip("0123456789")


def available_demographics(df: pd.DataFrame) -> List[str]:
    normalized = {_normalize_column_name(col).upper(): col for col in df.columns}
    return [normalized[name.upper()] for name in SOCIO_COLUMNS if name.upper() in normalized]


def detect_likert_columns(df: pd.DataFrame) -> List[str]:
    prefixes = tuple(LIKERT_PREFIX_LABELS.keys())
    likert_cols: List[str] = []
    for col in df.columns:
        upper = _normalize_column_name(col).upper()
        if upper.startswith(prefixes):
            likert_cols.append(col)
    return likert_cols


def friendly_question_label(column: str) -> str:
    upper = _normalize_column_name(column).upper()
    for prefix, label in LIKERT_PREFIX_LABELS.items():
        if upper.startswith(prefix):
            return f"{column} ({label})"
    return column


def to_likert_long(
    df: pd.DataFrame,
    likert_columns: Sequence[str] | None = None,
    extra_id_vars: Sequence[str] | None = None,
) -> pd.DataFrame:
    likert_cols = list(likert_columns) if likert_columns else detect_likert_columns(df)
    id_vars = available_demographics(df)
    if extra_id_vars:
        for col in extra_id_vars:
            if col in df.columns and col not in id_vars:
                id_vars.append(col)
    if "AgeClasse" in df.columns and "AgeClasse" not in id_vars:
        id_vars = [*id_vars, "AgeClasse"]
    melted = df.melt(
        id_vars=id_vars,
        value_vars=likert_cols,
        var_name="question_label",
        value_name="response_value",
    )
    melted["dimension_prefix"] = melted["question_label"].apply(_extract_prefix)
    melted["question_label"] = melted["question_label"].apply(friendly_question_label)
    return melted


def add_age_band(df: pd.DataFrame) -> pd.DataFrame:
    if "Age" not in df.columns:
        return df
    result = df.copy()
    result["Age"] = pd.to_numeric(result["Age"], errors="coerce")
    result["AgeClasse"] = pd.cut(
        result["Age"],
        bins=[0, 29, 39, 49, 59, np.inf],
        labels=["<30", "30-39", "40-49", "50-59", "60+"],
    )
    return result


def classify_distribution(series: pd.Series) -> str:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return "insufficient_data"
    counts = clean.value_counts(normalize=True)
    if counts.empty:
        return "insufficient_data"
    if counts.max() - counts.min() < 0.1:
        return "uniform"
    skewness = clean.skew()
    if skewness > 0.5:
        return "skew_right"
    if skewness < -0.5:
        return "skew_left"
    return "balanced"
