from __future__ import annotations

from typing import Dict, List

import pandas as pd

from src.services.survey_utils import LIKERT_PREFIX_LABELS


def likert_columns_by_prefix(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Group wide-format Likert columns (e.g. PGC2, EPUI1) by their prefix.

    Only prefixes known in `LIKERT_PREFIX_LABELS` are considered.
    """

    prefixes = tuple(LIKERT_PREFIX_LABELS.keys())
    out: Dict[str, List[str]] = {p: [] for p in prefixes}

    for col in df.columns:
        name = str(col).strip()
        upper = name.upper()
        for p in prefixes:
            if upper.startswith(p):
                out[p].append(col)
                break

    # Drop empty groups for convenience.
    return {p: cols for p, cols in out.items() if cols}


def compute_prefix_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-respondent mean score per Likert prefix.

    Returns a new DataFrame with columns like:
      - DIM_<PREFIX> (e.g. DIM_COM, DIM_EPUI)

    Missing/non-numeric values are coerced to NaN; a row’s mean is computed across available items.
    """

    groups = likert_columns_by_prefix(df)
    if not groups:
        raise ValueError("No Likert columns found to compute prefix scores")

    out = pd.DataFrame(index=df.index)
    for prefix, cols in groups.items():
        numeric = df[cols].apply(pd.to_numeric, errors="coerce")
        out[f"DIM_{prefix}"] = numeric.mean(axis=1)

    return out


def prefix_label(prefix: str) -> str:
    return LIKERT_PREFIX_LABELS.get(prefix, prefix)
