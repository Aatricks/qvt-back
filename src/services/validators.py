from typing import Iterable, List, Sequence

import pandas as pd


def missing_columns(df: pd.DataFrame, required: Iterable[str]) -> List[str]:
    required_set = {col for col in required}
    return sorted(list(required_set.difference(set(df.columns))))


def enforce_dimensions(df: pd.DataFrame, max_rows: int, max_columns: int) -> None:
    if len(df.index) > max_rows or len(df.columns) > max_columns:
        raise ValueError(
            f"Dataset too large: rows={len(df.index)}, cols={len(df.columns)}, "
            f"limits rows<={max_rows}, cols<={max_columns}"
        )


def check_likert_range(df: pd.DataFrame, column: str = "response_value") -> List[str]:
    if column not in df.columns:
        return [column]
    invalid = df[(df[column] < 1) | (df[column] > 5)]
    if invalid.empty:
        return []
    return [f"{column} out of range 1-5 in {len(invalid)} rows"]


def ensure_numeric(df: pd.DataFrame, columns: Sequence[str]) -> List[str]:
    issues: List[str] = []
    for col in columns:
        if col not in df.columns:
            issues.append(col)
            continue
        non_numeric = pd.to_numeric(df[col], errors="coerce")
        if non_numeric.isna().all():
            issues.append(f"{col} not numeric")
    return issues
