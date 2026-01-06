from typing import Iterable, List, Sequence, Union

import pandas as pd


def missing_columns(df: pd.DataFrame, required: Iterable[str]) -> List[str]:
    df_norm = {str(col).strip().upper() for col in df.columns}
    missing = []
    for col in required:
        if str(col).strip().upper() not in df_norm:
            missing.append(col)
    return sorted(missing)


def enforce_dimensions(df: pd.DataFrame, max_rows: int, max_columns: int) -> None:
    if len(df.index) > max_rows or len(df.columns) > max_columns:
        raise ValueError(
            f"Dataset too large: rows={len(df.index)}, cols={len(df.columns)}, "
            f"limits rows<={max_rows}, cols<={max_columns}"
        )


def check_likert_range(
    df: pd.DataFrame, column: Union[str, Sequence[str]] = "response_value"
) -> List[str]:
    columns = [column] if isinstance(column, str) else list(column)
    issues: List[str] = []
    for col in columns:
        if col not in df.columns:
            issues.append(col)
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        invalid = df[(numeric < 1) | (numeric > 5)]
        if not invalid.empty:
            issues.append(f"{col} out of range 1-5 in {len(invalid)} rows")
    return issues


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
