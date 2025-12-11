import csv
import io
import os
from typing import Optional

import pandas as pd

from src.config.settings import settings
from src.services.validators import enforce_dimensions


class UnsupportedFileType(ValueError):
    pass


def _detect_separator(sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";,|\t")
        return dialect.delimiter
    except csv.Error:
        return ","


def read_bytes_to_df(data: bytes, filename: Optional[str]) -> pd.DataFrame:
    extension = os.path.splitext(filename or "")[1].lower()
    buffer = io.BytesIO(data)
    if extension in {".xls", ".xlsx"}:
        df = pd.read_excel(buffer)
    elif extension in {".csv", ""}:
        sample = data[:1024].decode(errors="ignore")
        sep = _detect_separator(sample)
        df = pd.read_csv(io.BytesIO(data), sep=sep)
    else:
        raise UnsupportedFileType(f"Unsupported file type: {extension or 'unknown'}")
    enforce_dimensions(df, max_rows=settings.max_rows, max_columns=settings.max_columns)
    return df
