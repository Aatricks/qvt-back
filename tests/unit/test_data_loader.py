from pathlib import Path

import pytest

from src.services import data_loader

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_read_csv_with_autodetect():
    data = (FIXTURES / "pov_sample.csv").read_bytes()
    df = data_loader.read_bytes_to_df(data, "pov_sample.csv")
    assert not df.empty
    assert {"ID", "Sexe", "Age"}.issubset(df.columns)


def test_reject_unsupported_extension():
    with pytest.raises(data_loader.UnsupportedFileType):
        data_loader.read_bytes_to_df(b"bad", "data.txt")
