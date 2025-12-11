import pandas as pd

from src.services import validators


def test_missing_columns_detects_absent_fields():
    df = pd.DataFrame({"a": [1], "b": [2]})
    missing = validators.missing_columns(df, ["a", "c"])
    assert missing == ["c"]


def test_likert_range_flags_out_of_bounds():
    df = pd.DataFrame({"response_value": [1, 3, 6]})
    issues = validators.check_likert_range(df)
    assert issues
