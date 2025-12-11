from src.viz.registry import factory


def test_supported_keys_listed():
    keys = factory.list_keys()
    assert "time_series" in keys
    assert "likert_distribution" in keys
    assert "correlation_matrix" in keys
