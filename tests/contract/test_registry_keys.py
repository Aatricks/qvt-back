from src.viz.registry import factory


def test_supported_keys_listed():
    keys = factory.list_keys()
    assert "likert_distribution" in keys
    assert "distribution_anomalies" in keys
    assert "anova_significance" in keys
