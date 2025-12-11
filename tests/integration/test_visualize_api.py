from pathlib import Path

import pytest

from src.viz.registry import factory

FIXTURES = Path(__file__).parent.parent / "fixtures"


@pytest.mark.parametrize(
    "chart_key",
    ["likert_distribution", "distribution_anomalies", "anova_significance"],
)
def test_visualize_success(client, chart_key):
    files = {"hr_file": ("pov_sample.csv", (FIXTURES / "pov_sample.csv").read_bytes(), "text/csv")}
    response = client.post(f"/api/visualize/{chart_key}", files=files)
    assert response.status_code == 200
    payload = response.json()
    assert payload["chart_key"] == chart_key
    assert "spec" in payload


def test_visualize_unknown_key(client):
    response = client.post(
        "/api/visualize/unknown",
        files={"hr_file": ("pov_sample.csv", (FIXTURES / "pov_sample.csv").read_bytes(), "text/csv")},
    )
    assert response.status_code == 404
    payload = response.json()
    assert payload["code"] == "invalid_chart_key"
    assert factory.list_keys()
