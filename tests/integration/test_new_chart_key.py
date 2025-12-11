from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_new_chart_key_happy_path(client):
    response = client.post(
        "/api/visualize/example_new_chart",
        files={"hr_file": ("hr_valid.csv", (FIXTURES / "hr_valid.csv").read_bytes(), "text/csv")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["chart_key"] == "example_new_chart"
    assert "spec" in body


def test_unknown_chart_key_error(client):
    response = client.post(
        "/api/visualize/not_a_chart",
        files={"hr_file": ("hr_valid.csv", (FIXTURES / "hr_valid.csv").read_bytes(), "text/csv")},
    )
    assert response.status_code == 404
    body = response.json()
    assert body["code"] == "invalid_chart_key"
    assert "supported_chart_keys" in body
    assert "time_series" in body["supported_chart_keys"]
