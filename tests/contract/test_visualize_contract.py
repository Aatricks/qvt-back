from pathlib import Path

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_contract_visualize_returns_spec(client):
    response = client.post(
        "/api/visualize/time_series",
        files={"hr_file": ("hr_valid.csv", (FIXTURES / "hr_valid.csv").read_bytes(), "text/csv")},
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"chart_key", "generated_at", "spec"}
