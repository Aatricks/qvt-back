from pathlib import Path

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_contract_visualize_returns_spec(client):
    response = client.post(
        "/api/visualize/likert_distribution",
        files={"hr_file": ("pov_sample.csv", (FIXTURES / "pov_sample.csv").read_bytes(), "text/csv")},
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"chart_key", "generated_at", "spec"}
