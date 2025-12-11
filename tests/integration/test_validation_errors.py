from pathlib import Path

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_missing_columns_error(client):
    minimal_hr = b"ID,Sexe\n1,1"
    response = client.post(
        "/api/visualize/likert_distribution",
        files={"hr_file": ("hr_invalid.csv", minimal_hr, "text/csv")},
    )
    assert response.status_code == 400
    payload = response.json()
    assert payload["code"] == "missing_required_columns"


def test_likert_range_error(client):
    bad_survey = b"ID,Sexe,PGC2\n1,1,6"
    response = client.post(
        "/api/visualize/likert_distribution",
        files={
            "hr_file": ("pov_sample.csv", (FIXTURES / "pov_sample.csv").read_bytes(), "text/csv"),
            "survey_file": ("survey_bad.csv", bad_survey, "text/csv"),
        },
    )
    assert response.status_code == 400
    payload = response.json()
    assert payload["code"] == "invalid_value_range"
