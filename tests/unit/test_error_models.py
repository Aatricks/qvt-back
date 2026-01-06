from src.services.error_builder import build_error


def test_error_builder_structures_payload():
    payload = build_error("invalid_chart_key", "bad key", ["detail"], ["a", "b"])
    assert payload["code"] == "invalid_chart_key"
    assert payload["message"] == "bad key"
    assert payload["details"] == ["detail"]
    assert "supported_chart_keys" in payload
