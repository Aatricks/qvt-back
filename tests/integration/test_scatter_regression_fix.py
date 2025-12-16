from pathlib import Path
import json

FIXTURES = Path(__file__).parent.parent / "fixtures"

def test_scatter_regression_invalid_method(client):
    """
    Test that providing an invalid regression method (like 'spearman')
    does not crash the backend (500), but falls back to a default (linear) or succeeds.
    """
    files = {"hr_file": ("pov_sample.csv", (FIXTURES / "pov_sample.csv").read_bytes(), "text/csv")}
    
    # We pass 'spearman' which causes a SchemaValidationError in unmodified Altair code
    config = {
        "x_field": "EPUI1",
        "y_field": "EPUI2",
        "regression": True,
        "method": "spearman"
    }
    
    response = client.post(
        "/api/visualize/scatter_regression", 
        files=files, 
        data={"config": json.dumps(config)}
    )
    
    assert response.status_code == 200
    payload = response.json()
    assert payload["chart_key"] == "scatter_regression"
    spec = payload["spec"]
    
    # Verify in the spec that the transform_regression method is 'linear' (our fallback)
    # The spec structure for layers: layer -> [..., {transform: [{regression: ..., method: 'linear'}]}]
    found_regression = False
    
    # Helper to search for the regression transform in the spec
    # The spec might be a layer chart.
    layers = spec.get("layer", [])
    for layer in layers:
        transforms = layer.get("transform", [])
        for t in transforms:
            if "regression" in t:
                assert t["method"] == "linear"
                found_regression = True
    
    assert found_regression, "Regression transform should be present with method='linear'"
