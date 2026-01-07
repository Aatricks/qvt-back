import json
from pathlib import Path
import pytest

FIXTURES = Path(__file__).parent.parent / "fixtures"

def test_likert_distribution_faceting(client):
    files = {
        "hr_file": ("pov_sample.csv", (FIXTURES / "pov_sample.csv").read_bytes(), "text/csv")
    }
    # Test faceting by 'Sexe'
    config = json.dumps({"facet_field": "Sexe"})
    response = client.post("/api/visualize/likert_distribution", files=files, data={"config": config})
    
    assert response.status_code == 200
    payload = response.json()
    spec = payload["spec"]
    
    # Check if facet is in the spec
    assert "facet" in spec
    assert spec["facet"]["column"]["field"] == "Sexe"

def test_dimension_mean_std_scatter_segmentation(client):
    files = {
        "hr_file": ("pov_sample.csv", (FIXTURES / "pov_sample.csv").read_bytes(), "text/csv")
    }
    # Test segmentation by 'Sexe'
    config = json.dumps({"segment_field": "Sexe"})
    response = client.post("/api/visualize/dimension_mean_std_scatter", files=files, data={"config": config})
    
    assert response.status_code == 200
    payload = response.json()
    spec = payload["spec"]
    
    # In LayerChart or UnitChart, check for color encoding
    # For scatter plot, it should be in the encoding of the points layer
    if "layer" in spec:
        encodings = spec["layer"][0]["encoding"]
    else:
        encodings = spec["encoding"]
        
    assert encodings["color"]["field"] == "Sexe"

def test_dimension_ci_bars_facet_and_segment(client):
    files = {
        "hr_file": ("pov_sample.csv", (FIXTURES / "pov_sample.csv").read_bytes(), "text/csv")
    }
    # Test both facet and segment
    config = json.dumps({"facet_field": "Secteur", "segment_field": "Sexe"})
    response = client.post("/api/visualize/dimension_ci_bars", files=files, data={"config": config})
    
    assert response.status_code == 200
    payload = response.json()
    spec = payload["spec"]
    
    assert "facet" in spec
    assert spec["facet"]["column"]["field"] == "Secteur"
    # Color should be Sexe
    # In faceted charts, the inner spec is in 'spec'
    inner_spec = spec["spec"]
    if "layer" in inner_spec:
        # It's an errorbar + point layer
        color_field = inner_spec["layer"][0]["encoding"]["color"]["field"]
    else:
        color_field = inner_spec["encoding"]["color"]["field"]
    assert color_field == "Sexe"

def test_empty_filters_auto_comparison(client):
    files = {
        "hr_file": ("pov_sample.csv", (FIXTURES / "pov_sample.csv").read_bytes(), "text/csv")
    }
    # Provide filters with empty values
    filters = json.dumps({"Sexe": "", "Secteur": None})
    response = client.post("/api/visualize/dimension_ci_bars", files=files, data={"filters": filters})
    
    assert response.status_code == 200
    payload = response.json()
    spec = payload["spec"]
    
    # Sexe should have become segment_field (color)
    # Secteur should have become facet_field (facet)
    assert "facet" in spec
    assert spec["facet"]["column"]["field"] == "Secteur"
    
    inner_spec = spec["spec"]
    if "layer" in inner_spec:
        color_field = inner_spec["layer"][0]["encoding"]["color"]["field"]
    else:
        color_field = inner_spec["encoding"]["color"]["field"]
    assert color_field == "Sexe"
