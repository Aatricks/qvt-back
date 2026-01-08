
import pytest
import pandas as pd
import altair as alt
from src.viz.strategies.likert_distribution import LikertDistributionStrategy

def test_likert_distribution_structure():
    # Mock data matching the strategy expectations
    data = [
        {"dimension_prefix": "A", "question_label": "Q1", "response_value": 1},
        {"dimension_prefix": "A", "question_label": "Q1", "response_value": 2},
        {"dimension_prefix": "B", "question_label": "Q2", "response_value": 5},
    ]
    df = pd.DataFrame(data)
    
    strategy = LikertDistributionStrategy()
    
    # Generate spec
    spec = strategy.generate(
        data={"survey": df},
        config={"interactive_dimension": True, "segment_field": None, "facet_field": None},
        filters={},
        settings={}
    )
    
    # Check basic structure
    assert spec is not None
    assert "params" in spec
    # We removed the dummy facet, so "facet" should NOT be in spec for this case (facet_field=None)
    assert "facet" not in spec 
    
    # Check params
    params = spec["params"]
    param_names = [p["name"] for p in params]
    assert "dim_select" in param_names, "Renamed parameter 'dim_select' should be present"
    assert "dimension" not in param_names, "Old parameter 'dimension' should not be present"
    
    # Check that filter using 'dim_select' is at the TOP LEVEL
    # This reflects the latest fix to ensure signal visibility.
    assert "transform" in spec
    top_transforms = spec["transform"]
    
    # Check for the filter
    filter_present = False
    for t in top_transforms:
        if "filter" in t:
            f = t["filter"]
            if "dim_select" in str(f):
                filter_present = True
                break
    
    assert filter_present, "Filter referencing 'dim_select' should be at the top level"

def test_likert_distribution_faceted_structure():
    # Mock data
    data = [
        {"dimension_prefix": "A", "question_label": "Q1", "response_value": 1, "group": "G1"},
        {"dimension_prefix": "B", "question_label": "Q2", "response_value": 5, "group": "G2"},
    ]
    df = pd.DataFrame(data)
    
    strategy = LikertDistributionStrategy()
    
    # Generate spec with facet
    spec = strategy.generate(
        data={"survey": df},
        config={"interactive_dimension": True, "facet_field": "group"},
        filters={},
        settings={}
    )
    
    assert "facet" in spec
    assert "spec" in spec
    assert "transform" in spec["spec"], "Transform should be inside the faceted spec"
    
    # Check for the filter
    filter_present = False
    for t in spec["spec"]["transform"]:
        if "filter" in t:
            f = t["filter"]
            if "dim_select" in str(f):
                filter_present = True
                break
    
    assert filter_present, "Filter referencing 'dim_select' should be inside the faceted spec"

def test_likert_distribution_scoping():
    # Test ensuring that the signal usage is correct
    # This is implicitly checked by the structure above.
    pass
