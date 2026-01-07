
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
    assert "facet" in spec
    assert "spec" in spec
    
    # Check params
    params = spec["params"]
    param_names = [p["name"] for p in params]
    assert "dim_select" in param_names, "Renamed parameter 'dim_select' should be present"
    assert "dimension" not in param_names, "Old parameter 'dimension' should not be present"
    
    # Check that 'constant' is NOT used in transform_calculate inside the spec (since we moved it to pandas)
    # It might be in the data though.
    # The inner spec transform should NOT calculate constant.
    inner_spec = spec["spec"]
    transforms = inner_spec.get("transform", [])
    calc_constant = any(t.get("calculate") == "1" and t.get("as") == "constant" for t in transforms)
    assert not calc_constant, "constant should not be calculated in inner spec transform"
    
    # Check that filter using 'dim_select' is at the TOP LEVEL (facet chart)
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

def test_likert_distribution_scoping():
    # Test ensuring that the signal usage is correct
    # This is implicitly checked by the structure above.
    pass
