# qvt-back/tests/unit/test_chart_axis_layout.py

from pathlib import Path

import pandas as pd

from src.viz.strategies.dimension_ci_bars import DimensionCIBarsStrategy

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _load_pov() -> pd.DataFrame:
    """Load the POV sample fixture used across visualization tests."""
    return pd.read_csv(FIXTURES / "pov_sample.csv")


def _find_encoding_with_y(spec: dict) -> dict:
    """
    Recursively search a Vega-Lite spec dictionary for an `encoding` object that
    contains a `y` encoding and return that encoding dict.

    Raises an assertion error if none is found.
    """
    def _search(obj):
        if not isinstance(obj, dict):
            return None
        enc = obj.get("encoding")
        if isinstance(enc, dict) and "y" in enc:
            return enc
        for layer in obj.get("layer", []):
            found = _search(layer)
            if found is not None:
                return found
        return None

    result = _search(spec)
    assert result is not None, "No layer/encoding with 'y' found in spec"
    return result


def _has_height_step(spec: dict, step: int = 22) -> bool:
    """
    Recursively check whether the spec (or any nested layer) contains a
    'height' property with the provided 'step' value.
    """
    if not isinstance(spec, dict):
        return False
    height = spec.get("height")
    if isinstance(height, dict) and height.get("step") == step:
        return True
    for layer in spec.get("layer", []):
        if _has_height_step(layer, step=step):
            return True
    return False


def _find_left_padding(spec: dict) -> float | None:
    """
    Recursively search for a 'padding' dict with a 'left' value and return it (or None).
    """
    if not isinstance(spec, dict):
        return None
    pad = spec.get("padding")
    if isinstance(pad, dict) and pad.get("left") is not None:
        return pad.get("left")
    for layer in spec.get("layer", []):
        val = _find_left_padding(layer)
        if val is not None:
            return val
    return None




def test_dimension_ci_bars_axis_and_height():
    df = _load_pov()
    strat = DimensionCIBarsStrategy()
    spec = strat.generate({"survey": df}, {}, {}, {})
    enc = _find_encoding_with_y(spec)

    axis = enc["y"].get("axis") if isinstance(enc.get("y"), dict) else None
    assert axis is not None, "dimension_ci_bars: 'y' encoding must contain an 'axis' object"
    assert axis.get("labelLimit") == 260, "dimension_ci_bars: expected y axis labelLimit == 260"
    assert axis.get("labelPadding") == 8, "dimension_ci_bars: expected y axis labelPadding == 8"
    # Bars should be anchored to the left domain bound via x2=1 (default Likert lower bound)
    x2 = enc.get("x2")
    assert x2 is not None, "dimension_ci_bars: expected 'x2' anchor to be present"
    assert x2.get("datum") in (1, 1.0), "dimension_ci_bars: expected 'x2' anchor datum to be 1"
    left_pad = _find_left_padding(spec)
    assert left_pad is not None and float(left_pad) >= 120, "dimension_ci_bars: expected left padding >= 120"
    assert _has_height_step(spec, step=22), "dimension_ci_bars: expected chart height to include {'step': 22}"
