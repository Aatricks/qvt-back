from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _find_encoding_with_y(spec: dict) -> dict | None:
    """
    Recursively search the Vega-Lite spec dict for an encoding that contains a 'y' entry.
    Returns the encoding dict (i.e. value of 'encoding') if found, otherwise None.
    """
    if not isinstance(spec, dict):
        return None

    enc = spec.get("encoding")
    if isinstance(enc, dict) and "y" in enc:
        return enc

    for layer in spec.get("layer", []) or []:
        found = _find_encoding_with_y(layer)
        if found is not None:
            return found

    return None


def _has_height_step(spec: dict, step: int = 22) -> bool:
    """
    Recursively check whether the spec (or any nested layer) has a 'height' property
    with a dict value containing {'step': <step>}.
    """
    if not isinstance(spec, dict):
        return False

    h = spec.get("height")
    if isinstance(h, dict) and h.get("step") == step:
        return True

    for layer in spec.get("layer", []) or []:
        if _has_height_step(layer, step=step):
            return True

    return False


@pytest.mark.parametrize(
    "chart_key",
    [
        "dimension_ci_bars",
    ],
)
def test_visualize_axis_layout(client, chart_key):
    """
    Ensure the backend returns Vega-Lite specs that:
      - include a y-axis encoding with 'labelLimit' and 'labelPadding' set,
      - include a consistent 'height' step to avoid bars overlapping labels.
    """
    files = {"hr_file": ("pov_sample.csv", (FIXTURES / "pov_sample.csv").read_bytes(), "text/csv")}
    response = client.post(f"/api/visualize/{chart_key}", files=files)
    assert response.status_code == 200, f"API request failed for {chart_key}: {response.text}"

    payload = response.json()
    assert payload["chart_key"] == chart_key
    assert "spec" in payload

    spec = payload["spec"]
    enc = _find_encoding_with_y(spec)
    assert enc is not None, f"No 'y' encoding found in spec for {chart_key}"

    # The 'y' encoding must include an 'axis' with both labelLimit and labelPadding.
    y_enc = enc.get("y")
    assert isinstance(y_enc, dict), f"Unexpected 'y' encoding shape for {chart_key}"
    axis = y_enc.get("axis")
    assert isinstance(axis, dict), f"Missing 'axis' in y-encoding for {chart_key}"
    assert axis.get("labelLimit") == 260, f"{chart_key}: expected axis.labelLimit == 260"
    assert axis.get("labelPadding") == 8, f"{chart_key}: expected axis.labelPadding == 8"

    # Find a layer with an x2 anchor to ensure bars are not drawn starting from a baseline outside the domain.
    def _find_bar_with_x2(obj):
        if not isinstance(obj, dict):
            return None
        enc2 = obj.get("encoding")
        if isinstance(enc2, dict) and "x" in enc2 and "x2" in enc2:
            return enc2
        for layer in obj.get("layer", []) or []:
            found = _find_bar_with_x2(layer)
            if found is not None:
                return found
        return None

    def _find_left_padding(obj):
        """
        Recursively find a 'padding' dict with a 'left' key and return its numeric value (or None).
        This helps verify that the chart reserves left space for y-axis labels.
        """
        if not isinstance(obj, dict):
            return None
        pad = obj.get("padding")
        if isinstance(pad, dict) and pad.get("left") is not None:
            return pad.get("left")
        for layer in obj.get("layer", []) or []:
            val = _find_left_padding(layer)
            if val is not None:
                return val
        return None

    bar_enc = _find_bar_with_x2(spec)
    assert isinstance(bar_enc, dict), f"{chart_key}: expected a bar encoding with 'x2' anchor"
    x2_obj = bar_enc.get("x2")
    assert isinstance(x2_obj, dict) and x2_obj.get("value") is not None, f"{chart_key}: expected x2 anchor to have a numeric 'value'"
    # Ensure x2 anchor is non-negative
    assert float(x2_obj.get("value")) >= 0.0, f"{chart_key}: expected non-negative x2 anchor value"

    # Ensure left padding has been set to prevent bars overlapping y labels
    left_pad = _find_left_padding(spec)
    assert left_pad is not None and float(left_pad) >= 120, f"{chart_key}: expected left padding >= 120"

    # Ensure a consistent height step is present to space bars and labels properly.
    assert _has_height_step(spec, step=22), f"{chart_key}: expected chart height to include {{'step': 22}}"
