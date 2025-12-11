import altair as alt


def apply_theme() -> None:
    alt.themes.enable("none")
    alt.data_transformers.disable_max_rows()


BASE_ENCODING = {"color": alt.value("#1f77b4"), "opacity": alt.value(0.9)}
