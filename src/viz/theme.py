import altair as alt

def apply_theme() -> None:
    """Configures Altair with a professional, modern theme matching the app's UI."""

    FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
    PRIMARY_COLOR = "#4F46E5"
    AXIS_COLOR = "#94A3B8"  # slate-400
    GRID_COLOR = "#E2E8F0"  # slate-200
    LABEL_COLOR = "#475569" # slate-600
    TITLE_COLOR = "#1E293B" # slate-800

    def theme_config():
        return {
            "config": {
                "view": {
                    "stroke": "transparent",
                    "fill": "transparent",
                },
                "background": "transparent",
                "font": FONT,
                "title": {
                    "font": FONT,
                    "fontSize": 16,
                    "fontWeight": 700,
                    "anchor": "start",
                    "color": TITLE_COLOR,
                    "dy": -10,
                },
                "axis": {
                    "domainColor": AXIS_COLOR,
                    "gridColor": GRID_COLOR,
                    "labelColor": LABEL_COLOR,
                    "labelFont": FONT,
                    "labelFontSize": 11,
                    "titleColor": LABEL_COLOR,
                    "titleFont": FONT,
                    "titleFontSize": 12,
                    "titlePadding": 10,
                    "tickColor": AXIS_COLOR,
                },
                "legend": {
                    "labelColor": LABEL_COLOR,
                    "labelFont": FONT,
                    "labelFontSize": 11,
                    "titleColor": TITLE_COLOR,
                    "titleFont": FONT,
                    "titleFontSize": 12,
                    "titlePadding": 8,
                    "symbolSize": 100,
                    "padding": 10,
                },
                "range": {
                    "category": [
                        "#4F46E5", # indigo-600 (Primary)
                        "#10B981", # emerald-500
                        "#F59E0B", # amber-500
                        "#EF4444", # red-500
                        "#8B5CF6", # violet-500
                        "#EC4899", # pink-500
                        "#0EA5E9", # sky-500
                    ],
                    "comparison": [
                        "#1E293B", # slate-800 (Organization/Global)
                        "#4F46E5", # indigo-600
                        "#0EA5E9", # sky-500
                        "#10B981", # emerald-500
                        "#F59E0B", # amber-500
                        "#8B5CF6", # violet-500
                    ],
                    "diverging": [
                        "#DC2626", # red-600
                        "#FCA5A5", # red-300
                        "#F1F5F9", # slate-100 (Neutral)
                        "#93C5FD", # blue-300
                        "#2563EB", # blue-600
                    ],
                    "heatmap": ["#F1F5F9", "#4F46E5"]
                },
                "mark": {
                    "color": PRIMARY_COLOR,
                    "opacity": 0.85,
                },
                "bar": {
                    "cornerRadius": 2,
                    "binSpacing": 1,
                }
            }
        }

    alt.themes.register("qvcti_theme", theme_config)
    alt.theme.enable("qvcti_theme")
    alt.data_transformers.disable_max_rows()

# Helper constants for strategies
LIKERT_COLORS = ["#DC2626", "#FCA5A5", "#F1F5F9", "#93C5FD", "#2563EB"]
COMPARISON_COLORS = ["#1E293B", "#4F46E5", "#0EA5E9", "#10B981", "#F59E0B", "#8B5CF6"]
BASE_ENCODING = {"color": alt.value("#4F46E5"), "opacity": alt.value(0.85)}
