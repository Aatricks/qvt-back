from src.viz.registry import factory
from src.viz.strategies.anova_significance import AnovaSignificanceStrategy
from src.viz.strategies.benchmark_bullet import BenchmarkBulletStrategy
from src.viz.strategies.correlation_matrix import CorrelationMatrixStrategy
from src.viz.strategies.demographic_distribution import DemographicDistributionStrategy
from src.viz.strategies.dimension_boxplot import DimensionBoxplotStrategy
from src.viz.strategies.dimension_heatmap import DimensionHeatmapStrategy
from src.viz.strategies.dimension_mean_std_scatter import DimensionMeanStdScatterStrategy
from src.viz.strategies.dimension_summary import PracticeDimensionSummaryStrategy
from src.viz.strategies.dimension_ci_bars import DimensionCIBarsStrategy
from src.viz.strategies.distribution_anomalies import DistributionAnomaliesStrategy
from src.viz.strategies.eng_epui_quadrants import EngEpuiQuadrantsStrategy
from src.viz.strategies.example_new_chart import ExampleNewChartStrategy
from src.viz.strategies.action_priority_index import ActionPriorityIndexStrategy
from src.viz.strategies.leverage_scatter import LeverageScatterStrategy
from src.viz.strategies.importance_performance_matrix import ImportancePerformanceMatrixStrategy

from src.viz.strategies.likert_distribution import LikertDistributionStrategy
from src.viz.strategies.likert_item_heatmap import LikertItemHeatmapStrategy
from src.viz.strategies.scatter_regression import ScatterRegressionStrategy
from src.viz.strategies.time_series import TimeSeriesStrategy
from src.viz.strategies.time_series_ci import TimeSeriesCIStreamingStrategy

# Register default strategies at import time
factory.register("time_series", TimeSeriesStrategy())
factory.register("time_series_ci", TimeSeriesCIStreamingStrategy())
factory.register("likert_distribution", LikertDistributionStrategy())
factory.register("likert_item_heatmap", LikertItemHeatmapStrategy())
factory.register("correlation_matrix", CorrelationMatrixStrategy())
factory.register("distribution_anomalies", DistributionAnomaliesStrategy())
factory.register("anova_significance", AnovaSignificanceStrategy())
factory.register("dimension_summary", PracticeDimensionSummaryStrategy())
factory.register("dimension_heatmap", DimensionHeatmapStrategy())
factory.register("dimension_boxplot", DimensionBoxplotStrategy())
factory.register("dimension_mean_std_scatter", DimensionMeanStdScatterStrategy())
factory.register("dimension_ci_bars", DimensionCIBarsStrategy())
factory.register("scatter_regression", ScatterRegressionStrategy())
factory.register("eng_epui_quadrants", EngEpuiQuadrantsStrategy())
factory.register("demographic_distribution", DemographicDistributionStrategy())
factory.register("benchmark_bullet", BenchmarkBulletStrategy())
factory.register("example_new_chart", ExampleNewChartStrategy())
factory.register("action_priority_index", ActionPriorityIndexStrategy())
factory.register("leverage_scatter", LeverageScatterStrategy())
factory.register("importance_performance_matrix", ImportancePerformanceMatrixStrategy())
