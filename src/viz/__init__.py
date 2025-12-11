from src.viz.registry import factory
from src.viz.strategies.anova_significance import AnovaSignificanceStrategy
from src.viz.strategies.correlation_matrix import CorrelationMatrixStrategy
from src.viz.strategies.distribution_anomalies import DistributionAnomaliesStrategy
from src.viz.strategies.example_new_chart import ExampleNewChartStrategy
from src.viz.strategies.likert_distribution import LikertDistributionStrategy
from src.viz.strategies.time_series import TimeSeriesStrategy

# Register default strategies at import time
factory.register("time_series", TimeSeriesStrategy())
factory.register("likert_distribution", LikertDistributionStrategy())
factory.register("correlation_matrix", CorrelationMatrixStrategy())
factory.register("example_new_chart", ExampleNewChartStrategy())
factory.register("distribution_anomalies", DistributionAnomaliesStrategy())
factory.register("anova_significance", AnovaSignificanceStrategy())
