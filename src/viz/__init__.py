from src.viz.registry import factory
from src.viz.strategies.anova_significance import AnovaSignificanceStrategy
from src.viz.strategies.correlation_matrix import CorrelationMatrixStrategy
from src.viz.strategies.demographic_distribution import DemographicDistributionStrategy
from src.viz.strategies.dimension_mean_std_scatter import DimensionMeanStdScatterStrategy
from src.viz.strategies.dimension_summary import PracticeDimensionSummaryStrategy
from src.viz.strategies.dimension_ci_bars import DimensionCIBarsStrategy
from src.viz.strategies.distribution_anomalies import DistributionAnomaliesStrategy
from src.viz.strategies.eng_epui_quadrants import EngEpuiQuadrantsStrategy
from src.viz.strategies.action_priority_index import ActionPriorityIndexStrategy
from src.viz.strategies.leverage_scatter import LeverageScatterStrategy
from src.viz.strategies.importance_performance_matrix import ImportancePerformanceMatrixStrategy
from src.viz.strategies.predictive_simulation import PredictiveSimulationStrategy
from src.viz.strategies.clustering_profile import ClusteringProfileStrategy
from src.viz.strategies.likert_distribution import LikertDistributionStrategy

# Register default strategies at import time
factory.register("likert_distribution", LikertDistributionStrategy())
factory.register("correlation_matrix", CorrelationMatrixStrategy())
factory.register("distribution_anomalies", DistributionAnomaliesStrategy())
factory.register("anova_significance", AnovaSignificanceStrategy())
factory.register("dimension_summary", PracticeDimensionSummaryStrategy())
factory.register("dimension_mean_std_scatter", DimensionMeanStdScatterStrategy())
factory.register("dimension_ci_bars", DimensionCIBarsStrategy())
factory.register("eng_epui_quadrants", EngEpuiQuadrantsStrategy())
factory.register("demographic_distribution", DemographicDistributionStrategy())
factory.register("action_priority_index", ActionPriorityIndexStrategy())
factory.register("leverage_scatter", LeverageScatterStrategy())
factory.register("importance_performance_matrix", ImportancePerformanceMatrixStrategy())
factory.register("predictive_simulation", PredictiveSimulationStrategy())
factory.register("clustering_profile", ClusteringProfileStrategy())