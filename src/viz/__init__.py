from src.viz.registry import factory
from src.viz.strategies.anova_significance import AnovaSignificanceStrategy
from src.viz.strategies.clustering_profile import ClusteringProfileStrategy
from src.viz.strategies.correlation_matrix import CorrelationMatrixStrategy
from src.viz.strategies.demographic_distribution import DemographicDistributionStrategy
from src.viz.strategies.dimension_ci_bars import DimensionCIBarsStrategy
from src.viz.strategies.dimension_mean_std_scatter import DimensionMeanStdScatterStrategy
from src.viz.strategies.likert_distribution import LikertDistributionStrategy
from src.viz.strategies.action_priority_index import ActionPriorityIndexStrategy

# Register default strategies at import time
factory.register("likert_distribution", LikertDistributionStrategy())
factory.register("correlation_matrix", CorrelationMatrixStrategy())
factory.register("anova_significance", AnovaSignificanceStrategy())
factory.register("dimension_mean_std_scatter", DimensionMeanStdScatterStrategy())
factory.register("dimension_ci_bars", DimensionCIBarsStrategy())
factory.register("demographic_distribution", DemographicDistributionStrategy())
factory.register("clustering_profile", ClusteringProfileStrategy())
factory.register("action_priority_index", ActionPriorityIndexStrategy())
