from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd


class IVisualizationStrategy(ABC):
    """Strategy for producing Vega-Lite spec from dataframes.

    data: {"hr": DataFrame, "survey": Optional[DataFrame]}
    config: chart-specific configuration
    filters: optional filter parameters to apply before visualization

    To add a new chart: create a strategy class in src/viz/strategies/, implement generate,
    document required columns/config, and register the key in src/viz/__init__.py.
    """

    @abstractmethod
    def generate(
        self, data: Dict[str, pd.DataFrame], config: Dict[str, Any], filters: Dict[str, Any], settings: Any
    ) -> Dict[str, Any]:
        ...
