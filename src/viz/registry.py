from typing import Dict, Optional

from src.viz.base import IVisualizationStrategy


class VisualizationFactory:
    def __init__(self) -> None:
        self._strategies: Dict[str, IVisualizationStrategy] = {}

    def register(self, key: str, strategy: IVisualizationStrategy) -> None:
        self._strategies[key] = strategy

    def get(self, key: str) -> Optional[IVisualizationStrategy]:
        return self._strategies.get(key)

    def list_keys(self) -> list[str]:
        return sorted(self._strategies.keys())


factory = VisualizationFactory()
