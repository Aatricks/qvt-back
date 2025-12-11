from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ChartRequest(BaseModel):
    chart_key: str = Field(..., description="Registry key selecting the visualization strategy")
    filters: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional filters/segments applied before visualization"
    )
    config: Optional[Dict[str, Any]] = Field(
        default=None, description="Chart-specific configuration such as measure field"
    )


class VisualizationSpec(BaseModel):
    chart_key: str
    spec: Dict[str, Any]
    generated_at: datetime


class ValidationError(BaseModel):
    code: str
    message: str
    details: Optional[list[str]] = None
    supported_chart_keys: Optional[list[str]] = None
