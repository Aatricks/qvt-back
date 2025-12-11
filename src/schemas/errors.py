from typing import List, Optional

from pydantic import BaseModel, Field


class ErrorCode:
    INVALID_CHART_KEY = "invalid_chart_key"
    MISSING_REQUIRED_COLUMNS = "missing_required_columns"
    INVALID_VALUE_RANGE = "invalid_value_range"
    INVALID_FILE_TYPE = "invalid_file_type"
    PAYLOAD_ERROR = "payload_error"
    DATASET_TOO_LARGE = "dataset_too_large"


class ErrorResponse(BaseModel):
    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable description")
    details: Optional[List[str]] = Field(default=None, description="Specific field issues")
    supported_chart_keys: Optional[List[str]] = Field(
        default=None, description="Available chart keys when invalid chart key provided"
    )
