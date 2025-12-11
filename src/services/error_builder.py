from typing import List, Optional

from src.schemas.errors import ErrorResponse


def build_error(
    code: str,
    message: str,
    details: Optional[List[str]] = None,
    supported_keys: Optional[List[str]] = None,
) -> dict:
    return ErrorResponse(
        code=code,
        message=message,
        details=details or [],
        supported_chart_keys=supported_keys,
    ).dict()
