import json
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, Form, UploadFile, status
from fastapi.responses import JSONResponse

from src.schemas.visualize import ChartRequest
from src.services import visualize_service
from src.services.error_builder import build_error
from src.viz.registry import factory
from src.config.observability import log_event

router = APIRouter(tags=["visualize"])


@router.get("/visualize/supported-keys", response_model=list[str])
async def supported_keys() -> list[str]:
    return factory.list_keys()


@router.post("/visualize/{chart_type}", response_model=Dict[str, Any])
async def visualize(
    chart_type: str,
    hr_file: UploadFile = File(...),
    survey_file: Optional[UploadFile] = File(None),
    filters: Optional[str] = Form(None),
    config: Optional[str] = Form(None),
) -> Dict[str, Any]:
    try:
        parsed_filters = json.loads(filters) if filters else None
        parsed_config = json.loads(config) if config else None
    except json.JSONDecodeError as exc:
        error = build_error(
            code="payload_error",
            message="Invalid JSON payload in filters/config",
            details=[str(exc)],
            supported_keys=factory.list_keys(),
        )
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=error)

    # Debug log: record parsed filters/config for troubleshooting (non-blocking)
    try:
        log_event(
            "visualize.request_parsed",
            chart_key=chart_type,
            filters=parsed_filters,
            config=parsed_config,
        )
    except Exception:
        # Swallow any logging errors so they don't affect the API response
        pass

    request = ChartRequest(
        chart_key=chart_type,
        filters=parsed_filters,
        config=parsed_config,
    )
    try:
        spec = await visualize_service.generate_chart(
            request=request, hr_file=hr_file, survey_file=survey_file
        )
        return spec
    except visualize_service.UnknownChartKeyError as exc:
        error = build_error(
            code="invalid_chart_key",
            message=str(exc),
            details=[],
            supported_keys=factory.list_keys(),
        )
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content=error)
    except visualize_service.ValidationFailure as exc:
        error = build_error(
            code=exc.code,
            message=exc.message,
            details=exc.details,
            supported_keys=factory.list_keys(),
        )
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=error)
