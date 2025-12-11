from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import UploadFile

from src.config.observability import log_error, log_event, timed
from src.config.settings import settings
from src.schemas.datasets import HR_REQUIRED_COLUMNS, SURVEY_REQUIRED_COLUMNS
from src.schemas.visualize import ChartRequest
from src.services import data_loader
from src.services.survey_utils import detect_likert_columns
from src.services.validators import check_likert_range, missing_columns
from src.viz.registry import factory
import src.viz  # noqa: F401 ensures default strategies registered


class UnknownChartKeyError(KeyError):
    pass


class ValidationFailure(ValueError):
    def __init__(self, code: str, message: str, details: Optional[list[str]] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or []


async def _read_upload(file: UploadFile) -> bytes:
    return await file.read()


async def generate_chart(
    request: ChartRequest, hr_file: UploadFile, survey_file: Optional[UploadFile]
) -> Dict[str, Any]:
    strategy = factory.get(request.chart_key)
    if not strategy:
        log_error("invalid_chart_key", f"Unsupported chart key: {request.chart_key}")
        raise UnknownChartKeyError(f"Unsupported chart key: {request.chart_key}")

    with timed("load_hr_dataset"):
        hr_bytes = await _read_upload(hr_file)
        try:
            hr_df = data_loader.read_bytes_to_df(hr_bytes, hr_file.filename)
        except ValueError as exc:
            raise ValidationFailure(
                code="dataset_too_large", message="Dataset too large", details=[str(exc)]
            ) from exc

    missing_hr = missing_columns(hr_df, HR_REQUIRED_COLUMNS)
    if missing_hr:
        raise ValidationFailure(
            code="missing_required_columns",
            message="Missing required HR columns",
            details=missing_hr,
        )

    survey_df: Optional[pd.DataFrame] = None
    if survey_file:
        with timed("load_survey_dataset"):
            survey_bytes = await _read_upload(survey_file)
            try:
                survey_df = data_loader.read_bytes_to_df(survey_bytes, survey_file.filename)
            except ValueError as exc:
                raise ValidationFailure(
                    code="dataset_too_large", message="Dataset too large", details=[str(exc)]
                ) from exc
        missing_survey = missing_columns(survey_df, SURVEY_REQUIRED_COLUMNS)
        if missing_survey:
            raise ValidationFailure(
                code="missing_required_columns",
                message="Missing required survey columns",
                details=missing_survey,
            )
    elif detect_likert_columns(hr_df):
        # Single-file mode: reuse HR dataset for survey visualizations
        survey_df = hr_df

    if survey_df is not None:
        likert_errors = check_likert_range(survey_df, column=detect_likert_columns(survey_df))
        if likert_errors:
            raise ValidationFailure(
                code="invalid_value_range",
                message="Likert responses must be between 1 and 5",
                details=likert_errors,
            )

    with timed("generate_spec"):
        spec = strategy.generate(
            data={"hr": hr_df, "survey": survey_df},
            config=request.config or {},
            filters=request.filters or {},
            settings=settings,
        )

    log_event("chart_generated", chart_key=request.chart_key)
    return {
        "chart_key": request.chart_key,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "spec": spec,
    }
