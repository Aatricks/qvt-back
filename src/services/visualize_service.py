from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import UploadFile

import src.viz  # noqa: F401 ensures default strategies registered
from src.config.observability import log_error, log_event, timed
from src.config.settings import settings
from src.schemas.datasets import HR_REQUIRED_COLUMNS, SURVEY_REQUIRED_COLUMNS
from src.schemas.visualize import ChartRequest
from src.services import data_loader
from src.services.survey_utils import (
    add_age_band,
    add_seniority_band,
    detect_likert_columns,
)
from src.services.validators import check_likert_range, missing_columns
from src.viz.registry import factory

# Chart keys that require survey-style Likert data.
# Survey data can be provided either as:
#  - wide format (one row per respondent + many Likert item columns like PGC2, EPUI1, ...), or
#  - long format (question_label + response_value).
SURVEY_CHART_KEYS = {
    "likert_distribution",
    "anova_significance",
    "dimension_mean_std_scatter",
    "dimension_ci_bars",
    "clustering_profile",
}


def _has_long_survey_columns(df: pd.DataFrame) -> bool:
    df_norm = {str(c).strip().upper() for c in df.columns}
    return "QUESTION_LABEL" in df_norm and "RESPONSE_VALUE" in df_norm


def _require_survey_data_or_fail(hr_df: pd.DataFrame, survey_df: Optional[pd.DataFrame], chart_key: str) -> None:
    if chart_key not in SURVEY_CHART_KEYS:
        return
    if survey_df is not None:
        return
    # In single-file mode we can reuse hr_df as survey data only if it looks like survey.
    if detect_likert_columns(hr_df) or _has_long_survey_columns(hr_df):
        return
    raise ValidationFailure(
        code="missing_required_columns",
        message="Survey data required for this chart",
        details=[
            "Provide 'survey_file' (wide Likert columns like PGC2/EPUI1... or long columns question_label/response_value)",
            "OR upload a single file containing Likert columns (e.g., PGC2, COM3, ENG1, EPUI1...) as 'hr_file'",
        ],
    )


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
            # Standardize age/seniority grouping early
            hr_df = add_age_band(hr_df)
            hr_df = add_seniority_band(hr_df)
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
                # Standardize age/seniority grouping early
                survey_df = add_age_band(survey_df)
                survey_df = add_seniority_band(survey_df)
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

    # Single-file mode: reuse HR dataset for survey visualizations when it contains Likert columns
    if survey_df is None and (detect_likert_columns(hr_df) or _has_long_survey_columns(hr_df)):
        survey_df = hr_df

    # Enforce survey presence for survey-based charts (before strategy execution)
    _require_survey_data_or_fail(hr_df=hr_df, survey_df=survey_df, chart_key=request.chart_key)

    # 0. Pre-process filters: if a filter has an empty value, treat it as a request for comparison
    filters = request.filters or {}
    config = request.config or {}
    
    clean_filters = {}
    comparison_candidates = []
    
    for k, v in filters.items():
        # Treat None, empty string, or literal "null" as unspecified
        if v is None or str(v).strip() == "" or str(v).lower() == "null":
            comparison_candidates.append(k)
        else:
            clean_filters[k] = v
            
    # Auto-assign empty filters to comparison slots if not already explicitly set in config
    if comparison_candidates:
        config = config.copy()
        if not config.get("segment_field"):
            config["segment_field"] = comparison_candidates.pop(0)
        
        if comparison_candidates and not config.get("facet_field"):
            config["facet_field"] = comparison_candidates.pop(0)

    # 1. Cache lookup before validation/filtering (fast path)
    cache_key = _get_cache_key(
        request.chart_key, 
        {"hr": hr_df, "survey": survey_df}, 
        config, 
        clean_filters
    )
    if cache_key in _SPEC_CACHE:
        log_event("chart_cache_hit", chart_key=request.chart_key)
        cached = _SPEC_CACHE[cache_key].copy()
        cached["generated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        return cached

    if survey_df is not None:
        # Validate Likert range in both supported survey formats.
        if _has_long_survey_columns(survey_df):
            likert_errors = check_likert_range(survey_df, column="response_value")
        else:
            likert_cols = detect_likert_columns(survey_df)
            likert_errors = (
                check_likert_range(survey_df, column=likert_cols) if likert_cols else []
            )

        if likert_errors:
            raise ValidationFailure(
                code="invalid_value_range",
                message="Likert responses must be between 1 and 5",
                details=likert_errors,
            )

    # Apply filters robustly (handling string vs int mismatches) before passing to strategy
    # This ensures that frontend filters (always strings) match backend data (mixed types).
    hr_df = _apply_filters(hr_df, clean_filters)
    if survey_df is not None:
        survey_df = _apply_filters(survey_df, clean_filters)

    with timed("generate_spec"):
        try:
            spec = strategy.generate(
                data={"hr": hr_df, "survey": survey_df},
                config=config,
                filters={},  # Filters already applied
                settings=settings,
            )
        except ValueError as exc:
            # Convert strategy ValueError into a structured validation error so the API returns a 400.
            # Keep error codes aligned with the public contract.
            raise ValidationFailure(
                code="payload_error",
                message="Visualization generation failed",
                details=[str(exc)],
            ) from exc

    log_event("chart_generated", chart_key=request.chart_key)
    result = {
        "chart_key": request.chart_key,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "spec": spec,
    }

    # Populate cache
    if len(_SPEC_CACHE) >= MAX_CACHE_SIZE:
        _SPEC_CACHE.clear() # Simple eviction
    _SPEC_CACHE[cache_key] = result

    return result


# Cache for generated specs to avoid redundant heavy computations
_SPEC_CACHE: Dict[tuple, Any] = {}
MAX_CACHE_SIZE = 100


def _get_cache_key(chart_key: str, data: Dict[str, pd.DataFrame], config: Dict, filters: Dict) -> tuple:
    """Create a stable hashable key for request caching."""
    # Using shape and column tuple as a proxy for dataset identity
    data_id = tuple(
        (k, df.shape, tuple(df.columns)) 
        for k, df in data.items() if df is not None
    )
    # Convert dicts to sorted tuples for hashing
    config_id = tuple(sorted((k, str(v)) for k, v in config.items()))
    filter_id = tuple(sorted((k, str(v)) for k, v in filters.items()))
    return (chart_key, data_id, config_id, filter_id)


def _apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """Filter dataframe robustly, comparing values as strings."""
    if not filters or df.empty:
        return df
    
    # We don't copy immediately; pandas slicing returns a new object usually, 
    # but let's be safe if we modify it. 
    # Actually, we are just subsetting, so no deep copy needed unless we mutate.
    
    for key, value in filters.items():
        if key not in df.columns:
            continue
        
        # Cast column to string and strip whitespace for robust comparison against frontend string values
        # Note: this might be expensive for very large DFs, but ensures correctness.
        mask = df[key].astype(str).str.strip() == str(value).strip()
        df = df[mask]
        
    return df
