from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict


class Settings(BaseSettings):
    max_rows: int = Field(50000, description="Maximum allowed rows per dataset")
    max_columns: int = Field(200, description="Maximum allowed columns per dataset")
    request_timeout_seconds: int = Field(5, description="Timeout guard for request processing")
    log_level: str = Field("INFO", description="Logging level")
    cors_allow_origins: str = Field(
        "*",
        description="CORS allow origins for the API (use '*' or a comma-separated list)",
    )

    model_config = ConfigDict(env_prefix="QVCTI_", case_sensitive=False)


settings = Settings()
