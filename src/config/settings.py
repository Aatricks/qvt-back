from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    max_rows: int = Field(50000, description="Maximum allowed rows per dataset")
    max_columns: int = Field(20, description="Maximum allowed columns per dataset")
    request_timeout_seconds: int = Field(5, description="Timeout guard for request processing")
    log_level: str = Field("INFO", description="Logging level")

    class Config:
        env_prefix = "QVCTI_"
        case_sensitive = False


settings = Settings()
