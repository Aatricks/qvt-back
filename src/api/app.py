from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.visualize import router as visualize_router
from src.config.settings import settings

app = FastAPI(title="QVCTi Visualization API")


def _parse_cors_origins(value: str) -> tuple[list[str], bool]:
    raw = (value or "").strip()
    if raw == "*":
        # Credentials are not compatible with wildcard origins.
        return ["*"], False
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    # If user provided no valid origins, default to no origins.
    return origins, True


origins, allow_credentials = _parse_cors_origins(settings.cors_allow_origins)
if origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(visualize_router, prefix="/api")


@app.get("/health", tags=["health"])
async def health() -> dict:
    return {"status": "ok"}
