from fastapi import FastAPI

from src.api.visualize import router as visualize_router

app = FastAPI(title="QVCTi Visualization API")

app.include_router(visualize_router, prefix="/api")


@app.get("/health", tags=["health"])
async def health() -> dict:
    return {"status": "ok"}
