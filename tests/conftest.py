import pytest
from fastapi.testclient import TestClient

from src.api.app import app
import src.viz  # noqa: F401 ensures strategies registered


@pytest.fixture(scope="session")
def client() -> TestClient:
    return TestClient(app)
