import pytest
from fastapi.testclient import TestClient

import src.viz  # noqa: F401 ensures strategies registered
from src.api.app import app


@pytest.fixture(scope="session")
def client() -> TestClient:
    return TestClient(app)
