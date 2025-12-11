import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator

logger = logging.getLogger("qvcti")
logging.basicConfig(level=logging.INFO)


@contextmanager
def timed(operation: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info("op=%s duration_ms=%.2f", operation, duration_ms)


def log_event(event: str, **kwargs: Any) -> None:
    logger.info("event=%s %s", event, " ".join(f"{k}={v}" for k, v in kwargs.items()))


def log_error(event: str, message: str, **kwargs: Any) -> None:
    logger.error("event=%s message=\"%s\" %s", event, message, " ".join(f"{k}={v}" for k, v in kwargs.items()))
