import io
import os
import time

import pandas as pd
import pytest

TARGET_MS = 2000  # 2s budget
ROWS = 50000


@pytest.mark.performance
@pytest.mark.skipif(os.getenv("RUN_PERF_TESTS") != "1", reason="Performance tests disabled")
def test_visualize_time_budget(client):
    # Generate synthetic HR dataset in-memory
    df = pd.DataFrame(
        {
            "year": [2024] * ROWS,
            "absenteeism_rate": range(ROWS),
            "turnover_rate": range(ROWS),
            "department": ["Ops"] * ROWS,
        }
    )
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    data = buf.getvalue().encode()

    start = time.perf_counter()
    response = client.post(
        "/api/visualize/time_series",
        files={"hr_file": ("hr_large.csv", data, "text/csv")},
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert response.status_code == 200
    assert elapsed_ms <= TARGET_MS
