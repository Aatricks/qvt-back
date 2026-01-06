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
            "ID": range(ROWS),
            "Sexe": [1] * ROWS,
            "Age": [40] * ROWS,
            "Contrat": [1] * ROWS,
            "Temps": [1] * ROWS,
            "Encadre": [1] * ROWS,
            "Ancienne": [5] * ROWS,
            "Secteur": [1] * ROWS,
            "TailleOr": [2] * ROWS,
            "PGC2": [i % 5 + 1 for i in range(ROWS)],
        }
    )
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    data = buf.getvalue().encode()

    start = time.perf_counter()
    response = client.post(
        "/api/visualize/likert_distribution",
        files={"hr_file": ("hr_large.csv", data, "text/csv")},
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert response.status_code == 200
    assert elapsed_ms <= TARGET_MS
