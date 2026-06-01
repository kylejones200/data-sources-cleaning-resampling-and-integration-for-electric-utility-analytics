#!/usr/bin/env python3
"""Python vs Rust kernel benchmark."""

from __future__ import annotations

import time
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from compute_kernel import bucket_hourly_means  # noqa: E402

def main() -> None:
    n = 10000
    values = np.ascontiguousarray(np.sin(np.arange(n) * 0.01) + 50.0)
    bucket_ids = np.ascontiguousarray(np.arange(n) % 24, dtype=np.int64)
    n_buckets = 24
    t0 = time.perf_counter()
    for _ in range(200):
        bucket_hourly_means(values, bucket_ids, n_buckets)
    py_s = time.perf_counter() - t0
    try:
        import data_sources_cleaning_resampling_and_integration_for_electric_utility_analytics_rs as rs
    except ImportError:
        print("Build: maturin develop --release -m rust/py/Cargo.toml")
        print(f"Python {py_s:.3f}s")
        return
    rs_s = rs.bench_kernel_py(values, bucket_ids, n_buckets, 5000)
    print(f"Python {py_s:.3f}s Rust {rs_s:.3f}s speedup {py_s / max(rs_s, 1e-9):.1f}x")
    np.testing.assert_allclose(
        bucket_hourly_means(values, bucket_ids, n_buckets),
        np.asarray(rs.bucket_hourly_means_py(values, bucket_ids, n_buckets)),
        rtol=1e-10,
    )
    print("Correctness: OK")

if __name__ == "__main__":
    main()
