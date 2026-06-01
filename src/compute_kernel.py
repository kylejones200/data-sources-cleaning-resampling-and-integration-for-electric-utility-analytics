"""Mean value per bucket index."""

from __future__ import annotations

import numpy as np


def bucket_hourly_means(
    values: np.ndarray, bucket_ids: np.ndarray, n_buckets: int
) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    b = np.asarray(bucket_ids, dtype=np.int64)
    sums = np.zeros(n_buckets, dtype=float)
    counts = np.zeros(n_buckets, dtype=int)
    for val, bid in zip(v, b):
        idx = int(bid)
        if 0 <= idx < n_buckets:
            sums[idx] += val
            counts[idx] += 1
    out = np.zeros(n_buckets, dtype=float)
    mask = counts > 0
    out[mask] = sums[mask] / counts[mask]
    return out
