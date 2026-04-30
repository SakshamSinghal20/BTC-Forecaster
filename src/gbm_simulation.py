from __future__ import annotations

import numpy as np

from src.prediction import calculate_prediction_interval, simulate_gbm


def simulate_student_t(
    n: int,
    df: float,
    seed: int = 42,
    unit_variance: bool = False,
) -> np.ndarray:
    """Generate Student-t samples with an optional unit-variance normalization."""
    if df <= 2:
        raise ValueError("Student-t degrees of freedom must be greater than 2")
    if n < 1:
        raise ValueError("n must be positive")

    rng = np.random.default_rng(seed)
    samples = rng.standard_t(df, size=n)
    if unit_variance:
        samples = samples / np.sqrt(df / (df - 2))
    return samples


__all__ = ["calculate_prediction_interval", "simulate_gbm", "simulate_student_t"]

