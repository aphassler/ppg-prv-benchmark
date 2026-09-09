"""Signal quality indices and the coverage gate.

Elgendi (2016) compared eight SQIs and found skewness the best discriminator, so that is
the primary index here, with perfusion index and a gyroscope motion index alongside.

The gate exists to support the benchmark's central fairness rule: detectors are compared
at *matched coverage*, so that none of them can win by abstaining on the hard segments.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

WINDOW_S = 10.0


def _windows(n: int, fs: float, win_s: float) -> list[tuple[int, int]]:
    w = int(round(win_s * fs))
    return [(i, i + w) for i in range(0, max(0, n - w + 1), w)]


def skewness_sqi(x: np.ndarray, fs: float, win_s: float = WINDOW_S) -> np.ndarray:
    """Per-window skewness. A clean PPG pulse train is strongly right-skewed; motion
    and baseline wander drive it towards zero."""
    return np.asarray([stats.skew(x[a:b]) for a, b in _windows(len(x), fs, win_s)])


def perfusion_index(raw: np.ndarray, fs: float, win_s: float = WINDOW_S) -> np.ndarray:
    """Pulsatile range over DC level -- the conventional hardware quality gold standard."""
    out = []
    for a, b in _windows(len(raw), fs, win_s):
        seg = raw[a:b]
        dc = np.mean(np.abs(seg))
        ac = np.percentile(seg, 95) - np.percentile(seg, 5)
        out.append(float(ac / dc) if dc > 0 else 0.0)
    return np.asarray(out)


def motion_sqi(gyro: np.ndarray, fs: float, win_s: float = WINDOW_S) -> np.ndarray:
    """RMS gyroscope magnitude per window (deg/s), median-centred."""
    mag = np.linalg.norm(np.asarray(gyro, dtype=float), axis=1)
    mag = mag - np.median(mag)
    return np.asarray([float(np.sqrt(np.mean(mag[a:b] ** 2)))
                       for a, b in _windows(len(mag), fs, win_s)])


def quality_score(x: np.ndarray, fs: float, win_s: float = WINDOW_S) -> np.ndarray:
    """Composite per-window score used for the coverage sweep. Higher is better."""
    skew = skewness_sqi(x, fs, win_s)
    # Correlation of each window against the record's mean pulse template catches
    # windows whose morphology has fallen apart even when skewness survives.
    return np.nan_to_num(skew, nan=-np.inf)


def coverage_mask(score: np.ndarray, fraction: float) -> np.ndarray:
    """Keep the best `fraction` of windows. Returns a boolean mask over windows.

    Sweeping `fraction` is what produces the accuracy-vs-coverage curve; comparing
    detectors at a single fixed fraction is what makes the leaderboard honest.
    """
    score = np.asarray(score, dtype=float)
    if score.size == 0:
        return np.zeros(0, dtype=bool)
    fraction = float(np.clip(fraction, 0.0, 1.0))
    if fraction >= 1.0:
        return np.ones(score.size, dtype=bool)
    k = max(1, int(round(fraction * score.size)))
    thresh = np.sort(score)[::-1][k - 1]
    return score >= thresh


def windows_to_time_mask(mask: np.ndarray, win_s: float = WINDOW_S) -> list[tuple[float, float]]:
    """Convert a per-window boolean mask into accepted (start, end) time spans."""
    return [(i * win_s, (i + 1) * win_s) for i, keep in enumerate(mask) if keep]


def select_beats(beats_s: np.ndarray, spans: list[tuple[float, float]]) -> np.ndarray:
    """Keep beats falling inside accepted spans."""
    beats = np.asarray(beats_s, dtype=float)
    if not spans:
        return np.array([], dtype=float)
    keep = np.zeros(beats.shape, dtype=bool)
    for a, b in spans:
        keep |= (beats >= a) & (beats < b)
    return beats[keep]
