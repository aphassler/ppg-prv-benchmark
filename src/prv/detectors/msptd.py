"""MSPTD -- multi-scale peak and trough detection (Bishop & Ercole 2018).

A vectorised reimplementation of the algorithm, producing bit-identical output to
NeuroKit2's `ppg_findpeaks(method="bishop")` while running ~50x faster.

NeuroKit2 populates the local-maxima scalogram with a doubly nested Python loop -- for
each of ~L scales, for each of ~N samples -- which is roughly 90,000 interpreted
iterations for a single 6 s window at 100 Hz, and this benchmark evaluates ~100 windows
per record. The condition being tested,

    x[j] > x[j - k]  and  x[j] > x[j + k]

is a comparison between three shifted views of the same array, so an entire scale
resolves in one NumPy operation and the scalogram costs L operations rather than L*N
interpreted steps.

The equivalence is exact, not approximate: the same boolean matrix is produced by both
routes, so every downstream step is unchanged. `tests/test_msptd.py` asserts identical
peak and onset indices against NeuroKit2 on synthetic and real windows.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sg


def lms_scalograms(x: np.ndarray, L: int) -> tuple[np.ndarray, np.ndarray]:
    """Local-maxima and local-minima scalograms.

    Index convention follows the reference implementation exactly, including its scale
    range `1 <= k < L`, which leaves the final row of each matrix unpopulated.
    """
    n = len(x)
    m_max = np.full((L, n), False)
    m_min = np.full((L, n), False)

    for k in range(1, L):
        lo, hi = k + 1, n - k
        if hi <= lo:
            continue
        centre = x[lo:hi]
        left = x[lo - k:hi - k]
        right = x[lo + k:hi + k]
        m_max[k - 1, lo:hi] = (centre > left) & (centre > right)
        m_min[k - 1, lo:hi] = (centre < left) & (centre < right)

    return m_max, m_min


def detect_peaks_onsets(signal_window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (peaks, onsets) sample indices for one window."""
    x = np.asarray(signal_window, dtype=float)
    n = len(x)
    if n < 4:
        return np.array([], dtype=int), np.array([], dtype=int)

    ell = int(np.ceil(n / 2) - 1)
    if ell < 1:
        return np.array([], dtype=int), np.array([], dtype=int)

    x = sg.detrend(x, type="linear")
    m_max, m_min = lms_scalograms(x, ell)

    # Scale carrying the most local extrema.
    lambda_max = int(np.argmax(m_max.sum(axis=1)))
    lambda_min = int(np.argmax(m_min.sum(axis=1)))

    # Keep scales up to lambda, then a column is an extremum only if it held at every
    # retained scale.
    peaks = np.where((~m_max[: lambda_max + 1, :]).sum(axis=0) == 0)[0].astype(int)
    onsets = np.where((~m_min[: lambda_min + 1, :]).sum(axis=0) == 0)[0].astype(int)
    return peaks, onsets


def detect(x: np.ndarray, fs: float) -> np.ndarray:
    """Return systolic peak sample indices for one window.

    Windowing is applied by `detectors.windowed()`; the scalogram is O(N^2) in the
    window length, so MSPTD is never run over a whole record.
    """
    peaks, _ = detect_peaks_onsets(x)
    return peaks
