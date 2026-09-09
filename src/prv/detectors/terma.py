"""TERMA / ERMA -- Elgendi's two event-related moving averages.

From Elgendi et al., "Systolic peak detection in acceleration photoplethysmograms
measured from emergency responders in tropical conditions" (PLoS ONE 2013). Two moving
averages of different length -- one matched to the systolic peak, one to the whole beat
-- generate blocks of interest whose maxima are the systolic peaks.

Cheapest of the five approaches, which is why both reports shortlist it for mobile.
Cross-validated against NeuroKit2's `method="elgendi"` in scripts/01_qc_reference.py.
"""
from __future__ import annotations

import numpy as np

W1_S = 0.111       # systolic peak duration
W2_S = 0.667       # beat duration
BETA = 0.02        # offset scaling for the threshold
MIN_DELAY_S = 0.30 # refractory period between accepted peaks (<= 200 bpm)


def _moving_average(x: np.ndarray, w: int) -> np.ndarray:
    w = max(1, w)
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="same")


def detect(x: np.ndarray, fs: float, w1_s: float = W1_S, w2_s: float = W2_S,
           beta: float = BETA, min_delay_s: float = MIN_DELAY_S) -> np.ndarray:
    """Return systolic peak sample indices."""
    x = np.asarray(x, dtype=float)
    if x.size < int(2 * fs):
        return np.array([], dtype=int)

    # Clip to positives then square: emphasises the systolic rise, discards the
    # diastolic downstroke entirely.
    clipped = np.clip(x, 0, None)
    squared = clipped**2

    ma_peak = _moving_average(squared, int(round(w1_s * fs)))
    ma_beat = _moving_average(squared, int(round(w2_s * fs)))

    alpha = beta * float(np.mean(squared))
    thr1 = ma_beat + alpha

    # Blocks of interest, and the minimum width that makes one a real pulse.
    blocks = ma_peak > thr1
    thr2 = int(round(w1_s * fs))

    peaks: list[int] = []
    edges = np.diff(blocks.astype(np.int8))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0] + 1)
    if blocks[0]:
        starts.insert(0, 0)
    if blocks[-1]:
        ends.append(len(blocks))

    for s, e in zip(starts, ends):
        if e - s >= thr2:
            peaks.append(s + int(np.argmax(x[s:e])))

    # Refractory period. Elgendi's published rule stops at the block-width test, but
    # without a minimum delay the dicrotic notch opens a second block of its own on
    # well-perfused waves: on record s3 that inflated the beat count to 806 against 603
    # reference beats. Keeping the taller of two too-close candidates matches both the
    # physiology and NeuroKit2's reference implementation.
    if not peaks:
        return np.array([], dtype=int)
    min_gap = int(round(min_delay_s * fs))
    kept: list[int] = [peaks[0]]
    for p in peaks[1:]:
        if p - kept[-1] < min_gap:
            if x[p] > x[kept[-1]]:
                kept[-1] = p
        else:
            kept.append(p)

    return np.asarray(kept, dtype=int)
