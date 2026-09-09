"""Alignment of PPG beats to reference ECG beats, and derivation of paired intervals.

Charlton's benchmark matches detected beats to ECG beats within +/-150 ms. That window
is a *matching* tolerance, not a statement about acceptable timing accuracy -- a point
the German report makes explicitly and which is worth restating, because a detector can
score a perfect F1 at 150 ms while its intervals are useless for RMSSD.

Pulse arrival time at the finger is roughly 200-300 ms, so a constant lag is estimated
and removed before matching; otherwise the tolerance window would not even be centred.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TOLERANCE_S = 0.150
LAG_SEARCH = (-0.10, 0.60)   # plausible finger PAT range, with margin


@dataclass
class MatchResult:
    ref_idx: np.ndarray      # indices into the reference beat array
    test_idx: np.ndarray     # indices into the detected beat array
    n_ref: int
    n_test: int
    lag_s: float

    @property
    def tp(self) -> int:
        return len(self.ref_idx)

    @property
    def fn(self) -> int:
        return self.n_ref - self.tp

    @property
    def fp(self) -> int:
        return self.n_test - self.tp


def estimate_lag(ref: np.ndarray, test: np.ndarray, fs_grid: float = 200.0) -> float:
    """Constant PPG-minus-ECG delay, by cross-correlating smoothed impulse trains.

    More robust than a median nearest-neighbour difference, which breaks down when the
    detector emits many spurious beats.
    """
    ref = np.asarray(ref, dtype=float)
    test = np.asarray(test, dtype=float)
    if ref.size < 3 or test.size < 3:
        return 0.0

    lo, hi = LAG_SEARCH
    # For each candidate lag, how many reference beats find a partner within tolerance.
    best_lag, best_score = 0.0, -1
    coarse = np.arange(lo, hi, 1.0 / fs_grid)
    for lag in coarse:
        shifted = test - lag
        idx = np.searchsorted(shifted, ref)
        idx = np.clip(idx, 1, len(shifted) - 1)
        d = np.minimum(np.abs(shifted[idx] - ref), np.abs(shifted[idx - 1] - ref))
        score = int(np.sum(d <= 0.05))
        if score > best_score:
            best_lag, best_score = float(lag), score

    # Refine: median residual of the beats that matched at the coarse optimum.
    shifted = test - best_lag
    idx = np.clip(np.searchsorted(shifted, ref), 1, len(shifted) - 1)
    cand = np.where(np.abs(shifted[idx] - ref) < np.abs(shifted[idx - 1] - ref),
                    shifted[idx], shifted[idx - 1])
    resid = cand - ref
    keep = np.abs(resid) <= 0.05
    return best_lag + float(np.median(resid[keep])) if keep.sum() >= 3 else best_lag


def match_beats(ref: np.ndarray, test: np.ndarray, tolerance: float = TOLERANCE_S,
                lag: float | None = None) -> MatchResult:
    """One-to-one greedy nearest matching within `tolerance`."""
    ref = np.asarray(ref, dtype=float)
    test = np.asarray(test, dtype=float)
    if ref.size == 0 or test.size == 0:
        return MatchResult(np.array([], int), np.array([], int), len(ref), len(test), 0.0)

    if lag is None:
        lag = estimate_lag(ref, test)
    shifted = test - lag

    # Candidate pairs within tolerance, then greedily accept closest-first so that
    # each reference beat and each detection is used at most once.
    pairs: list[tuple[float, int, int]] = []
    for i, t in enumerate(ref):
        lo = np.searchsorted(shifted, t - tolerance, side="left")
        hi = np.searchsorted(shifted, t + tolerance, side="right")
        for j in range(lo, hi):
            pairs.append((abs(shifted[j] - t), i, j))

    pairs.sort()
    used_ref: set[int] = set()
    used_test: set[int] = set()
    mi: list[int] = []
    mj: list[int] = []
    for _, i, j in pairs:
        if i in used_ref or j in used_test:
            continue
        used_ref.add(i)
        used_test.add(j)
        mi.append(i)
        mj.append(j)

    order = np.argsort(mi)
    return MatchResult(
        ref_idx=np.asarray(mi, dtype=int)[order],
        test_idx=np.asarray(mj, dtype=int)[order],
        n_ref=len(ref),
        n_test=len(test),
        lag_s=float(lag),
    )


def paired_intervals(ref: np.ndarray, test: np.ndarray, m: MatchResult
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RR and PP intervals for beat pairs that are consecutive in *both* series.

    Requiring consecutiveness on both sides is what keeps a missed beat from silently
    contributing a doubled interval to the IBI error, which would conflate a detection
    failure with a timing failure.

    Returns (rr, pp, ref_idx) where ref_idx is the index of the *first* reference beat
    of each interval. That key is what lets intervals from different detectors be
    aligned to each other: two detectors' interval arrays have different lengths and
    cover different beats, so aligning them by position rather than by reference beat
    silently compares unrelated heartbeats.
    """
    ref = np.asarray(ref, dtype=float)
    test = np.asarray(test, dtype=float)
    if m.tp < 2:
        return np.array([]), np.array([]), np.array([], dtype=int)

    ok = (np.diff(m.ref_idx) == 1) & (np.diff(m.test_idx) == 1)
    a, b = m.ref_idx[:-1][ok], m.ref_idx[1:][ok]
    c, d = m.test_idx[:-1][ok], m.test_idx[1:][ok]
    return ref[b] - ref[a], test[d] - test[c], a
