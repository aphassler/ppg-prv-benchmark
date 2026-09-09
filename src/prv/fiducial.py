"""Fiducial point location and sub-sample refinement.

Two jobs:

1. **Normalise across detectors.** Detectors anchor on different things -- the MSPTD
   family and TERMA on systolic peaks, qppg on pulse onsets. Every anchor is expanded to
   an (onset, peak) pulse pair so that all detectors can be scored at the *same*
   fiducial. Without this, qppg would be penalised for measuring a different point on
   the wave rather than for detecting beats worse.

2. **Beat the sample grid.** At the 100 Hz operating rate one sample is 10 ms, the same
   order as the RMSSD being measured. Sub-sample refinement is therefore not a
   refinement at all but a precondition; experiment E2 quantifies exactly this.

Fiducial choices follow Peralta et al. (2019), who found mid-amplitude, the
first-derivative apex and the tangent intersection superior to the apex and the foot.
"""
from __future__ import annotations

import numpy as np

FIDUCIALS = ("peak", "mid_upslope", "max_slope", "foot")
DEFAULT_FIDUCIAL = "mid_upslope"   # MSPTDfast v2's own published choice

MAX_UPSLOPE_S = 0.40   # widest plausible onset->peak separation


def _parabolic(y: np.ndarray, i: int) -> float:
    """Sub-sample extremum position by fitting a parabola through three points."""
    if i <= 0 or i >= len(y) - 1:
        return float(i)
    a, b, c = float(y[i - 1]), float(y[i]), float(y[i + 1])
    denom = a - 2.0 * b + c
    if denom == 0.0:
        return float(i)
    delta = 0.5 * (a - c) / denom
    return float(i) + delta if abs(delta) <= 1.0 else float(i)


def to_pulses(x: np.ndarray, fs: float, anchors: np.ndarray, anchor: str) -> np.ndarray:
    """Expand anchors into (onset, peak) index pairs. Returns an (n, 2) int array."""
    x = np.asarray(x, dtype=float)
    anchors = np.asarray(anchors, dtype=int)
    anchors = anchors[(anchors >= 0) & (anchors < len(x))]
    if anchors.size == 0:
        return np.empty((0, 2), dtype=int)

    span = int(round(MAX_UPSLOPE_S * fs))
    pairs: list[tuple[int, int]] = []

    if anchor == "peak":
        prev = 0
        for p in anchors:
            lo = max(prev, p - span, 0)
            if p <= lo:
                prev = p
                continue
            onset = lo + int(np.argmin(x[lo : p + 1]))
            if onset < p:
                pairs.append((onset, int(p)))
            prev = p
    else:  # anchor == "onset"
        for k, o in enumerate(anchors):
            hi = min(len(x), o + span + 1)
            if k + 1 < len(anchors):
                hi = min(hi, int(anchors[k + 1]))
            if hi <= o + 1:
                continue
            peak = o + int(np.argmax(x[o:hi]))
            if peak > o:
                pairs.append((int(o), int(peak)))

    return np.asarray(pairs, dtype=int) if pairs else np.empty((0, 2), dtype=int)


def locate(x: np.ndarray, fs: float, anchors: np.ndarray, anchor: str,
           kind: str = DEFAULT_FIDUCIAL) -> np.ndarray:
    """Return fiducial times in seconds, sub-sample refined.

    kind: one of FIDUCIALS.
    """
    if kind not in FIDUCIALS:
        raise ValueError(f"unknown fiducial {kind!r}; expected one of {FIDUCIALS}")

    x = np.asarray(x, dtype=float)
    pulses = to_pulses(x, fs, anchors, anchor)
    if pulses.size == 0:
        return np.array([], dtype=float)

    if kind == "max_slope":
        dx = np.gradient(x)

    out: list[float] = []
    for onset, peak in pulses:
        if kind == "peak":
            pos = _parabolic(x, int(peak))

        elif kind == "foot":
            pos = _parabolic(x, int(onset))

        elif kind == "mid_upslope":
            # Linear crossing of the half-amplitude level on the rising edge. This is
            # inherently sub-sample and needs no parabola.
            lo_v, hi_v = x[onset], x[peak]
            target = 0.5 * (lo_v + hi_v)
            seg = x[onset : peak + 1]
            above = np.where(seg >= target)[0]
            if above.size == 0:
                pos = float(peak)
            else:
                j = int(above[0])
                if j == 0:
                    pos = float(onset)
                else:
                    y0, y1 = seg[j - 1], seg[j]
                    frac = 0.0 if y1 == y0 else (target - y0) / (y1 - y0)
                    pos = float(onset + j - 1 + frac)

        else:  # max_slope
            if peak <= onset:
                pos = float(peak)
            else:
                j = onset + int(np.argmax(dx[onset : peak + 1]))
                pos = _parabolic(dx, int(j))

        out.append(pos / fs)

    return np.asarray(out, dtype=float)
