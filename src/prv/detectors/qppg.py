"""qppg -- systolic upslope detection via Zong's slope sum function.

Ported from the description in Zong et al., "An open-source algorithm to detect onset
of arterial blood pressure pulses" (Comput Cardiol 2003), which is the basis of the
qppg detector benchmarked by Charlton et al. (2022) as MSPTD's co-winner.

Unlike the multiscale detectors this searches for *pulse onsets* using the expected
characteristics of a systolic upslope, so it returns onset anchors rather than peaks.

Note on licensing: the original qppg C code descends from WFDB and is GPL. This is an
independent NumPy implementation written from the published method description.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

SSF_WINDOW_S = 0.128       # Zong's 128 ms slope-sum accumulation window
LEARNING_S = 10.0          # initial period used to seed the threshold
EYE_CLOSING_S = 0.30       # refractory period after an accepted onset
LOWPASS_HZ = 16.0
ONSET_FRACTION = 0.01      # SSF fraction defining the true foot of the upslope
BACKSEARCH_FACTOR = 1.5    # re-search with a lowered threshold beyond this gap


def _slope_sum(x: np.ndarray, fs: float) -> np.ndarray:
    """Accumulate only the positive first differences over a sliding window, which
    amplifies the systolic upslope and suppresses the diastolic decay."""
    du = np.diff(x, prepend=x[0])
    du[du < 0] = 0.0
    w = max(1, int(round(SSF_WINDOW_S * fs)))
    kernel = np.ones(w)
    # 'full' then trim keeps each output aligned with the END of its window,
    # which is where Zong's threshold crossing is defined.
    return np.convolve(du, kernel)[: len(x)]


def _find_onset(ssf: np.ndarray, cross: int, fs: float) -> int:
    """Walk back from a threshold crossing to where the slope sum has essentially
    vanished -- that point is the foot of the upslope."""
    search = max(0, cross - int(round(0.30 * fs)))
    seg = ssf[search : cross + 1]
    if seg.size == 0:
        return cross
    thresh = ONSET_FRACTION * float(np.max(seg))
    below = np.where(seg <= thresh)[0]
    return search + int(below[-1]) if below.size else search


def detect(x: np.ndarray, fs: float) -> np.ndarray:
    """Return pulse onset sample indices."""
    x = np.asarray(x, dtype=float)
    if x.size < int(2 * fs):
        return np.array([], dtype=int)

    nyq = fs / 2.0
    if LOWPASS_HZ < nyq * 0.95:
        sos = signal.butter(2, LOWPASS_HZ / nyq, btype="low", output="sos")
        xf = signal.sosfiltfilt(sos, x)
    else:
        xf = x

    ssf = _slope_sum(xf, fs)

    n_learn = min(len(ssf), int(round(LEARNING_S * fs)))
    learn = ssf[:n_learn]
    # Zong seeds from the learning period; 3x the mean is his published multiplier.
    threshold = 3.0 * float(np.mean(learn)) if np.mean(learn) > 0 else float(np.max(ssf)) * 0.2
    if threshold <= 0:
        return np.array([], dtype=int)

    eye = int(round(EYE_CLOSING_S * fs))
    onsets: list[int] = []
    intervals: list[float] = []

    i = 1
    n = len(ssf)
    last_accept = -eye
    while i < n:
        if ssf[i] > threshold and ssf[i - 1] <= threshold and (i - last_accept) >= eye:
            # Peak of this slope-sum excursion, then walk back to the foot.
            hi = min(n, i + int(round(0.15 * fs)))
            local_max_idx = i + int(np.argmax(ssf[i:hi])) if hi > i else i
            onset = _find_onset(ssf, local_max_idx, fs)

            if onsets and onset <= onsets[-1]:
                i += 1
                continue

            if onsets:
                intervals.append((onset - onsets[-1]) / fs)
            onsets.append(onset)
            last_accept = onset

            # Adapt the threshold towards the amplitude of accepted pulses.
            threshold = 0.8 * threshold + 0.2 * (0.6 * float(ssf[local_max_idx]))

            # Backsearch: a gap much longer than recent intervals means a pulse was
            # missed, so re-scan it at a lowered threshold.
            if len(intervals) >= 3:
                median_int = float(np.median(intervals[-8:]))
                gap_limit = BACKSEARCH_FACTOR * median_int * fs
                if len(onsets) >= 2 and (onsets[-1] - onsets[-2]) > gap_limit:
                    lo, hi2 = onsets[-2] + eye, onsets[-1] - eye
                    if hi2 > lo:
                        seg = ssf[lo:hi2]
                        low_thr = 0.5 * threshold
                        if seg.size and float(np.max(seg)) > low_thr:
                            cand = lo + int(np.argmax(seg))
                            recovered = _find_onset(ssf, cand, fs)
                            if onsets[-2] < recovered < onsets[-1]:
                                onsets.insert(-1, recovered)
            i = onset + eye
        else:
            i += 1

    return np.unique(np.asarray(onsets, dtype=int))
