"""Interbeat-interval artifact correction.

Both reports claim correction is the single largest lever on HRV accuracy; experiment E3
tests that directly by running the same detectors through all three strategies.

The German report's warning is respected here: aggressive correction can erase genuine
variability, so every function reports how many beats it touched. A RMSSD computed after
correcting 15% of beats is a different epistemic object from one that touched 0.5%, and
the benchmark refuses to let those be compared without saying so.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Deliberately wide: a hard physiological screen, not an outlier test. The narrow
# 600-1500 ms window Zuern et al. used suits a resting adult cohort but would mangle
# athletes or bradycardia, so it is not adopted as a default.
HARD_MIN_S = 0.30
HARD_MAX_S = 2.00


@dataclass
class Correction:
    beats: np.ndarray        # corrected beat times, seconds (full series, for matching)
    ibi: np.ndarray          # corrected intervals, seconds
    times: np.ndarray        # beat time associated with each retained interval
    n_input: int
    n_corrected: int

    @property
    def fraction_corrected(self) -> float:
        return self.n_corrected / self.n_input if self.n_input else np.nan


def none(beats_s: np.ndarray) -> Correction:
    """No correction beyond the hard physiological screen."""
    beats = np.asarray(beats_s, dtype=float)
    ibi = np.diff(beats)
    keep = (ibi >= HARD_MIN_S) & (ibi <= HARD_MAX_S)
    return Correction(beats, ibi[keep], beats[1:][keep], len(ibi), int((~keep).sum()))


def median_mad(beats_s: np.ndarray, window: int = 9, n_mad: float = 5.0) -> Correction:
    """Local median/MAD outlier rejection over a sliding window of beats."""
    beats = np.asarray(beats_s, dtype=float)
    ibi = np.diff(beats)
    if ibi.size < window:
        return none(beats)

    half = window // 2
    padded = np.pad(ibi, half, mode="edge")
    med = np.array([np.median(padded[i : i + window]) for i in range(len(ibi))])
    mad = np.array([np.median(np.abs(padded[i : i + window] - med[i])) for i in range(len(ibi))])
    # 1.4826 rescales the MAD to a Gaussian-equivalent standard deviation.
    scale = 1.4826 * mad
    scale[scale <= 0] = np.finfo(float).eps

    ok = (np.abs(ibi - med) <= n_mad * scale) & (ibi >= HARD_MIN_S) & (ibi <= HARD_MAX_S)
    return Correction(beats, ibi[ok], beats[1:][ok], len(ibi), int((~ok).sum()))


def lipponen_tarvainen(beats_s: np.ndarray, fs: float = 1000.0) -> Correction:
    """Kubios-style correction (Lipponen & Tarvainen 2019) via NeuroKit2.

    Uses time-varying thresholds on the dRR series plus beat classification, correcting
    extra, missed and ectopic beats rather than simply deleting outliers.
    """
    import warnings

    import neurokit2 as nk

    beats = np.asarray(beats_s, dtype=float)
    if beats.size < 6:
        return none(beats)

    samples = np.round(beats * fs).astype(int)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, corrected = nk.signal_fixpeaks(
                samples, sampling_rate=int(fs), iterative=True, method="Kubios", show=False
            )
        corrected = np.asarray(corrected, dtype=float) / fs
    except Exception:
        return median_mad(beats)

    corrected = np.unique(corrected)
    if corrected.size < 3:
        return none(beats)

    ibi = np.diff(corrected)
    keep = (ibi >= HARD_MIN_S) & (ibi <= HARD_MAX_S)
    # Count how far the series moved: added, removed or shifted beats all count.
    n_changed = int(abs(len(corrected) - len(beats)) + np.sum(~keep))
    return Correction(corrected, ibi[keep], corrected[1:][keep],
                      max(len(np.diff(beats)), 1), n_changed)


STRATEGIES = {"none": none, "median_mad": median_mad, "lipponen_tarvainen": lipponen_tarvainen}
DEFAULT_STRATEGY = "lipponen_tarvainen"
