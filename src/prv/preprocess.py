"""Signal conditioning.

The pipeline operating rate is 100 Hz: the 500 Hz source is anti-alias decimated by 5
before any detector sees it. At 100 Hz one sample is 10 ms, the same order as the RMSSD
values being measured, which is why sub-sample refinement in `fiducial.py` matters.
"""
from __future__ import annotations

import numpy as np
from scipy import signal, stats

FS_PIPELINE = 100.0

# Charlton's benchmark band. Wide enough to preserve the systolic upslope that the
# mid-upslope and max-dP/dt fiducials depend on.
BAND_LOW = 0.5
BAND_HIGH = 8.0


def decimate_to(x: np.ndarray, fs_in: float, fs_out: float) -> tuple[np.ndarray, float]:
    """Anti-alias decimate. Uses an integer factor where possible, else polyphase resample.

    The DC pedestal is removed first. Raw PTT-PPG samples sit around 74000 with only a
    ~150-unit pulsatile component, and the anti-alias filter's edge behaviour on that
    step produces transients tens of thousands of units tall -- which then swamp every
    amplitude-based threshold downstream and make the simple detectors find no beats at
    all. Removing the offset costs nothing here because the output is normalised anyway.
    """
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    if np.isclose(fs_in, fs_out):
        return x, fs_in

    ratio = fs_in / fs_out
    if np.isclose(ratio, round(ratio)) and round(ratio) >= 2:
        q = int(round(ratio))
        # zero_phase avoids the group delay that would bias every beat time.
        return signal.decimate(x, q, ftype="fir", zero_phase=True), fs_out

    # Non-integer ratios (e.g. 500 -> 30 Hz) go through rational resampling.
    from fractions import Fraction

    frac = Fraction(fs_out / fs_in).limit_denominator(1000)
    return signal.resample_poly(x, frac.numerator, frac.denominator), fs_out


def bandpass(x: np.ndarray, fs: float, low: float = BAND_LOW, high: float = BAND_HIGH,
             order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass. Zero-phase is essential: any phase distortion
    would shift fiducial points and show up as IBI error.

    The linear detrend beforehand is not cosmetic. Raw PTT-PPG samples sit on a DC
    pedestal of ~74000 with a pulsatile AC component of ~150, and running a 0.5 Hz
    high-pass across that step produces an edge transient tens of standard deviations
    tall -- which then swamps every amplitude threshold downstream.
    """
    nyq = fs / 2.0
    high = min(high, nyq * 0.95)
    x = signal.detrend(np.asarray(x, dtype=float), type="linear")
    sos = signal.butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    return signal.sosfiltfilt(sos, x, padlen=min(3 * (2 * order + 1), len(x) - 1))


def normalise(x: np.ndarray) -> np.ndarray:
    sd = np.std(x)
    return (x - np.mean(x)) / sd if sd > 0 else x - np.mean(x)


def prepare(raw: np.ndarray, fs_in: float, fs_out: float = FS_PIPELINE,
            low: float = BAND_LOW, high: float = BAND_HIGH) -> tuple[np.ndarray, float]:
    """Full front end: decimate to the operating rate, band-pass, normalise."""
    x, fs = decimate_to(np.asarray(raw, dtype=float), fs_in, fs_out)
    x = bandpass(x, fs, low, high)
    return normalise(x), fs


def motion_index(gyro: np.ndarray, fs: float, win_s: float = 10.0) -> np.ndarray:
    """Per-window RMS gyroscope power, used to grade residual motion within the
    sitting records (experiment E4)."""
    mag = np.linalg.norm(np.asarray(gyro, dtype=float), axis=1)
    mag = mag - np.median(mag)
    n = max(1, int(round(win_s * fs)))
    nwin = len(mag) // n
    if nwin == 0:
        return np.array([np.sqrt(np.mean(mag**2))])
    trimmed = mag[: nwin * n].reshape(nwin, n)
    return np.sqrt(np.mean(trimmed**2, axis=1))


def detect_polarity(x: np.ndarray, fs: float) -> float:
    """Return +1 if the waveform already has PPG orientation, -1 if it is inverted.

    A correctly oriented photoplethysmogram has a fast systolic upstroke and a slow
    diastolic decay, so its first derivative is strongly right-skewed: a large brief
    positive excursion against a small prolonged negative one. Inverting the wave flips
    the sign of that skew, which makes it a cheap detector-free polarity test.

    This matters more than it looks. The PTT-PPG channels are transmission pulse
    oximetry -- more blood means more absorption and therefore a *lower* sample value --
    so the raw signal arrives upside down. Left uncorrected, every fiducial
    (mid-upslope, max dP/dt, foot) would be located on the diastolic decay instead of
    the systolic rise, and the benchmark would still report plausible F1 scores while
    measuring the wrong point on the wave entirely.
    """
    x = np.asarray(x, dtype=float)
    d = np.diff(bandpass(x - np.mean(x), fs))
    sk = float(stats.skew(d))
    return 1.0 if sk >= 0 else -1.0
