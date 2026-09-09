"""Shared fixtures: a synthetic PPG whose beat times are known exactly.

Ground truth from real data is only ever as good as its annotations, so the unit tests
run against a generated waveform where the correct answer is known by construction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def synth_ppg(duration_s: float = 60.0, fs: float = 100.0, hr_bpm: float = 60.0,
              rr_sd_s: float = 0.03, noise: float = 0.0, seed: int = 0,
              dc: float = 0.0) -> tuple[np.ndarray, np.ndarray, float]:
    """Generate a PPG-like waveform with a fast systolic upstroke, a dicrotic notch and
    a slow diastolic decay.

    Returns (signal, true systolic peak times in seconds, fs).
    """
    rng = np.random.default_rng(seed)
    mean_rr = 60.0 / hr_bpm

    beats: list[float] = []
    t = 1.0
    while t < duration_s - 1.0:
        beats.append(t)
        t += float(rng.normal(mean_rr, rr_sd_s))
    beats_arr = np.asarray(beats)

    n = int(round(duration_s * fs))
    t_axis = np.arange(n) / fs
    sig = np.zeros(n)

    rise, decay = 0.12, 0.35
    for b in beats_arr:
        rel = t_axis - b
        # Systolic: a narrow raised-cosine rising to its apex at `rel = 0`.
        sysl = np.where((rel > -rise) & (rel <= 0),
                        0.5 * (1 + np.cos(np.pi * rel / rise)), 0.0)
        # Diastolic decay plus a dicrotic bump, which is what makes naive peak pickers
        # double-count if they have no refractory period.
        dias = np.where(rel > 0, np.exp(-rel / decay), 0.0)
        notch = np.where(rel > 0, 0.25 * np.exp(-((rel - 0.28) ** 2) / (2 * 0.05**2)), 0.0)
        sig += sysl + dias + notch

    if noise > 0:
        sig = sig + rng.normal(0, noise, n)
    return sig + dc, beats_arr, fs


@pytest.fixture
def clean_ppg():
    return synth_ppg(duration_s=60.0, fs=100.0, hr_bpm=60.0, noise=0.0)


@pytest.fixture
def noisy_ppg():
    return synth_ppg(duration_s=60.0, fs=100.0, hr_bpm=72.0, noise=0.05, seed=3)
