"""Front-end tests.

The DC-pedestal test is the important one here: it locks in the bug that made every
amplitude-threshold detector find zero beats on the real data.
"""
from __future__ import annotations

import numpy as np
import pytest
from conftest import synth_ppg

from prv import preprocess


def test_decimation_preserves_heart_rate():
    sig, beats, fs = synth_ppg(duration_s=60.0, fs=500.0, hr_bpm=72.0)
    out, fs_out = preprocess.decimate_to(sig, 500.0, 100.0)
    assert fs_out == 100.0
    assert len(out) == pytest.approx(len(sig) / 5, rel=0.01)

    from scipy import signal as sg

    f, p = sg.welch(out - out.mean(), fs=100.0, nperseg=2048)
    assert f[np.argmax(p)] == pytest.approx(72.0 / 60.0, abs=0.05)


def test_decimation_survives_large_dc_offset():
    """Raw PTT-PPG sits on a ~74000 DC pedestal with a ~150 pulsatile component.

    Decimating that directly produces edge transients tens of thousands of units tall,
    which then swamp every amplitude threshold downstream. Regression guard.
    """
    sig, _, _ = synth_ppg(duration_s=60.0, fs=500.0, hr_bpm=72.0)
    scaled = sig * 150.0 + 74000.0

    out, _ = preprocess.decimate_to(scaled, 500.0, 100.0)
    # No sample may sit absurdly far from the bulk of the distribution.
    assert np.max(np.abs(out)) < 20 * np.std(out)


def test_prepare_output_is_normalised_and_transient_free():
    sig, _, _ = synth_ppg(duration_s=60.0, fs=500.0, hr_bpm=72.0)
    x, fs = preprocess.prepare(sig * 150.0 + 74000.0, 500.0, 100.0)
    assert fs == 100.0
    assert np.std(x) == pytest.approx(1.0, abs=1e-6)
    assert np.abs(np.mean(x)) < 1e-6
    assert np.max(np.abs(x)) < 10.0


def test_bandpass_is_zero_phase():
    """Any group delay would shift every fiducial and appear as IBI bias."""
    sig, beats, fs = synth_ppg(duration_s=30.0, fs=100.0, hr_bpm=60.0)
    filt = preprocess.bandpass(sig, fs)
    lags = np.arange(-50, 51)
    xc = [np.corrcoef(sig[50:-50], np.roll(filt, int(k))[50:-50])[0, 1] for k in lags]
    assert lags[int(np.argmax(xc))] == 0


def test_detect_polarity_on_synthetic_wave():
    sig, _, fs = synth_ppg(duration_s=30.0, fs=100.0, hr_bpm=60.0)
    assert preprocess.detect_polarity(sig, fs) == 1.0
    assert preprocess.detect_polarity(-sig, fs) == -1.0


def test_detect_polarity_ignores_dc_offset():
    sig, _, fs = synth_ppg(duration_s=30.0, fs=100.0, hr_bpm=60.0)
    assert preprocess.detect_polarity(-sig * 150.0 + 74000.0, fs) == -1.0


def test_motion_index_responds_to_movement():
    still = np.zeros((3000, 3))
    moving = np.random.default_rng(0).normal(0, 5, (3000, 3))
    assert np.median(preprocess.motion_index(still, 100.0)) < \
        np.median(preprocess.motion_index(moving, 100.0))
