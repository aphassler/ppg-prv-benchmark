"""Known-answer tests for the HRV indices, and behavioural tests for the detectors."""
from __future__ import annotations

import numpy as np
import pytest
from conftest import synth_ppg

from prv import correct, hrv, preprocess, sqi
from prv.detectors import ALL_KEYS, REGISTRY


# ----------------------------------------------------------------------- HRV
def test_time_domain_against_hand_computed_values():
    ibi = np.array([0.800, 0.820, 0.780, 0.810, 0.790])
    out = hrv.time_domain(ibi)
    ms = ibi * 1000.0
    assert out["mean_nn"] == pytest.approx(np.mean(ms))
    assert out["sdnn"] == pytest.approx(np.std(ms, ddof=1))
    assert out["rmssd"] == pytest.approx(np.sqrt(np.mean(np.diff(ms) ** 2)))
    assert out["mean_hr"] == pytest.approx(60.0 / np.mean(ibi))


def test_rmssd_is_zero_for_a_constant_tachogram():
    out = hrv.time_domain(np.full(50, 0.85))
    assert out["rmssd"] == pytest.approx(0.0)
    assert out["sdnn"] == pytest.approx(0.0)
    assert out["pnn50"] == pytest.approx(0.0)


def test_pnn50_known_answer():
    # Alternating 800/900 ms: every successive difference is 100 ms, so pNN50 = 100%.
    ibi = np.tile([0.800, 0.900], 25)
    assert hrv.time_domain(ibi)["pnn50"] == pytest.approx(100.0)


def test_poincare_identity_holds():
    """SD1^2 + SD2^2 == 2*SDNN^2 is an algebraic identity, not an approximation."""
    rng = np.random.default_rng(1)
    ibi = rng.normal(0.85, 0.04, 400)
    t, nl = hrv.time_domain(ibi), hrv.nonlinear(ibi)
    assert nl["sd1"] ** 2 + nl["sd2"] ** 2 == pytest.approx(2 * t["sdnn"] ** 2, rel=1e-6)


def test_frequency_domain_recovers_an_injected_respiratory_peak():
    """A 0.25 Hz modulation must land in HF, not LF."""
    n, mean_rr = 500, 0.8
    t = np.cumsum(np.full(n, mean_rr))
    ibi = mean_rr + 0.03 * np.sin(2 * np.pi * 0.25 * t)
    out = hrv.frequency_domain(ibi, np.cumsum(ibi), method="welch")
    assert out["hf"] > 5 * out["lf"]


def test_sample_entropy_orders_random_above_periodic():
    rng = np.random.default_rng(2)
    periodic = np.tile([800.0, 820.0], 100)
    random = rng.normal(810, 20, 200)
    assert hrv.sample_entropy(periodic) < hrv.sample_entropy(random)


# ----------------------------------------------------------------- correction
def test_correction_flags_how_much_it_changed():
    beats = np.cumsum(np.full(60, 0.85))
    clean = correct.none(beats)
    assert clean.n_corrected == 0
    assert clean.fraction_corrected == pytest.approx(0.0)


def test_median_mad_removes_an_injected_outlier():
    beats = np.cumsum(np.full(80, 0.85))
    beats[40] += 0.35                        # a badly mislocated beat
    out = correct.median_mad(beats)
    assert out.n_corrected >= 1
    assert np.max(out.ibi) < 1.15


def test_hard_screen_rejects_impossible_intervals():
    beats = np.array([0.0, 0.1, 0.2, 3.0, 3.85, 4.70])   # 100 ms and 2.8 s gaps
    out = correct.none(beats)
    assert np.all((out.ibi >= correct.HARD_MIN_S) & (out.ibi <= correct.HARD_MAX_S))


# ------------------------------------------------------------------ detectors
@pytest.mark.parametrize("key", ALL_KEYS)
def test_every_detector_finds_the_right_beat_count_on_clean_signal(key):
    sig, beats, fs = synth_ppg(duration_s=60.0, fs=100.0, hr_bpm=60.0, noise=0.0)
    x = preprocess.normalise(preprocess.bandpass(sig, fs))
    found = REGISTRY[key](x, fs)
    assert len(found) == pytest.approx(len(beats), rel=0.05), (
        f"{key}: found {len(found)}, expected {len(beats)}"
    )


@pytest.mark.parametrize("key", ALL_KEYS)
def test_no_detector_is_fooled_by_the_dicrotic_notch(key):
    """The synthetic wave carries a dicrotic bump; double-counting it would roughly
    double the beat count. This is the failure the TERMA refractory period fixed."""
    sig, beats, fs = synth_ppg(duration_s=60.0, fs=100.0, hr_bpm=60.0)
    x = preprocess.normalise(preprocess.bandpass(sig, fs))
    assert len(REGISTRY[key](x, fs)) < 1.4 * len(beats)


@pytest.mark.parametrize("key", ALL_KEYS)
def test_detectors_return_sorted_unique_indices(key):
    sig, _, fs = synth_ppg(duration_s=30.0, fs=100.0, hr_bpm=75.0, noise=0.02)
    x = preprocess.normalise(preprocess.bandpass(sig, fs))
    found = REGISTRY[key](x, fs)
    assert np.all(np.diff(found) > 0)
    assert found.dtype.kind == "i"


def test_detectors_tolerate_a_flat_signal():
    for key in ALL_KEYS:
        assert len(REGISTRY[key](np.zeros(3000), 100.0)) < 50


# ------------------------------------------------------------------------ SQI
def test_coverage_mask_keeps_the_requested_fraction():
    score = np.arange(100, dtype=float)
    assert sqi.coverage_mask(score, 0.5).sum() == 50
    assert sqi.coverage_mask(score, 1.0).sum() == 100


def test_coverage_mask_keeps_the_best_windows():
    score = np.array([5.0, 1.0, 4.0, 2.0, 3.0])
    kept = sqi.coverage_mask(score, 0.6)
    assert kept.tolist() == [True, False, True, False, True]
