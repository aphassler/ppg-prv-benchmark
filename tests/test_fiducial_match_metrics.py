"""Tests for fiducial location, beat matching and the metric definitions."""
from __future__ import annotations

import numpy as np
import pytest
from conftest import synth_ppg

from prv import fiducial, match, metrics, preprocess
from prv.detectors import REGISTRY


# ------------------------------------------------------------------ fiducial
def test_parabolic_interpolation_finds_true_apex():
    """A parabola sampled off-grid must resolve to its analytic apex."""
    true_apex = 10.37
    idx = np.arange(21)
    y = -((idx - true_apex) ** 2)
    assert fiducial._parabolic(y, int(np.argmax(y))) == pytest.approx(true_apex, abs=1e-9)


def test_fiducials_are_ordered_within_the_pulse():
    """foot < mid-upslope < max-slope-ish < peak, by construction of a systolic rise."""
    sig, beats, fs = synth_ppg(duration_s=30.0, fs=100.0, hr_bpm=60.0)
    x = preprocess.normalise(preprocess.bandpass(sig, fs))
    anchors = REGISTRY["msptdfast"](x, fs)
    times = {k: fiducial.locate(x, fs, anchors, "peak", k) for k in fiducial.FIDUCIALS}
    n = min(len(v) for v in times.values())
    assert n > 20
    assert np.all(times["foot"][:n] < times["mid_upslope"][:n])
    assert np.all(times["mid_upslope"][:n] < times["peak"][:n])


def test_mid_upslope_is_sub_sample():
    """Mid-upslope crossings must not quantise onto the sample grid."""
    sig, _, fs = synth_ppg(duration_s=30.0, fs=100.0, hr_bpm=67.0)
    x = preprocess.normalise(preprocess.bandpass(sig, fs))
    anchors = REGISTRY["msptdfast"](x, fs)
    t = fiducial.locate(x, fs, anchors, "peak", "mid_upslope")
    residual = np.abs(t * fs - np.round(t * fs))
    assert np.median(residual) > 0.01


def test_onset_anchored_detector_yields_same_fiducial():
    """qppg anchors on onsets; it must still be scoreable at the peak fiducial,
    otherwise it would be penalised for measuring a different point on the wave."""
    sig, _, fs = synth_ppg(duration_s=30.0, fs=100.0, hr_bpm=60.0)
    x = preprocess.normalise(preprocess.bandpass(sig, fs))
    from_peak = fiducial.locate(x, fs, REGISTRY["msptdfast"](x, fs), "peak", "peak")
    from_onset = fiducial.locate(x, fs, REGISTRY["qppg"](x, fs), "onset", "peak")
    m = match.match_beats(from_peak, from_onset, tolerance=0.05, lag=0.0)
    assert m.tp / len(from_peak) > 0.95


# ------------------------------------------------------------------ matching
def test_estimate_lag_recovers_known_delay():
    ref = np.cumsum(np.full(120, 0.85))
    for true_lag in (0.05, 0.25, 0.40):
        assert match.estimate_lag(ref, ref + true_lag) == pytest.approx(true_lag, abs=0.005)


def test_match_is_one_to_one():
    ref = np.cumsum(np.full(50, 0.9))
    duplicated = np.sort(np.concatenate([ref, ref + 0.01]))
    m = match.match_beats(ref, duplicated, lag=0.0)
    assert m.tp == len(ref)
    assert len(set(m.test_idx.tolist())) == m.tp
    assert m.fp == len(duplicated) - len(ref)


def test_paired_intervals_skips_missed_beats():
    """A dropped beat must not contribute a doubled interval to the IBI error --
    that would score a detection failure as a timing failure."""
    ref = np.cumsum(np.full(30, 0.8))
    test = np.delete(ref, 10)
    m = match.match_beats(ref, test, lag=0.0)
    rr, pp, ref_idx = match.paired_intervals(ref, test, m)
    assert np.all(np.abs(pp - rr) < 1e-9)
    assert len(rr) == len(ref) - 3  # the gap removes the pairs either side
    assert len(ref_idx) == len(rr)
    assert 9 not in ref_idx.tolist() and 10 not in ref_idx.tolist()


# ------------------------------------------------------------------- metrics
def test_detection_metrics_known_answer():
    ref = np.cumsum(np.full(100, 0.8))
    m = match.match_beats(ref, ref[:90], lag=0.0)
    d = metrics.detection_metrics(m)
    assert d["tp"] == 90 and d["fn"] == 10 and d["fp"] == 0
    assert d["sensitivity"] == pytest.approx(0.9)
    assert d["ppv"] == pytest.approx(1.0)
    assert d["f1"] == pytest.approx(2 * 0.9 / 1.9)


def test_ibi_metrics_known_answer():
    rr = np.full(50, 0.800)
    pp = np.full(50, 0.810)     # a constant +10 ms offset
    out = metrics.ibi_metrics(rr, pp)
    assert out["ibi_bias"] == pytest.approx(10.0)
    assert out["ibi_mae"] == pytest.approx(10.0)
    assert out["ibi_rmse"] == pytest.approx(10.0)
    assert out["ibi_sd"] == pytest.approx(0.0, abs=1e-9)


def test_bland_altman_known_answer():
    a = np.zeros(200)
    b = np.full(200, 5.0)
    b[::2] = 3.0                # mean 4, sd 1
    ba = metrics.bland_altman(a, b)
    assert ba["bias"] == pytest.approx(4.0)
    assert ba["loa_upper"] == pytest.approx(4.0 + 1.96, abs=0.02)


def test_lins_ccc_penalises_bias_where_pearson_does_not():
    rng = np.random.default_rng(0)
    a = rng.normal(50, 10, 300)
    b = a + 20.0                # perfectly correlated, badly biased
    assert np.corrcoef(a, b)[0, 1] == pytest.approx(1.0)
    assert metrics.lins_ccc(a, b) < 0.6


def test_consensus_floor_beats_individual_detectors():
    """The consensus must be at least as good as a typical member, and must key on
    reference-beat index: aligning by position averages unrelated heartbeats and
    inflates the floor above every individual detector."""
    rng = np.random.default_rng(4)
    n = 200
    truth = rng.normal(0.85, 0.04, n)
    rr_by_ref = {i: float(truth[i]) for i in range(n)}

    per_det = {}
    for d in range(5):
        # Each detector sees a different, ragged subset of the intervals.
        idx = np.sort(rng.choice(n, size=n - 20 - d * 5, replace=False))
        noise = rng.normal(0, 0.008, len(idx))
        per_det[f"d{d}"] = (idx, truth[idx] + noise)

    floor = metrics.consensus_floor(per_det, rr_by_ref)
    single = np.sqrt(np.mean((per_det["d0"][1] - truth[per_det["d0"][0]]) ** 2)) * 1000
    assert floor["floor_rmse"] < single
    assert floor["n_intervals"] > 100
