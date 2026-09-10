"""The vectorised MSPTD must stay bit-identical to the reference implementation.

This is the test that licenses the optimisation. `src/prv/detectors/msptd.py` replaces
NeuroKit2's doubly nested Python loop with shifted-array comparisons; the claim is
exact equivalence, not approximation, so the test asserts array equality rather than
closeness. If NeuroKit2 ever changes its algorithm this test is where it surfaces.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest
from conftest import synth_ppg

from prv import preprocess
from prv.detectors import msptd

nk_bishop = importlib.import_module("neurokit2.ppg.ppg_findpeaks")._ppg_findpeaks_bishop


def windows_from_synthetic(n_windows: int = 8, win: int = 600):
    sig, _, fs = synth_ppg(duration_s=90.0, fs=100.0, hr_bpm=68.0, noise=0.02, seed=5)
    x = preprocess.normalise(preprocess.bandpass(sig, fs))
    step = (len(x) - win) // n_windows
    return [x[i * step: i * step + win] for i in range(n_windows)]


@pytest.mark.parametrize("seg_index", range(8))
def test_peaks_and_onsets_match_reference(seg_index):
    seg = windows_from_synthetic()[seg_index]
    ref_peaks, ref_onsets = nk_bishop(seg)
    new_peaks, new_onsets = msptd.detect_peaks_onsets(seg)
    assert np.array_equal(np.asarray(ref_peaks, dtype=int), new_peaks)
    assert np.array_equal(np.asarray(ref_onsets, dtype=int), new_onsets)


@pytest.mark.parametrize("n", [4, 5, 7, 16, 37, 120, 331, 600, 601])
def test_matches_reference_at_awkward_lengths(n):
    """Scale range and matrix shape both depend on N; off-by-one errors hide here."""
    rng = np.random.default_rng(n)
    seg = np.sin(np.linspace(0, 8 * np.pi, n)) + 0.1 * rng.normal(0, 1, n)
    ref_peaks, ref_onsets = nk_bishop(seg)
    new_peaks, new_onsets = msptd.detect_peaks_onsets(seg)
    assert np.array_equal(np.asarray(ref_peaks, dtype=int), new_peaks)
    assert np.array_equal(np.asarray(ref_onsets, dtype=int), new_onsets)


def test_matches_reference_on_pure_noise():
    rng = np.random.default_rng(99)
    for _ in range(5):
        seg = rng.normal(0, 1, 400)
        ref_peaks, _ = nk_bishop(seg)
        new_peaks, _ = msptd.detect_peaks_onsets(seg)
        assert np.array_equal(np.asarray(ref_peaks, dtype=int), new_peaks)


def test_degenerate_inputs_do_not_raise():
    for seg in (np.zeros(0), np.zeros(3), np.zeros(50), np.ones(50)):
        peaks, onsets = msptd.detect_peaks_onsets(seg)
        assert peaks.dtype.kind == "i" and onsets.dtype.kind == "i"


def test_scalogram_leaves_final_scale_unpopulated():
    """The reference loops `range(1, L)`, so row L-1 is never written. Reproducing that
    exactly matters: it shifts which scale argmax selects."""
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 200)
    m_max, m_min = msptd.lms_scalograms(x, 99)
    assert not m_max[-1].any()
    assert not m_min[-1].any()
