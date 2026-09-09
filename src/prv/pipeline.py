"""End-to-end PRV pipeline: one record, one configuration, one row of results.

Order of stages is fixed and deliberate:

    decimate -> band-pass -> detect -> fiducial (+sub-sample) -> SQI gate -> correct -> HRV

Note the placement of the sub-sample refinement. The compass report recommends splining
the PPG up to >=250 Hz *before* running MSPTDfast, which would discard the very
optimisation that makes MSPTDfast fast (it downsamples to 20 Hz internally). Detection
happens at the operating rate and only the fiducial is refined against the higher-rate
grid -- detect low, refine high.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import correct as correct_mod
from . import fiducial as fiducial_mod
from . import hrv as hrv_mod
from . import match as match_mod
from . import metrics as metrics_mod
from . import preprocess, sqi
from .detectors import REGISTRY
from .io import Record


@dataclass
class RunResult:
    record: str
    detector: str
    fiducial: str
    correction: str
    coverage_target: float
    fs: float
    beats_s: np.ndarray = field(repr=False)
    ref_beats_s: np.ndarray = field(repr=False)
    rr_s: np.ndarray = field(repr=False)
    pp_s: np.ndarray = field(repr=False)
    pair_ref_idx: np.ndarray = field(repr=False, default_factory=lambda: np.array([], int))
    row: dict = field(default_factory=dict, repr=False)


def run(rec: Record, detector_key: str, *, fiducial: str = fiducial_mod.DEFAULT_FIDUCIAL,
        correction: str = correct_mod.DEFAULT_STRATEGY, coverage: float = 1.0,
        fs: float = preprocess.FS_PIPELINE, sub_sample: bool = True) -> RunResult:
    det = REGISTRY[detector_key]

    # --- front end -----------------------------------------------------------
    x, fs_actual = preprocess.prepare(rec.ppg, rec.fs_native, fs)

    # --- detection (timed for the cost axis) ---------------------------------
    t0 = time.perf_counter()
    anchors = det(x, fs_actual)
    detect_s = time.perf_counter() - t0

    if anchors.size < 5:
        return RunResult(rec.name, detector_key, fiducial, correction, coverage, fs_actual,
                         np.array([]), rec.ecg_beats_s, np.array([]), np.array([]),
                         np.array([], dtype=int),
                         {"record": rec.name, "detector": detector_key, "failed": True})

    # --- fiducial ------------------------------------------------------------
    beats = fiducial_mod.locate(x, fs_actual, anchors, det.anchor, fiducial)
    if not sub_sample:
        # Snap back onto the sample grid: this is the E2 control that isolates how much
        # of the accuracy comes from sub-sample interpolation alone.
        beats = np.round(beats * fs_actual) / fs_actual

    # --- quality gate --------------------------------------------------------
    score = sqi.quality_score(x, fs_actual)
    mask = sqi.coverage_mask(score, coverage)
    spans = sqi.windows_to_time_mask(mask)
    if coverage < 1.0:
        beats_gated = sqi.select_beats(beats, spans)
        ref_gated = sqi.select_beats(rec.ecg_beats_s, spans)
    else:
        beats_gated, ref_gated = beats, rec.ecg_beats_s

    analysable = (len(spans) * sqi.WINDOW_S) / rec.duration_s if rec.duration_s else np.nan

    # --- correction ----------------------------------------------------------
    corr = correct_mod.STRATEGIES[correction](beats_gated)
    # The reference gets the hard physiological screen only; it is the gold standard
    # and must not be "corrected" towards the thing being tested.
    ref_corr = correct_mod.none(ref_gated)

    # --- levels A-B ----------------------------------------------------------
    m = match_mod.match_beats(ref_corr.beats, corr.beats)
    rr, pp, pair_ref_idx = match_mod.paired_intervals(ref_corr.beats, corr.beats, m)

    row: dict = {
        "record": rec.name,
        "detector": detector_key,
        "detector_label": det.label,
        "fiducial": fiducial,
        "correction": correction,
        "coverage_target": coverage,
        "fs": fs_actual,
        "sub_sample": sub_sample,
        "failed": False,
        "n_detected": int(len(beats)),
        "n_ref": int(len(ref_corr.beats)),
        "analysable_fraction": analysable,
        "fraction_corrected": corr.fraction_corrected,
        "detect_s_per_min": detect_s / (rec.duration_s / 60.0) if rec.duration_s else np.nan,
    }
    row.update(metrics_mod.detection_metrics(m))
    row.update(metrics_mod.ibi_metrics(rr, pp))

    # --- level D: HRV endpoints ---------------------------------------------
    ref_hrv = hrv_mod.all_indices(ref_corr.ibi, ref_corr.times)
    test_hrv = hrv_mod.all_indices(corr.ibi, corr.times)
    row.update(metrics_mod.hrv_error(ref_hrv, test_hrv, hrv_mod.HRV_KEYS))

    return RunResult(rec.name, detector_key, fiducial, correction, coverage, fs_actual,
                     corr.beats, ref_corr.beats, rr, pp, pair_ref_idx, row)
