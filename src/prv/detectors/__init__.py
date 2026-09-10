"""Beat detector registry.

Every detector takes a band-passed PPG at the pipeline rate and returns integer sample
indices plus an anchor type. The anchor type matters: MSPTD-family detectors and TERMA
land on systolic peaks, while qppg lands on pulse onsets. `fiducial.py` normalises both
to a common fiducial so the comparison between detectors is fair.
"""
from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from . import msptd as _msptd
from . import qppg as _qppg
from . import terma as _terma

AnchorType = Literal["peak", "onset"]


@dataclass(frozen=True)
class Detector:
    key: str
    label: str
    anchor: AnchorType
    fn: Callable[[np.ndarray, float], np.ndarray]
    source: str

    def __call__(self, x: np.ndarray, fs: float) -> np.ndarray:
        return np.asarray(self.fn(x, fs), dtype=int)


def windowed(fn: Callable[[np.ndarray, float], np.ndarray], win_s: float = 6.0,
             overlap_s: float = 1.0, refractory_s: float = 0.25
             ) -> Callable[[np.ndarray, float], np.ndarray]:
    """Apply a detector over overlapping windows and merge the results.

    Needed for the original MSPTD, whose local-maxima scalogram is O(N^2/2): on a 508 s
    record at 100 Hz that is a ~10 GB matrix. Charlton's own benchmark applies detectors
    to windows for the same reason, and MSPTDfast v2 makes its 6 s window an explicit
    design parameter. Using that same 6 s window for MSPTD keeps the MSPTD-vs-v2
    comparison honest: the only remaining differences are v2's own optimisations
    (20 Hz downsampling and the >30 bpm scale limit), which is exactly what we want to
    measure.
    """

    def run(x: np.ndarray, fs: float) -> np.ndarray:
        n = len(x)
        w = int(round(win_s * fs))
        step = max(1, w - int(round(overlap_s * fs)))
        if n <= w:
            return np.asarray(fn(x, fs), dtype=int)

        found: list[int] = []
        for start in range(0, n, step):
            stop = min(n, start + w)
            if stop - start < int(round(2.0 * fs)):
                break
            seg = x[start:stop]
            try:
                idx = np.asarray(fn(seg, fs), dtype=int)
            except Exception:
                continue
            # Drop detections in the overlap margin except at the very edges, so a beat
            # is claimed by the window that sees it with the most context.
            margin = 0 if start == 0 else int(round(0.5 * overlap_s * fs))
            keep = idx >= margin
            found.extend((idx[keep] + start).tolist())

        if not found:
            return np.array([], dtype=int)
        merged = np.unique(np.asarray(found, dtype=int))
        # Collapse duplicates that survived the margin rule.
        out = [int(merged[0])]
        min_gap = int(round(refractory_s * fs))
        for i in merged[1:]:
            if i - out[-1] >= min_gap:
                out.append(int(i))
        return np.asarray(out, dtype=int)

    return run


def _nk_findpeaks(method: str) -> Callable[[np.ndarray, float], np.ndarray]:
    def run(x: np.ndarray, fs: float) -> np.ndarray:
        import neurokit2 as nk

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # NeuroKit2's charlton path indexes with sampling_rate, so it must be int.
            out = nk.ppg_findpeaks(x, sampling_rate=int(round(fs)), method=method, show=False)
        return np.asarray(out["PPG_Peaks"], dtype=int)

    return run


def _heartpy_detect(x: np.ndarray, fs: float) -> np.ndarray:
    import heartpy as hp

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            wd, _ = hp.process(np.asarray(x, dtype=float), sample_rate=float(fs))
        except Exception:
            return np.array([], dtype=int)
    peaks = np.asarray(wd.get("peaklist", []), dtype=int)
    removed = set(np.asarray(wd.get("removed_beats", []), dtype=int).tolist())
    # HeartPy flags rejected beats rather than dropping them; honour its own judgement.
    return np.array([p for p in peaks if p not in removed], dtype=int)


def _findpeaks_baseline(x: np.ndarray, fs: float) -> np.ndarray:
    """The naive baseline: scipy.find_peaks with a physiological refractory period,
    as used by Zuern et al. Included to show what the sophisticated detectors buy."""
    from scipy.signal import find_peaks

    # 0.3 s refractory => at most 200 bpm.
    peaks, _ = find_peaks(x, distance=int(round(0.3 * fs)), prominence=0.3 * np.std(x))
    return np.asarray(peaks, dtype=int)


REGISTRY: dict[str, Detector] = {
    "msptdfast": Detector(
        key="msptdfast",
        label="MSPTDfast v2",
        anchor="peak",
        fn=_nk_findpeaks("charlton"),
        source="NeuroKit2 (author-contributed; Charlton 2025)",
    ),
    "msptd": Detector(
        key="msptd",
        label="MSPTD",
        anchor="peak",
        fn=windowed(_msptd.detect, win_s=6.0, overlap_s=1.0),
        source="own vectorised port (Bishop & Ercole 2018), 6 s windows; "
               "verified bit-identical to NeuroKit2 method='bishop'",
    ),
    "qppg": Detector(
        key="qppg",
        label="qppg (slope-sum)",
        anchor="onset",
        fn=_qppg.detect,
        source="own port (Zong 2003)",
    ),
    "terma": Detector(
        key="terma",
        label="TERMA",
        anchor="peak",
        fn=_terma.detect,
        source="own port (Elgendi 2013)",
    ),
    "heartpy": Detector(
        key="heartpy",
        label="HeartPy",
        anchor="peak",
        fn=_heartpy_detect,
        source="HeartPy package",
    ),
    "findpeaks": Detector(
        key="findpeaks",
        label="find_peaks baseline",
        anchor="peak",
        fn=_findpeaks_baseline,
        source="baseline (Zuern-style)",
    ),
}

MAIN_FIVE = ["msptdfast", "msptd", "qppg", "terma", "heartpy"]
ALL_KEYS = MAIN_FIVE + ["findpeaks"]
