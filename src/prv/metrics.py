"""The six levels of the validation protocol.

Levels follow the structure the German report prescribes and the published benchmarks
skip: detection F1 is kept only for comparability with Charlton 2022, while the ranking
runs on beat-to-beat IBI error and HRV endpoint error at matched coverage.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from .match import MatchResult


# ---------------------------------------------------------------- Level A: detection
def detection_metrics(m: MatchResult) -> dict[str, float]:
    tp, fp, fn = m.tp, m.fp, m.fn
    se = tp / (tp + fn) if (tp + fn) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    f1 = 2 * se * ppv / (se + ppv) if (se and ppv and (se + ppv) > 0) else np.nan
    return {"tp": tp, "fp": fp, "fn": fn, "sensitivity": se, "ppv": ppv, "f1": f1,
            "lag_ms": m.lag_s * 1000.0}


# -------------------------------------------------------------- Level B: beat timing
def ibi_metrics(rr_s: np.ndarray, pp_s: np.ndarray) -> dict[str, float]:
    rr = np.asarray(rr_s, dtype=float) * 1000.0
    pp = np.asarray(pp_s, dtype=float) * 1000.0
    ok = np.isfinite(rr) & np.isfinite(pp)
    rr, pp = rr[ok], pp[ok]
    if rr.size < 2:
        return {k: np.nan for k in ("ibi_mae", "ibi_rmse", "ibi_bias", "ibi_sd", "n_ibi")}
    d = pp - rr
    return {
        "ibi_mae": float(np.mean(np.abs(d))),
        "ibi_rmse": float(np.sqrt(np.mean(d**2))),
        "ibi_bias": float(np.mean(d)),
        "ibi_sd": float(np.std(d, ddof=1)),
        "n_ibi": int(rr.size),
    }


# ------------------------------------------------------------- Level C: agreement
def bland_altman(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Bias and 95% limits of agreement for b relative to a (reference)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 3:
        return {"bias": np.nan, "loa_lower": np.nan, "loa_upper": np.nan, "n": int(a.size)}
    d = b - a
    bias, sd = float(np.mean(d)), float(np.std(d, ddof=1))
    return {"bias": bias, "loa_lower": bias - 1.96 * sd, "loa_upper": bias + 1.96 * sd,
            "n": int(a.size)}


def lins_ccc(a: np.ndarray, b: np.ndarray) -> float:
    """Lin's concordance correlation coefficient: penalises departure from the identity
    line, unlike Pearson r which a proportionally biased method can still ace."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 3:
        return np.nan
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    cov = np.cov(a, b, ddof=1)[0, 1]
    denom = va + vb + (np.mean(a) - np.mean(b)) ** 2
    return float(2 * cov / denom) if denom > 0 else np.nan


def tost_equivalence(a: np.ndarray, b: np.ndarray, bound_frac: float = 0.10
                     ) -> dict[str, float]:
    """Two one-sided tests. Equivalence bound is a fraction of the reference mean --
    an equivalence claim needs a bound, and stating it is the point."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 5:
        return {"tost_p": np.nan, "equivalent": np.nan, "bound": np.nan}
    d = b - a
    bound = bound_frac * float(np.mean(np.abs(a)))
    se = float(np.std(d, ddof=1) / np.sqrt(len(d)))
    if se == 0:
        return {"tost_p": 0.0, "equivalent": 1.0, "bound": bound}
    df = len(d) - 1
    p_low = stats.t.sf((np.mean(d) + bound) / se, df)
    p_high = stats.t.cdf((np.mean(d) - bound) / se, df)
    p = float(max(p_low, p_high))
    return {"tost_p": p, "equivalent": float(p < 0.05), "bound": bound}


# ---------------------------------------------------------- Level D: HRV endpoints
def hrv_error(ref: dict[str, float], test: dict[str, float], keys: list[str]
              ) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in keys:
        r, t = ref.get(k, np.nan), test.get(k, np.nan)
        out[f"{k}_ref"] = r
        out[f"{k}_test"] = t
        out[f"{k}_err"] = t - r if (np.isfinite(r) and np.isfinite(t)) else np.nan
        out[f"{k}_abserr"] = abs(t - r) if (np.isfinite(r) and np.isfinite(t)) else np.nan
        out[f"{k}_pcterr"] = (100.0 * (t - r) / r) if (np.isfinite(r) and np.isfinite(t)
                                                       and r != 0) else np.nan
    return out


# ------------------------------------------------- PRV floor (physiology vs algorithm)
def consensus_floor(per_detector: dict[str, tuple[np.ndarray, np.ndarray]],
                    rr_by_ref: dict[int, float], min_detectors: int = 3
                    ) -> dict[str, float]:
    """Estimate the irreducible PRV-vs-HRV error.

    A constant pulse arrival time cancels inside an interval, so the residual PPI-RRI
    error is detector jitter *combined with* beat-to-beat PAT variability. Taking the
    median PPI across detectors gives a consensus fiducial whose remaining error is
    dominated by physiology rather than by any single algorithm; each detector's excess
    RMSE over this floor is then its own attributable contribution.

    Intervals are keyed by reference-beat index, never by position: detectors produce
    different numbers of intervals covering different beats, so a positional median
    would average unrelated heartbeats together and inflate the floor beyond any
    individual detector.

    `per_detector` maps detector key -> (ref_idx, pp_seconds).
    This is an estimate, not ground truth: it assumes detector errors are independent
    enough for the median to cancel them. It is reported as such.
    """
    votes: dict[int, list[float]] = {}
    for ref_idx, pp in per_detector.values():
        for i, v in zip(np.asarray(ref_idx, dtype=int), np.asarray(pp, dtype=float)):
            if np.isfinite(v):
                votes.setdefault(int(i), []).append(float(v))

    keys = [k for k, v in votes.items() if len(v) >= min_detectors and k in rr_by_ref]
    if len(keys) < 5:
        return {"floor_rmse": np.nan, "floor_mae": np.nan, "n_intervals": len(keys),
                "n_detectors": len(per_detector)}

    consensus = np.array([np.median(votes[k]) for k in keys])
    truth = np.array([rr_by_ref[k] for k in keys])
    d = (consensus - truth) * 1000.0
    return {"floor_rmse": float(np.sqrt(np.mean(d**2))),
            "floor_mae": float(np.mean(np.abs(d))),
            "n_intervals": len(keys), "n_detectors": len(per_detector)}
