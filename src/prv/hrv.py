"""HRV / PRV indices computed directly from an interval tachogram.

Frequency-domain indices are computed two ways: PCHIP resampling to 4 Hz plus Welch
(the conventional route), and Lomb-Scargle on the irregular series (which tolerates the
gaps left by rejected beats, as Zuern et al. used). Reporting both matters because the
two disagree exactly when coverage is poor -- which is the regime this benchmark cares
about.

Band definitions are the Task Force (1996) ones: LF 0.04-0.15 Hz, HF 0.15-0.40 Hz.
"""
from __future__ import annotations

import numpy as np
from scipy import interpolate, signal

LF_BAND = (0.04, 0.15)
HF_BAND = (0.15, 0.40)
RESAMPLE_HZ = 4.0


def time_domain(ibi_s: np.ndarray) -> dict[str, float]:
    ibi = np.asarray(ibi_s, dtype=float)
    ibi = ibi[np.isfinite(ibi)]
    if ibi.size < 3:
        return {k: np.nan for k in ("mean_nn", "sdnn", "rmssd", "pnn50", "sdsd", "mean_hr")}
    ms = ibi * 1000.0
    d = np.diff(ms)
    return {
        "mean_nn": float(np.mean(ms)),
        "sdnn": float(np.std(ms, ddof=1)),
        "rmssd": float(np.sqrt(np.mean(d**2))),
        "pnn50": float(100.0 * np.mean(np.abs(d) > 50.0)),
        "sdsd": float(np.std(d, ddof=1)) if d.size > 1 else np.nan,
        "mean_hr": float(60.0 / np.mean(ibi)),
    }


def nonlinear(ibi_s: np.ndarray) -> dict[str, float]:
    ibi = np.asarray(ibi_s, dtype=float)
    ibi = ibi[np.isfinite(ibi)]
    out = {k: np.nan for k in ("sd1", "sd2", "sd1_sd2", "sampen", "dfa_a1")}
    if ibi.size < 10:
        return out
    ms = ibi * 1000.0
    d = np.diff(ms)
    sd1 = float(np.sqrt(0.5) * np.std(d, ddof=1))
    sdnn = float(np.std(ms, ddof=1))
    sd2_sq = 2.0 * sdnn**2 - sd1**2
    sd2 = float(np.sqrt(sd2_sq)) if sd2_sq > 0 else np.nan
    out["sd1"], out["sd2"] = sd1, sd2
    out["sd1_sd2"] = sd1 / sd2 if sd2 and np.isfinite(sd2) and sd2 > 0 else np.nan
    out["sampen"] = sample_entropy(ms)
    out["dfa_a1"] = dfa_alpha1(ms)
    return out


def sample_entropy(x: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < m + 2:
        return np.nan
    r = r_factor * np.std(x, ddof=1)
    if r <= 0:
        return np.nan

    def count(mm: int) -> int:
        # Chebyshev distance between all embedded vector pairs, excluding self-matches.
        emb = np.lib.stride_tricks.sliding_window_view(x, mm)
        total = 0
        for i in range(len(emb) - 1):
            dist = np.max(np.abs(emb[i + 1 :] - emb[i]), axis=1)
            total += int(np.sum(dist <= r))
        return total

    a, b = count(m + 1), count(m)
    if a == 0 or b == 0:
        return np.nan
    return float(-np.log(a / b))


def dfa_alpha1(x: np.ndarray, lo: int = 4, hi: int = 16) -> float:
    """Short-term detrended fluctuation analysis scaling exponent."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < hi * 2:
        return np.nan
    y = np.cumsum(x - np.mean(x))
    scales = np.arange(lo, hi + 1)
    fluct = []
    for s in scales:
        nseg = n // s
        if nseg < 2:
            fluct.append(np.nan)
            continue
        seg = y[: nseg * s].reshape(nseg, s)
        t = np.arange(s)
        resid = []
        for row in seg:
            coef = np.polyfit(t, row, 1)
            resid.append(row - np.polyval(coef, t))
        fluct.append(float(np.sqrt(np.mean(np.asarray(resid) ** 2))))
    fluct = np.asarray(fluct, dtype=float)
    ok = np.isfinite(fluct) & (fluct > 0)
    if ok.sum() < 3:
        return np.nan
    slope = np.polyfit(np.log(scales[ok]), np.log(fluct[ok]), 1)[0]
    return float(slope)


def _bandpower(freqs: np.ndarray, psd: np.ndarray, band: tuple[float, float]) -> float:
    sel = (freqs >= band[0]) & (freqs < band[1])
    return float(np.trapezoid(psd[sel], freqs[sel])) if sel.sum() > 1 else np.nan


def frequency_domain(ibi_s: np.ndarray, times_s: np.ndarray | None = None,
                     method: str = "welch") -> dict[str, float]:
    """LF/HF power. method: 'welch' (PCHIP -> 4 Hz) or 'lombscargle' (irregular)."""
    ibi = np.asarray(ibi_s, dtype=float)
    out = {k: np.nan for k in ("lf", "hf", "lf_hf", "total_power")}
    if ibi.size < 20:
        return out

    if times_s is None:
        times_s = np.cumsum(ibi)
    t = np.asarray(times_s, dtype=float)
    ms = ibi * 1000.0
    ok = np.isfinite(t) & np.isfinite(ms)
    t, ms = t[ok], ms[ok]
    if t.size < 20 or (t[-1] - t[0]) < 60.0:
        return out

    if method == "welch":
        grid = np.arange(t[0], t[-1], 1.0 / RESAMPLE_HZ)
        vals = interpolate.PchipInterpolator(t, ms)(grid)
        vals = vals - np.mean(vals)
        nper = min(len(vals), int(RESAMPLE_HZ * 120))
        freqs, psd = signal.welch(vals, fs=RESAMPLE_HZ, nperseg=nper,
                                  noverlap=nper // 2, detrend="linear")
    else:
        freqs = np.linspace(0.01, 0.5, 500)
        vals = ms - np.mean(ms)
        psd = signal.lombscargle(t, vals, 2 * np.pi * freqs, normalize=False)
        psd = psd * 2.0 / len(t)

    lf, hf = _bandpower(freqs, psd, LF_BAND), _bandpower(freqs, psd, HF_BAND)
    out.update(
        lf=lf,
        hf=hf,
        lf_hf=(lf / hf) if (hf and np.isfinite(hf) and hf > 0) else np.nan,
        total_power=_bandpower(freqs, psd, (0.0033, 0.40)),
    )
    return out


def all_indices(ibi_s: np.ndarray, times_s: np.ndarray | None = None) -> dict[str, float]:
    out: dict[str, float] = {}
    out.update(time_domain(ibi_s))
    out.update(nonlinear(ibi_s))
    out.update(frequency_domain(ibi_s, times_s, method="welch"))
    ls = frequency_domain(ibi_s, times_s, method="lombscargle")
    out.update({f"{k}_ls": v for k, v in ls.items()})
    return out


HRV_KEYS = ["mean_nn", "sdnn", "rmssd", "pnn50", "sdsd", "sd1", "sd2", "sd1_sd2",
            "sampen", "dfa_a1", "lf", "hf", "lf_hf"]
