"""Quality control before any benchmarking.

Checks, in order:
  1. record integrity (channels, rate, duration) and .atr plausibility
  2. that decimation 500 -> 100 Hz preserves every pulse
  3. which pleth_* channel has the best SNR -> the primary channel for the benchmark
  4. that the from-scratch TERMA agrees with NeuroKit2's reference "elgendi"
  5. that the from-scratch qppg finds a plausible number of beats
  6. that the best pipeline reaches F1 > 0.99 on the cleanest record

Step 6 is the one that matters most: if alignment or matching were broken, every
detector would look bad and the benchmark would report a plausible-looking lie.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import signal as sg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prv import io, match, pipeline, preprocess  # noqa: E402
from prv.detectors import REGISTRY  # noqa: E402
from prv.detectors import terma as terma_mod  # noqa: E402

RESULTS = ROOT / "results"


def cardiac_snr_db(x: np.ndarray, fs: float, hr_hz: float) -> float:
    """Ratio of power in the cardiac band and its harmonics to everything else."""
    f, p = sg.welch(x, fs=fs, nperseg=min(len(x), 2048))
    band = (f > hr_hz * 0.7) & (f < hr_hz * 3.5)
    return float(10 * np.log10(np.sum(p[band]) / max(np.sum(p[~band]), 1e-12)))


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    records = io.available_records()
    if not records:
        print("no records in data/ -- run scripts/00_fetch_data.py first")
        return 1
    print(f"{len(records)} records available\n")

    problems: list[str] = []
    report: dict = {"n_records": len(records), "records": records}

    # ---------------------------------------------------------------- 1. integrity
    print("[1] record integrity")
    durations: list[float] = []
    hr_meta_delta: list[float] = []
    for name in records:
        r = io.load_record(name)
        durations.append(r.duration_s)
        rr = np.diff(r.ecg_beats_s)
        bad = int(np.sum((rr < 0.30) | (rr > 2.00)))
        hr_atr = 60.0 / float(np.median(rr))
        hrs = [float(r.meta[k]) for k in ("hr_1_start", "hr_1_end") if k in r.meta]
        if hrs:
            hr_meta_delta.append(abs(hr_atr - float(np.mean(hrs))))
        if r.fs_native != 500.0:
            problems.append(f"{name}: fs={r.fs_native}")
        if bad > 0.02 * len(rr):
            problems.append(f"{name}: {bad}/{len(rr)} implausible RR intervals")
    print(f"    duration {np.min(durations):.0f}-{np.max(durations):.0f} s "
          f"(median {np.median(durations):.0f})")
    if hr_meta_delta:
        print(f"    |HR(.atr) - HR(header)| median {np.median(hr_meta_delta):.1f} bpm")
        report["hr_meta_delta_median_bpm"] = float(np.median(hr_meta_delta))
    report["duration_s_median"] = float(np.median(durations))

    # ------------------------------------------------ 2 & 3. decimation + channel SNR
    print("\n[2/3] decimation integrity and channel selection")
    probe = records[: min(6, len(records))]
    snr_by_channel: dict[str, list[float]] = {c: [] for c in io.PLETH_CHANNELS}
    decim_delta: list[int] = []
    for name in probe:
        for ch in io.PLETH_CHANNELS:
            r = io.load_record(name, ppg_channel=ch)
            hr_hz = float(np.median(1.0 / np.diff(r.ecg_beats_s)))
            x100, _ = preprocess.prepare(r.ppg, r.fs_native, 100.0)
            snr_by_channel[ch].append(cardiac_snr_db(x100, 100.0, hr_hz))
            if ch == "pleth_2":
                x500, fs500 = preprocess.prepare(r.ppg, r.fs_native, r.fs_native)
                b500 = REGISTRY["msptdfast"](x500, fs500) / fs500
                b100 = REGISTRY["msptdfast"](x100, 100.0) / 100.0
                # Agreement of the beats themselves: a count that happens to match tells
                # us nothing if decimation moved or replaced pulses.
                mm = match.match_beats(b500, b100, tolerance=0.05, lag=0.0)
                decim_delta.append(1.0 - mm.tp / max(len(b500), 1))

    medians = {c: float(np.median(v)) for c, v in snr_by_channel.items()}
    best_channel = max(medians, key=lambda k: medians[k])
    for c, v in sorted(medians.items(), key=lambda kv: -kv[1]):
        print(f"    {c}: cardiac-band SNR {v:6.2f} dB")
    print(f"    -> primary channel: {best_channel}")
    worst = float(max(decim_delta))
    print(f"    decimation 500->100 Hz: worst beat loss {worst * 100:.2f}% "
          f"(median {np.median(decim_delta) * 100:.2f}%)")
    if worst > 0.02:
        problems.append(f"decimation loses up to {worst * 100:.1f}% of beats")
    report["channel_snr_db"] = medians
    report["primary_channel"] = best_channel
    report["decimation_beat_loss_worst"] = worst

    # -------------------------------------------- 4. TERMA vs NeuroKit2 reference
    print("\n[4] own TERMA vs NeuroKit2 reference implementation")
    import neurokit2 as nk

    agreements: list[float] = []
    for name in probe:
        r = io.load_record(name, ppg_channel=best_channel)
        x, fs = preprocess.prepare(r.ppg, r.fs_native, 100.0)
        mine = terma_mod.detect(x, fs) / fs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = np.asarray(
                nk.ppg_findpeaks(x, sampling_rate=fs, method="elgendi")["PPG_Peaks"],
                dtype=int,
            ) / fs
        m = match.match_beats(ref, mine, tolerance=0.05, lag=0.0)
        se = m.tp / max(len(ref), 1)
        ppv = m.tp / max(len(mine), 1)
        agreements.append(min(se, ppv))
        print(f"    {name}: Se={se * 100:5.1f}% PPV={ppv * 100:5.1f}% "
              f"({len(mine)} vs {len(ref)} beats)")
    if float(np.mean(agreements)) < 0.90:
        problems.append(
            f"TERMA port agrees with NeuroKit2 only {np.mean(agreements) * 100:.0f}%"
        )
    report["terma_agreement"] = float(np.mean(agreements))

    # --------------------------------------------------- 5. qppg beat-count sanity
    print("\n[5] own qppg beat-count plausibility")
    ratios: list[float] = []
    for name in probe:
        r = io.load_record(name, ppg_channel=best_channel)
        x, fs = preprocess.prepare(r.ppg, r.fs_native, 100.0)
        n = len(REGISTRY["qppg"](x, fs))
        ratio = n / len(r.ecg_beats_s)
        ratios.append(ratio)
        print(f"    {name}: {n} onsets vs {len(r.ecg_beats_s)} ECG beats "
              f"(ratio {ratio:.3f})")
    if not (0.90 <= float(np.mean(ratios)) <= 1.10):
        problems.append(f"qppg beat-count ratio {np.mean(ratios):.3f} outside 0.90-1.10")
    report["qppg_count_ratio"] = float(np.mean(ratios))

    # ------------------------------------------------ 6. pipeline sanity (the big one)
    print("\n[6] pipeline sanity: MSPTDfast on the cleanest record")
    best: tuple[str, float, dict] | None = None
    for name in records[: min(10, len(records))]:
        r = io.load_record(name, ppg_channel=best_channel)
        res = pipeline.run(r, "msptdfast", correction="none")
        f1 = float(res.row.get("f1", np.nan))
        if best is None or (np.isfinite(f1) and f1 > best[1]):
            best = (name, f1, res.row)
    assert best is not None
    print(f"    cleanest: {best[0]}  F1={best[1] * 100:.2f}%  "
          f"IBI RMSE={best[2]['ibi_rmse']:.1f} ms  lag={best[2]['lag_ms']:.0f} ms")
    if not (np.isfinite(best[1]) and best[1] > 0.99):
        problems.append(f"best-case F1 only {best[1] * 100:.2f}% -- alignment suspect")
    report["best_case_record"] = best[0]
    report["best_case_f1"] = float(best[1])
    report["best_case_ibi_rmse_ms"] = float(best[2]["ibi_rmse"])
    report["best_case_lag_ms"] = float(best[2]["lag_ms"])

    # ------------------------------------------------------------------- verdict
    print("\n" + "=" * 62)
    if problems:
        print("QC PROBLEMS:")
        for p in problems:
            print(f"  ! {p}")
    else:
        print("all QC checks passed")
    report["problems"] = problems
    (RESULTS / "qc_report.json").write_text(json.dumps(report, indent=2))
    print(f"written {RESULTS / 'qc_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
