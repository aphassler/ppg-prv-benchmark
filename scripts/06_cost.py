"""Measure detector cost properly: serial, single process, repeated.

Timing collected inside the parallel sweep is worthless as a cost measure. With 20
workers contending for 10 physical cores, a detector's wall clock reflects scheduling
pressure rather than work done -- in one run MSPTDfast measured 6x *more* expensive than
MSPTD, which does roughly 47x more scalogram work. So cost gets its own pass, with the
machine otherwise idle and one process only.

It also settles a comparability question the optimisation created. MSPTD here is a
vectorised port; MSPTDfast still runs NeuroKit2's reference loops. A cost column mixing
the two is not an algorithmic comparison, so both MSPTD implementations are timed and
reported side by side.

    python scripts/06_cost.py --repeats 5
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prv import io, preprocess  # noqa: E402
from prv.detectors import ALL_KEYS, REGISTRY, windowed  # noqa: E402

RESULTS = ROOT / "results"


def primary_channel() -> str:
    qc = RESULTS / "qc_report.json"
    if qc.exists():
        try:
            return json.loads(qc.read_text()).get("primary_channel", "pleth_2")
        except Exception:
            pass
    return "pleth_2"


def time_fn(fn, x, fs, repeats: int) -> tuple[float, float]:
    """Return (best, median) seconds over `repeats` runs.

    The best of several runs is the cleanest estimate of the work itself; the median is
    reported alongside so an unstable measurement is visible rather than hidden.
    """
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(x, fs)
        times.append(time.perf_counter() - t0)
    return float(np.min(times)), float(np.median(times))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    channel = primary_channel()
    records = io.available_records()[: args.records]
    if not records:
        print("no records available")
        return 1

    # The reference MSPTD, for the like-for-like comparison against MSPTDfast.
    nk_findpeaks = importlib.import_module("neurokit2.ppg.ppg_findpeaks")

    def msptd_reference(x, fs):
        return np.asarray(nk_findpeaks._ppg_findpeaks_bishop(x)[0], dtype=int)

    candidates = [(k, REGISTRY[k].label, REGISTRY[k]) for k in ALL_KEYS]
    candidates.append(
        ("msptd_reference", "MSPTD (NeuroKit2 reference loops)",
         windowed(msptd_reference, win_s=6.0, overlap_s=1.0))
    )

    print(f"cost: {len(records)} records x {args.repeats} repeats, serial, "
          f"channel={channel}\n")

    rows = []
    for name in records:
        rec = io.load_record(name, ppg_channel=channel)
        x, fs = preprocess.prepare(rec.ppg, rec.fs_native, 100.0)
        minutes = rec.duration_s / 60.0
        for key, label, fn in candidates:
            best, med = time_fn(fn, x, fs, args.repeats)
            rows.append({"record": name, "detector": key, "detector_label": label,
                         "best_s": best, "median_s": med,
                         "best_s_per_min": best / minutes,
                         "median_s_per_min": med / minutes})

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "detector_cost.csv", index=False)

    summary = df.groupby("detector_label").agg(
        s_per_min_best=("best_s_per_min", "median"),
        s_per_min_median=("median_s_per_min", "median"),
    ).sort_values("s_per_min_best")
    print(summary.to_string(float_format=lambda v: f"{v:10.4f}"))

    ref = summary.loc["MSPTD (NeuroKit2 reference loops)", "s_per_min_best"]
    fast = summary.loc["MSPTDfast v2", "s_per_min_best"]
    ours = summary.loc["MSPTD", "s_per_min_best"]
    print(f"\nMSPTDfast v2 vs MSPTD reference implementation: {100 * fast / ref:.1f}% "
          f"(Charlton reports 5.3-35.9% across datasets)")
    print(f"vectorised MSPTD vs its reference implementation: {ref / ours:.0f}x faster")
    print(f"MSPTDfast v2 vs vectorised MSPTD: {100 * fast / ours:.0f}% "
          f"-- not like-for-like, MSPTDfast still runs the reference loops")
    print(f"\nwrote {RESULTS / 'detector_cost.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
