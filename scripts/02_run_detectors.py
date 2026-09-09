"""Main benchmark sweep: every detector on every record, at several coverage levels.

Writes results/benchmark_main.csv (one row per record x detector x coverage) and
results/prv_floor.csv (the consensus-fiducial estimate of the irreducible PRV error).

The coverage sweep exists so the leaderboard can be read at *matched* coverage. A
detector that abstains on the hard segments will look excellent at 60% coverage and
mediocre at 100%; comparing detectors at different coverage is the mistake the German
report calls out, so the sweep makes it visible instead of hiding it.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prv import io, metrics, pipeline  # noqa: E402
from prv.detectors import ALL_KEYS  # noqa: E402

RESULTS = ROOT / "results"
COVERAGES = [1.0, 0.9, 0.75, 0.5]
DEFAULT_CHANNEL = "pleth_2"


def primary_channel() -> str:
    qc = RESULTS / "qc_report.json"
    if qc.exists():
        try:
            return json.loads(qc.read_text()).get("primary_channel", DEFAULT_CHANNEL)
        except Exception:
            pass
    return DEFAULT_CHANNEL


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=int, default=0, help="limit record count (0 = all)")
    ap.add_argument("--correction", default="lipponen_tarvainen")
    ap.add_argument("--coverages", default="", help="comma-separated; default 1.0,0.9,0.75,0.5")
    ap.add_argument("--suffix", default="", help="suffix for the output CSV names")
    args = ap.parse_args()

    coverages = ([float(c) for c in args.coverages.split(",")] if args.coverages
                 else COVERAGES)

    warnings.filterwarnings("ignore")
    RESULTS.mkdir(exist_ok=True)

    channel = primary_channel()
    records = io.available_records()
    if args.records:
        records = records[: args.records]
    if not records:
        print("no records available")
        return 1

    print(f"benchmark: {len(records)} records x {len(ALL_KEYS)} detectors "
          f"x {len(coverages)} coverage levels, channel={channel}, "
          f"correction={args.correction}")

    rows: list[dict] = []
    floor_rows: list[dict] = []

    for ri, name in enumerate(records, 1):
        rec = io.load_record(name, ppg_channel=channel)
        print(f"[{ri}/{len(records)}] {name} "
              f"({rec.duration_s:.0f}s, {len(rec.ecg_beats_s)} ref beats, "
              f"polarity {rec.polarity:+.0f})", flush=True)

        for cov in coverages:
            per_det: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            rr_by_ref: dict[int, float] = {}

            for key in ALL_KEYS:
                res = pipeline.run(rec, key, correction=args.correction, coverage=cov)
                row = dict(res.row)
                row["channel"] = channel
                rows.append(row)
                if res.pp_s.size > 5:
                    per_det[key] = (res.pair_ref_idx, res.pp_s)
                    # Reference intervals keyed by their first beat, so detectors that
                    # cover different beats can still be compared interval by interval.
                    for i, v in zip(res.pair_ref_idx, res.rr_s):
                        rr_by_ref.setdefault(int(i), float(v))

            # PRV floor: the consensus fiducial across detectors, whose residual error
            # against ECG is dominated by pulse-arrival-time variability rather than by
            # any single algorithm.
            if rr_by_ref and len(per_det) >= 3:
                fl = metrics.consensus_floor(per_det, rr_by_ref)
                fl.update(record=name, coverage_target=cov)
                floor_rows.append(fl)

    df = pd.DataFrame(rows)
    main_out = RESULTS / f"benchmark_main{args.suffix}.csv"
    df.to_csv(main_out, index=False)
    pd.DataFrame(floor_rows).to_csv(RESULTS / f"prv_floor{args.suffix}.csv", index=False)
    print(f"\nwrote {len(df)} rows -> {main_out}")

    ok = df[(df.coverage_target == 1.0) & (~df.failed)]
    summary = ok.groupby("detector_label").agg(
        f1=("f1", "median"),
        ibi_rmse=("ibi_rmse", "median"),
        ibi_mae=("ibi_mae", "median"),
        rmssd_abserr=("rmssd_abserr", "median"),
        sdnn_abserr=("sdnn_abserr", "median"),
        s_per_min=("detect_s_per_min", "median"),
    ).sort_values("ibi_rmse")
    print("\nmedian across records at 100% coverage:")
    print(summary.to_string(float_format=lambda v: f"{v:8.3f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
