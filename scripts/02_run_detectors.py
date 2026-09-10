"""Main benchmark sweep: every detector on every record, at several coverage levels.

Writes results/benchmark_main{suffix}.csv (one row per record x detector x coverage) and
results/prv_floor{suffix}.csv (the consensus-fiducial estimate of the irreducible PRV
error).

The coverage sweep exists so the leaderboard can be read at *matched* coverage. A
detector that abstains on the hard segments will look excellent at 60% coverage and
mediocre at 100%; comparing detectors at different coverage is the mistake the German
report calls out, so the sweep makes it visible instead of hiding it.

Records are independent, so the sweep fans out across processes. Within a record the
front end and every detector are computed once and reused across coverage levels --
detection does not depend on coverage, and it is roughly two thirds of the cost.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

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


def process_record(job: tuple[str, str, str, list[float]]) -> tuple[list[dict], list[dict]]:
    """One record, every detector, every coverage level. Runs in a worker process.

    Defined at module level so Windows' spawn start method can pickle it.
    """
    name, channel, correction, coverages = job
    warnings.filterwarnings("ignore")

    rec = io.load_record(name, ppg_channel=channel)
    cache = pipeline.build_cache(rec, ALL_KEYS)

    rows: list[dict] = []
    floors: list[dict] = []
    for cov in coverages:
        per_det: dict[str, tuple] = {}
        rr_by_ref: dict[int, float] = {}
        for key in ALL_KEYS:
            res = pipeline.run(rec, key, correction=correction, coverage=cov, cache=cache)
            row = dict(res.row)
            row["channel"] = channel
            rows.append(row)
            if res.pp_s.size > 5:
                per_det[key] = (res.pair_ref_idx, res.pp_s)
                for i, v in zip(res.pair_ref_idx, res.rr_s):
                    rr_by_ref.setdefault(int(i), float(v))

        if rr_by_ref and len(per_det) >= 3:
            fl = metrics.consensus_floor(per_det, rr_by_ref)
            fl.update(record=name, coverage_target=cov)
            floors.append(fl)

    return rows, floors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=int, default=0, help="limit record count (0 = all)")
    ap.add_argument("--correction", default="lipponen_tarvainen")
    ap.add_argument("--coverages", default="", help="comma-separated; default 1.0,0.9,0.75,0.5")
    ap.add_argument("--suffix", default="", help="suffix for the output CSV names")
    ap.add_argument("--jobs", type=int, default=0,
                    help="worker processes; 0 = auto (physical cores), 1 = serial")
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

    # Measured on a 10-core/20-thread Xeon: 5 workers 16.3 s, 10 workers 11.2 s,
    # 20 workers 9.3 s. The stages are a mix of NumPy and interpreted work, so the
    # hyperthreads do earn their keep. Capping at the record count avoids idle workers,
    # and each holds only ~7 MB of signal so memory is never the constraint.
    auto = os.cpu_count() or 2
    jobs = max(1, min(args.jobs or auto, len(records)))

    print(f"benchmark: {len(records)} records x {len(ALL_KEYS)} detectors "
          f"x {len(coverages)} coverage levels, channel={channel}, "
          f"correction={args.correction}, jobs={jobs}")

    jobspec = [(name, channel, args.correction, coverages) for name in records]
    rows: list[dict] = []
    floor_rows: list[dict] = []
    t0 = time.perf_counter()

    if jobs == 1:
        for i, spec in enumerate(jobspec, 1):
            r, f = process_record(spec)
            rows += r
            floor_rows += f
            print(f"[{i}/{len(records)}] {spec[0]}", flush=True)
    else:
        # Each worker re-imports NeuroKit2 (~5 s) under spawn, so keep the pool for the
        # whole sweep rather than creating one per record. map preserves input order,
        # which keeps the output CSV byte-comparable between serial and parallel runs.
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for i, (r, f) in enumerate(pool.map(process_record, jobspec), 1):
                rows += r
                floor_rows += f
                print(f"[{i}/{len(records)}] {records[i - 1]}", flush=True)

    elapsed = time.perf_counter() - t0

    df = pd.DataFrame(rows)
    main_out = RESULTS / f"benchmark_main{args.suffix}.csv"
    df.to_csv(main_out, index=False)
    pd.DataFrame(floor_rows).to_csv(RESULTS / f"prv_floor{args.suffix}.csv", index=False)
    print(f"\nwrote {len(df)} rows -> {main_out}")
    print(f"wall time {elapsed:.1f} s on {jobs} worker(s) "
          f"({elapsed / len(records):.2f} s per record)")

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
