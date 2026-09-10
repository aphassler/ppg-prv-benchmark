"""Sub-experiments E1-E4, each targeting a specific claim in the research reports.

E1  fiducial point      -- Peralta et al. 2019: mid-amplitude / max-dP/dt beat apex+foot
E2  sampling rate       -- the 100 Hz operating point vs 50/30/25 Hz, +/- sub-sample refinement
E3  artifact correction -- "correction is the single largest lever on HRV accuracy"
E4  motion              -- residual motion within the sitting records, graded by gyroscope

Records are independent, so this fans out across processes. Within a record, detection is
cached per sampling rate: E1 varies only the fiducial and E3 only the correction, and
neither sits upstream of the detector, so all four experiments share one set of detections.
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

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prv import correct, fiducial, io, pipeline, sqi  # noqa: E402
from prv.detectors import ALL_KEYS  # noqa: E402

RESULTS = ROOT / "results"

PRIMARY_DETECTOR = "msptdfast"
E2_RATES = [100.0, 50.0, 30.0, 25.0]
E2_DIAGNOSTIC_CEILING = 500.0   # not part of the delivered pipeline; bounds what 100 Hz costs


def primary_channel() -> str:
    qc = RESULTS / "qc_report.json"
    if qc.exists():
        try:
            return json.loads(qc.read_text()).get("primary_channel", "pleth_2")
        except Exception:
            pass
    return "pleth_2"


def process_record(job: tuple[str, str]) -> tuple[list, list, list, list]:
    """All four experiments for one record. Runs in a worker process."""
    name, channel = job
    warnings.filterwarnings("ignore")

    rec = io.load_record(name, ppg_channel=channel)
    cache = pipeline.build_cache(rec, ALL_KEYS)

    e1: list[dict] = []
    e2: list[dict] = []
    e3: list[dict] = []
    e4: list[dict] = []

    # ---------------------------------------------------------------- E1
    for fid in fiducial.FIDUCIALS:
        r = pipeline.run(rec, PRIMARY_DETECTOR, fiducial=fid, correction="none", cache=cache)
        if not r.row.get("failed"):
            e1.append(r.row)

    # ---------------------------------------------------------------- E2
    for fs_hz in E2_RATES + [E2_DIAGNOSTIC_CEILING]:
        # Detection depends on the rate, so each rate gets its own cache; sub-sample
        # refinement sits downstream of detection and reuses it.
        rate_cache = (cache if np.isclose(fs_hz, cache.fs)
                      else pipeline.build_cache(rec, [PRIMARY_DETECTOR], fs=fs_hz))
        for sub in (True, False):
            r = pipeline.run(rec, PRIMARY_DETECTOR, correction="none",
                             fs=fs_hz, sub_sample=sub, cache=rate_cache)
            if not r.row.get("failed"):
                row = dict(r.row)
                row["is_ceiling"] = fs_hz == E2_DIAGNOSTIC_CEILING
                e2.append(row)

    # ---------------------------------------------------------------- E3
    for strat in correct.STRATEGIES:
        for det in ALL_KEYS:
            r = pipeline.run(rec, det, correction=strat, cache=cache)
            if not r.row.get("failed"):
                e3.append(r.row)

    # ---------------------------------------------------------------- E4
    # Residual motion inside a "resting" record, from the gyroscope on the same clock.
    motion = sqi.motion_sqi(rec.gyro, rec.fs_native)
    base = pipeline.run(rec, PRIMARY_DETECTOR, correction="none", cache=cache)
    e4.append({"record": name,
               "gyro_rms_median": float(np.median(motion)),
               "gyro_rms_p90": float(np.percentile(motion, 90)),
               "f1": base.row.get("f1"), "ibi_rmse": base.row.get("ibi_rmse"),
               "rmssd_abserr": base.row.get("rmssd_abserr"),
               "sdnn_abserr": base.row.get("sdnn_abserr")})

    return e1, e2, e3, e4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=0, help="worker processes; 0 = auto, 1 = serial")
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    RESULTS.mkdir(exist_ok=True)

    channel = primary_channel()
    records = io.available_records()
    if args.records:
        records = records[: args.records]

    jobs = max(1, min(args.jobs or (os.cpu_count() or 2), len(records)))
    print(f"experiments on {len(records)} records, channel={channel}, jobs={jobs}\n")

    jobspec = [(name, channel) for name in records]
    e1: list[dict] = []
    e2: list[dict] = []
    e3: list[dict] = []
    e4: list[dict] = []
    t0 = time.perf_counter()

    def collect(res) -> None:
        a, b, c, d = res
        e1.extend(a)
        e2.extend(b)
        e3.extend(c)
        e4.extend(d)

    if jobs == 1:
        for i, spec in enumerate(jobspec, 1):
            collect(process_record(spec))
            print(f"[{i}/{len(records)}] {spec[0]}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for i, res in enumerate(pool.map(process_record, jobspec), 1):
                collect(res)
                print(f"[{i}/{len(records)}] {records[i - 1]}", flush=True)

    elapsed = time.perf_counter() - t0

    for tag, rows in (("e1_fiducial", e1), ("e2_sampling", e2),
                      ("e3_correction", e3), ("e4_motion", e4)):
        pd.DataFrame(rows).to_csv(RESULTS / f"{tag}{args.suffix}.csv", index=False)
        print(f"wrote results/{tag}{args.suffix}.csv ({len(rows)} rows)")
    print(f"wall time {elapsed:.1f} s on {jobs} worker(s)")

    # ------------------------------------------------------------- quick summaries
    d1 = pd.DataFrame(e1)
    if not d1.empty:
        print("\nE1 fiducial (median across records):")
        print(d1.groupby("fiducial")[["ibi_rmse", "rmssd_abserr", "sdnn_abserr"]]
              .median().sort_values("ibi_rmse").to_string(float_format=lambda v: f"{v:8.2f}"))

    d2 = pd.DataFrame(e2)
    if not d2.empty:
        print("\nE2 sampling rate (median IBI RMSE, ms):")
        print(d2.pivot_table(index="fs", columns="sub_sample", values="ibi_rmse",
                             aggfunc="median").to_string(float_format=lambda v: f"{v:8.2f}"))

    d3 = pd.DataFrame(e3)
    if not d3.empty:
        print("\nE3 correction (median across records):")
        print(d3.groupby(["detector_label", "correction"])[
            ["ibi_rmse", "rmssd_abserr", "fraction_corrected"]]
            .median().to_string(float_format=lambda v: f"{v:8.2f}"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
