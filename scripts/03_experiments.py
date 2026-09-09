"""Sub-experiments E1-E4, each targeting a specific claim in the research reports.

E1  fiducial point      -- Peralta et al. 2019: mid-amplitude / max-dP/dt beat apex+foot
E2  sampling rate       -- the 100 Hz operating point vs 50/30/25 Hz, +/- sub-sample refinement
E3  artifact correction -- "correction is the single largest lever on HRV accuracy"
E4  motion              -- residual motion within the sitting records, graded by gyroscope
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

from prv import correct, fiducial, io, pipeline, sqi  # noqa: E402

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=int, default=0)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    RESULTS.mkdir(exist_ok=True)

    channel = primary_channel()
    records = io.available_records()
    if args.records:
        records = records[: args.records]
    print(f"experiments on {len(records)} records, channel={channel}\n")

    e1: list[dict] = []
    e2: list[dict] = []
    e3: list[dict] = []
    e4: list[dict] = []

    for ri, name in enumerate(records, 1):
        rec = io.load_record(name, ppg_channel=channel)
        print(f"[{ri}/{len(records)}] {name}", flush=True)

        # ---------------------------------------------------------------- E1
        for fid in fiducial.FIDUCIALS:
            r = pipeline.run(rec, PRIMARY_DETECTOR, fiducial=fid, correction="none")
            if not r.row.get("failed"):
                e1.append(r.row)

        # ---------------------------------------------------------------- E2
        for fs_hz in E2_RATES + [E2_DIAGNOSTIC_CEILING]:
            for sub in (True, False):
                r = pipeline.run(rec, PRIMARY_DETECTOR, correction="none",
                                 fs=fs_hz, sub_sample=sub)
                if not r.row.get("failed"):
                    row = dict(r.row)
                    row["is_ceiling"] = fs_hz == E2_DIAGNOSTIC_CEILING
                    e2.append(row)

        # ---------------------------------------------------------------- E3
        for strat in correct.STRATEGIES:
            for det in ("msptdfast", "terma", "findpeaks"):
                r = pipeline.run(rec, det, correction=strat)
                if not r.row.get("failed"):
                    e3.append(r.row)

        # ---------------------------------------------------------------- E4
        # Residual motion inside a "resting" record, from the gyroscope on the same clock.
        gyro_rms = float(np.median(sqi.motion_sqi(rec.gyro, rec.fs_native)))
        gyro_p90 = float(np.percentile(sqi.motion_sqi(rec.gyro, rec.fs_native), 90))
        base = pipeline.run(rec, PRIMARY_DETECTOR, correction="none")
        e4.append({"record": name, "gyro_rms_median": gyro_rms, "gyro_rms_p90": gyro_p90,
                   "f1": base.row.get("f1"), "ibi_rmse": base.row.get("ibi_rmse"),
                   "rmssd_abserr": base.row.get("rmssd_abserr"),
                   "sdnn_abserr": base.row.get("sdnn_abserr")})

    for tag, rows in (("e1_fiducial", e1), ("e2_sampling", e2),
                      ("e3_correction", e3), ("e4_motion", e4)):
        pd.DataFrame(rows).to_csv(RESULTS / f"{tag}.csv", index=False)
        print(f"wrote results/{tag}.csv ({len(rows)} rows)")

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
