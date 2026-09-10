"""Verify an optimised run reproduces a reference run exactly.

Speeding a benchmark up is only worth anything if the numbers do not move. This
compares two result CSVs field by field, on every row and every column, and reports the
largest discrepancy found rather than a pass/fail summary that could hide one.

Timing columns are excluded by name: they are measurements of the machine, not of the
signal, and are expected to differ.

    python scripts/05_verify.py results/reference/benchmark_main.csv \
                                results/benchmark_main_fast.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Wall-clock measurements, not results.
TIMING_COLUMNS = {"detect_s_per_min"}

# Columns identifying a row, used to align the two frames before comparing.
KEY_COLUMNS = ["record", "detector", "coverage_target", "fiducial", "correction"]


def compare(ref_path: Path, new_path: Path, tol: float) -> int:
    ref = pd.read_csv(ref_path)
    new = pd.read_csv(new_path)

    print(f"reference : {ref_path}  ({len(ref)} rows, {len(ref.columns)} cols)")
    print(f"candidate : {new_path}  ({len(new)} rows, {len(new.columns)} cols)\n")

    problems: list[str] = []

    if len(ref) != len(new):
        problems.append(f"row count differs: {len(ref)} vs {len(new)}")
    missing = set(ref.columns) - set(new.columns)
    added = set(new.columns) - set(ref.columns)
    if missing:
        problems.append(f"columns missing from candidate: {sorted(missing)}")
    if added:
        problems.append(f"columns only in candidate: {sorted(added)}")

    keys = [k for k in KEY_COLUMNS if k in ref.columns and k in new.columns]
    ref = ref.sort_values(keys).reset_index(drop=True)
    new = new.sort_values(keys).reset_index(drop=True)

    for k in keys:
        if not ref[k].equals(new[k]):
            problems.append(f"row identity differs on column {k!r}")

    shared = [c for c in ref.columns if c in new.columns and c not in TIMING_COLUMNS]
    numeric, exact, worst = 0, 0, []

    for col in shared:
        a, b = ref[col], new[col]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            numeric += 1
            av, bv = a.to_numpy(float), b.to_numpy(float)
            both_nan = np.isnan(av) & np.isnan(bv)
            if not np.array_equal(np.isnan(av), np.isnan(bv)):
                problems.append(f"{col}: NaN pattern differs")
            d = np.abs(av - bv)
            d[both_nan] = 0.0
            if np.all(d == 0):
                exact += 1
            mx = float(np.nanmax(d)) if d.size else 0.0
            worst.append((mx, col))
            if mx > tol:
                idx = int(np.nanargmax(d))
                problems.append(
                    f"{col}: max |diff| {mx:.3e} > tol {tol:.0e} "
                    f"at row {idx} ({ref.loc[idx, keys[0]] if keys else idx}) "
                    f"ref={av[idx]!r} new={bv[idx]!r}"
                )
        else:
            if not a.astype(str).equals(b.astype(str)):
                problems.append(f"{col}: non-numeric values differ")

    worst.sort(reverse=True)
    print(f"compared {len(shared)} columns ({numeric} numeric), "
          f"{len(ref)} rows = {len(shared) * len(ref):,} field comparisons")
    print(f"bit-identical numeric columns: {exact}/{numeric}")
    print(f"excluded as timing: {sorted(TIMING_COLUMNS)}\n")

    print("largest absolute differences by column:")
    for mx, col in worst[:8]:
        flag = "exact" if mx == 0 else f"{mx:.3e}"
        print(f"  {col:24s} {flag}")

    print()
    if problems:
        print("MISMATCH")
        for p in problems:
            print(f"  ! {p}")
        return 1
    print(f"IDENTICAL — every compared field agrees within {tol:.0e}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reference", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--tol", type=float, default=0.0,
                    help="absolute tolerance; default 0 requires bit-identical results")
    args = ap.parse_args()

    for p in (args.reference, args.candidate):
        if not p.exists():
            print(f"missing: {p}")
            return 2
    return compare(args.reference, args.candidate, args.tol)


if __name__ == "__main__":
    sys.exit(main())
