"""Download the 22 sitting records of the PhysioNet Pulse Transit Time PPG dataset.

Only .hea/.dat/.atr are fetched (~150 MB); the CSV/ mirror, which is most of the
dataset's 2.9 GB, is skipped. Files are verified against the published SHA256SUMS.txt.
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

BASE = "https://physionet.org/files/pulse-transit-time-ppg/1.1.0"
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RECORDS = [f"s{i}_sit" for i in range(1, 23)]
EXTS = (".hea", ".dat", ".atr")


def fetch(rel: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE}/{rel}"
    with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)

    # Aux files: README gives the pleth_* wavelength mapping; SHA256SUMS lets us verify.
    for aux in ("README.txt", "SHA256SUMS.txt", "RECORDS", "LICENSE.txt"):
        target = DATA / aux
        if not target.exists():
            print(f"fetching {aux}", flush=True)
            fetch(aux, target)

    sums: dict[str, str] = {}
    for line in (DATA / "SHA256SUMS.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            sums[parts[1].lstrip("*./")] = parts[0]

    failures: list[str] = []
    for rec in RECORDS:
        for ext in EXTS:
            name = rec + ext
            target = DATA / name
            if not target.exists():
                print(f"fetching {name}", flush=True)
                fetch(name, target)
            expected = sums.get(name)
            if expected is None:
                print(f"  ! no checksum published for {name}")
                continue
            got = sha256(target)
            if got != expected:
                failures.append(name)
                print(f"  ! CHECKSUM MISMATCH {name}")
                print(f"      expected {expected}")
                print(f"      got      {got}")

    total_mb = sum(p.stat().st_size for p in DATA.glob("s*_sit.*")) / 1e6
    print(f"\n{len(RECORDS)} records, {total_mb:.0f} MB in {DATA}")
    if failures:
        print(f"FAILED checksum: {failures}")
        return 1
    print("all checksums OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
