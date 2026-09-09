"""Loading of PTT-PPG records.

The PPG under test is decimated to the pipeline operating rate (100 Hz); the ECG
reference beats are kept at the native 500 Hz so the gold standard is never degraded
alongside the signal being measured.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wfdb

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

FS_NATIVE = 500.0
SIT_RECORDS = [f"s{i}_sit" for i in range(1, 23)]

# Channel layout of the WFDB record (see the dataset README).
PLETH_CHANNELS = ["pleth_1", "pleth_2", "pleth_3", "pleth_4", "pleth_5", "pleth_6"]
GYRO_CHANNELS = ["g_x", "g_y", "g_z"]
ACC_CHANNELS = ["a_x", "a_y", "a_z"]


@dataclass
class Record:
    """One PTT-PPG recording.

    Attributes
    ----------
    ppg : raw (undecimated) PPG for the selected channel, native 500 Hz.
    ecg_beats_s : reference R-peak times in seconds, from the manually verified .atr.
    gyro, acc : IMU channels at native rate, used for the motion index.
    meta : the key/value pairs from the header comment line.
    """

    name: str
    ppg: np.ndarray
    ppg_channel: str
    fs_native: float
    ecg: np.ndarray
    ecg_beats_s: np.ndarray
    gyro: np.ndarray
    acc: np.ndarray
    meta: dict[str, str]
    polarity: float = 1.0

    @property
    def duration_s(self) -> float:
        return len(self.ppg) / self.fs_native


def parse_header_meta(comments: list[str]) -> dict[str, str]:
    """Pull the <key>: value pairs out of the header comment line."""
    text = " ".join(comments)
    return dict(re.findall(r"<([^>]+)>:\s*([^<]+?)(?=\s*<|$)", text))


def load_record(name: str, ppg_channel: str = "pleth_2", data_dir: Path | None = None) -> Record:
    data_dir = data_dir or DATA
    rec = wfdb.rdrecord(str(data_dir / name))
    ann = wfdb.rdann(str(data_dir / name), "atr")

    fs = float(rec.fs)
    sig_names = list(rec.sig_name)

    def col(ch: str) -> np.ndarray:
        return np.asarray(rec.p_signal[:, sig_names.index(ch)], dtype=float)

    ppg = col(ppg_channel)
    # The pleth channels are transmission pulse oximetry and arrive inverted: more
    # blood means more absorption and a lower sample value. Orient them so that the
    # systolic peak is a maximum, otherwise every fiducial lands on the wrong phase.
    from .preprocess import detect_polarity

    polarity = detect_polarity(ppg, fs)
    ppg = ppg * polarity

    ecg = col("ecg")
    gyro = np.column_stack([col(c) for c in GYRO_CHANNELS])
    acc = np.column_stack([col(c) for c in ACC_CHANNELS])

    # .atr holds R peaks; keep only true beat annotations, in seconds.
    beat_symbols = set("NLRBAaJSVrFejnE/fQ?")
    keep = np.array([s in beat_symbols for s in ann.symbol], dtype=bool)
    beats = np.asarray(ann.sample, dtype=float)[keep] / fs

    return Record(
        name=name,
        ppg=ppg,
        ppg_channel=ppg_channel,
        fs_native=fs,
        ecg=ecg,
        ecg_beats_s=beats,
        gyro=gyro,
        acc=acc,
        meta=parse_header_meta(rec.comments or []),
        polarity=polarity,
    )


def available_records(data_dir: Path | None = None) -> list[str]:
    """Records whose .dat is present *and* complete.

    The size check matters while the download is still running: a truncated .dat
    otherwise reaches wfdb and fails deep inside its block decoder.
    """
    data_dir = data_dir or DATA
    out: list[str] = []
    for r in SIT_RECORDS:
        dat, hea = data_dir / f"{r}.dat", data_dir / f"{r}.hea"
        if not (dat.exists() and hea.exists() and (data_dir / f"{r}.atr").exists()):
            continue
        try:
            first = hea.read_text().splitlines()[0].split()
            n_sig, n_samp = int(first[1]), int(first[3])
        except (IndexError, ValueError):
            continue
        # format 212 packs two 12-bit samples into three bytes.
        expected = int(np.ceil(n_sig * n_samp * 1.5))
        if dat.stat().st_size >= expected:
            out.append(r)
    return out
