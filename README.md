# PPG-PRV Benchmark

End-to-end benchmark of photoplethysmography beat detectors for **pulse rate variability**,
scored on beat-to-beat interval error and HRV endpoint error rather than on beat-detection F1.

[![CI](https://github.com/aphassler/ppg-prv-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/aphassler/ppg-prv-benchmark/actions/workflows/ci.yml)

## Why this exists

The published PPG beat-detection benchmarks — Charlton et al.
([2022](https://doi.org/10.1088/1361-6579/ac826d), [2025](https://doi.org/10.1088/1361-6579/adb89e))
above all — report **F1 and heart-rate MAPE**. Almost none report beat-to-beat **IBI RMSE** or
the error in the HRV indices people actually consume. That gap matters, because a detector can
score a near-perfect F1 at a ±150 ms matching tolerance while its intervals are useless for
RMSSD: 150 ms of slack is roughly five times a typical resting RMSSD.

This repository closes that gap for one well-defined case — **finger PPG at rest, 100 Hz** —
by running five detectors through identical downstream stages against a synchronous ECG gold
standard, and ranking them on the errors that matter.

## What is measured

Six levels, following the validation protocol that the source literature prescribes but
rarely executes:

| Level | Metric | Role |
| --- | --- | --- |
| A | Se / PPV / F1 at ±150 ms | secondary, for comparability with Charlton 2022 |
| B | **IBI MAE, RMSE, bias, SD** on matched beats | **primary** |
| C | Bland–Altman bias and 95% LoA, Lin's CCC, TOST equivalence | agreement |
| D | **RMSSD, SDNN, pNN50, SD1/SD2, LF/HF, SampEn, DFA α1 error** | **primary** |
| E | Coverage, % corrected beats, accuracy-vs-coverage curve | honesty |
| F | Execution time per minute of signal | deployment cost |

Level E is not decoration. A detector can post excellent accuracy by abstaining on the hard
segments, so the leaderboard is read at **matched coverage** and the trade-off curve is
published alongside it.

### Separating algorithm error from physiology

A constant pulse arrival time cancels inside an interval, so the residual PPI−RRI error is
detector jitter *combined with* beat-to-beat PAT variability — which is physiology, not a bug.
The benchmark estimates an irreducible **PRV floor** as the element-wise median PPI across
detectors (a consensus fiducial), and reports each detector's **excess over that floor** as its
attributable contribution. This is an estimate, labelled as such: it assumes detector errors are
independent enough for the median to cancel them.

## The five approaches

Each is a full pipeline — `decimate → band-pass → detect → fiducial → sub-sample refine →
SQI gate → correct → HRV`. Only the detector stage differs.

| Approach | Implementation | Why included |
| --- | --- | --- |
| **MSPTDfast v2** | NeuroKit2 `method="charlton"` (author-contributed) | best-evidenced open detector |
| **MSPTD** | NeuroKit2 `method="bishop"`, 6 s windows | the ancestor; isolates what v2's optimisations cost |
| **qppg / Zong slope-sum** | **from scratch** ([`qppg.py`](src/prv/detectors/qppg.py)) | Charlton's co-winner; no faithful Python port existed |
| **TERMA** | **from scratch** ([`terma.py`](src/prv/detectors/terma.py)) | lowest compute; cross-validated against NeuroKit2 |
| **HeartPy** | `heartpy` package | lightweight reference implementation |
| *find_peaks baseline* | ours | the floor — shows what the sophistication buys |

`pyPPG` (Aboy++) is **not** included: it pins `Python == 3.10` with `numpy==1.23.2`, which is
incompatible with this stack. It would need its own virtualenv.

## Data

[PhysioNet Pulse Transit Time PPG v1.1.0](https://physionet.org/content/pulse-transit-time-ppg/1.1.0/)
— the 22 `sit` records only (~150 MB of the 2.9 GB total).

- Finger PPG, left index finger, two sites × three wavelengths, and **`pleth_6` (green,
  537 nm, proximal phalanx)** selected empirically as the best-SNR channel.
- Synchronous ECG with **manually verified** `.atr` R-peak annotations — a real gold standard.
- 18 channels at 500 Hz; ~8.5 min per record, so a full 5-minute short-term HRV window fits.
- IMU on the same clock, used to grade residual motion inside the "resting" records.
- Open Data Commons ODbL.

**The PPG is decimated to 100 Hz** as the first pipeline stage — the rate a real wearable would
ship, and the German source report's recommended default. The **ECG reference stays at 500 Hz**:
degrading the gold standard alongside the signal under test would hide the very error being
measured.

## Two findings that changed the implementation

Both were caught by the QC stage, and both would have produced a plausible-looking but wrong
benchmark:

1. **The raw PPG arrives inverted.** These are transmission pulse-oximetry channels — more
   blood means more absorption and a *lower* sample value. Left uncorrected, every fiducial
   (mid-upslope, max dP/dt, foot) lands on the diastolic decay instead of the systolic rise,
   while F1 still looks fine. Detected automatically from first-derivative skew and confirmed
   against pulse morphology and physiological PAT on all 22 records.
2. **`scipy.signal.decimate` explodes on a DC pedestal.** Raw samples sit around 74 000 with a
   ~150 pulsatile component; the anti-alias filter's edge behaviour on that step produced a
   transient 44 standard deviations tall, which swamped every amplitude threshold and made
   TERMA, HeartPy, qppg and `find_peaks` return *zero* beats. The dataset README quietly
   confirms DC removal is required.

## Reproducing

```bash
pip install -e ".[dev]"
python scripts/00_fetch_data.py       # ~150 MB, verified against SHA256SUMS.txt
python scripts/01_qc_reference.py     # must pass before any result is believable
python scripts/02_run_detectors.py    # main sweep
python scripts/03_experiments.py      # E1-E4
python scripts/04_analyse.py          # tables in results/, figures in figures/
```

Unit tests run entirely on a synthetic PPG with known beat times, so they need no download:

```bash
pytest -q
```

## Repository layout

```
src/prv/            library: io, preprocess, detectors/, fiducial, sqi, correct, hrv, match, metrics, pipeline
scripts/            00 fetch -> 01 QC -> 02 benchmark -> 03 experiments -> 04 analyse
tests/              48 tests against a synthetic PPG with known ground truth
results/            CSV tables (git-ignored except summaries)
figures/            generated plots
```

## Scope and limitations

- 22 healthy adults (mean age 28.5 y), finger PPG, seated, high SNR. This is a **favourable**
  case: the numbers are an upper bound, not a free-living expectation.
- Nothing here transfers automatically to wrist PPG, ambulatory recording, camera PPG, or
  clinical populations — and the largest study in the field (Kantrowitz et al. 2025, n=931)
  found systematic PRV−HRV disagreement even at rest, in a more heterogeneous cohort.
- **PRV is not HRV.** Results are reported as pulse rate variability throughout.

## Licence

MIT (see [LICENSE](LICENSE)). The dataset carries its own ODbL licence. Note that the original
qppg C implementation descends from GPL WFDB code; [`qppg.py`](src/prv/detectors/qppg.py) here is
an independent NumPy implementation written from the published method description.
