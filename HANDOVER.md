# Handover — PPG-PRV Benchmark

For an agent picking this up cold. Read this before touching the code; several things
here cost hours to discover and are not obvious from reading the source.

- **Repo**: `github.com/aphassler/ppg-prv-benchmark` (private, `main`, CI green on 3.11/3.12)
- **Local**: `C:\Users\Maximus\Documents\Claude_Code\PPG-based_PRV`
- **Report**: https://claude.ai/code/artifact/e6900d88-e0c8-4439-a571-6765d1abfce2
- **Status**: complete and self-consistent. Everything below is finished work, not WIP.

---

## 1. What this is

Published PPG beat-detection benchmarks (Charlton 2022/2025) report **F1 and HR MAPE**,
never beat-to-beat IBI RMSE or HRV endpoint error. This repo closes that gap for one
regime — **finger PPG, seated, 100 Hz** — by running five detectors through identical
downstream stages against a synchronous ECG gold standard and ranking on the errors that
actually matter.

It began as a fact-check of two research reports that are still in the repo
(`compass_artifact_*.md`, English/smartphone-focused; `deep-research-report_ppg-based_prv.md`,
German/broader). Keep both — the report's errata section cites them.

**Terminology discipline**: this is PRV (pulse rate variability), not HRV. The distinction
is load-bearing here, not pedantry — see finding 2 below.

---

## 2. Results you do not need to re-derive

Median across 22 records, 100 % coverage. Full tables in `results/`.

| Detector | F1 % | IBI RMSE (LT) | IBI RMSE (med/MAD) | RMSSD err (LT) | RMSSD err (med/MAD) | s/min |
|---|---:|---:|---:|---:|---:|---:|
| MSPTDfast v2 | 99.92 | 7.15 | 6.06 | **1.57** | 2.40 | 0.02 |
| MSPTD | 99.90 | 6.92 | 6.06 | 1.86 | 1.96 | 0.89 |
| TERMA | 99.88 | 7.05 | 6.08 | 3.30 | 2.44 | 0.00 |
| HeartPy | 99.82 | 7.28 | 6.09 | 4.52 | 2.09 | 0.02 |
| qppg | 99.21 | 7.26 | **5.99** | 13.98 | 2.11 | 0.00 |
| find_peaks baseline | 99.88 | 6.96 | 6.30 | 2.29 | 2.40 | 0.00 |

**Three findings.**

1. **F1 saturates.** Spread across detectors: F1 0.99×, IBI RMSE 1.05×, **RMSSD error
   8.92×**, SDNN error 4.49×. F1 has no dynamic range left in this regime and cannot rank
   anything. This is the whole point of the project.
2. **The detector contributes ~nothing to interval timing.** Consensus PRV floor =
   **6.92 ms** RMSE; best detector = 6.92 ms; excess **+0.00 ms**; whole field within
   0.36 ms. The ~7 ms residual is pulse-arrival-time variability — the PRV/HRV gap — and
   no better algorithm closes it.
3. **The correction stage decides the ranking.** Correction is the largest lever (qppg
   RMSSD error: 23.81 ms uncorrected → 2.11 ms with median/MAD). But
   **Lipponen–Tarvainen, which both source reports recommend, lost to a plain median/MAD
   rule** on every detector's IBI RMSE and 4 of 6 RMSSD errors.

**Experiments.** E1: systolic apex fiducial costs **26×** RMSSD error vs mid-upslope
(3.47 → 91.12 ms; IBI RMSE 6.06 → 29.67 ms). E2: degradation without sub-sample
interpolation is *entirely* quantisation — `T/√12·√2` ⊕ floor predicts every rate to
within 0.24 ms, no fitted parameters — and interpolation holds IBI RMSE at ~6.1 ms from
500 Hz down to 25 Hz. E3: see finding 3. E4: within the resting records, the
highest-motion tercile carries **11× the RMSSD error at unchanged F1**.

---

## 3. Gotchas — the expensive knowledge

**These are the reason a naive reimplementation produces a plausible-looking wrong answer.**

1. **The raw PPG arrives inverted.** PTT-PPG `pleth_*` are transmission pulse oximetry:
   more blood → more absorption → *lower* sample value. Uncorrected, every fiducial lands
   on the diastolic decay instead of the systolic rise **while F1 still looks fine**.
   `preprocess.detect_polarity()` decides it from first-derivative skew; `io.load_record()`
   applies it and records `Record.polarity` (it is `-1.0` for all 22 records). Verified
   against pulse morphology (rise 130–200 ms vs decay 360–700 ms) and physiological PAT
   (292–334 ms inverted vs an implausible 170 ms as-is).

2. **`scipy.signal.decimate` explodes on a DC pedestal.** Raw samples sit at ~74,000 with
   a ~150 pulsatile component. Decimating that directly produced an edge transient **44
   standard deviations tall**, which swamped every amplitude threshold and made TERMA,
   HeartPy, qppg and `find_peaks` return **zero** beats. `decimate_to()` removes the mean
   first. Regression test: `test_decimation_survives_large_dc_offset`.

3. **MSPTD is O(N²).** Its scalogram on a 508 s record at 100 Hz is a ~10 GB matrix. It is
   wrapped in `detectors.windowed()` at 6 s / 1 s overlap — the same window MSPTDfast v2
   uses, which also makes the MSPTD-vs-v2 comparison isolate exactly v2's optimisations.
   MSPTDfast, TERMA, HeartPy, qppg and find_peaks are all O(N) and run globally.

4. **NeuroKit2's `charlton` method requires an integer `sampling_rate`.** It indexes with
   it; a float raises `TypeError` deep inside. Wrapper already casts.

5. **The PRV floor must be keyed by reference-beat index, never by position.** Detectors
   produce different numbers of intervals covering different beats. A positional median
   averages unrelated heartbeats and returns a floor *worse than every individual
   detector* (43.7 ms vs the correct 6.92 ms). `paired_intervals()` returns
   `(rr, pp, ref_idx)` for this. Regression test:
   `test_consensus_floor_beats_individual_detectors`.

6. **`paired_intervals` requires consecutiveness in *both* series.** Otherwise a missed
   beat contributes a doubled interval to the IBI error, scoring a *detection* failure as
   a *timing* failure.

7. **Why qppg fails under LT.** It misses ~4 % of beats. Each miss leaves a doubled
   interval (~1.7 s at 70 bpm) that **passes the 2.0 s hard physiological screen**, and
   each corrupts two successive differences. Three such intervals in 582 took one record's
   RMSSD from 30 → 88 ms. LT did not repair them. **This is not a wiring bug** — on a
   synthetic series with 3 deliberately deleted beats the same NeuroKit2 call inserts all
   three correctly. Report it as a property of that implementation on this data, not a
   refutation of the method; Kubios's own implementation may differ.

8. **`pleth_6` (green 537 nm, proximal phalanx)** was chosen empirically by cardiac-band
   SNR (18.12 dB, best of six). `01_qc_reference.py` re-derives it and writes it to
   `results/qc_report.json`; scripts 02/03 read it from there.

9. **`pyPPG` (Aboy++) will not install** — pins `Python == 3.10`, `numpy==1.23.2`,
   `scipy==1.9.1`. Needs its own venv. Deliberately excluded.

10. **Process note**: a patch that added an output-suffix flag silently didn't match and a
    second sweep overwrote 13 minutes of results. **Verify string patches actually applied
    before relying on them.** Results are now suffixed: `benchmark_main.csv` = LT,
    `benchmark_main_medmad.csv` = median/MAD.

---

## 4. Layout

```
src/prv/
  io.py          WFDB load, channel select, .atr ref beats (kept at 500 Hz), polarity applied
  preprocess.py  decimate 500→100 Hz, bandpass 0.5–8 Hz zero-phase, normalise, detect_polarity
  detectors/     registry + windowed() wrapper; qppg.py and terma.py are from-scratch ports
  fiducial.py    peak / mid_upslope / max_slope / foot + parabolic sub-sample refinement
  sqi.py         skewness SQI, perfusion index, gyro motion index, coverage_mask
  correct.py     none / median_mad / lipponen_tarvainen (all report fraction_corrected)
  hrv.py         time, frequency (PCHIP→4 Hz Welch *and* Lomb–Scargle), non-linear
  match.py       constant-lag estimation + ±150 ms one-to-one matching + paired_intervals
  metrics.py     levels A–F, Bland–Altman, Lin's CCC, TOST, consensus_floor
  pipeline.py    run() = one record × one config → one result row
scripts/         00 fetch → 01 QC → 02 benchmark → 03 experiments → 04 analyse
tests/           49 tests on a synthetic PPG with known beat times (no download needed)
```

**Pipeline order is deliberate**: `decimate → bandpass → detect → fiducial → sub-sample
refine → SQI gate → correct → HRV`. Detect at the low rate, refine the fiducial against
the high-rate grid. Do *not* upsample before detection — that is errata #5 in the report,
and it discards MSPTDfast's entire speed advantage.

---

## 5. Running it

```bash
pip install -e ".[dev]"
python scripts/00_fetch_data.py       # 22 sit records, ~150 MB, SHA256-verified
python scripts/01_qc_reference.py     # MUST pass before any result is believable
python scripts/02_run_detectors.py    # ~14 min; --correction, --coverages, --suffix
python scripts/03_experiments.py      # ~25 min; E1–E4
python scripts/04_analyse.py          # tables → results/, figures → figures/
pytest -q && ruff check src tests scripts
```

`data/` is git-ignored (140 MB local). Re-fetch is idempotent and checksum-verified.

**QC step 6 is the critical guard**: MSPTDfast on the cleanest record must reach F1 > 0.99.
It currently gets **99.95 %, IBI RMSE 3.3 ms, PAT lag 234 ms**. If that regresses, the
alignment or matching is broken — do not interpret any downstream number until it passes.

---

## 6. Open threads

Roughly in order of value.

1. **Add the 22 `walk` records** (`io.SIT_RECORDS` → include `_walk`; ~150 MB more). This
   is the single highest-value extension. "Detector choice is irrelevant" is a finding
   *about the resting regime* — precisely where published F1 tables all cluster above
   99 %. The same detectors separate sharply under exercise in Charlton's own benchmark.
   Right now that contrast is argued, not measured. E4 approximates it with residual
   motion inside the sitting records, but n=22 in terciles of 7.
2. **`lins_ccc` and `tost_equivalence` are implemented and tested but never called** by
   any script. Level C of the stated protocol is therefore only half-delivered
   (Bland–Altman is surfaced, CCC and equivalence testing are not). Wiring them into
   `04_analyse.py` is cheap.
3. **Nine of thirteen HRV endpoints are computed and stored but unanalysed** — `sd1`,
   `sd2`, `sd1_sd2`, `sampen`, `dfa_a1`, `lf`, `hf`, `sdsd`, `mean_nn` all sit in
   `benchmark_main.csv` as `*_abserr` columns. The Zuern paper's finding (SD2 excellent,
   SampEn/DFAα1 weak) is directly testable from data already on disk, no re-run needed.
4. **The coverage sweep is collected but under-used.** Four levels (1.0/0.9/0.75/0.5) ×
   528 rows exist and `figures/coverage_accuracy.png` is generated, but the report does
   not discuss it and the leaderboard is only read at 100 %. The matched-coverage argument
   is the methodological guard the German report demands — it deserves to be shown.
5. **Investigate *why* LT fails here.** Its thresholds are relative (quartile-deviation
   based, `c1=0.13`, `c2=0.17`); the hypothesis is that at realistic resting variability
   the doubled intervals fall below threshold. Worth confirming against Kubios proper
   before the claim is published anywhere beyond this repo.
6. **The 26× apex penalty deserves a per-beat look.** RMSE 29.67 ms with a 160 ms max
   error suggests the apex search sometimes lands on the dicrotic notch, not just broad-peak
   jitter. A few annotated waveform plots would settle it.
7. **`pyPPG`/Aboy++ in a Python 3.10 venv**, if the sixth detector is wanted.

---

## 7. Conventions

- **Commits**: imperative subject, then *why* before *what*. Trailer
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- **Lint**: `ruff check src tests scripts` must be clean (line length 100, `E,F,W,I,UP,B`).
- **Tests**: run on synthetic PPG only — CI needs no PhysioNet download. Keep it that way.
- **Comments explain *why*, not *what*.** Several comments in `preprocess.py`,
  `correct.py` and `metrics.py` encode the gotchas above; do not strip them.
- **Never degrade the ECG reference.** It stays at native 500 Hz while the PPG under test
  is decimated. Degrading both hides the error being measured.
- The published report is a **hand-written artifact** (`report/benchmark.html`), not
  generated from the CSVs. If numbers change, it must be updated by hand — republish to
  the same URL by passing it as `url`.
