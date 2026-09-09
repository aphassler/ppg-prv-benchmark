# Best-Performing PPG Algorithms for IBI Extraction and HRV on Smartphones

## TL;DR
- **Adopt MSPTD (or its efficient variant MSPTDfast / the fast qppg implementation, qppgfast) as the beat detector**, run on a signal band-pass filtered to ~0.5–8 Hz, followed by sub-sample peak interpolation (parabolic / cubic-spline) and Lipponen–Tarvainen (Kubios-style) IBI artifact correction. In the definitive head-to-head benchmark (Charlton et al. 2022, 15 detectors × 8 datasets vs ECG), MSPTD and qppg were the top performers, reaching median F1 ≈ 99.9% on clean data and remaining highest under motion.
- **Artifact handling and fiducial-point/interpolation choices dominate HRV error, not the peak detector.** At rest, PPG-derived RMSSD/SDNN agree closely with ECG (pooled absolute standardized errors 0.188 for RMSSD and 0.134 for SDNN in a 43-study meta-analysis); during motion RMSSD error balloons (relative MAE ~34% for wrist PPG) because a single misdetected beat corrupts successive-difference metrics.
- **Smartphone-camera PPG is viable for HRV only at rest.** At 30 fps the frame quantization imposes a ±16.6 ms peak ambiguity (PRV RMSE floor ~16–20 ms even after spline interpolation to 500 Hz); interpolation plus outlier correction brings beat-to-beat SD error to ~5–8 ms and makes RMSSD/SDNN usable at rest, but HRV during movement is unreliable.

## Key Findings

1. **Winner: MSPTD / MSPTDfast, with qppgfast as a complementary alternative.** The definitive benchmark is Charlton et al. (2022, *Physiol. Meas.* 43:085007), which assessed 15 open-source detectors on 8 datasets against ECG-derived reference beats using F1 (harmonic mean of sensitivity and PPV) with a ±150 ms matching tolerance. MSPTD and qppg ranked best, with *complementary* characteristics stated verbatim: "MSPTD tended to have a higher positive predictive value, whereas qppg tended to have higher sensitivity." Algorithmically, "MSPTD searches for peaks without using any prior knowledge of the characteristics of PPG pulse waves. In contrast, qppg searches for systolic upslopes based on their expected characteristics." Eight detectors performed well at rest — AMPD, MSPTD, qppg, PWD, ERMA, SPAR, ABD, and HeartPy — with F1 ≥ 90% on hospital and at-rest wearable data. Performance degraded predictably: F1 55%–91% during exercise; 84%–96% in neonates vs 98%–99% in adults; 92%–97% in AF vs 99%–100% in normal sinus rhythm.

2. **MSPTDfast makes MSPTD mobile-feasible.** MSPTDfast (v.2, 2025, *Physiol. Meas.* 46:035002) downsamples to 20 Hz and restricts scalogram scales to heart rates >30 bpm. Its execution time was **5.3% of MSPTD's on CapnoBase and 15.1% on BIDMC** with an equal F1-score (highest F1 alongside MSPTD in benchmarking). The earlier v.1 (medRxiv 2024) cut execution time 64.4% vs MSPTD (F1 87.8% vs 87.7%) and ran only 30.0% slower than qppgfast. Both are MIT-licensed. This resolves the historical objection that the accurate MSPTD was too heavy for wearables/phones.

3. **Fiducial point matters: use maximum-first-derivative or mid-amplitude, not the systolic apex or the foot.** Multiple pulse-rate-variability (PRV) studies find the systolic peak (apex) and the pulse foot/onset give the largest jitter and worst RR agreement; the maximum of the first derivative (max slope), the mid-amplitude point, and the intersecting-tangents point give the lowest error and highest correlation with ECG HRV.

4. **Artifact/ectopic IBI correction is the single largest lever on HRV accuracy.** The Lipponen–Tarvainen (2019) algorithm — Kubios's default, using time-varying thresholds on the dRR series plus beat classification and cubic-spline replacement — sharply reduces RMSSD error. Without correction, a few seconds of noise destroy RMSSD/pNN50/HF even when mean HR and SDNN appear acceptable.

5. **Sampling rate: coarse PPG is largely recoverable with interpolation.** With parabolic or cubic-spline interpolation, 20–50 Hz PPG approaches 250–1000 Hz accuracy for time-domain HRV; without interpolation ≥250 Hz is needed. A 30 fps phone camera imposes a hard ±16.6 ms timing floor that interpolation cannot fully remove.

## Details

### 1. Algorithm classes and named algorithms

**Classical DSP / fiducial detectors (all in the PPG-beats toolbox):**
- **Aboy ABD (2005)** — strong band-pass around an HR estimate, differentiation, 75th-percentile peak picking. GPL.
- **Zong slope-sum function (SSF, 2003)** — windowed/weighted slope-sum + adaptive thresholding + backsearch; detected 99.31% of 368,364 MIT-BIH Polysomnographic beats. Conceptual basis of the qppg family.
- **qppg / qppgfast** — systolic-upslope search via slope-sum and adaptive thresholding, designed specifically for cardiovascular pulse waves. Top-2 in the Charlton benchmark; qppgfast is among the fastest detectors tested.
- **Elgendi ERMA / TERMA (2013 / 2016)** — two event-related moving averages (MApeak 111 ms emphasising systolic peaks, MAbeat 667 ms emphasising beats) with an offset threshold; band-pass 0.5–8 Hz, rectify, square. Reported **99.84% sensitivity and 99.89% positive predictivity** on 40 heat-stressed subjects. MIT. Slightly outperformed Billauer's, Li's and Zong's detectors in that study.
- **Shin adaptive-threshold (ATmax/ATmin, 2009)**; **Karlen Incremental Merge Segmentation (IMS, 2012)**, designed for mobile; **Argüello-Prada Peak Detection Algorithm (PDA, 2018)**; **Pulse Wave Delineator (PWD)**; **Orphanidou COppg percentile detector (2015)**.
- **AMPD (Scholkmann 2012)** and **MSPTD (Bishop & Ercole 2018)** — multiscale local-maxima (and, in MSPTD, also local-minima) scalogram; parameter-free, works on any quasi-periodic signal. MSPTD extends AMPD with trough detection plus efficiency optimisations. MIT.
- **SPAR (Aston/Charlton)** — symmetric projection attractor reconstruction; **SWT** stationary-wavelet; **WFD** wavelet foot delineation.

**SQI / motion rejection:** perfusion index (the conventional gold standard), skewness SQI, kurtosis, relative power, entropy, zero-crossing, and template-matching/systolic-detector-match. Elgendi (2016, *Bioengineering* 3(4):21; 106 recordings of 60 s) found the **skewness index best of eight SQIs, with F1 scores of 86.0% / 87.2% / 79.1%** for the three quality-discrimination tasks. Orphanidou provides a template-correlation SQI. Open toolbox: vital_sqi (statistical, HRV-based, and waveform SQIs with JSON rulesets).

**IBI/ectopic correction:** Malik, Karlsson, Kamath, Acar percentage/median filters; **Lipponen–Tarvainen (2019)** time-varying dRR thresholds + novel beat classification (Kubios default; medium threshold ≈0.25 s corrects with cubic-spline interpolation). Interpolation method for missing beats: cubic spline generally best; pchip/linear for higher missing fractions.

**Deep learning:** U-Net segmentation (TAU — Temporal Attentive U-Net for PPG peaks), CNN/LSTM IBI estimators, and transformers — PPG-PT (interpretable pre-trained transformer) reached **median F1 98.04%** on the MIMIC PERform testing set, comparable to qppg (96.9%) and MSPTD (97.5%). PPG-to-ECG reconstruction models (CardioGAN, ReHeartNet DC-BiLSTM, hybrid attention CNN-BiLSTM) target R-peak beat-timing error rather than F1. pyPPG's improved peak detector scored F1 88.19% on 2,054 polysomnography recordings (>91M reference beats) and fiducial-point MAE <10 ms.

### 2. Head-to-head benchmark results (Charlton 2022; median F1 %, ±150 ms tolerance)

| Dataset / condition | MSPTD | qppg | Range (all 15) |
|---|---|---|---|
| CapnoBase (clean, hospital) | 99.9 | 99.9 | 97.1–99.9 |
| BIDMC (clean, hospital) | 99.7 | 99.6 | 93.4–99.7 |
| MIMIC PERform Testing | 97.5 | 96.9 | 59.0–97.5 |
| — adults | 98.5 | 98.0 | 91.9–98.5 |
| — neonates | 95.9 | 95.2 | 50.7–95.9 |
| MIMIC AF (AF) | 96.7 | 97.1 | 75.3–97.1 |
| MIMIC AF (non-AF) | 99.7 | 99.6 | 91.3–99.7 |
| WESAD meditation | 98.2 | 98.3 | 71.5–98.3 |
| WESAD stress | 70.1 | 68.7 | 17.9–70.1 |
| PPG-DaLiA sitting | 95.1 | 95.1 | 63.1–95.5 |
| PPG-DaLiA working | 81.2 | 80.0 | 40.3–81.4 |
| PPG-DaLiA walking | 72.1 | 76.9 | 31.2–76.9 |
| PPG-DaLiA cycling | 87.1 | 90.6 | 33.6–90.6 |

Notes: on MIMIC PERform Testing the top group was MSPTD/AMPD/qppg/ABD/Pulses (96.6–97.5%); on PPG-DaLiA "working" it was PWD/MSPTD/AMPD/ABD/qppg/WFD (80.0–81.4%). On the more dynamic PPG-DaLiA activities (cycling, walking) qppg's higher sensitivity beat MSPTD. Exact F1 values for detectors other than MSPTD/qppg appear only as box-plot medians in the paper's figures, not tabulated. The paper reports **HR MAPE (mean absolute percentage error), not IBI MAE** — e.g. MSPTD HR MAPE 0.2% (CapnoBase), 2.4% (MIMIC Testing), 13.2% (WESAD stress), 19.1% (PPG-DaLiA walking). MSPTDfast (v.2) benchmarking: F1 96.8% on MIMIC PERform Testing and 84.6% on WESAD, among the most efficient algorithms tested.

### 3. IBI-level and HRV agreement metrics (from separate validation studies)
- **Wrist PPG, post-surgical patients:** beat-to-beat mean error −1.34 ms and MAE 10.4 ms for good-quality beats; relative MAE of HRV indices: SDNN 9.1%, TRI 11.4%, TINN 11.1% (low) but RMSSD 34.3%, pNN50 139%, NN50 188% (high). Non-linear SD2 7.5%, DFA α1 8.25%, DFA α2 4.71% (accurate) vs SD1 34.3% (poor). Prior work (Tarniceriu) reported sinus-rhythm beat-to-beat MAE 7.34 ms; Parak reported 5.94 ms after artifact correction.
- **Meta-analysis (43 studies, healthy populations):** pooled absolute standardized error **0.188 (95% CI 0.066–0.309; I²=11.69%) for RMSSD** and **0.134 (95% CI 0.014–0.255; I²=0%) for SDNN**. RMSSD and HF power are systematically overestimated by PPG-derived PRV.
- **WHOOP wrist PPG:** HR bias ≤0.39±0.38% (LOA ≤1.56%); Ln RMSSD bias 1.66±1.80% (LOA ±5.93%) with a 200 ms / proprietary filter — approaching the smallest worthwhile change, so interpret against its own precision.
- **Smartphone camera (Guo et al.):** 14 of 16 HRV parameters correlated r>0.7 with ECG; 7 (AVNN, TP, VLF, LF, HF, nLF, nHF) within acceptable agreement limits.
- **Clinical caution:** a large diverse-population study found statistically significant PPG-PRV vs ECG-HRV disagreement (p<0.001), e.g. RMSSD bias 3–4 ms and SDNN bias 7–12 ms across disease groups — "PRV is not the same as HRV."

### 4. Fiducial-point evidence
- **Optimal fiducial points (tilt-table; forehead + finger PPG):** the mid-amplitude point (n_M), the first-derivative apex (max slope), and the tangent-intersection point (n_T) give the lowest PRV–HRV relative error and highest correlation/reliability; the apex (systolic peak) and foot are significantly worse (Wilcoxon-significant differences; poorest Bland-Altman agreement). n_M proposed as most accurate.
- **Finger PPG (rest / mild exercise / mild mental stress):** HRV estimation feasible at rest and under mild mental stress but with large errors during mild physical exercise; accuracy "very dependent on outlier correction and fiducial point selection," and SDNN more robust than RMSSD.
- **Smartphone study (contact camera PPG):** SPP-vs-RR error minimized using the fiducial at the **maximum of the first derivative (FP2)**; 30 Hz sampling error compensated by interpolation.
- Béres–Hejjel: pulse-arrival-time precision is highest (minimum relative precision error) at the ½-amplitude point and worst at the base/foot; required sampling rate is index-dependent.

### 5. Smartphone-camera-specific evidence
- **Frame-rate floor:** at 30 fps, inter-frame spacing is 33.3 ms → ±16.6 ms peak ambiguity; DistancePPG reports a PRV RMSE floor of ~16–20 ms for fair skin at rest that "no amount of interpolation…can completely recover." Higher-frame-rate cameras (100 fps) reduce this but cut exposure/SNR.
- **Android real-time app (Renderscript GPU, 30 Hz):** worst-case beat-to-beat SD error 7.81±3.81 ms; device- and posture-dependent (Samsung S5 < Motorola X). A related smartphone method reports SDE between smartphone-PP and RR ~5.4 ms after interpolation.
- **Peng et al.:** 20–30 fps camera PPG good for mean HR but HRV parameters inaccurate.
- **Hardware:** iPhone 6s supports 30/60/240 fps depending on resolution — use the highest frame rate available. Skin tone, ambient light, and contact pressure materially affect signal quality.
- **Validated apps/tools:** Camera-based PRV via HRV4Training (Plews et al.: trivial RMSSD differences vs Polar/ECG, r≈1.00), CameraHRV (RMSSD MAPE ~17.5% vs chest-strap 2.2%), Cardiio (HR RMSE ~1 bpm at rest). Note HR validation is far stronger than HRV validation across the literature.

### 6. Preprocessing that materially improves IBI accuracy
- **Band-pass:** 0.5–8 Hz (Charlton used 0.67–8.0 Hz to remove non-cardiac frequencies); third-order Butterworth 0.5–8 Hz is a common choice, followed by per-window z-normalization.
- **Interpolation / upsampling:** cubic-spline or parabolic peak interpolation. Baek: 20 Hz + interpolation ≈ 250 Hz ECG for HRV. Béres–Hejjel: RMSSD accurate down to ~10 ms decimation (or 100 ms with interpolation) in normal-variability signals; AVNN tolerant to 5 Hz. Cross-study consensus: RAE <5% for SDNN/RMSSD is achievable at ~20–50 Hz with interpolation; without interpolation, keep ≥250 Hz (Task Force recommendation 250–500 Hz).
- **Detrending** and IBI resampling to 4 Hz (cubic spline) before AR/FFT for frequency-domain metrics.

### 7. Open-source implementations and portability
| Tool | Language | Licence | Notes for mobile |
|---|---|---|---|
| PPG-beats | MATLAB | MIT / GPL (per-detector) | 15 detectors + full benchmark framework; reference implementations to port |
| pyPPG | Python | MIT | fiducial MAE <10 ms; validated on MESA/PPG-BP; real-time capable |
| NeuroKit2 | Python | MIT | multiple PPG detectors + Lipponen–Tarvainen correction + HRV suite |
| HeartPy | Python | MIT | adaptive threshold, robust to clipping/motion, real-time/offline |
| BioSPPy | Python | BSD | general biosignal pipeline |
| vital_sqi | Python | (open) | SQI computation + rulesets |
| Kubios | commercial | proprietary | Lipponen–Tarvainen correction, matched-filter beat detection |

For deployment: MSPTDfast and qppgfast are algorithmically simple enough to port to C/C++/Swift/Kotlin; MSPTDfast's 20 Hz downsampling and scale restriction were explicitly designed for efficiency. HeartPy is a lightweight pure-Python option good for prototyping. Neural detectors (PPG-PT transformer, U-Net) should be exported to TFLite/CoreML and are worthwhile only where an NPU is available; MIT/GPL licence terms must be checked per detector (some PPG-beats members are GPL, which affects closed-source distribution — prefer the MIT-licensed AMPD/MSPTD/MSPTDfast/ERMA/HeartPy for commercial apps).

## Recommendations

**Ranked shortlist of beat detectors for smartphone deployment:** 1) **MSPTDfast (v.2)** — best accuracy/efficiency trade-off, MIT; 2) **qppgfast** — complementary (higher sensitivity), best when motion is expected, but check licence; 3) **HeartPy** — lightweight, easy Python/mobile prototyping, MIT; 4) **ERMA/TERMA** — low compute, MIT; 5) **PPG-PT transformer** — only if an NPU is available and you need best-in-class F1.

**Recommended pipeline:**
1. Acquire at the highest available frame rate (≥60 fps; 240 fps ideal on iPhone). Use fingertip contact PPG with the flash on; validate finger placement (pixel-intensity gate) and contact pressure.
2. Band-pass filter 0.5–8 Hz (Butterworth), detrend, z-normalize per window.
3. Upsample/interpolate the PPG (cubic spline to ≥250 Hz) before fiducial detection.
4. Detect beats with **MSPTDfast** (fall back to qppgfast under motion). Locate the fiducial at the **maximum of the first derivative** (or mid-amplitude), **not** the systolic apex or foot.
5. Refine each fiducial with **parabolic interpolation** for sub-sample timing.
6. Compute an SQI (skewness + perfusion index) per 10–30 s window; discard low-quality windows.
7. Apply **Lipponen–Tarvainen** IBI correction (medium threshold); replace ectopic/missed/extra beats via cubic spline.
8. Compute HRV; prefer **RMSSD** at rest (report on log scale), and treat **SDNN** as the more robust index; resample IBIs to 4 Hz for LF/HF.

**Expected error:** at rest with a good signal, ~5–8 ms beat-to-beat SD and RMSSD/SDNN within acceptable agreement limits; on a 30 fps camera, a ~16 ms PRV RMSE floor persists. During motion, expect RMSSD relative errors >30% — **gate HRV output to stationary recordings only.**

**Thresholds that change the recommendation:**
- Frame rate <30 fps, or SQI below threshold, or accelerometer/motion detected → suppress HRV output (report HR only).
- If targeting frequency-domain HRV (LF/HF), require ≥120 s of clean data and ≥250 Hz effective (interpolated) resolution; otherwise report time-domain only.
- If an on-device NPU is available and best-in-class F1 under noise is required, switch the detector stage to a PPG-PT-style transformer or U-Net; otherwise stay with MSPTDfast.

## Caveats and Confidence
- The Charlton benchmark reports F1 and HR MAPE, **not IBI MAE**; IBI/HRV error magnitudes are inferred from separate, smaller validation studies and are device- and cohort-dependent.
- **"PRV ≠ HRV":** even with perfect beat detection, pulse-transit-time variability and respiration introduce systematic PPG-vs-ECG differences (RMSSD/HF overestimation), and disagreement grows in older, female, and clinical populations.
- Deep-learning detectors report strong F1 but generally **lack independent mobile-latency/power validation**; treat their headline numbers as promising, not deployment-proven.
- Smartphone-camera HRV numbers come largely from small, healthy, at-rest cohorts; generalization to motion, darker skin tones, and clinical populations is weakly evidenced.

**Overall confidence:** **HIGH** that MSPTD/qppg (and MSPTDfast/qppgfast) are the best-evidenced beat detectors and that artifact correction dominates HRV error. **HIGH** that 20–50 Hz PPG + interpolation suffices for time-domain HRV at rest. **MODERATE** on exact smartphone IBI-error magnitudes (small, device-specific studies) and on the precise fiducial-point ranking (consistent direction — favouring max-slope/mid-amplitude over apex/foot — but modest sample sizes). **MODERATE-LOW** on deep-learning approaches' real-world mobile performance.