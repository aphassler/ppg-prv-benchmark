"""Tables and figures from the cached benchmark outputs.

Regenerates everything in results/ and figures/ without re-running any detector.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prv import metrics  # noqa: E402

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# Colour-blind-safe qualitative palette.
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "figure.autolayout": True,
})

ORDER = ["MSPTDfast v2", "MSPTD", "qppg (slope-sum)", "TERMA", "HeartPy",
         "find_peaks baseline"]


def _order(labels) -> list[str]:
    return [c for c in ORDER if c in set(labels)]


def leaderboard(df: pd.DataFrame, coverage: float = 1.0) -> pd.DataFrame:
    ok = df[(df.coverage_target == coverage) & (~df.failed)]
    agg = ok.groupby("detector_label").agg(
        n=("record", "nunique"),
        f1_pct=("f1", lambda s: 100 * s.median()),
        ibi_mae_ms=("ibi_mae", "median"),
        ibi_rmse_ms=("ibi_rmse", "median"),
        rmssd_mae_ms=("rmssd_abserr", "median"),
        sdnn_mae_ms=("sdnn_abserr", "median"),
        pnn50_mae=("pnn50_abserr", "median"),
        lfhf_mae=("lf_hf_abserr", "median"),
        corrected_pct=("fraction_corrected", lambda s: 100 * s.median()),
        cost_s_per_min=("detect_s_per_min", "median"),
    )
    return agg.reindex(_order(agg.index)).sort_values("ibi_rmse_ms")


def fig_leaderboard(df: pd.DataFrame) -> None:
    lb = leaderboard(df)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, (col, title, unit) in zip(axes, [
        ("ibi_rmse_ms", "Beat-to-beat IBI RMSE", "ms"),
        ("rmssd_mae_ms", "RMSSD absolute error", "ms"),
        ("f1_pct", "Beat-detection F1", "%"),
    ]):
        d = lb.sort_values(col, ascending=(col != "f1_pct"))
        ax.barh(range(len(d)), d[col], color=PALETTE[: len(d)])
        ax.set_yticks(range(len(d)))
        ax.set_yticklabels(d.index, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(unit)
        ax.set_title(title, fontsize=10)
        for i, v in enumerate(d[col]):
            ax.text(v, i, f" {v:.1f}", va="center", fontsize=7.5)
    fig.suptitle("Detector leaderboard at 100% coverage (median over records)",
                 fontsize=11, y=1.04)
    fig.savefig(FIGURES / "leaderboard.png", bbox_inches="tight")
    plt.close(fig)


def fig_f1_vs_hrv(df: pd.DataFrame) -> None:
    """The central claim of the German report: F1 does not predict HRV accuracy."""
    ok = df[(df.coverage_target == 1.0) & (~df.failed)].dropna(
        subset=["f1", "rmssd_abserr", "ibi_rmse"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, (ycol, ylab) in zip(axes, [("ibi_rmse", "IBI RMSE (ms)"),
                                       ("rmssd_abserr", "RMSSD absolute error (ms)")]):
        for i, lab in enumerate(_order(ok.detector_label.unique())):
            sub = ok[ok.detector_label == lab]
            ax.scatter(100 * sub.f1, sub[ycol], s=18, alpha=0.75,
                       color=PALETTE[i % len(PALETTE)], label=lab)
        r = np.corrcoef(ok.f1, ok[ycol])[0, 1]
        ax.set_xlabel("beat-detection F1 (%)")
        ax.set_ylabel(ylab)
        ax.set_yscale("log")
        ax.set_title(f"r = {r:+.2f}", fontsize=10)
    axes[1].legend(fontsize=7, loc="upper left", framealpha=0.9)
    fig.suptitle("Beat-detection F1 against the errors that actually matter",
                 fontsize=11, y=1.03)
    fig.savefig(FIGURES / "f1_vs_hrv_error.png", bbox_inches="tight")
    plt.close(fig)


def fig_coverage(df: pd.DataFrame) -> None:
    ok = df[~df.failed]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, (col, ylab) in zip(axes, [("ibi_rmse", "IBI RMSE (ms)"),
                                      ("rmssd_abserr", "RMSSD abs. error (ms)")]):
        for i, lab in enumerate(_order(ok.detector_label.unique())):
            sub = ok[ok.detector_label == lab].groupby("coverage_target")[col].median()
            ax.plot(100 * sub.index, sub.values, "o-", ms=4,
                    color=PALETTE[i % len(PALETTE)], label=lab)
        ax.set_xlabel("coverage retained (%)")
        ax.set_ylabel(ylab)
        ax.invert_xaxis()
    axes[1].legend(fontsize=7)
    fig.suptitle("Accuracy against coverage: comparisons are only fair at equal coverage",
                 fontsize=11, y=1.03)
    fig.savefig(FIGURES / "coverage_accuracy.png", bbox_inches="tight")
    plt.close(fig)


def fig_bland_altman(df: pd.DataFrame) -> None:
    ok = df[(df.coverage_target == 1.0) & (~df.failed)]
    best = leaderboard(df).index[0]
    sub = ok[ok.detector_label == best]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, (key, name) in zip(axes, [("rmssd", "RMSSD"), ("sdnn", "SDNN")]):
        a = sub[f"{key}_ref"].to_numpy(float)
        b = sub[f"{key}_test"].to_numpy(float)
        m = np.isfinite(a) & np.isfinite(b)
        a, b = a[m], b[m]
        ba = metrics.bland_altman(a, b)
        ax.scatter((a + b) / 2, b - a, s=22, color=PALETTE[0], alpha=0.8)
        ax.axhline(ba["bias"], color=PALETTE[3], lw=1.4,
                   label=f"bias {ba['bias']:+.1f} ms")
        for loa in ("loa_lower", "loa_upper"):
            ax.axhline(ba[loa], color=PALETTE[3], ls="--", lw=1)
        ax.axhline(0, color="0.4", lw=0.8)
        ax.set_xlabel(f"mean of ECG and PPG {name} (ms)")
        ax.set_ylabel(f"PPG - ECG {name} (ms)")
        ax.set_title(f"{name}: LoA [{ba['loa_lower']:.1f}, {ba['loa_upper']:.1f}] ms",
                     fontsize=10)
        ax.legend(fontsize=7.5)
    fig.suptitle(f"Bland-Altman, {best}, one point per record", fontsize=11, y=1.03)
    fig.savefig(FIGURES / "bland_altman.png", bbox_inches="tight")
    plt.close(fig)


def fig_experiments() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))

    e1p = RESULTS / "e1_fiducial.csv"
    if e1p.exists():
        d = pd.read_csv(e1p)
        g = d.groupby("fiducial")["ibi_rmse"].median().sort_values()
        axes[0].bar(range(len(g)), g.values, color=PALETTE[: len(g)])
        axes[0].set_xticks(range(len(g)))
        axes[0].set_xticklabels(g.index, rotation=20, ha="right", fontsize=8)
        axes[0].set_ylabel("IBI RMSE (ms)")
        axes[0].set_title("E1  fiducial point", fontsize=10)

    e2p = RESULTS / "e2_sampling.csv"
    if e2p.exists():
        d = pd.read_csv(e2p)
        for i, (sub, lab) in enumerate([(True, "with sub-sample interp."),
                                        (False, "on the sample grid")]):
            g = d[d.sub_sample == sub].groupby("fs")["ibi_rmse"].median()
            axes[1].plot(g.index, g.values, "o-", ms=4, color=PALETTE[i], label=lab)
        axes[1].set_xscale("log")
        axes[1].set_xticks([25, 30, 50, 100, 500])
        axes[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axes[1].set_xlabel("sampling rate (Hz)")
        axes[1].set_ylabel("IBI RMSE (ms)")
        axes[1].set_title("E2  sampling rate", fontsize=10)
        axes[1].legend(fontsize=7.5)

    e3p = RESULTS / "e3_correction.csv"
    if e3p.exists():
        d = pd.read_csv(e3p)
        piv = d.pivot_table(index="correction", columns="detector_label",
                            values="rmssd_abserr", aggfunc="median")
        piv = piv.reindex(["none", "median_mad", "lipponen_tarvainen"])
        x = np.arange(len(piv))
        w = 0.8 / max(len(piv.columns), 1)
        for i, c in enumerate(piv.columns):
            axes[2].bar(x + i * w, piv[c].values, w, label=c, color=PALETTE[i % len(PALETTE)])
        axes[2].set_xticks(x + w * (len(piv.columns) - 1) / 2)
        axes[2].set_xticklabels(piv.index, rotation=20, ha="right", fontsize=8)
        axes[2].set_ylabel("RMSSD abs. error (ms)")
        axes[2].set_title("E3  artifact correction", fontsize=10)
        axes[2].legend(fontsize=7)

    fig.savefig(FIGURES / "experiments.png", bbox_inches="tight")
    plt.close(fig)


def correction_comparison() -> pd.DataFrame | None:
    """Leaderboards side by side under the two correction strategies.

    Worth its own table because the correction stage turned out to reorder the ranking
    entirely -- a detector's apparent quality is not separable from what repairs its
    mistakes, which is precisely the claim the source reports make and never test.
    """
    lt_p, mm_p = RESULTS / "benchmark_main.csv", RESULTS / "benchmark_main_medmad.csv"
    if not (lt_p.exists() and mm_p.exists()):
        return None

    frames = []
    for path, label in [(lt_p, "Lipponen-Tarvainen"), (mm_p, "median/MAD")]:
        lb = leaderboard(pd.read_csv(path))
        lb = lb[["f1_pct", "ibi_rmse_ms", "rmssd_mae_ms", "sdnn_mae_ms"]].copy()
        lb["correction"] = label
        frames.append(lb.reset_index())

    out = pd.concat(frames)
    wide = out.pivot(index="detector_label", columns="correction",
                     values=["ibi_rmse_ms", "rmssd_mae_ms"])
    wide.columns = [f"{a}__{b}" for a, b in wide.columns]
    wide = wide.reindex(_order(wide.index))
    wide.to_csv(RESULTS / "correction_comparison.csv")
    return wide


def discrimination(df: pd.DataFrame) -> pd.DataFrame:
    """How much does each metric actually separate the detectors?

    Correlation alone understates the problem. The point is that F1 *saturates*: on
    resting finger PPG every detector sits within a fraction of a percentage point of
    every other, while the errors that matter for PRV span most of an order of
    magnitude. A metric whose best-to-worst spread is 1.007x cannot rank anything.
    """
    lb = leaderboard(df)
    rows = []
    for col, name, lower_better in [
        ("f1_pct", "beat-detection F1", False),
        ("ibi_rmse_ms", "IBI RMSE", True),
        ("rmssd_mae_ms", "RMSSD error", True),
        ("sdnn_mae_ms", "SDNN error", True),
    ]:
        v = lb[col].dropna()
        if v.empty:
            continue
        best, worst = (v.min(), v.max()) if lower_better else (v.max(), v.min())
        rows.append({
            "metric": name,
            "best": float(best),
            "worst": float(worst),
            "spread_ratio": float(worst / best) if best else np.nan,
            "cv_pct": float(100 * v.std() / v.mean()),
        })
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "discrimination.csv", index=False)
    return out


def sampling_quantisation_model() -> pd.DataFrame | None:
    """Compare the measured sampling-rate degradation with quantisation theory.

    Placing a beat on a grid of step T is uniform quantisation, so its timing SD is
    T/sqrt(12). An interval is the difference of two independently quantised beat times,
    which scales that by sqrt(2). Adding the irreducible floor in quadrature gives a
    parameter-free prediction of the whole E2 curve -- and if the measurement matches it,
    the degradation really is quantisation and nothing else.
    """
    p = RESULTS / "e2_sampling.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    raw = d[~d.sub_sample].groupby("fs")["ibi_rmse"].median()
    interp = d[d.sub_sample].groupby("fs")["ibi_rmse"].median()
    if raw.empty:
        return None

    floor = float(interp.min())
    rows = []
    for fs_hz, measured in raw.items():
        T = 1000.0 / float(fs_hz)
        predicted = float(np.sqrt((T / np.sqrt(12) * np.sqrt(2)) ** 2 + floor**2))
        rows.append({"fs": fs_hz, "step_ms": T, "predicted_rmse_ms": predicted,
                     "measured_rmse_ms": float(measured),
                     "residual_ms": predicted - float(measured),
                     "with_interp_rmse_ms": float(interp.get(fs_hz, np.nan))})
    out = pd.DataFrame(rows).sort_values("fs")
    out.to_csv(RESULTS / "e2_quantisation_model.csv", index=False)
    return out


def main() -> int:
    FIGURES.mkdir(exist_ok=True)
    main_csv = RESULTS / "benchmark_main.csv"
    if not main_csv.exists():
        print("run scripts/02_run_detectors.py first")
        return 1

    df = pd.read_csv(main_csv)
    lb = leaderboard(df)
    lb.to_csv(RESULTS / "leaderboard.csv")

    print("=" * 96)
    print("LEADERBOARD  (median across records, 100% coverage, Lipponen-Tarvainen correction)")
    print("=" * 96)
    print(lb.to_string(float_format=lambda v: f"{v:9.2f}"))

    floor_p = RESULTS / "prv_floor.csv"
    summary: dict = {"leaderboard": json.loads(lb.to_json(orient="index"))}
    if floor_p.exists():
        fl = pd.read_csv(floor_p)
        fl1 = fl[fl.coverage_target == 1.0]
        floor = float(fl1.floor_rmse.median())
        print(f"\nconsensus PRV floor (estimated physiological limit): "
              f"{floor:.2f} ms RMSE, {float(fl1.floor_mae.median()):.2f} ms MAE")
        best_rmse = float(lb.ibi_rmse_ms.min())
        print(f"best detector IBI RMSE {best_rmse:.2f} ms -> "
              f"excess over floor {best_rmse - floor:+.2f} ms")
        summary["prv_floor_rmse_ms"] = floor
        summary["best_excess_over_floor_ms"] = best_rmse - floor

    ok = df[(df.coverage_target == 1.0) & (~df.failed)].dropna(
        subset=["f1", "rmssd_abserr", "ibi_rmse"])
    r_ibi = float(np.corrcoef(ok.f1, ok.ibi_rmse)[0, 1])
    r_rmssd = float(np.corrcoef(ok.f1, ok.rmssd_abserr)[0, 1])
    print(f"\ncorrelation of F1 with IBI RMSE:      r = {r_ibi:+.3f}")
    print(f"correlation of F1 with RMSSD error:   r = {r_rmssd:+.3f}")
    summary["r_f1_ibi_rmse"] = r_ibi
    summary["r_f1_rmssd_error"] = r_rmssd

    cc = correction_comparison()
    if cc is not None:
        print("\nThe correction stage reorders the ranking (median IBI RMSE / RMSSD error, ms)")
        print(cc.to_string(float_format=lambda v: f"{v:9.2f}"))
        summary["correction_comparison"] = json.loads(cc.to_json(orient="index"))

    disc = discrimination(df)
    print("\nDiscriminative power: best-to-worst spread across detectors")
    print(disc.to_string(index=False, float_format=lambda v: f"{v:10.3f}"))
    summary["discrimination"] = json.loads(disc.to_json(orient="records"))

    for name, fn in [("leaderboard", fig_leaderboard), ("f1_vs_hrv_error", fig_f1_vs_hrv),
                     ("coverage_accuracy", fig_coverage), ("bland_altman", fig_bland_altman)]:
        try:
            fn(df)
        except Exception as exc:
            print(f"  ! figure {name} failed: {exc}")
    try:
        fig_experiments()
    except Exception as exc:
        print(f"  ! figure experiments failed: {exc}")

    qm = sampling_quantisation_model()
    if qm is not None:
        print("\nE2: measured sampling-rate degradation vs quantisation theory")
        print("    (prediction is parameter-free given the floor; no fitting)")
        print(qm.to_string(index=False, float_format=lambda v: f"{v:10.2f}"))
        summary["e2_quantisation_max_residual_ms"] = float(qm.residual_ms.abs().max())

    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nfigures -> {FIGURES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
