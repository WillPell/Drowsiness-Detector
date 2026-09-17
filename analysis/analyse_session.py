import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

STATE_COLOUR = {
    "AWAKE": "#22c55e",
    "SLIGHTLY_DROWSY": "#eab308",
    "DROWSY": "#f97316",
    "CRITICAL": "#ef4444",
}

REQUIRED_COLUMNS = {
    "elapsed_s", "raw_ear", "smoothed_ear", "perclos",
    "drowsiness_score", "state", "blink_rate",
}

PERCLOS_RISK_THRESHOLD = 40.0
BAR_WIDTH = 50


def load_session(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Session contains no rows")
    return df


def generate_figure(df: pd.DataFrame, out_path: str):
    t = df["elapsed_s"]

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True,
                             gridspec_kw={"hspace": 0.25})
    fig.suptitle("Drowsiness Detection - Session Analysis",
                 fontsize=14, fontweight="bold", y=0.98)

    ax = axes[0]
    ax.plot(t, df["raw_ear"], alpha=0.3, color="#94a3b8", linewidth=0.8, label="Raw EAR")
    ax.plot(t, df["smoothed_ear"], color="#3b82f6", linewidth=1.5, label="Smoothed EAR")
    ax.set_ylabel("Eye Aspect Ratio")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    ax.plot(t, df["drowsiness_score"], color="#8b5cf6", linewidth=1.5)
    for level, state in ((25, "SLIGHTLY_DROWSY"), (50, "DROWSY"), (75, "CRITICAL")):
        ax.axhline(level, color=STATE_COLOUR[state], ls="--", lw=0.8)
    ax.set_ylabel("Drowsiness Score")
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.2)

    for state_name, colour in STATE_COLOUR.items():
        mask = df["state"] == state_name
        if mask.any():
            starts = t[mask & ~mask.shift(1, fill_value=False)]
            ends = t[mask & ~mask.shift(-1, fill_value=False)]
            for start, end in zip(starts, ends):
                ax.axvspan(start, end, alpha=0.10, color=colour)

    ax = axes[2]
    ax.plot(t, df["perclos"] * 100, color="#f59e0b", linewidth=1.5)
    ax.axhline(PERCLOS_RISK_THRESHOLD, color="#ef4444", ls="--", lw=0.8,
               label="40% threshold")
    ax.set_ylabel("PERCLOS (%)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)

    ax = axes[3]
    ax.plot(t, df["blink_rate"], color="#10b981", linewidth=1.5)
    ax.axhspan(15, 20, alpha=0.08, color="#10b981", label="Normal range")
    ax.set_ylabel("Blinks / min")
    ax.set_xlabel("Elapsed Time (seconds)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)

    patches = [Patch(facecolor=c, label=s, alpha=0.6) for s, c in STATE_COLOUR.items()]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.01))

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Figure saved -> {out_path}")


def longest_run_seconds(df: pd.DataFrame, state: str) -> float:
    mask = df["state"] == state
    if not mask.any():
        return 0.0

    t = df["elapsed_s"]
    group = (mask != mask.shift()).cumsum()
    runs = t[mask].groupby(group[mask])
    return float(max(run.iloc[-1] - run.iloc[0] for _, run in runs))


def generate_report(df: pd.DataFrame, out_path: str):
    duration = float(df["elapsed_s"].iloc[-1])
    frames = len(df)
    frame_rate = frames / duration if duration > 0 else 0.0

    critical_share = float((df["state"] == "CRITICAL").mean()) * 100
    longest_critical = longest_run_seconds(df, "CRITICAL")

    if critical_share > 5.0 or longest_critical > 10.0:
        verdict = "HIGH RISK - sustained critical drowsiness detected"
    elif critical_share > 0.0 or float((df["state"] == "DROWSY").mean()) > 0.20:
        verdict = "MODERATE RISK - repeated drowsiness episodes"
    else:
        verdict = "LOW RISK - driver appeared generally alert"

    lines = [
        "=" * 60,
        "  DROWSINESS DETECTION - SESSION REPORT",
        "=" * 60,
        "",
        f"  Duration           : {duration / 60:.1f} minutes ({duration:.0f} seconds)",
        f"  Total frames       : {frames}",
        f"  Avg frame rate     : {frame_rate:.1f} FPS",
        "",
        "  -- Drowsiness Metrics " + "-" * 28,
        f"  Average score      : {df['drowsiness_score'].mean():.1f} / 100",
        f"  Peak score         : {df['drowsiness_score'].max():.1f} / 100",
        f"  Average EAR        : {df['smoothed_ear'].mean():.3f}",
        f"  Average PERCLOS    : {df['perclos'].mean() * 100:.1f}%",
        f"  Average blink rate : {df['blink_rate'].mean():.1f} blinks/min",
        "",
        "  -- State Distribution " + "-" * 28,
    ]

    for state in STATE_COLOUR:
        share = float((df["state"] == state).mean())
        filled = int(round(share * BAR_WIDTH))
        bar = "#" * filled + "." * (BAR_WIDTH - filled)
        lines.append(f"  {state:<20} {bar} {share * 100:>5.1f}%")

    lines += [
        "",
        "  -- Risk Assessment " + "-" * 31,
        f"  Time in CRITICAL   : {critical_share:.1f}%",
        f"  Longest critical   : {longest_critical:.1f} seconds",
        f"  {verdict}",
        "",
        "=" * 60,
        "",
    ]

    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(f"  Report saved -> {out_path}")


def analyse(csv_path: str):
    print(f"\nAnalysing: {csv_path}")
    df = load_session(csv_path)
    base = os.path.splitext(csv_path)[0]
    generate_figure(df, base + "_analysis.png")
    generate_report(df, base + "_report.txt")


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m analysis.analyse_session <session.csv> [session2.csv ...]")
        return 1

    exit_code = 0
    for path in sys.argv[1:]:
        if not os.path.isfile(path):
            print(f"File not found: {path}")
            exit_code = 1
            continue
        try:
            analyse(path)
        except (ValueError, pd.errors.ParserError) as exc:
            print(f"  Skipped {path}: {exc}")
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
