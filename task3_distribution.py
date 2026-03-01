"""
task3_distribution.py
=====================
Task 3 – Distribution Modelling & Tail Integrity (25%)

PROCESSING STEP:
  Reads raw Parquet files → selects best industrial station →
  computes distribution statistics, KDE values, CCDF values →
  saves all processed results to data/processed/

OUTPUT FILES:
  data/processed/task3_stats.parquet      – summary statistics per station
  data/processed/task3_kde.parquet        – KDE curve (x, density) values
  data/processed/task3_ccdf.parquet       – CCDF curve (value, exceedance) values
  outputs/task3_kde_peak.png
  outputs/task3_ccdf_tail.png
"""

import logging
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import gaussian_kde

RAW_DATA_DIR       = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
OUTPUT_DIR         = Path("outputs")

PM25_THRESHOLD    = 35.0
EXTREME_THRESHOLD = 200.0

PARAM_ALIASES = {
    "pm25": "pm25", "pm2.5": "pm25", "pm2_5": "pm25",
    "pm10": "pm10",
    "no2": "no2", "nitrogen dioxide": "no2",
    "o3": "o3", "ozone": "o3",
    "temperature": "temperature", "temp": "temperature",
    "humidity": "humidity", "relativehumidity": "humidity", "rh": "humidity",
}

plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "font.size": 10,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s]  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("task3")


def load_pm25_all() -> pl.DataFrame:
    """Load all raw PM2.5 data from parquet files."""
    files = sorted(RAW_DATA_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files in {RAW_DATA_DIR}")

    log.info(f"Scanning {len(files)} raw station files ...")
    frames = [
        pl.scan_parquet(f).select(
            ["station_id", "zone_type", "parameter", "value"]
        )
        for f in files
    ]
    df = pl.concat(frames).collect()

    df = df.with_columns(
        pl.col("parameter").str.to_lowercase().str.strip_chars()
          .replace(PARAM_ALIASES).alias("parameter")
    )

    pm25 = df.filter(
        (pl.col("parameter") == "pm25") &
        pl.col("value").is_not_null() &
        (pl.col("value") > 0)
    )
    log.info(f"Found {len(pm25):,} valid PM2.5 readings across "
             f"{pm25['station_id'].n_unique()} stations")
    return pm25


def select_best_station(pm25: pl.DataFrame) -> tuple:
    """
    PROCESSING STEP 1:
    Select the industrial station with the most PM2.5 readings.
    Falls back to any station if no industrial stations found.
    """
    counts = (
        pm25.group_by(["station_id", "zone_type"])
            .agg(pl.len().alias("n"))
            .sort("n", descending=True)
    )

    industrial = counts.filter(pl.col("zone_type") == "Industrial")
    if len(industrial) > 0:
        best = industrial.row(0, named=True)
    else:
        best = counts.row(0, named=True)

    station_id = best["station_id"]
    zone       = best["zone_type"]
    n          = best["n"]
    log.info(f"Selected station: {station_id} ({zone}) – {n:,} readings")

    values = (
        pm25.filter(pl.col("station_id") == station_id)["value"].to_numpy()
    )
    return station_id, zone, values


def compute_statistics(station_id: str, zone: str, values: np.ndarray) -> pl.DataFrame:
    """
    PROCESSING STEP 2:
    Compute full distribution statistics for the selected station.
    Saves as a processed summary table.
    """
    log.info("Computing distribution statistics ...")

    stats = {
        "station_id"  : station_id,
        "zone_type"   : zone,
        "n"           : len(values),
        "mean"        : float(np.mean(values)),
        "median"      : float(np.median(values)),
        "std"         : float(np.std(values)),
        "p05"         : float(np.percentile(values, 5)),
        "p25"         : float(np.percentile(values, 25)),
        "p75"         : float(np.percentile(values, 75)),
        "p95"         : float(np.percentile(values, 95)),
        "p99"         : float(np.percentile(values, 99)),
        "p999"        : float(np.percentile(values, 99.9)),
        "max"         : float(np.max(values)),
        "p_health"    : float(np.mean(values > PM25_THRESHOLD)),
        "p_extreme"   : float(np.mean(values > EXTREME_THRESHOLD)),
        "hours_health_per_year"  : float(np.mean(values > PM25_THRESHOLD) * 8760),
        "hours_extreme_per_year" : float(np.mean(values > EXTREME_THRESHOLD) * 8760),
    }

    for k, v in stats.items():
        if k not in ("station_id", "zone_type"):
            log.info(f"  {k:>30}  {v}")

    stats_df = pl.DataFrame({k: [v] for k, v in stats.items()})
    return stats_df


def compute_kde(values: np.ndarray) -> pl.DataFrame:
    """
    PROCESSING STEP 3:
    Compute KDE curve values and save as processed table.
    x_eval and density columns represent the full KDE curve.
    """
    log.info("Computing KDE curve ...")
    x_max  = min(float(np.percentile(values, 99.9)), 600)
    x_eval = np.linspace(0, x_max, 2000)
    density = gaussian_kde(values, bw_method="scott")(x_eval)

    kde_df = pl.DataFrame({
        "x_value"    : x_eval.tolist(),
        "density"    : density.tolist(),
    })
    log.info(f"KDE curve: {len(kde_df)} points, x range 0–{x_max:.1f} µg/m³")
    return kde_df


def compute_ccdf(values: np.ndarray) -> pl.DataFrame:
    """
    PROCESSING STEP 4:
    Compute empirical CCDF (Complementary CDF) values and save as processed table.
    For each value x, stores P(X > x).
    """
    log.info("Computing CCDF curve ...")
    sorted_vals = np.sort(values)
    exceedance  = 1.0 - np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

    # keep only points with positive exceedance for log-log plotting
    valid = exceedance > 0
    ccdf_df = pl.DataFrame({
        "pm25_value"  : sorted_vals[valid].tolist(),
        "exceedance"  : exceedance[valid].tolist(),
    })
    log.info(f"CCDF curve: {len(ccdf_df)} points")
    return ccdf_df


def save_processed(stats_df, kde_df, ccdf_df):
    """Save all processed datasets to data/processed/"""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    stats_path = PROCESSED_DATA_DIR / "task3_stats.parquet"
    kde_path   = PROCESSED_DATA_DIR / "task3_kde.parquet"
    ccdf_path  = PROCESSED_DATA_DIR / "task3_ccdf.parquet"

    stats_df.write_parquet(stats_path)
    kde_df.write_parquet(kde_path)
    ccdf_df.write_parquet(ccdf_path)

    log.info(f"Saved processed stats → {stats_path}")
    log.info(f"Saved processed KDE   → {kde_path}  ({len(kde_df)} curve points)")
    log.info(f"Saved processed CCDF  → {ccdf_path}  ({len(ccdf_df)} curve points)")


def plot_kde(kde_df: pl.DataFrame, stats_df: pl.DataFrame):
    """Generate KDE plot from processed curve data."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    x       = kde_df["x_value"].to_numpy()
    density = kde_df["density"].to_numpy()
    p99     = float(stats_df["p99"][0])
    median  = float(stats_df["median"][0])
    station = stats_df["station_id"][0]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.fill_between(x, density, alpha=0.2, color="#457b9d")
    ax.plot(x, density, color="#457b9d", linewidth=2)
    ax.axvline(PM25_THRESHOLD, color="#e63946", linestyle="--", linewidth=1.5,
               label=f"Health threshold ({PM25_THRESHOLD} µg/m³)")
    ax.axvline(p99, color="#2d6a4f", linestyle=":", linewidth=1.5,
               label=f"99th percentile ({p99:.1f} µg/m³)")
    ax.axvline(median, color="#457b9d", linestyle="-.", linewidth=1,
               label=f"Median ({median:.1f} µg/m³)", alpha=0.7)
    ax.set_xlabel("PM2.5 concentration (µg/m³)", fontsize=11)
    ax.set_ylabel("Probability density", fontsize=11)
    ax.set_title(f"Kernel Density Estimate – Station {station}\n"
                 f"Shows peak/modal pollution level (linear scale)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    out = OUTPUT_DIR / "task3_kde_peak.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved KDE plot → {out}")


def plot_ccdf(ccdf_df: pl.DataFrame, stats_df: pl.DataFrame):
    """Generate CCDF plot from processed curve data."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    x          = ccdf_df["pm25_value"].to_numpy()
    exceedance = ccdf_df["exceedance"].to_numpy()
    p99        = float(stats_df["p99"][0])
    p_health   = float(stats_df["p_health"][0])
    p_extreme  = float(stats_df["p_extreme"][0])
    station    = stats_df["station_id"][0]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, exceedance, color="#457b9d", linewidth=2)
    ax.axvline(PM25_THRESHOLD, color="#e63946", linestyle="--", linewidth=1.5,
               label=f"Health threshold – P(exceed) = {p_health*100:.1f}%")
    ax.axvline(p99, color="#2d6a4f", linestyle=":", linewidth=1.5,
               label=f"99th percentile ({p99:.1f} µg/m³)")
    if p_extreme > 0:
        ax.axvline(EXTREME_THRESHOLD, color="#f4a261", linestyle="-.", linewidth=1.5,
                   label=f"Extreme hazard ({EXTREME_THRESHOLD} µg/m³)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("PM2.5 concentration (µg/m³)  [log scale]", fontsize=11)
    ax.set_ylabel("P(PM2.5 > x)  [log scale]", fontsize=11)
    ax.set_title(f"Complementary CDF – Station {station}\n"
                 f"Honest tail view – rare events clearly visible (log-log scale)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    out = OUTPUT_DIR / "task3_ccdf_tail.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved CCDF plot → {out}")


def print_analysis(stats_df):
    s = stats_df.row(0, named=True)
    print(f"""
================================================================================
  TASK 3 – ANALYSIS
================================================================================

Processed datasets saved to:
  data/processed/task3_stats.parquet  – summary statistics
  data/processed/task3_kde.parquet    – KDE curve (2000 points)
  data/processed/task3_ccdf.parquet   – CCDF curve ({s['n']} points)

Station analysed : {s['station_id']} ({s['zone_type']})
Total readings   : {s['n']:,}

Key statistics:
  Mean PM2.5          : {s['mean']:.2f} µg/m³
  Median PM2.5        : {s['median']:.2f} µg/m³
  95th percentile     : {s['p95']:.2f} µg/m³
  99th percentile     : {s['p99']:.2f} µg/m³   ← answer to the task
  Maximum recorded    : {s['max']:.2f} µg/m³
  P(> {PM25_THRESHOLD} µg/m³)      : {s['p_health']*100:.2f}%  ({s['hours_health_per_year']:.0f} hours/year)
  P(> {EXTREME_THRESHOLD} µg/m³)    : {s['p_extreme']*100:.4f}%  ({s['hours_extreme_per_year']:.1f} hours/year)

Which plot is more honest for rare events?

  KDE (linear scale) – GOOD for modal peak, BAD for tails
    The tail above ~100 µg/m³ is compressed into a near-invisible
    sliver. Bin-width choice can hide or exaggerate tail features.

  CCDF (log-log scale) – HONEST for rare hazard events
    (a) No binning – avoids histogram artefacts entirely
    (b) Equal visual space per decade on both axes
    (c) P(PM2.5 > x) is directly readable for any threshold
    (d) Straight line on log-log = power-law tail (physically meaningful)
    (e) Cannot be manipulated by axis or bin choices

  Conclusion: The 99th percentile is {s['p99']:.1f} µg/m³.
  In {s['p_health']*100:.2f}% of hours the station exceeds the WHO health threshold.
================================================================================
""")


def run_task3():
    log.info("Starting Task 3 – Distribution Modelling & Tail Integrity")
    pm25                     = load_pm25_all()
    station_id, zone, values = select_best_station(pm25)
    stats_df                 = compute_statistics(station_id, zone, values)
    kde_df                   = compute_kde(values)
    ccdf_df                  = compute_ccdf(values)
    save_processed(stats_df, kde_df, ccdf_df)
    plot_kde(kde_df, stats_df)
    plot_ccdf(ccdf_df, stats_df)
    print_analysis(stats_df)
    log.info("Task 3 complete.")


if __name__ == "__main__":
    run_task3()
