# 🌍 Urban Environmental Intelligence Challenge

### Assignment 2 — Data Science for Software Engineers

> **End-to-end air-quality analytics pipeline**: download live data from the OpenAQ v3 API, apply dimensionality reduction, temporal pattern mining, distribution modelling, and visual integrity auditing — and explore everything through an interactive Streamlit dashboard.

---

## ✨ Highlights

| Capability | Detail |
|---|---|
| **Live Data** | 100 stations discovered via OpenAQ v3 API, 44 yielded usable hourly sensor readings |
| **Big-Data Ready** | Polars lazy evaluation + Parquet columnar storage — sub-second queries on millions of rows |
| **Four Analytical Tasks** | PCA · Temporal heatmaps · KDE & CCDF distributions · Visual integrity audit |
| **Interactive Dashboard** | Dark-themed Streamlit GUI with Plotly charts, live metrics, and data tables |
| **Reproducible** | Single `python main.py` command regenerates every processed file and chart |

---

## 📁 Project Structure

```
project/
│
├── downloader.py              # Step 0 – OpenAQ v3 API download (resume + retry)
├── task1_dimensionality.py    # Task 1 – PCA dimensionality reduction
├── task2_temporal.py          # Task 2 – High-density temporal heatmaps
├── task3_distribution.py      # Task 3 – KDE + CCDF distribution analysis
├── task4_visual_integrity.py  # Task 4 – Lie Factor audit + small multiples
├── dashboard.py               # Interactive Streamlit dashboard
├── main.py                    # Master pipeline orchestrator
├── requirements.txt           # Python dependencies
│
├── data/
│   ├── raw/                   # One .parquet per station (44 files)
│   ├── processed/             # Task outputs (8 parquet files)
│   └── download_state.json    # Tracks download progress (auto-created)
│
└── outputs/                   # Static chart PNGs (8 files)
    ├── task1_pca_scatter.png
    ├── task1_pca_loadings.png
    ├── task2_heatmap_hourly.png
    ├── task2_heatmap_seasonal.png
    ├── task3_kde_peak.png
    ├── task3_ccdf_tail.png
    ├── task4_small_multiples.png
    └── task4_bivariate_matrix.png
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Set your OpenAQ API key

```bash
export OPENAQ_API_KEY="your_key_here"
```

### 3. Run the pipeline

```bash
# Full pipeline: download + all 4 tasks
python main.py

# Skip download (if data/raw/ already has parquet files)
python main.py --skip-download

# Run a specific task only
python main.py --task 1   # PCA
python main.py --task 2   # Temporal heatmaps
python main.py --task 3   # Distribution modelling
python main.py --task 4   # Visual integrity audit
```

### 4. Launch the dashboard

```bash
streamlit run dashboard.py
```

---

## 📊 Task Summaries

### Task 1 — Dimensionality Reduction (PCA)

**Goal:** Reduce 6 air-quality parameters (PM2.5, PM10, NO₂, O₃, Temperature, Humidity) to 2 principal components.

- **Method:** StandardScaler → PCA (2 components) on annual means per station
- **Stations:** 44 (all stations; missing parameters filled via mean imputation)
- **Key Finding:** PC1 captures the pollution axis (PM10, PM2.5, NO₂ load positively), PC2 captures the meteorological axis (Temperature vs Humidity — opposing signs)
- **Outputs:** Scatter plot colour-coded by zone type (Industrial / Residential) + loading bar chart

### Task 2 — High-Density Temporal Analysis

**Goal:** Build heatmaps revealing hourly and seasonal PM2.5 violation rhythms.

- **Method:** Binary violation flag (PM2.5 > 35 µg/m³, WHO 24-hr guideline) → aggregated by station × hour-of-day and station × day-of-year
- **Stations:** 36 (only stations with PM2.5 sensor data)
- **Key Finding:** Seasonal variation dominates hourly variation — winter temperature inversions and heating activity drive exceedances
- **Chart Justification:** 36 stations × 8,760 hours = unreadable as line charts → heatmap maximises data-ink ratio

### Task 3 — Distribution Modelling & Tail Integrity

**Goal:** Model PM2.5 concentration distribution and answer: *Which plot honestly depicts rare extreme events?*

- **Method:** Kernel Density Estimate (linear scale) + Complementary CDF (log-log scale) on the highest-data industrial station
- **Station:** 1 (deep-dive on station 26)
- **Key Finding:** KDE excels at showing modal concentration but compresses the hazardous tail into near-invisibility; the CCDF on log-log is the honest depiction — no binning artefacts, every decade gets equal visual space, and exceedance probability is directly readable

### Task 4 — Visual Integrity Audit

**Goal:** Calculate the Lie Factor of a 3D bar chart and propose a distortion-free alternative.

- **Method:** Tufte's Lie Factor calculation (LF = 1.5 → 50% visual distortion) → Small Multiples scatter (one panel per geographic region) + Bivariate colour matrix legend
- **Stations:** 36
- **Key Finding:** 3D perspective causes bars to appear ~50% larger than the data warrants, combined with occlusion and wasted data-ink → replaced with small multiples achieving LF = 1.0

---

## 🔢 Station Count Explained

The dashboard shows different station counts on different pages. This is **intentional**, not a bug:

| Page | Count | Reason |
|---|---|---|
| Task 1 (PCA) | **44** | Uses all 6 parameters; missing values are imputed |
| Task 2 (Temporal) | **36** | Filters to PM2.5-only; 8 stations lack PM2.5 sensors |
| Task 3 (Distribution) | **1** | Deep-dive on highest-data industrial station |
| Task 4 (Visual Integrity) | **36** | PM2.5-based summary → same 36 stations |
| Overview | **36** | Uses Task 4 summary file |

**Discovery pipeline:** 100 stations queried → 44 had usable data → 36 of those had PM2.5 sensors.

---

## 🛠️ Technical Design Decisions

### Big-Data Handling with Polars

- All data stored as **Parquet** (columnar, zstd-compressed) — 5–10× smaller than CSV
- **`pl.scan_parquet()`** enables lazy evaluation — Polars reads only needed columns/rows
- Predicate pushdown filters are applied before data reaches memory

### Download Resilience

- **Resume support:** `download_state.json` tracks each station's status (`done` / `failed` / `no_data`); re-running skips completed stations
- **Rate-limit handling:** HTTP 429 → 65-second back-off + retry
- **Network retry:** Connection errors trigger exponential back-off (5s → 10s → … → 300s max), up to 10 retries
- **Crash protection:** Each API page is appended to a temp JSONL file immediately — a mid-download crash only loses the current page

### No Graphical Ducks (Tufte Principles)

- **No 3D effects**, shadows, or decorative grids
- `axes.spines.top = False`, `axes.spines.right = False` applied globally
- Sequential colourmap only (**YlOrRd**) — monotonically increasing luminance ensures visual order matches data order
- Rainbow / jet colormaps explicitly rejected (non-monotonic luminance creates false perceptual boundaries)

---

## 📋 Dependencies

| Package | Purpose |
|---|---|
| `polars` | Fast columnar DataFrames with lazy/streaming evaluation |
| `numpy` | Numerical computing |
| `scikit-learn` | PCA, StandardScaler |
| `scipy` | `gaussian_kde` for distribution modelling |
| `matplotlib` | Static chart generation (all 8 task outputs) |
| `plotly` | Interactive charts in the Streamlit dashboard |
| `streamlit` | Web-based interactive dashboard |
| `requests` | OpenAQ API calls with retry logic |
| `tqdm` | Progress bars |

---

## 📄 License

This project was completed as part of the **Data Science for Software Engineers** course. For academic use only.
