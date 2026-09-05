# 🚀 Dataset Autopilot

**Autonomous Data-Auditing, Leakage Detection, and Remediation Engine for Tabular Datasets**

Dataset Autopilot is an end-to-end data auditing engine that automatically profiles tabular datasets, detects data-quality anomalies (missingness patterns, extreme outliers, invalid types), discovers ML-specific risks (target leakage, train-test contamination, covariate drift via PSI/KS tests, subgroup fairness disparities), and runs controlled empirical A/B experiments to quantify their exact impact on baseline predictive models.

---

## 🌟 Key Features

1. **Intelligent Ingestion & Profiling**:
   - Universal loader for CSV, TSV, Parquet, JSON, and Excel.
   - Comprehensive column statistics: mean, median, standard deviation, IQR fences, quantiles (p5–p95), skewness, kurtosis, and 10-bin distribution histograms.
   - Semantic type detection (Email, Phone, SSN, Credit Card, IP, URL, UUID, Postal Code, Datetime, Currency).
   - Structural role inference (`id_candidate`, `target`, `numeric_feature`, `categorical_feature`, `constant`).

2. **Data Quality & Informative Missingness (MNAR)**:
   - Identifies non-random missingness correlating with target or covariates using Chi-Square and Cramér's V.
   - Robust outlier detection via 1.5× and 3.0× IQR fences and Z-scores (>3σ).
   - Zero-variance constant column detection and high-cardinality flags.

3. **Machine Learning Risk Detection**:
   - **Target Leakage**: Calculates single-feature ROC-AUC and Spearman/Pearson correlation to flag features with predictive dominance (AUC > 0.88).
   - **Temporal & Post-Event Leakage**: Detects timestamps or semantic attributes recorded post-target event.
   - **Train-Test Contamination**: Identifies shared identifiers or duplicate instances across partitions.

4. **Distribution & Drift Analysis**:
   - Computes **Population Stability Index (PSI)** with 10-bin discretization.
   - Two-sample Kolmogorov-Smirnov (KS) tests for numeric features and Jensen-Shannon divergence for categoricals.

5. **Subgroup Bias & Group Fairness**:
   - Evaluates model performance across demographic slices (Gender, Age brackets, Regions, Tiers).
   - Flags slices deviating by >15% from overall population baseline.

6. **Controlled Remediation Experiments**:
   - Evaluates a controlled Random Forest oracle model with 5-fold cross-validation.
   - Runs automated A/B remediation tests:
     - **Leakage Removal**: Retrains without suspect features (quantifies AUC drop to confirm leakage).
     - **Imputation Strategy**: Compares Median vs. Mean imputation.
     - **Outlier Remediation**: Tests 1st/99th percentile winsorization vs. raw data.

7. **Evidence Reporting & Composite Health Score (0–100)**:
   - Overall Health Score with category breakdown (Quality 30%, Leakage 30%, Drift 20%, Fairness 20%) and Letter Grade (A+ to F).
   - Natural language executive summaries via **Google Gemini API** (`gemini-2.5-flash`) with zero-cost deterministic template fallback.
   - Standalone interactive HTML report and JSON contract export.

8. **Modern React Web Dashboard**:
   - Dark & light glassmorphic UI built with Vite + React + TypeScript + Vanilla CSS tokens.
   - Real-time animated audit progress stepper.
   - Interactive data dictionary, distribution overlays, leakage leaderboards, and experiment scorecards.

---

## 🏗️ Architecture

```
dataset_autopilot/
├── dataset_autopilot/            # Core Engine & Analysis Modules
│   ├── config.py                 # Thresholds, penalties, weights
│   ├── schemas.py                # Pydantic models & OpenAPI contracts
│   ├── utils.py                  # PSI, Cramér's V, safe JSON math
│   ├── ingestion.py              # File loader (CSV, Parquet, JSON, base64)
│   ├── profiling.py              # Column statistics and histograms
│   ├── schema.py                 # Semantic type detection & role inference
│   ├── quality.py                # Missingness (MNAR), outliers (IQR), duplicates
│   ├── leakage.py                # Single-feature AUC/corr & post-event leakage
│   ├── drift.py                  # PSI calculation & KS 2-sample tests
│   ├── bias.py                   # Subgroup fairness & slice disparities
│   ├── models.py                 # Controlled Random Forest oracle models
│   ├── experiments.py            # Controlled A/B remediation experiments
│   ├── reporting.py              # Composite Health Score, Gemini summaries, HTML export
│   ├── engine.py                 # Master Autopilot Orchestrator
│   ├── api.py                    # FastAPI REST Server
│   └── main.py                   # CLI Entrypoint & server runner
├── data/samples/                 # 4 Curated Benchmark Datasets
│   ├── telecom_churn_leakage.csv # Injected target leakage & post-event feature
│   ├── credit_risk_drift.csv     # Injected time-based distribution drift
│   ├── titanic_quality_audit.csv # Injected MNAR missingness & extreme outliers
│   └── recruitment_bias_slices.csv# Injected demographic slice disparity
├── frontend/                     # Interactive Vite + React Dashboard
└── tests/                        # 14 Pytest Unit & Integration Tests
```

---

## ⚡ Quick Start

### 1. Installation

```bash
# Clone or navigate to the directory
cd "d:/Dataset autopilot"

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend
npm install
cd ..
```

### 2. Run Autonomous Audit via CLI

```bash
# Audit a CSV dataset and save JSON + HTML reports
python -m dataset_autopilot.main audit --file data/samples/telecom_churn_leakage.csv --target churn --output report.json --html report.html
```

### 3. Launch FastAPI Server & React Dashboard

**Terminal 1 (Backend API):**
```bash
python -m dataset_autopilot.main serve --port 8000
# API docs available at http://127.0.0.1:8000/docs
```

**Terminal 2 (Frontend UI):**
```bash
cd frontend
npm run dev
# Dashboard available at http://localhost:5173
```

---

## 🧪 Pre-built Benchmark Datasets

The dashboard includes 4 built-in demo datasets for instant 1-click evaluation:

| Benchmark Dataset | Primary ML Challenge | Injected Issues |
| :--- | :--- | :--- |
| **Telecom Customer Churn** | Target Leakage Demo | Post-event feature `refund_issued_after_churn` (AUC 0.993 $\rightarrow$ 0.966 upon removal) |
| **Credit Risk & Loan Default** | Distribution Drift Demo | Macroeconomic shift in `annual_income` & `debt_to_income_ratio` across time (PSI > 0.28) |
| **Titanic Passenger Survival** | Data Quality & Outliers | Informative missingness in `age` and extreme fare outliers (> 3.0× IQR) |
| **Recruitment & Hiring** | Slice Fairness & Bias | Subgroup disparity across experience & candidate demographic slices |

---

## 🔌 REST API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `GET /api/samples` | GET | List available benchmark datasets |
| `POST /api/samples/{id}/analyze` | POST | Trigger 1-click audit on a benchmark dataset |
| `POST /api/analyze` | POST | Upload and audit a custom CSV / Parquet / JSON file |
| `GET /api/status/{job_id}` | GET | Poll job status, progress percentage, and current stage |
| `GET /api/results/{job_id}` | GET | Retrieve full structured Audit Report JSON |
| `GET /api/experiments/{job_id}` | GET | Retrieve experiment scorecard and baseline metrics |
| `GET /api/export/{job_id}/html` | GET | Download standalone interactive HTML audit report |

---

## 🛡️ Testing & Validation

Run the test suite with pytest:

```bash
python -m pytest tests/ -v
```

All 14 tests validate ingestion, profiling, semantic typing, quality checks, leakage detection, drift analysis, slice fairness, oracle modeling, and end-to-end API execution.
