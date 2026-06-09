"""
config.py
---------
Central configuration for the Airline Loyalty Analytics System.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

# ── Raw Data File Names (EXACT names from Google Drive) ────────────────────────
LOYALTY_FILE  = "Customer Loyalty History.csv"
FLIGHT_FILE   = "Customer Flight Activity.csv"
CALENDAR_FILE = "Calendar.csv"
DICT_FILE     = "Airline Loyalty Data Dictionary.csv"

# ── Churn Definition ───────────────────────────────────────────────────────────
CHURN_INACTIVITY_MONTHS    = 6
OBSERVATION_WINDOW_MONTHS  = 12
PREDICTION_WINDOW_MONTHS   = 3

# ── Train/Test Split ───────────────────────────────────────────────────────────
TRAIN_CUTOFF_YEAR  = 2017
TRAIN_CUTOFF_MONTH = 6

# ── Model Hyperparameters ──────────────────────────────────────────────────────
RANDOM_STATE = 42
N_JOBS = -1

LOGISTIC_PARAMS = {
    "C": 1.0,
    "max_iter": 1000,
    "solver": "lbfgs",
    "random_state": RANDOM_STATE,
}

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 8,
    "min_samples_leaf": 10,
    "random_state": RANDOM_STATE,
    "n_jobs": N_JOBS,
}

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
    "n_jobs": N_JOBS,
}

# ── Segmentation ───────────────────────────────────────────────────────────────
N_CLUSTERS_MIN   = 2
N_CLUSTERS_MAX   = 10
N_CLUSTERS_FINAL = 6

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_LEVEL  = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

# ── Business Thresholds ────────────────────────────────────────────────────────
HIGH_CHURN_RISK_THRESHOLD   = 0.70
MEDIUM_CHURN_RISK_THRESHOLD = 0.40

# ── Standardised Column Names (after cleaning) ─────────────────────────────────
# These match the lowercased/underscored versions of actual CSV headers
CUSTOMER_ID_COL = "loyalty_number"
DATE_COL        = "year_month"
TARGET_COL      = "churned"

# ── Actual CSV column names (raw, before standardisation) ─────────────────────
# Customer Flight Activity columns
FLIGHT_COLS_RAW = [
    "Loyalty Number", "Year", "Month", "Total Flights",
    "Distance", "Points Accumulated", "Points Redeemed",
    "Dollar Cost Points Redeemed"
]

# Customer Loyalty History columns
LOYALTY_COLS_RAW = [
    "Loyalty Number", "Country", "Province", "City", "Postal Code",
    "Gender", "Education", "Salary", "Marital Status", "Loyalty Card",
    "CLV", "Enrollment Type", "Enrollment Year", "Enrollment Month",
    "Cancellation Year", "Cancellation Month"
]