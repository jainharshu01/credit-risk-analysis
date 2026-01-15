# src/config.py

# ------------------
# General settings
# ------------------
RANDOM_SEED = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "loan_status"

# ------------------
# Paths
# ------------------
RAW_DATA_PATH = "data/raw/credit_data.csv"
PROCESSED_DATA_PATH = "data/processed/cleaned_data.csv"

# ------------------
# Model parameters
# ------------------
LOGISTIC_PARAMS = {
    "max_iter": 1000,
    "random_state": RANDOM_SEED
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 10,
    "random_state": RANDOM_SEED
}

XGBOOST_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 5,
    "random_state": RANDOM_SEED
}

# Outlier handling configuration

CLIP_COLS = [
    "revol_util",   # percentage
    "dti"           # ratio
]

LOG_COLS = [
    "loan_amnt",
    "annual_inc",
    "revol_bal",
    "tot_cur_bal"
]

IQR_COLS = [
    "installment",
    "credit_age"
]
