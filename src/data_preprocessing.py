import pandas as pd
import numpy as np
from pathlib import Path
def load_raw_data():
    data_path = Path("data/raw/credit_data.csv")
    df = pd.read_csv(data_path, low_memory=False)
    return df
def drop_identifier_columns(df):
    drop_cols = [
        "id",
        "member_id",
        "url",
        "desc",
        "title",
        "zip_code",
        "addr_state",
        "emp_title"
    ]
    return df.drop(columns=drop_cols, errors="ignore")
def drop_leakage_columns(df):
    leakage_cols = [
        "out_prncp",
        "out_prncp_inv",
        "total_pymnt",
        "total_pymnt_inv",
        "total_rec_prncp",
        "total_rec_int",
        "total_rec_late_fee",
        "recoveries",
        "collection_recovery_fee",
        "last_pymnt_amnt",
        "last_pymnt_d",
        "next_pymnt_d",
        "last_credit_pull_d",
        "last_fico_range_high",
        "last_fico_range_low",
        "settlement_status",
        "settlement_date",
        "settlement_amount",
        "settlement_percentage",
        "settlement_term",
        "hardship_loan_status"
    ]
    return df.drop(columns=leakage_cols, errors="ignore")
def create_binary_target(df):
    df = df.copy()

    # keep only final loan outcomes
    final_status = [
        'Fully Paid',
        'Charged Off',
        'Default',
        'Does not meet the credit policy. Status:Charged Off',
        'Does not meet the credit policy. Status:Fully Paid'
    ]

    df = df[df['loan_status'].isin(final_status)]

    # map to binary target
    df['target'] = df['loan_status'].replace({
        'Fully Paid': 0,
        'Does not meet the credit policy. Status:Fully Paid': 0,
        'Charged Off': 1,
        'Default': 1,
        'Does not meet the credit policy. Status:Charged Off': 1
    })

    return df
def fix_invalid_values(df):
    df["dti"] = df["dti"].clip(lower=0)
    df["revol_util"] = df["revol_util"].clip(upper=100)
    return df
def drop_high_missing_columns(df, threshold=0.6):
    missing_ratio = df.isnull().mean()
    drop_cols = missing_ratio[missing_ratio > threshold].index
    return df.drop(columns=drop_cols)
def drop_low_variance_columns(df, threshold=0.98):
    """
    Drops columns where a single value accounts for more than `threshold`
    proportion of the data (near-constant features).
    """
    low_variance_cols = []

    for col in df.columns:
        top_freq = df[col].value_counts(normalize=True, dropna=False).iloc[0]
        if top_freq > threshold:
            low_variance_cols.append(col)

    df = df.drop(columns=low_variance_cols)

    print(f"Dropped {len(low_variance_cols)} low-variance columns:")
    print(low_variance_cols)

    return df
def drop_highly_correlated_features(df, target_col, threshold=0.85):
    # Only numeric columns
    numeric_df = df.select_dtypes(include="number")

    # Correlation matrix
    corr_matrix = numeric_df.corr()

    # Correlation of each feature with target
    target_corr = corr_matrix[target_col].abs()

    # Upper triangle mask
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    drop_cols = set()

    for col in upper.columns:
        for row in upper.index:
            if abs(upper.loc[row, col]) > threshold:
                # Drop the feature less correlated with target
                if target_corr[row] < target_corr[col]:
                    drop_cols.add(row)
                else:
                    drop_cols.add(col)

    df = df.drop(columns=list(drop_cols), errors="ignore")
    print(f"Dropped {len(drop_cols)} highly correlated features:")
    print(drop_cols)

    return df


from src.config import CLIP_COLS, LOG_COLS, IQR_COLS

def handle_outliers(df, 
                    clip_cols=CLIP_COLS, 
                    log_cols=LOG_COLS, 
                    iqr_cols=IQR_COLS, 
                    factor=1.5):
    """
    Handles outliers using different strategies:
    - clip_cols: columns to clip to valid ranges
    - log_cols: columns to log-transform
    - iqr_cols: columns to apply IQR capping
    """
    df = df.copy()

    # 1. Clip bounded variables
    if clip_cols:
        for col in clip_cols:
            if col in df.columns:
                df[col] = df[col].clip(lower=0)

    # 2. Log-transform skewed monetary variables
    if log_cols:
        for col in log_cols:
            if col in df.columns:
                df[col] = np.log1p(df[col])  # log(1 + x) avoids log(0)

    # 3. IQR cap remaining numeric variables
    if iqr_cols:
        for col in iqr_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - factor * IQR
                upper = Q3 + factor * IQR
                df[col] = df[col].clip(lower, upper)

    return df

def clean_term_column(df):
    """
    Converts term from '36 months' → 36
    """
    df = df.copy()
    if 'term' in df.columns:
        df['term'] = df['term'].str.extract(r'(\d+)').astype(int)
    return df


from datetime import datetime

def create_credit_age_feature(df):
    """
    Converts earliest_cr_line to credit_age (in years) and drops the original column.
    """
    df = df.copy()

    # Step 1: Convert to datetime
    df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], errors='coerce')

    # Step 2: Calculate credit age in years
    current_date = datetime.now()
    df['credit_age'] = (current_date - df['earliest_cr_line']).dt.days / 365.25

    # Step 3: Drop original column
    df.drop('earliest_cr_line', axis=1, inplace=True)

    # Step 4: Round for readability
    df['credit_age'] = df['credit_age'].round(2)

    return df

def handle_months_since_features(df):
    """
    Handles 'months since' and delinquency-related features:
    - Fills NaN with -1 (meaning 'no such event')
    - Clips extreme upper values at 99th percentile
    """
    df = df.copy()

    months_features = [
        'inq_last_6mths',
        'mths_since_last_delinq',
        'delinq_2yrs',
        'mths_since_rcnt_il',
        'mo_sin_old_rev_tl_op',
        'mo_sin_rcnt_tl',
        'mths_since_recent_inq',
        'mo_sin_old_il_acct'
    ]

    for col in months_features:
        if col in df.columns:
            # Step 1: NaN means "never happened"
            df[col] = df[col].fillna(-1)

            # Step 2: Clip extreme upper outliers
            upper = df[col].quantile(0.99)
            df[col] = df[col].clip(upper=upper)

    return df

    import numpy as np

def clean_emp_length(df):
    """
    Converts emp_length from string format to numeric years.
    Handles:
    - '< 1 year'  -> 0.5
    - '10+ years' -> 10
    - 'X years'   -> X
    Fills missing values with median.
    """
    df = df.copy()

    def parse_emp_length(x):
        if pd.isnull(x):
            return np.nan
        if x == '< 1 year':
            return 0.5
        if x == '10+ years':
            return 10
        return int(x.split()[0])

    if 'emp_length' in df.columns:
        df['emp_length'] = df['emp_length'].apply(parse_emp_length)

        # Fill missing values with median
        median_val = df['emp_length'].median()
        df['emp_length'] = df['emp_length'].fillna(median_val)

    return df

def handle_remaining_nulls(df):
    """
    Handles remaining missing values after all domain-specific cleaning.
    
    Strategy:
    - Median imputation for continuous ratios/percentages
    - Zero imputation for count-like or event-based features
    """
    df = df.copy()

    # Columns where median makes sense
    median_cols = [
        'revol_util',
        'pct_tl_nvr_dlq'
    ]

    for col in median_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Columns where 0 means "no event / no activity"
    zero_fill_cols = [
        'percent_bc_gt_75',
        'dti',
        'num_rev_accts'
    ]

    for col in zero_fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df

def final_feature_pruning(df):
    """
    Drops manually identified redundant or leakage-prone features.
    This step enforces a clean, interpretable and compact feature set.
    """

    drop_cols = [
        # Absolute leakage
        "loan_status",
        "issue_d",
        "funded_amnt",

        # Redundancy: grade vs sub_grade
        "grade",

        # Revolving credit redundancy
        "percent_bc_gt_75",
        "bc_open_to_buy",
        "bc_util",

        # Excessive account counters
        "num_bc_sats",
        "num_il_tl",
        "num_op_rev_tl",
        "num_tl_op_past_12m",

        # Delinquency overlap
        "num_accts_ever_120_pd",
        "num_tl_90g_dpd_24m",
        "num_tl_120dpd_2m",

        # Credit age duplicates
        "mo_sin_old_il_acct",

        # Optional: too granular months counters
        "mo_sin_rcnt_tl",
        "mths_since_recent_bc",
        "mths_since_recent_inq"
    ]

    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")

    print(f"Final feature pruning done. Dropped {len(drop_cols)} columns.")
    return df


def save_processed_data(df):
    output_path = Path("data/processed/cleaned_data.csv")
    df.to_csv(output_path, index=False)


def run_preprocessing():
    print("Loading raw data...")
    df = load_raw_data()

    print("Dropping identifier columns...")
    df = drop_identifier_columns(df)

    print("Dropping leakage columns...")
    df = drop_leakage_columns(df)

    print("Creating binary target...")
    df = create_binary_target(df)

    print("Fixing invalid values...")
    df = fix_invalid_values(df)

    # Feature creation & semantic cleaning FIRST
    print("Cleaning term column...")
    df = clean_term_column(df)

    print("Transforming earliest_cr_line to credit_age...")
    df = create_credit_age_feature(df)

    print("Handling months_since features...")
    df = handle_months_since_features(df)

    print("Cleaning emp_length...")
    df = clean_emp_length(df)

    print("Handling remaining null values...")
    df = handle_remaining_nulls(df)

    # Structural pruning AFTER feature creation
    print("Dropping high-missing columns...")
    df = drop_high_missing_columns(df)

    print("Dropping low-variance columns...")
    df = drop_low_variance_columns(df)

    print("Dropping highly correlated features...")
    df = drop_highly_correlated_features(df, target_col="target")

    print("Dropping redundant features...")
    df= final_feature_pruning(df)

    print("Final NaN cleanup...")
    df = df.dropna(subset=["target"])
    df = df.dropna()

    print("Saving processed data...")
    save_processed_data(df)

    print("Preprocessing completed successfully ✅")

if __name__ == "__main__":
    run_preprocessing()
