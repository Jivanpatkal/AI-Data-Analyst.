import pandas as pd
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# DATA CLEANING MODULE
# All cleaning functions return a cleaned DataFrame + a log of changes made
# ══════════════════════════════════════════════════════════════════════════════

def remove_duplicates(df):
    """
    Remove duplicate rows from the DataFrame.
    Returns: cleaned df, count of rows removed
    """
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    return df, removed


def fix_column_names(df):
    """
    Standardize column names:
    - Strip whitespace
    - Replace spaces with underscores
    - Lowercase everything
    """
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^\w]", "", regex=True)
    )
    return df


def fix_data_types(df):
    """
    Automatically detect and fix data types:
    - Try converting object columns that look numeric
    - Parse columns with 'date' in their name to datetime
    Returns: cleaned df, list of columns converted
    """
    converted = []

    for col in df.columns:
        # Try numeric conversion
        if df[col].dtype == "object":
            try:
                df[col] = pd.to_numeric(df[col].str.replace(",", ""), errors="raise")
                converted.append(f"{col} → numeric")
                continue
            except Exception:
                pass

            # Try datetime conversion for date-like column names
            if any(kw in col.lower() for kw in ["date", "time", "day", "month", "year"]):
                try:
                    df[col] = pd.to_datetime(df[col], errors="raise")
                    converted.append(f"{col} → datetime")
                except Exception:
                    pass

    return df, converted


def handle_missing_values(df, strategy="auto"):
    """
    Fill missing values based on column data type.
    
    strategy options:
    - 'auto'   : numeric → mean, categorical → mode
    - 'median' : numeric → median, categorical → mode
    - 'drop'   : drop rows with any missing values

    Returns: cleaned df, dict of {column: action_taken}
    """
    fill_log = {}

    if strategy == "drop":
        before = len(df)
        df = df.dropna()
        fill_log["rows_dropped"] = before - len(df)
        return df, fill_log

    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count == 0:
            continue

        if df[col].dtype in ["float64", "int64", "float32", "int32"]:
            # Numeric: fill with mean or median
            if strategy == "median":
                fill_val = df[col].median()
                df[col] = df[col].fillna(fill_val)
                fill_log[col] = f"filled {null_count} nulls with median ({fill_val:.2f})"
            else:
                fill_val = df[col].mean()
                df[col] = df[col].fillna(fill_val)
                fill_log[col] = f"filled {null_count} nulls with mean ({fill_val:.2f})"
        else:
            # Categorical: fill with mode
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val[0])
                fill_log[col] = f"filled {null_count} nulls with mode ('{mode_val[0]}')"
            else:
                df[col] = df[col].fillna("Unknown")
                fill_log[col] = f"filled {null_count} nulls with 'Unknown'"

    return df, fill_log


def handle_outliers(df, method="iqr"):
    """
    Detect and cap outliers in numeric columns using IQR method.
    Values beyond 1.5×IQR from Q1/Q3 are capped (Winsorization).
    
    Returns: cleaned df, dict of {column: count_capped}
    """
    outlier_log = {}
    numeric_cols = df.select_dtypes(include=["number"]).columns

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        # Count outliers before capping
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()

        if n_outliers > 0:
            # Cap values instead of removing (safer)
            df[col] = df[col].clip(lower=lower, upper=upper)
            outlier_log[col] = f"{n_outliers} outliers capped [{lower:.2f} – {upper:.2f}]"

    return df, outlier_log


def clean_dataset(df, missing_strategy="auto", handle_outliers_flag=True):
    """
    Master cleaning function — runs all cleaning steps in order.
    
    Returns:
        df_clean   : fully cleaned DataFrame
        clean_log  : dict with detailed log of every action taken
    """
    clean_log = {}

    # Step 1: Fix column names
    df = fix_column_names(df)
    clean_log["column_names"] = "Standardized (lowercase, underscores)"

    # Step 2: Remove duplicates
    df, dupes_removed = remove_duplicates(df)
    clean_log["duplicates_removed"] = dupes_removed

    # Step 3: Fix data types
    df, type_conversions = fix_data_types(df)
    clean_log["type_conversions"] = type_conversions if type_conversions else ["None needed"]

    # Step 4: Handle missing values
    df, fill_log = handle_missing_values(df, strategy=missing_strategy)
    clean_log["missing_values"] = fill_log if fill_log else {"status": "No missing values found"}

    # Step 5: Handle outliers (optional)
    if handle_outliers_flag:
        df, outlier_log = handle_outliers(df)
        clean_log["outliers"] = outlier_log if outlier_log else {"status": "No significant outliers found"}

    return df, clean_log