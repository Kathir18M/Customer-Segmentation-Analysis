"""
RFM Analysis and Preprocessing Module
=====================================
Calculates customer-level Recency, Frequency, and Monetary (RFM) metrics,
evaluates skewness, applies variance-stabilizing log transforms, and scales features.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def calculate_rfm(df_clean: pd.DataFrame, snapshot_date: pd.Timestamp = None) -> pd.DataFrame:
    """
    Aggregates cleaned transactions into customer-level RFM metrics.

    Parameters
    ----------
    df_clean : pd.DataFrame
        Cleaned transactions DataFrame.
    snapshot_date : pd.Timestamp, optional
        Fixed reference date for recency calculation.
        Defaults to max(InvoiceDate) + 1 day.

    Returns
    -------
    pd.DataFrame
        Customer-level RFM metrics (CustomerID, Recency, Frequency, Monetary).
    """
    if snapshot_date is None:
        snapshot_date = df_clean["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df_clean.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum")
    ).reset_index()

    # Ensure consistent typing
    rfm["CustomerID"] = rfm["CustomerID"].astype(int)
    rfm["Recency"] = rfm["Recency"].astype(int)
    rfm["Frequency"] = rfm["Frequency"].astype(int)
    rfm["Monetary"] = rfm["Monetary"].round(2)

    return rfm


def preprocess_rfm(rfm_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, StandardScaler]:
    """
    Prepares RFM features for distance-based clustering algorithms.
    Applies np.log1p() to mitigate severe positive skewness, then applies StandardScaler.
    Preserves CustomerID separately.

    Parameters
    ----------
    rfm_df : pd.DataFrame
        Customer-level RFM DataFrame.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray, StandardScaler]
        (rfm_log_transformed, rfm_scaled_matrix, fitted_scaler)
    """
    feature_cols = ["Recency", "Frequency", "Monetary"]
    
    # 1. Log transformation
    rfm_log = np.log1p(rfm_df[feature_cols].copy())

    # 2. Standardization
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)

    return rfm_log, rfm_scaled, scaler


def validate_rfm(rfm_df: pd.DataFrame, df_clean: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """
    Validates RFM calculations against transaction records.

    Parameters
    ----------
    rfm_df : pd.DataFrame
        Aggregated RFM DataFrame.
    df_clean : pd.DataFrame
        Cleaned transactions DataFrame.

    Returns
    -------
    dict[str, tuple[str, str]]
        Validation check name -> (status: PASS/FAIL, message)
    """
    results = {}

    # Check 1: Unique CustomerID count matches
    unique_cust_clean = df_clean["CustomerID"].nunique()
    cust_rfm_count = len(rfm_df)
    results["Unique Customer Count Match"] = (
        "PASS" if unique_cust_clean == cust_rfm_count else "FAIL",
        f"Cleaned unique customers: {unique_cust_clean}, RFM rows: {cust_rfm_count}"
    )

    # Check 2: No duplicate CustomerID
    dup_cust = rfm_df["CustomerID"].duplicated().sum()
    results["No Duplicate CustomerID Check"] = (
        "PASS" if dup_cust == 0 else "FAIL",
        f"{dup_cust} duplicate CustomerIDs in RFM table"
    )

    # Check 3: Monetary sum matches total revenue
    total_rev_clean = round(df_clean["Revenue"].sum(), 2)
    total_monetary_rfm = round(rfm_df["Monetary"].sum(), 2)
    diff = abs(total_rev_clean - total_monetary_rfm)
    results["Monetary Revenue Sum Check"] = (
        "PASS" if diff < 1.0 else "FAIL",
        f"Total transaction revenue: £{total_rev_clean:,.2f}, Total RFM monetary: £{total_monetary_rfm:,.2f} (diff: £{diff:.2f})"
    )

    # Check 4: Positive Recency, Frequency, Monetary
    neg_r = (rfm_df["Recency"] < 0).sum()
    neg_f = (rfm_df["Frequency"] <= 0).sum()
    neg_m = (rfm_df["Monetary"] <= 0).sum()
    results["Valid RFM Bounds Check"] = (
        "PASS" if (neg_r == 0 and neg_f == 0 and neg_m == 0) else "FAIL",
        f"Non-positive values found: Recency={neg_r}, Frequency={neg_f}, Monetary={neg_m}"
    )

    return results
