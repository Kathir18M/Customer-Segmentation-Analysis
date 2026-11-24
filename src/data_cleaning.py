"""
Data Cleaning and Data Quality Module
====================================
Provides robust data quality auditing, missing value detection, anomaly screening,
and structured multi-stage data cleaning for transaction records.
"""

import pandas as pd
import numpy as np


def investigate_data_quality(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs comprehensive data quality checks on the raw transaction dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Raw transactions DataFrame.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (missing_values_report, data_quality_report)
    """
    total_rows = len(df)
    
    # 1. Missing values report
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / total_rows) * 100
    missing_report = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": missing_counts.values,
        "Missing Percentage (%)": missing_pct.round(4).values
    }).sort_values(by="Missing Values", ascending=False).reset_index(drop=True)

    # 2. Detailed anomaly and quality checks
    exact_duplicates = df.duplicated().sum()
    duplicate_invoices = df["InvoiceNo"].duplicated().sum()
    missing_customer = df["CustomerID"].isnull().sum()
    missing_desc = df["Description"].isnull().sum()
    
    cancellation_invoices = df["InvoiceNo"].astype(str).str.startswith("C", na=False).sum()
    neg_quantity = (df["Quantity"] < 0).sum()
    zero_quantity = (df["Quantity"] == 0).sum()
    neg_price = (df["UnitPrice"] < 0).sum()
    zero_price = (df["UnitPrice"] == 0).sum()
    
    # Check invalid dates
    invalid_dates = pd.to_datetime(df["InvoiceDate"], errors="coerce").isnull().sum()
    
    quality_checks = [
        {"Investigation Item": "Total Original Records", "Count / Value": total_rows, "Percentage (%)": 100.0, "Description": "Total raw transaction rows"},
        {"Investigation Item": "Total Columns", "Count / Value": df.shape[1], "Percentage (%)": "-", "Description": "Original column count"},
        {"Investigation Item": "Exact Duplicate Rows", "Count / Value": exact_duplicates, "Percentage (%)": round((exact_duplicates / total_rows) * 100, 4), "Description": "Identical row duplicates"},
        {"Investigation Item": "Duplicate Invoice Numbers", "Count / Value": duplicate_invoices, "Percentage (%)": round((duplicate_invoices / total_rows) * 100, 4), "Description": "Multi-item invoice entries"},
        {"Investigation Item": "Missing CustomerID", "Count / Value": missing_customer, "Percentage (%)": round((missing_customer / total_rows) * 100, 4), "Description": "Guest / anonymous transactions"},
        {"Investigation Item": "Missing Description", "Count / Value": missing_desc, "Percentage (%)": round((missing_desc / total_rows) * 100, 4), "Description": "Items without product description"},
        {"Investigation Item": "Cancellation Invoices ('C')", "Count / Value": cancellation_invoices, "Percentage (%)": round((cancellation_invoices / total_rows) * 100, 4), "Description": "Cancelled or refunded orders"},
        {"Investigation Item": "Negative Quantity", "Count / Value": neg_quantity, "Percentage (%)": round((neg_quantity / total_rows) * 100, 4), "Description": "Returns and stock adjustments"},
        {"Investigation Item": "Zero Quantity", "Count / Value": zero_quantity, "Percentage (%)": round((zero_quantity / total_rows) * 100, 4), "Description": "Orders with zero units"},
        {"Investigation Item": "Negative UnitPrice", "Count / Value": neg_price, "Percentage (%)": round((neg_price / total_rows) * 100, 4), "Description": "Accounting debt adjustments (e.g., bad debt)"},
        {"Investigation Item": "Zero UnitPrice", "Count / Value": zero_price, "Percentage (%)": round((zero_price / total_rows) * 100, 4), "Description": "Promotional or free sample items"},
        {"Investigation Item": "Invalid Transaction Dates", "Count / Value": invalid_dates, "Percentage (%)": 0.0, "Description": "Unparseable timestamp values"},
        {"Investigation Item": "Unique Customer IDs", "Count / Value": df["CustomerID"].nunique(dropna=True), "Percentage (%)": "-", "Description": "Unique registered customers in raw data"},
        {"Investigation Item": "Unique Invoices", "Count / Value": df["InvoiceNo"].nunique(), "Percentage (%)": "-", "Description": "Unique invoice numbers in raw data"},
        {"Investigation Item": "Unique Countries", "Count / Value": df["Country"].nunique(), "Percentage (%)": "-", "Description": "Distinct geographical markets"}
    ]
    
    quality_report = pd.DataFrame(quality_checks)
    return missing_report, quality_report


def clean_dataset(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cleans raw transaction records for customer-level purchase and RFM analysis.
    Preserves raw data integrity and logs each removal step.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw transactions DataFrame.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (df_clean, cleaning_report_df)
    """
    df_clean = df_raw.copy(deep=True)
    cleaning_steps = []

    def record_step(step_name: str, before: int, after: int, reason: str):
        cleaning_steps.append({
            "Cleaning Operation": step_name,
            "Records Before": before,
            "Records After": after,
            "Records Removed": before - after,
            "Reason for Exclusion": reason
        })

    # 1. Remove exact duplicate records
    b = len(df_clean)
    df_clean = df_clean.drop_duplicates().copy()
    record_step("Remove exact duplicates", b, len(df_clean), "Redundant repeated transactions recorded identically")

    # 2. Exclude records without CustomerID
    b = len(df_clean)
    df_clean = df_clean.dropna(subset=["CustomerID"]).copy()
    record_step("Remove missing CustomerID", b, len(df_clean), "Cannot attribute purchases to individual customers for RFM analysis")

    # 3. Exclude cancellation invoices
    b = len(df_clean)
    cancellation_mask = df_clean["InvoiceNo"].astype(str).str.startswith("C", na=False)
    df_clean = df_clean[~cancellation_mask].copy()
    record_step("Remove cancellation invoices", b, len(df_clean), "Cancelled orders (invoices starting with 'C') excluded from positive purchase RFM")

    # 4. Exclude non-positive quantities
    b = len(df_clean)
    df_clean = df_clean[df_clean["Quantity"] > 0].copy()
    record_step("Remove non-positive quantities", b, len(df_clean), "Non-positive purchase volumes represent returns or adjustments")

    # 5. Exclude non-positive unit prices
    b = len(df_clean)
    df_clean = df_clean[df_clean["UnitPrice"] > 0].copy()
    record_step("Remove non-positive unit prices", b, len(df_clean), "Free items, damaged goods, or accounting write-offs")

    # 6. Type conversions & date validation
    df_clean["InvoiceDate"] = pd.to_datetime(df_clean["InvoiceDate"], errors="coerce")
    df_clean["CustomerID"] = df_clean["CustomerID"].astype(int)
    
    b = len(df_clean)
    df_clean = df_clean.dropna(subset=["InvoiceDate"]).copy()
    record_step("Remove invalid transaction dates", b, len(df_clean), "Timestamps that could not be parsed to valid datetime")

    # 7. Line-level revenue calculation
    df_clean["Revenue"] = df_clean["Quantity"] * df_clean["UnitPrice"]

    cleaning_report_df = pd.DataFrame(cleaning_steps)
    return df_clean, cleaning_report_df


def validate_cleaned_data(df_clean: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """
    Automated assertion checks for the cleaned dataset.

    Parameters
    ----------
    df_clean : pd.DataFrame
        Cleaned transactions DataFrame.

    Returns
    -------
    dict[str, tuple[str, str]]
        Dictionary of check name -> (status: PASS/FAIL, message)
    """
    results = {}

    # Check 1: Missing CustomerID
    missing_cust = df_clean["CustomerID"].isnull().sum()
    results["Missing CustomerID Check"] = ("PASS" if missing_cust == 0 else "FAIL", f"{missing_cust} missing CustomerIDs found")

    # Check 2: Exact duplicates
    duplicates = df_clean.duplicated().sum()
    results["Duplicate Records Check"] = ("PASS" if duplicates == 0 else "FAIL", f"{duplicates} duplicate rows found")

    # Check 3: Cancellation invoices
    cancellations = df_clean["InvoiceNo"].astype(str).str.startswith("C", na=False).sum()
    results["Cancellation Invoices Check"] = ("PASS" if cancellations == 0 else "FAIL", f"{cancellations} cancellation invoices found")

    # Check 4: Positive quantities
    invalid_qty = (df_clean["Quantity"] <= 0).sum()
    results["Positive Quantity Check"] = ("PASS" if invalid_qty == 0 else "FAIL", f"{invalid_qty} non-positive quantities found")

    # Check 5: Positive unit prices
    invalid_price = (df_clean["UnitPrice"] <= 0).sum()
    results["Positive UnitPrice Check"] = ("PASS" if invalid_price == 0 else "FAIL", f"{invalid_price} non-positive unit prices found")

    # Check 6: Valid dates
    invalid_dates = df_clean["InvoiceDate"].isnull().sum()
    results["Valid InvoiceDate Check"] = ("PASS" if invalid_dates == 0 else "FAIL", f"{invalid_dates} invalid timestamps found")

    # Check 7: Revenue calculation accuracy
    computed_rev = df_clean["Quantity"] * df_clean["UnitPrice"]
    diff = np.abs(df_clean["Revenue"] - computed_rev).max()
    results["Line Revenue Accuracy Check"] = ("PASS" if diff < 1e-5 else "FAIL", f"Maximum discrepancy is {diff}")

    return results
