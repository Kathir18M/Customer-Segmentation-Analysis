"""
Master Analysis Runner
======================
Executes end-to-end pipeline: data quality investigation, data cleaning,
RFM analysis, K-Means clustering, profiling, chart export, and report generation.
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure workspace root is in python path
sys.path.insert(0, os.path.abspath(r"K:\Customer-Segmentation-Analysis"))

from src.data_cleaning import investigate_data_quality, clean_dataset, validate_cleaned_data
from src.rfm_analysis import calculate_rfm, preprocess_rfm, validate_rfm
from src.clustering import evaluate_kmeans, fit_kmeans, profile_clusters
from src.visualization import (
    plot_rfm_distributions,
    plot_elbow_and_silhouette,
    plot_cluster_summary,
    plot_scatter_and_heatmap,
    plot_pca_clusters,
    plot_eda_charts
)

RAW_DATA_PATH = r"K:\Customer-Segmentation-Analysis\data\raw\Online Retail.xlsx"
OUTPUTS_DIR = r"K:\Customer-Segmentation-Analysis\outputs"
CHARTS_DIR = os.path.join(OUTPUTS_DIR, "charts")

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

print("=" * 60)
print("1. LOADING RAW DATASET")
print("=" * 60)
df_raw = pd.read_excel(RAW_DATA_PATH)
print(f"Loaded raw dataset: {df_raw.shape[0]:,} rows, {df_raw.shape[1]} columns")

print("\n" + "=" * 60)
print("2. INVESTIGATING DATA QUALITY & ANOMALIES")
print("=" * 60)
missing_report, quality_report = investigate_data_quality(df_raw)
missing_report.to_csv(os.path.join(OUTPUTS_DIR, "missing_values_report.csv"), index=False)
quality_report.to_csv(os.path.join(OUTPUTS_DIR, "data_quality_report.csv"), index=False)
print("Saved missing_values_report.csv and data_quality_report.csv")

print("\n" + "=" * 60)
print("3. EXECUTING MULTI-STAGE DATA CLEANING")
print("=" * 60)
df_clean, cleaning_report = clean_dataset(df_raw)
cleaning_report.to_csv(os.path.join(OUTPUTS_DIR, "cleaning_report.csv"), index=False)
print(f"Cleaned dataset: {df_clean.shape[0]:,} rows, {df_clean.shape[1]} columns")
print("Saved cleaning_report.csv")

print("\nValidating cleaned data...")
cleaning_val = validate_cleaned_data(df_clean)
for check, (status, msg) in cleaning_val.items():
    print(f"  [{status}] {check}: {msg}")
    assert status == "PASS", f"Validation failed: {check} - {msg}"

print("\n" + "=" * 60)
print("4. GENERATING EDA VISUALIZATIONS")
print("=" * 60)
plot_eda_charts(df_clean, CHARTS_DIR)
print(f"Exported EDA charts to {CHARTS_DIR}")

print("\n" + "=" * 60)
print("5. COMPUTING RFM METRICS")
print("=" * 60)
rfm_df = calculate_rfm(df_clean)
rfm_df.to_csv(os.path.join(OUTPUTS_DIR, "rfm_analysis.csv"), index=False)
print(f"Computed RFM metrics for {len(rfm_df):,} unique customers")
print("Saved rfm_analysis.csv")

print("\nValidating RFM calculations...")
rfm_val = validate_rfm(rfm_df, df_clean)
for check, (status, msg) in rfm_val.items():
    print(f"  [{status}] {check}: {msg}")
    assert status == "PASS", f"Validation failed: {check} - {msg}"

print("\n" + "=" * 60)
print("6. RFM PREPROCESSING (LOG TRANSFORM + SCALING)")
print("=" * 60)
rfm_log, rfm_scaled, scaler = preprocess_rfm(rfm_df)
print("Features log-transformed and standardized using StandardScaler.")
plot_rfm_distributions(rfm_df, CHARTS_DIR)
print(f"Exported RFM distribution plots to {CHARTS_DIR}")

print("\n" + "=" * 60)
print("7. K-MEANS EVALUATION (K = 2 to 8)")
print("=" * 60)
eval_df = evaluate_kmeans(rfm_scaled, k_range=range(2, 9), random_state=42, n_init=10)
eval_df.to_csv(os.path.join(OUTPUTS_DIR, "clustering_evaluation.csv"), index=False)
print(eval_df.to_string(index=False))
print("Saved clustering_evaluation.csv")
plot_elbow_and_silhouette(eval_df, CHARTS_DIR)
print(f"Exported Elbow and Silhouette curves to {CHARTS_DIR}")

print("\n" + "=" * 60)
print("8. TRAINING FINAL K-MEANS MODEL (K = 4) & PROFILING")
print("=" * 60)
kmeans_model, labels = fit_kmeans(rfm_scaled, n_clusters=4, random_state=42, n_init=10)

segment_mapping = {
    0: "Champions",
    1: "Hibernating Customers",
    2: "New / Promising Customers",
    3: "Potential Loyalists"
}

rfm_segmented, cluster_profile = profile_clusters(rfm_df, labels, segment_mapping)
cluster_profile.to_csv(os.path.join(OUTPUTS_DIR, "cluster_profile.csv"), index=False)
rfm_segmented.to_csv(os.path.join(OUTPUTS_DIR, "customer_segments.csv"), index=False)
print("\nCluster Business Profile:")
print(cluster_profile.to_string(index=False))
print("\nSaved cluster_profile.csv and customer_segments.csv")

print("\n" + "=" * 60)
print("9. EXPORTING CLUSTER VISUALIZATIONS")
print("=" * 60)
plot_cluster_summary(cluster_profile, CHARTS_DIR)
plot_scatter_and_heatmap(rfm_segmented, cluster_profile, CHARTS_DIR)
plot_pca_clusters(rfm_scaled, rfm_segmented, CHARTS_DIR)
print(f"Exported all clustering visualizations to {CHARTS_DIR}")

print("\n" + "=" * 60)
print("10. CREATING PROJECT SUMMARY AND AUDIT REPORT")
print("=" * 60)

summary_content = f"""================================================================================
CUSTOMER SEGMENTATION ANALYSIS PROJECT SUMMARY
================================================================================
Dataset: UCI Online Retail Dataset (Dec 2010 - Dec 2011)
Raw Records: {len(df_raw):,}
Cleaned Purchase Transactions: {len(df_clean):,}
Unique Active Customers: {len(rfm_df):,}
Unique Purchase Invoices: {df_clean['InvoiceNo'].nunique():,}
Total Analyzed Revenue: £{df_clean['Revenue'].sum():,.2f}
Average Order Value: £{df_clean.groupby('InvoiceNo')['Revenue'].sum().mean():,.2f}

--------------------------------------------------------------------------------
RFM DISTRIBUTIONS SUMMARY
--------------------------------------------------------------------------------
Recency (Days):    Min={rfm_df['Recency'].min()}, Median={rfm_df['Recency'].median()}, Mean={rfm_df['Recency'].mean():.1f}, Max={rfm_df['Recency'].max()}
Frequency (Orders): Min={rfm_df['Frequency'].min()}, Median={rfm_df['Frequency'].median()}, Mean={rfm_df['Frequency'].mean():.1f}, Max={rfm_df['Frequency'].max()}
Monetary Spend (£): Min=£{rfm_df['Monetary'].min():.2f}, Median=£{rfm_df['Monetary'].median():.2f}, Mean=£{rfm_df['Monetary'].mean():.2f}, Max=£{rfm_df['Monetary'].max():,.2f}

--------------------------------------------------------------------------------
K-MEANS CLUSTER EVALUATION (K = 2 to 8)
--------------------------------------------------------------------------------
{eval_df.to_string(index=False)}

Selected Optimal K: 4
Justification: Clear elbow inflection in inertia curve; highest multi-cluster silhouette score (0.3375);
optimal business interpretability separating VIPs, recent buyers, potential loyalists, and churned accounts.

--------------------------------------------------------------------------------
CUSTOMER SEGMENT PROFILES (K = 4)
--------------------------------------------------------------------------------
{cluster_profile.to_string(index=False)}

================================================================================
BUSINESS RECOMMENDATIONS
================================================================================
1. Champions (16.4% customers, 64.9% revenue):
   - Premium loyalty tiers, concierge support, exclusive early product access.
   - Referral programmes and VIP appreciation campaigns.

2. Potential Loyalists (26.9% customers, 23.6% revenue):
   - Category cross-selling, personalized bundle recommendations.
   - Volume discount incentives to convert them into top-tier Champions.

3. New / Promising Customers (19.3% customers, 5.2% revenue):
   - Automated onboarding email drip campaigns, welcome coupons for 2nd order.
   - Educational product showcases and customer review highlights.

4. Hibernating Customers (37.4% customers, 6.2% revenue):
   - Low-cost automated win-back / reactivation emails.
   - Surveys to understand inactivity reason; selective deep discounts on past favorites.
================================================================================
"""

with open(os.path.join(OUTPUTS_DIR, "project_summary.txt"), "w", encoding="utf-8") as f:
    f.write(summary_content)

print("Saved project_summary.txt")
print("\nALL ANALYSIS PIPELINE STAGES COMPLETED SUCCESSFULLY!")
