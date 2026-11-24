# Customer Segmentation Analysis using RFM and K-Means Clustering

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Project%20Status-Complete%20%26%20Validated-brightgreen.svg)]()
[![Dataset](https://img.shields.io/badge/Dataset-UCI%20Online%20Retail-orange.svg)](https://archive.ics.uci.edu/dataset/352/online+retail)

A production-grade, end-to-end data science and machine learning project that performs comprehensive customer segmentation on over **541,000 real-world transaction records** from a UK-based online retailer. Leveraging **Recency, Frequency, and Monetary (RFM)** behavioral modeling and unsupervised **K-Means clustering**, this project segments customers into distinct, actionable behavioral cohorts to optimize marketing ROI, maximize retention, and accelerate revenue growth.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Business Problem](#business-problem)
- [Dataset Attribution & Information](#dataset-attribution--information)
- [Technology Stack](#technology-stack)
- [Project Architecture](#project-architecture)
- [Data Cleaning Methodology](#data-cleaning-methodology)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [RFM Feature Engineering](#rfm-feature-engineering)
- [Feature Preprocessing](#feature-preprocessing)
- [K-Means Clustering & Model Evaluation](#k-means-clustering--model-evaluation)
- [Customer Segment Profiles](#customer-segment-profiles)
- [Strategic Marketing Recommendations](#strategic-marketing-recommendations)
- [Key Visualizations](#key-visualizations)
- [Validation & Quality Assurance](#validation--quality-assurance)
- [Installation & Execution Guide](#installation--execution-guide)
- [Project Audit Summary](#project-audit-summary)
- [Author & License](#author--license)

---

## Project Overview

In competitive e-commerce environments, treat-all-customers-the-same marketing strategies yield suboptimal conversion rates and wasteful customer acquisition costs. By segmenting customers according to their demonstrated purchasing habits—how recently they transacted, how frequently they order, and how much monetary value they generate—e-commerce businesses can tailor communication, incentivize repeat purchases, and proactively mitigate churn.

This repository provides an enterprise-ready pipeline incorporating:
1. Multi-stage anomaly screening, auditing, and cleaning of raw transactional logs.
2. Mathematically grounded RFM metric computation at the customer level.
3. Variance-stabilizing logarithmic feature transformations and standardization.
4. Objective cluster count optimization via the Elbow Method and Silhouette Analysis ($K=2$ to $8$).
5. Meaningful business segment profiling with quantitative contribution metrics.
6. Modular, production-tested Python source code and automated validation suites.

---

## Business Problem

Online retailers face three primary commercial challenges:
1. **Revenue Concentration Risk:** Understanding what percentage of customers drive the bulk of top-line revenue (Pareto Principle).
2. **Customer Churn:** Identifying previously active buyers who have disengaged before they lapse permanently.
3. **Growth Potential:** Distinguishing promising newer buyers from casual one-time shoppers to deploy targeted upsell campaigns.

By clustering active buyers into data-driven behavioral personas, marketing and CRM teams can deliver personalized retention, reactivation, and VIP engagement campaigns that demonstrably improve Customer Lifetime Value (CLV).

---

## Dataset Attribution & Information

The analysis is conducted on the canonical **Online Retail** dataset from the UCI Machine Learning Repository:

- **Dataset Name:** Online Retail
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online+retail)
- **DOI:** 10.24432/C5BW33
- **Period:** December 1, 2010 to December 9, 2011
- **Domain:** UK-based online gifts and homeware retail
- **Currency:** British Pound Sterling (£ / GBP)
- **Original Dimensions:** 541,909 transaction records × 8 attributes

### Original Attributes
| Field | Data Type | Description |
| :--- | :--- | :--- |
| `InvoiceNo` | String / Object | 6-digit transaction identifier. Prefixed with `'C'` if cancelled |
| `StockCode` | String / Object | 5-digit unique product SKU code |
| `Description` | String / Object | Product description |
| `Quantity` | Integer | Quantity of SKU units ordered per transaction |
| `InvoiceDate` | Datetime | Timestamp of transaction |
| `UnitPrice` | Float | Unit price in Sterling (£) |
| `CustomerID` | Float / Int | 5-digit unique identifier for registered customers |
| `Country` | String / Object | Customer residence country |

---

## Technology Stack

- **Core Analytics & Data Manipulation:** `Python 3.10+`, `pandas`, `NumPy`, `SciPy`
- **Machine Learning & Preprocessing:** `scikit-learn` (`KMeans`, `StandardScaler`, `silhouette_score`, `PCA`)
- **Data Visualization:** `Matplotlib`, `Seaborn`
- **Execution & Development:** `Jupyter Notebook`, `openpyxl`
- **Version Control:** `Git`, `GitHub`

---

## Project Architecture

```text
Customer-Segmentation-Analysis/
├── data/
│   ├── raw/                              # Original Online Retail.xlsx (gitignored)
│   ├── processed/                        # Cached cleaned transaction files
│   └── README.md                         # Detailed dataset documentation & download guide
│
├── notebooks/
│   └── Customer_Segmentation_Analysis.ipynb  # Comprehensive, executed 20-phase notebook
│
├── outputs/
│   ├── charts/                           # 15 high-resolution (300 DPI) publication charts
│   │   ├── recency_distribution.png
│   │   ├── frequency_distribution.png
│   │   ├── monetary_distribution.png
│   │   ├── elbow_method.png
│   │   ├── silhouette_score_comparison.png
│   │   ├── customer_count_by_cluster.png
│   │   ├── revenue_by_cluster.png
│   │   ├── segment_revenue_contribution.png
│   │   ├── recency_vs_frequency.png
│   │   ├── frequency_vs_monetary.png
│   │   ├── cluster_comparison_heatmap.png
│   │   ├── pca_clusters_visualization.png
│   │   ├── monthly_revenue_trend.png
│   │   ├── country_revenue_distribution.png
│   │   └── top_10_products_revenue.png
│   ├── customer_segments.csv             # Customer RFM metrics with assigned business segments
│   ├── rfm_analysis.csv                  # Aggregated customer-level RFM metrics
│   ├── cleaning_report.csv               # Multi-stage cleaning audit log
│   ├── clustering_evaluation.csv         # Inertia & Silhouette scores for K=2..8
│   ├── cluster_profile.csv               # Mean/median metrics and financial contributions
│   ├── data_quality_report.csv           # 15-point data quality screening
│   ├── missing_values_report.csv         # Missingness counts and percentages
│   ├── project_summary.txt               # Executive textual summary
│   └── project_audit_report.txt          # Formal validation audit certificate
│
├── src/                                  # Modular, reusable Python package
│   ├── __init__.py                       # Package initialization
│   ├── data_cleaning.py                  # Auditing, cleaning, and validation functions
│   ├── rfm_analysis.py                   # RFM aggregation, log transform, and scaling
│   ├── clustering.py                     # Multi-K evaluation, K-Means fitting, and profiling
│   └── visualization.py                  # Publication-ready plotting routines
│
├── run_analysis.py                       # Automated end-to-end pipeline execution script
├── README.md                             # Comprehensive project documentation
├── requirements.txt                      # Project dependency specification
└── .gitignore                            # Comprehensive ignore rules
```

---

## Data Cleaning Methodology

Raw transactional records contain returns, cancellations, missing customer identifiers, and administrative adjustments. Every exclusion step was empirically investigated and recorded:

| Cleaning Step | Records Before | Records After | Records Removed | Reason for Exclusion |
| :--- | :---: | :---: | :---: | :--- |
| **1. Remove exact duplicates** | 541,909 | 536,641 | 5,268 | Identical multi-entry duplicate transaction rows |
| **2. Remove missing CustomerID** | 536,641 | 401,604 | 135,037 | Guest checkouts; cannot be attributed to customer cohorts |
| **3. Remove cancellations ('C')** | 401,604 | 392,732 | 8,872 | Cancelled or refunded orders excluded from purchase RFM |
| **4. Remove non-positive quantities** | 392,732 | 392,732 | 0 | Returns/inventory adjustments (covered by cancellations) |
| **5. Remove non-positive unit prices** | 392,732 | 392,692 | 40 | Free promotional samples or internal adjustments |
| **6. Remove invalid dates** | 392,692 | 392,692 | 0 | All remaining timestamps verified as valid datetimes |

**Result:** A verified purchase transaction dataset of **392,692 records** across **4,338 unique customers** and **18,532 distinct purchase orders**.

---

## Exploratory Data Analysis (EDA)

Key commercial benchmarks calculated from cleaned purchase records:
- **Total Valid Revenue:** £8,887,208.89
- **Total Purchase Invoices:** 18,532
- **Unique Customers:** 4,338
- **Average Order Value (AOV):** £479.56
- **Top Country:** United Kingdom (£7,285,024.64, representing 82.0% of total revenue), followed by Netherlands, EIRE, Germany, and France.
- **Top Product by Revenue:** *REGENCY CAKESTAND 3 TIER* (£142,592.95) and *WHITE HANGING HEART T-LIGHT HOLDER* (£100,352.15).

---

## RFM Feature Engineering

Customer transactions were aggregated to calculate core RFM metrics:
- **Recency ($R$):** Days elapsed from the customer's most recent purchase date to the reference snapshot date (`2011-12-10`, one day after the maximum observed transaction date).
- **Frequency ($F$):** Total count of unique purchase invoice numbers (`InvoiceNo.nunique()`). Product line items within the same invoice are not counted as multiple orders.
- **Monetary ($M$):** Total net transaction spend ($\sum \text{Quantity} \times \text{UnitPrice}$) in GBP.

---

## Feature Preprocessing

Raw RFM distributions exhibit severe right-skewness and extreme outliers:
- **Skewness Before Transformation:** Recency = `1.25`, Frequency = `18.00`, Monetary = `19.39`.
- **Log Transformation:** Applied $y = \ln(1 + x)$ via `np.log1p()` to stabilize variance across all three dimensions.
- **Feature Scaling:** Applied `StandardScaler` to the log-transformed space ($\mu = 0, \sigma = 1$). `CustomerID` is preserved separately and never passed to the clustering algorithm.

---

## K-Means Clustering & Model Evaluation

K-Means models were evaluated across $K \in [2, 8]$ using both the **Elbow Method (Inertia)** and the **Silhouette Score** with `random_state=42` and `n_init=10`:

| Clusters ($K$) | Inertia (WCSS) | Silhouette Score | Evaluation & Decision |
| :---: | :---: | :---: | :--- |
| **2** | 6,483.59 | 0.4328 | High silhouette but oversimplified; clusters only active vs inactive |
| **3** | 4,869.49 | 0.3365 | Moderate separation; conflates recent one-time buyers with loyalists |
| **4** | **3,939.05** | **0.3375** | **Selected Optimal Model:** Distinct elbow curvature, highest multi-cluster silhouette score, and superior business actionability |
| **5** | 3,296.71 | 0.3162 | Diminishing silhouette; introduces redundant split |
| **6** | 2,855.76 | 0.3124 | Sub-segmentation creates small, hard-to-target clusters |
| **7** | 2,548.82 | 0.3092 | Declining silhouette |
| **8** | 2,336.34 | 0.3033 | Lowest silhouette |

---

## Customer Segment Profiles

Evaluating the final 4-cluster model on **original-scale** RFM metrics reveals distinct behavioral personas:

| Cluster | Segment Persona | Customers | % Customers | Total Revenue (£) | % Revenue | Mean Recency | Median Recency | Mean Frequency | Median Frequency | Mean Spend | Median Spend |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | **Champions** | 713 | 16.44% | £5,766,757.07 | **64.89%** | 12.2 days | 8.0 days | 13.8 orders | 10.0 orders | £8,088.02 | £3,722.82 |
| **1** | **Hibernating Customers** | 1,622 | **37.39%** | £553,099.77 | 6.22% | 181.5 days | 174.5 days | 1.3 orders | 1.0 orders | £341.00 | £296.09 |
| **2** | **New / Promising Customers** | 837 | 19.29% | £466,479.03 | 5.25% | 17.7 days | 17.0 days | 2.2 orders | 2.0 orders | £557.32 | £480.31 |
| **3** | **Potential Loyalists** | 1,166 | 26.88% | £2,100,873.02 | **23.64%** | 71.6 days | 57.0 days | 4.1 orders | 4.0 orders | £1,801.78 | £1,345.20 |

---

## Strategic Marketing Recommendations

```
+-------------------------------------------------------------------------------------------------+
|                                 SEGMENT ACTION MATRIX                                           |
+---------------------------+---------------------------------------+-----------------------------+
| Segment                   | Behavioral Traits                     | Recommended CRM Strategy    |
+---------------------------+---------------------------------------+-----------------------------+
| Champions                 | Transacted ~12 days ago; ~14 orders;   | VIP concierge programs,     |
| (16.4% users, 64.9% rev)  | Average spend > £8,000.               | exclusive product previews, |
|                           | Critical core accounts.               | referral advocate incentives|
+---------------------------+---------------------------------------+-----------------------------+
| Potential Loyalists       | Transacted ~71 days ago; ~4 orders;   | Targeted cross-sell bundles,|
| (26.9% users, 23.6% rev)  | Substantial spend (~£1,800).          | threshold-based volume      |
|                           | Prime conversion targets.             | discounts to push into VIP  |
+---------------------------+---------------------------------------+-----------------------------+
| New / Promising Customers | Highly recent (~18 days); 2 orders;   | Automated welcome drip,     |
| (19.3% users, 5.2% rev)   | Low-to-moderate initial spend.        | 2nd-order discount coupon,  |
|                           | Responsive onboarding window.         | curated onboarding reviews  |
+---------------------------+---------------------------------------+-----------------------------+
| Hibernating Customers     | Lapsed > 6 months (~182 days);        | Automated win-back series,  |
| (37.4% users, 6.2% rev)   | Single order history (~1.3 orders).   | reactivation surveys,       |
|                           | At risk of permanent churn.           | aggressive clearance deals  |
+---------------------------+---------------------------------------+-----------------------------+
```

---

## Key Visualizations

All figures are rendered at 300 DPI and stored in `outputs/charts/`:

| Visualization | Description | Preview |
| :--- | :--- | :--- |
| **Elbow & Silhouette Analysis** | WCSS inertia elbow inflection and silhouette score evaluation across $K=2..8$ | `outputs/charts/elbow_method.png` |
| **Segment Revenue Share** | Donut chart detailing monetary contribution per segment | `outputs/charts/segment_revenue_contribution.png` |
| **Customer Count by Segment** | Bar chart showing customer volume across each cluster | `outputs/charts/customer_count_by_cluster.png` |
| **Recency vs. Frequency** | Scatter plot showing behavioral separation of cohorts | `outputs/charts/recency_vs_frequency.png` |
| **Frequency vs. Monetary** | Log-scaled scatter plot showing value concentration | `outputs/charts/frequency_vs_monetary.png` |
| **Cluster RFM Heatmap** | Min-max scaled intensity matrix comparing segment averages | `outputs/charts/cluster_comparison_heatmap.png` |
| **2D PCA Cluster Map** | Dimensionality reduction capturing 93.9% of RFM variance | `outputs/charts/pca_clusters_visualization.png` |
| **Monthly Revenue Trend** | Longitudinal monthly revenue trend over 12 months | `outputs/charts/monthly_revenue_trend.png` |

---

## Validation & Quality Assurance

Automated assertion checks are executed continuously:
- **Raw Row Count Assertion:** 541,909 rows verified.
- **Null CustomerID Assertion:** 0 null customer records in cleaned data (`PASS`).
- **Cancellation Check:** 0 cancellation invoices in cleaned data (`PASS`).
- **Positive Bounds Check:** All quantities and unit prices $> 0$ (`PASS`).
- **Revenue Reconciliation Check:** Total cleaned revenue (£8,887,208.89) equals RFM monetary sum with £0.00 difference (`PASS`).
- **Cluster Coverage Check:** 100% of the 4,338 customers are assigned to valid business segments (`PASS`).

---

## Installation & Execution Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Kathir18M/Customer-Segmentation-Analysis.git
cd Customer-Segmentation-Analysis
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Obtain Dataset
Place `Online Retail.xlsx` in `data/raw/`. If absent, run the automated pipeline runner or download from [UCI Machine Learning Repository](https://archive.ics.uci.edu/static/public/352/online+retail.zip).

### 4. Execute Analysis Pipeline
```bash
python run_analysis.py
```

### 5. Launch Interactive Notebook
```bash
jupyter notebook notebooks/Customer_Segmentation_Analysis.ipynb
```

---

## Project Audit Summary

- **Repository Structure:** Fully verified and compliant with industry standards.
- **Automated Audit Status:** **10 / 10 CHECKS PASSED** (`outputs/project_audit_report.txt`).
- **Data Integrity:** 100% verified against original UCI benchmarks.
- **Code Modularity:** Reusable, well-typed functions in `src/` with clear docstrings and error handling.

---

## Author & License

- **Author:** [Kathiravan M](https://github.com/Kathir18M)
- **GitHub Repository:** [Customer-Segmentation-Analysis](https://github.com/Kathir18M/Customer-Segmentation-Analysis)
- **License:** Open for academic and professional research.
