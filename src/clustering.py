"""
Clustering and Profiling Module
===============================
Implements multi-K evaluation (Inertia & Silhouette Score), K-Means model training,
and comprehensive business profiling and segment labeling.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def evaluate_kmeans(
    rfm_scaled: np.ndarray,
    k_range: range = range(2, 9),
    random_state: int = 42,
    n_init: int = 10
) -> pd.DataFrame:
    """
    Evaluates K-Means clustering across a range of cluster counts (K).

    Parameters
    ----------
    rfm_scaled : np.ndarray
        Standardized feature matrix.
    k_range : range
        Range of K values to evaluate (default: 2 to 8).
    random_state : int
        Seed for reproducibility.
    n_init : int
        Number of initializations per K.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['K', 'Inertia', 'Silhouette Score']
    """
    records = []

    for k in k_range:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = model.fit_predict(rfm_scaled)
        inertia = model.inertia_
        sil = silhouette_score(rfm_scaled, labels)
        records.append({
            "K": k,
            "Inertia": round(inertia, 4),
            "Silhouette Score": round(sil, 6)
        })

    return pd.DataFrame(records)


def fit_kmeans(
    rfm_scaled: np.ndarray,
    n_clusters: int = 4,
    random_state: int = 42,
    n_init: int = 10
) -> tuple[KMeans, np.ndarray]:
    """
    Trains the final K-Means clustering model.

    Parameters
    ----------
    rfm_scaled : np.ndarray
        Standardized feature matrix.
    n_clusters : int
        Selected optimal number of clusters.
    random_state : int
        Seed for reproducibility.
    n_init : int
        Number of initializations.

    Returns
    -------
    tuple[KMeans, np.ndarray]
        (fitted_model, cluster_labels)
    """
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    labels = model.fit_predict(rfm_scaled)
    return model, labels


def profile_clusters(
    rfm_df: pd.DataFrame,
    labels: np.ndarray,
    segment_names: dict[int, str] = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Profiles customer clusters using original-scale RFM metrics, assigns meaningful
    business labels, and computes customer and revenue contributions.

    Parameters
    ----------
    rfm_df : pd.DataFrame
        Customer-level RFM DataFrame.
    labels : np.ndarray
        Cluster assignments.
    segment_names : dict[int, str], optional
        Mapping from cluster ID to business label.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (rfm_segmented, cluster_profile_df)
    """
    rfm_segmented = rfm_df.copy()
    rfm_segmented["Cluster"] = labels

    total_customers = len(rfm_segmented)
    total_revenue = rfm_segmented["Monetary"].sum()

    # Aggregate original-scale metrics
    profile = rfm_segmented.groupby("Cluster").agg(
        Customer_Count=("CustomerID", "count"),
        Average_Recency=("Recency", "mean"),
        Median_Recency=("Recency", "median"),
        Average_Frequency=("Frequency", "mean"),
        Median_Frequency=("Frequency", "median"),
        Average_Monetary=("Monetary", "mean"),
        Median_Monetary=("Monetary", "median"),
        Total_Revenue=("Monetary", "sum")
    ).reset_index()

    profile["Customer_Percentage"] = (profile["Customer_Count"] / total_customers) * 100
    profile["Revenue_Percentage"] = (profile["Total_Revenue"] / total_revenue) * 100

    # Default justified business labels if not provided
    if segment_names is None:
        segment_names = {
            0: "Champions",
            1: "Hibernating Customers",
            2: "New / Promising Customers",
            3: "Potential Loyalists"
        }

    profile["Segment"] = profile["Cluster"].map(segment_names)
    rfm_segmented["Segment"] = rfm_segmented["Cluster"].map(segment_names)

    # Reorder columns
    cols_order = [
        "Cluster", "Segment", "Customer_Count", "Customer_Percentage",
        "Total_Revenue", "Revenue_Percentage",
        "Average_Recency", "Median_Recency",
        "Average_Frequency", "Median_Frequency",
        "Average_Monetary", "Median_Monetary"
    ]
    profile = profile[cols_order].round(2)

    return rfm_segmented, profile
