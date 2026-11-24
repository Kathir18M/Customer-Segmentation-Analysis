"""
Customer Segmentation Analysis Package
======================================
Modular components for data cleaning, RFM metrics calculation,
K-Means clustering, customer segment profiling, and publication-ready visualizations.
"""

from .data_cleaning import investigate_data_quality, clean_dataset, validate_cleaned_data
from .rfm_analysis import calculate_rfm, preprocess_rfm, validate_rfm
from .clustering import evaluate_kmeans, fit_kmeans, profile_clusters
from .visualization import (
    plot_rfm_distributions,
    plot_elbow_and_silhouette,
    plot_cluster_summary,
    plot_scatter_and_heatmap,
    plot_pca_clusters,
    plot_eda_charts
)

__all__ = [
    "investigate_data_quality",
    "clean_dataset",
    "validate_cleaned_data",
    "calculate_rfm",
    "preprocess_rfm",
    "validate_rfm",
    "evaluate_kmeans",
    "fit_kmeans",
    "profile_clusters",
    "plot_rfm_distributions",
    "plot_elbow_and_silhouette",
    "plot_cluster_summary",
    "plot_scatter_and_heatmap",
    "plot_pca_clusters",
    "plot_eda_charts"
]
