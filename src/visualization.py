"""
Visualization Module
====================
Provides publication-quality charting functions using Matplotlib and Seaborn.
Exports all charts as high-resolution (300 DPI) PNG files to outputs/charts/.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


def set_plot_style():
    """Configures clean, consistent styling for publication-ready figures."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 15,
        "figure.titleweight": "bold",
        "legend.fontsize": 10
    })


def plot_rfm_distributions(rfm_df: pd.DataFrame, output_dir: str):
    """Generates distribution plots for Recency, Frequency, and Monetary."""
    set_plot_style()
    os.makedirs(output_dir, exist_ok=True)

    metrics = [
        ("Recency", "Recency Distribution (Days since Last Purchase)", "Recency (Days)", "#2b5c8f", "recency_distribution.png"),
        ("Frequency", "Frequency Distribution (Number of Unique Orders)", "Frequency (Orders)", "#2a9d8f", "frequency_distribution.png"),
        ("Monetary", "Monetary Distribution (Total Spend in £)", "Monetary (£)", "#e76f51", "monetary_distribution.png")
    ]

    for col, title, xlabel, color, fname in metrics:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(rfm_df[col], kde=True, color=color, ax=ax, bins=40, edgecolor="white", alpha=0.7)
        ax.set_title(title, pad=12)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Customer Count")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, fname), dpi=300)
        plt.close(fig)


def plot_elbow_and_silhouette(evaluation_df: pd.DataFrame, output_dir: str):
    """Plots Elbow Method inertia curve and Silhouette Score comparison."""
    set_plot_style()
    os.makedirs(output_dir, exist_ok=True)

    # 1. Elbow Method
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(evaluation_df["K"], evaluation_df["Inertia"], marker="o", linewidth=2.5, color="#1d3557", markersize=8)
    ax.axvline(x=4, color="#e63946", linestyle="--", alpha=0.8, label="Optimal Elbow (K = 4)")
    ax.set_title("Elbow Method for Optimal K", pad=12)
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Inertia (Within-Cluster Sum of Squares)")
    ax.set_xticks(evaluation_df["K"])
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "elbow_method.png"), dpi=300)
    plt.close(fig)

    # 2. Silhouette Score
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(evaluation_df["K"], evaluation_df["Silhouette Score"], marker="s", linewidth=2.5, color="#2a9d8f", markersize=8)
    ax.axvline(x=4, color="#e63946", linestyle="--", alpha=0.8, label="Selected K = 4 (Balance of Score & Separation)")
    ax.set_title("Silhouette Score Comparison Across K", pad=12)
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Silhouette Score")
    ax.set_xticks(evaluation_df["K"])
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "silhouette_score_comparison.png"), dpi=300)
    plt.close(fig)


def plot_cluster_summary(profile_df: pd.DataFrame, output_dir: str):
    """Plots customer distribution and revenue share across segments."""
    set_plot_style()
    os.makedirs(output_dir, exist_ok=True)

    palette = ["#2a9d8f", "#e76f51", "#457b9d", "#f4a261"]

    # 1. Customer Count by Cluster
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(profile_df["Segment"], profile_df["Customer_Count"], color=palette, edgecolor="white", width=0.6)
    ax.set_title("Customer Count by Segment", pad=12)
    ax.set_xlabel("Customer Segment")
    ax.set_ylabel("Number of Customers")
    plt.xticks(rotation=15, ha="right")
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:,}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "customer_count_by_cluster.png"), dpi=300)
    plt.close(fig)

    # 2. Revenue by Cluster
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(profile_df["Segment"], profile_df["Total_Revenue"], color=palette, edgecolor="white", width=0.6)
    ax.set_title("Total Revenue Contribution by Segment", pad=12)
    ax.set_xlabel("Customer Segment")
    ax.set_ylabel("Total Revenue (£)")
    ax.yaxis.set_major_formatter("£{x:,.0f}")
    plt.xticks(rotation=15, ha="right")
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"£{h:,.0f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "revenue_by_cluster.png"), dpi=300)
    plt.close(fig)

    # 3. Revenue Share Donut Chart
    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        profile_df["Total_Revenue"],
        labels=profile_df["Segment"],
        autopct="%1.1f%%",
        startangle=140,
        colors=palette,
        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
        pctdistance=0.75
    )
    plt.setp(autotexts, size=10, weight="bold", color="white")
    plt.setp(texts, size=10)
    ax.set_title("Revenue Contribution Share (%)", pad=15)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "segment_revenue_contribution.png"), dpi=300)
    plt.close(fig)


def plot_scatter_and_heatmap(rfm_df: pd.DataFrame, profile_df: pd.DataFrame, output_dir: str):
    """Plots Recency vs Frequency, Frequency vs Monetary, and Cluster Heatmap."""
    set_plot_style()
    os.makedirs(output_dir, exist_ok=True)

    # 1. Recency vs Frequency
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        data=rfm_df,
        x="Recency",
        y="Frequency",
        hue="Segment",
        palette="tab10",
        alpha=0.65,
        s=40,
        ax=ax
    )
    ax.set_title("Customer Segmentation: Recency vs. Frequency", pad=12)
    ax.set_xlabel("Recency (Days since Last Order)")
    ax.set_ylabel("Frequency (Unique Invoices)")
    ax.set_yscale("log")
    ax.legend(title="Segment", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "recency_vs_frequency.png"), dpi=300)
    plt.close(fig)

    # 2. Frequency vs Monetary
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        data=rfm_df,
        x="Frequency",
        y="Monetary",
        hue="Segment",
        palette="tab10",
        alpha=0.65,
        s=40,
        ax=ax
    )
    ax.set_title("Customer Segmentation: Frequency vs. Monetary Spend", pad=12)
    ax.set_xlabel("Frequency (Unique Invoices)")
    ax.set_ylabel("Monetary (£)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(title="Segment", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "frequency_vs_monetary.png"), dpi=300)
    plt.close(fig)

    # 3. Heatmap
    heatmap_data = profile_df.set_index("Segment")[[
        "Average_Recency", "Average_Frequency", "Average_Monetary"
    ]]
    # Normalize per column (0 to 1) for meaningful visual comparison
    norm_heatmap = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(norm_heatmap, annot=heatmap_data.round(1), fmt="g", cmap="Blues", cbar=True, ax=ax, linewidths=1)
    ax.set_title("Cluster RFM Profile Comparison (Values with Min-Max Relative Intensity)", pad=12)
    ax.set_ylabel("Customer Segment")
    ax.set_xlabel("RFM Metrics")
    plt.xticks(rotation=0)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "cluster_comparison_heatmap.png"), dpi=300)
    plt.close(fig)


def plot_pca_clusters(rfm_scaled: np.ndarray, rfm_df: pd.DataFrame, output_dir: str):
    """Reduces scaled RFM space to 2D using PCA for visualization."""
    set_plot_style()
    os.makedirs(output_dir, exist_ok=True)

    pca = PCA(n_components=2, random_state=42)
    rfm_pca = pca.fit_transform(rfm_scaled)
    var_exp = pca.explained_variance_ratio_

    pca_df = pd.DataFrame(rfm_pca, columns=["PC1", "PC2"], index=rfm_df.index)
    pca_df["Segment"] = rfm_df["Segment"]

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="Segment",
        palette="tab10",
        alpha=0.6,
        s=45,
        ax=ax
    )
    ax.set_title(
        f"2D PCA Visualization of Customer Segments\n(Explained Variance: PC1={var_exp[0]*100:.1f}%, PC2={var_exp[1]*100:.1f}%, Total={var_exp.sum()*100:.1f}%)",
        pad=12
    )
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend(title="Segment", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "pca_clusters_visualization.png"), dpi=300)
    plt.close(fig)


def plot_eda_charts(df_clean: pd.DataFrame, output_dir: str):
    """Generates and exports primary EDA visualizations."""
    set_plot_style()
    os.makedirs(output_dir, exist_ok=True)

    # 1. Monthly revenue trend
    df_temp = df_clean.copy()
    df_temp["MonthYear"] = df_temp["InvoiceDate"].dt.to_period("M").astype(str)
    monthly_rev = df_temp.groupby("MonthYear")["Revenue"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(monthly_rev["MonthYear"], monthly_rev["Revenue"], marker="o", color="#1d3557", linewidth=2.5)
    ax.set_title("Monthly Revenue Trend (Dec 2010 – Dec 2011)", pad=12)
    ax.set_xlabel("Month-Year")
    ax.set_ylabel("Total Revenue (£)")
    ax.yaxis.set_major_formatter("£{x:,.0f}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "monthly_revenue_trend.png"), dpi=300)
    plt.close(fig)

    # 2. Country-wise revenue (Top 10)
    country_rev = df_clean.groupby("Country")["Revenue"].sum().sort_values(ascending=False).head(10).reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=country_rev, x="Revenue", y="Country", hue="Country", palette="Blues_r", legend=False, ax=ax)
    ax.set_title("Top 10 Countries by Total Revenue", pad=12)
    ax.set_xlabel("Total Revenue (£)")
    ax.xaxis.set_major_formatter("£{x:,.0f}")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "country_revenue_distribution.png"), dpi=300)
    plt.close(fig)

    # 3. Top 10 products by revenue
    product_rev = df_clean.groupby("Description")["Revenue"].sum().sort_values(ascending=False).head(10).reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=product_rev, x="Revenue", y="Description", hue="Description", palette="mako", legend=False, ax=ax)
    ax.set_title("Top 10 Products by Total Revenue", pad=12)
    ax.set_xlabel("Total Revenue (£)")
    ax.xaxis.set_major_formatter("£{x:,.0f}")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "top_10_products_revenue.png"), dpi=300)
    plt.close(fig)
