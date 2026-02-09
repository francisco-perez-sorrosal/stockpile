"""Characterization tests for stock_clusters.py clustering logic.

Tests use synthetic data to verify the clustering functions produce
expected output shapes and types.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add the scripts directory to the path so we can import stock_clusters
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from stock_clusters import (
    cluster_stocks,
    enrich_clusters_with_info,
    find_elbow,
    label_clusters,
)


@pytest.fixture
def synthetic_metrics() -> pd.DataFrame:
    """Create a synthetic DataFrame mimicking real metrics data.

    Three distinct groups:
    - High return, low volatility (cluster A)
    - Low return, high volatility (cluster B)
    - Moderate return, moderate volatility (cluster C)
    """
    np.random.seed(42)
    rows = []

    # Group A: high return, low volatility
    for i in range(5):
        rows.append({
            "Ticker": f"HIGH{i}",
            "Returns": 0.3 + np.random.normal(0, 0.02),
            "Volatility": 0.1 + np.random.normal(0, 0.01),
        })

    # Group B: low return, high volatility
    for i in range(5):
        rows.append({
            "Ticker": f"LOW{i}",
            "Returns": -0.1 + np.random.normal(0, 0.02),
            "Volatility": 0.5 + np.random.normal(0, 0.02),
        })

    # Group C: moderate return, moderate volatility
    for i in range(5):
        rows.append({
            "Ticker": f"MID{i}",
            "Returns": 0.1 + np.random.normal(0, 0.02),
            "Volatility": 0.25 + np.random.normal(0, 0.02),
        })

    return pd.DataFrame(rows)


@pytest.fixture
def small_metrics() -> pd.DataFrame:
    """Minimal DataFrame for edge case testing."""
    return pd.DataFrame({
        "Ticker": ["A", "B", "C", "D"],
        "Returns": [0.1, 0.2, -0.1, 0.3],
        "Volatility": [0.15, 0.3, 0.4, 0.1],
    })


class TestClusterStocks:
    """Verify cluster_stocks() applies K-means and returns correct structure."""

    def test_returns_dataframe(self, synthetic_metrics):
        result = cluster_stocks(synthetic_metrics, n_clusters=3)
        assert isinstance(result, pd.DataFrame)

    def test_has_cluster_column(self, synthetic_metrics):
        result = cluster_stocks(synthetic_metrics, n_clusters=3)
        assert "Cluster" in result.columns

    def test_preserves_original_columns(self, synthetic_metrics):
        result = cluster_stocks(synthetic_metrics, n_clusters=3)
        for col in ["Ticker", "Returns", "Volatility"]:
            assert col in result.columns

    def test_correct_number_of_clusters(self, synthetic_metrics):
        n_clusters = 3
        result = cluster_stocks(synthetic_metrics, n_clusters=n_clusters)
        unique_clusters = result["Cluster"].nunique()
        assert unique_clusters == n_clusters

    def test_preserves_row_count(self, synthetic_metrics):
        result = cluster_stocks(synthetic_metrics, n_clusters=3)
        assert len(result) == len(synthetic_metrics)

    def test_cluster_values_are_integers(self, synthetic_metrics):
        result = cluster_stocks(synthetic_metrics, n_clusters=3)
        # vq returns integer cluster indices
        assert all(isinstance(v, (int, np.integer)) for v in result["Cluster"])

    def test_does_not_modify_input(self, synthetic_metrics):
        original = synthetic_metrics.copy()
        cluster_stocks(synthetic_metrics, n_clusters=3)
        pd.testing.assert_frame_equal(synthetic_metrics, original)

    def test_two_clusters(self, small_metrics):
        result = cluster_stocks(small_metrics, n_clusters=2)
        assert result["Cluster"].nunique() == 2


class TestLabelClusters:
    """Verify label_clusters() produces descriptive labels for each cluster."""

    def test_returns_dict(self, synthetic_metrics):
        clustered = cluster_stocks(synthetic_metrics, n_clusters=3)
        labels = label_clusters(clustered)
        assert isinstance(labels, dict)

    def test_one_label_per_cluster(self, synthetic_metrics):
        clustered = cluster_stocks(synthetic_metrics, n_clusters=3)
        labels = label_clusters(clustered)
        assert len(labels) == clustered["Cluster"].nunique()

    def test_labels_are_strings(self, synthetic_metrics):
        clustered = cluster_stocks(synthetic_metrics, n_clusters=3)
        labels = label_clusters(clustered)
        for label in labels.values():
            assert isinstance(label, str)

    def test_label_format_contains_return_and_vol(self, synthetic_metrics):
        """Labels follow the pattern '{ReturnLevel} Return, {VolLevel} Vol'."""
        clustered = cluster_stocks(synthetic_metrics, n_clusters=3)
        labels = label_clusters(clustered)
        for label in labels.values():
            assert "Return" in label
            assert "Vol" in label

    def test_label_return_categories(self):
        """Verify the four return categories based on median comparison."""
        # Strong: avg_return >= median * 1.5
        # Moderate: avg_return >= median
        # Low: avg_return >= 0
        # Negative: avg_return < 0
        df = pd.DataFrame({
            "Ticker": ["A", "B"],
            "Returns": [0.5, -0.1],
            "Volatility": [0.2, 0.3],
            "Cluster": [0, 1],
        })
        labels = label_clusters(df)
        # Median return = (0.5 + -0.1) / 2 = 0.2
        # Cluster 0: avg=0.5, >= 0.2*1.5=0.3 -> Strong
        # Cluster 1: avg=-0.1, < 0 -> Negative
        assert "Strong" in labels[0]
        assert "Negative" in labels[1]

    def test_label_volatility_categories(self):
        """Verify the three volatility categories based on median comparison."""
        df = pd.DataFrame({
            "Ticker": ["A", "B", "C"],
            "Returns": [0.1, 0.1, 0.1],
            "Volatility": [0.1, 0.3, 0.6],
            "Cluster": [0, 1, 2],
        })
        labels = label_clusters(df)
        # Median vol = 0.3
        # Cluster 0: avg=0.1, < 0.3 -> Low Vol
        # Cluster 1: avg=0.3, >= 0.3 -> Moderate Vol
        # Cluster 2: avg=0.6, >= 0.3*1.5=0.45 -> High Vol
        assert "Low Vol" in labels[0]
        assert "Moderate Vol" in labels[1]
        assert "High Vol" in labels[2]


class TestFindElbow:
    """Verify find_elbow() returns distortions for each k value."""

    def test_returns_list(self, synthetic_metrics):
        data = np.asarray([
            np.asarray(synthetic_metrics["Returns"]),
            np.asarray(synthetic_metrics["Volatility"]),
        ]).T
        data_whitened = data / data.std(axis=0)
        k_range = range(2, 6)
        distortions = find_elbow(data_whitened, k_range)
        assert isinstance(distortions, list)

    def test_correct_length(self, synthetic_metrics):
        data = np.asarray([
            np.asarray(synthetic_metrics["Returns"]),
            np.asarray(synthetic_metrics["Volatility"]),
        ]).T
        data_whitened = data / data.std(axis=0)
        k_range = range(2, 6)
        distortions = find_elbow(data_whitened, k_range)
        assert len(distortions) == len(k_range)

    def test_distortions_are_positive(self, synthetic_metrics):
        data = np.asarray([
            np.asarray(synthetic_metrics["Returns"]),
            np.asarray(synthetic_metrics["Volatility"]),
        ]).T
        data_whitened = data / data.std(axis=0)
        k_range = range(2, 6)
        distortions = find_elbow(data_whitened, k_range)
        for d in distortions:
            assert d >= 0

    def test_distortions_generally_decrease(self, synthetic_metrics):
        """More clusters should generally reduce distortion."""
        data = np.asarray([
            np.asarray(synthetic_metrics["Returns"]),
            np.asarray(synthetic_metrics["Volatility"]),
        ]).T
        data_whitened = data / data.std(axis=0)
        k_range = range(2, 8)
        distortions = find_elbow(data_whitened, k_range)
        # First distortion should be larger than last
        assert distortions[0] > distortions[-1]


class TestEnrichClustersWithInfo:
    """Verify enrich_clusters_with_info() adds metadata columns."""

    def test_adds_name_column(self, synthetic_metrics):
        clustered = cluster_stocks(synthetic_metrics, n_clusters=3)
        labels = label_clusters(clustered)
        ticker_info = {
            "HIGH0": {"name": "High Corp 0", "sector": "Tech"},
        }
        enriched = enrich_clusters_with_info(clustered, ticker_info, labels)
        assert "Name" in enriched.columns

    def test_adds_sector_column(self, synthetic_metrics):
        clustered = cluster_stocks(synthetic_metrics, n_clusters=3)
        labels = label_clusters(clustered)
        ticker_info = {}
        enriched = enrich_clusters_with_info(clustered, ticker_info, labels)
        assert "Sector" in enriched.columns

    def test_adds_cluster_label_column(self, synthetic_metrics):
        clustered = cluster_stocks(synthetic_metrics, n_clusters=3)
        labels = label_clusters(clustered)
        ticker_info = {}
        enriched = enrich_clusters_with_info(clustered, ticker_info, labels)
        assert "ClusterLabel" in enriched.columns

    def test_name_defaults_to_ticker_when_missing(self):
        """When ticker_info lacks a ticker, Name defaults to the ticker symbol."""
        df = pd.DataFrame({
            "Ticker": ["AAPL", "UNKNOWN"],
            "Returns": [0.1, 0.2],
            "Volatility": [0.2, 0.3],
            "Cluster": [0, 0],
        })
        ticker_info = {"AAPL": {"name": "Apple Inc.", "sector": "Tech"}}
        labels = {0: "Test Label"}
        enriched = enrich_clusters_with_info(df, ticker_info, labels)
        assert enriched.loc[enriched["Ticker"] == "AAPL", "Name"].iloc[0] == "Apple Inc."
        assert enriched.loc[enriched["Ticker"] == "UNKNOWN", "Name"].iloc[0] == "UNKNOWN"

    def test_cluster_label_maps_correctly(self):
        df = pd.DataFrame({
            "Ticker": ["A", "B"],
            "Returns": [0.1, 0.2],
            "Volatility": [0.2, 0.3],
            "Cluster": [0, 1],
        })
        labels = {0: "Low Risk", 1: "High Risk"}
        enriched = enrich_clusters_with_info(df, {}, labels)
        assert enriched.loc[enriched["Cluster"] == 0, "ClusterLabel"].iloc[0] == "Low Risk"
        assert enriched.loc[enriched["Cluster"] == 1, "ClusterLabel"].iloc[0] == "High Risk"

    def test_does_not_modify_input(self, synthetic_metrics):
        clustered = cluster_stocks(synthetic_metrics, n_clusters=3)
        original = clustered.copy()
        labels = label_clusters(clustered)
        enrich_clusters_with_info(clustered, {}, labels)
        pd.testing.assert_frame_equal(clustered, original)

    def test_preserves_row_count(self, synthetic_metrics):
        clustered = cluster_stocks(synthetic_metrics, n_clusters=3)
        labels = label_clusters(clustered)
        enriched = enrich_clusters_with_info(clustered, {}, labels)
        assert len(enriched) == len(clustered)
