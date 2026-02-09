"""Characterization tests for stock_clusters.py cache reading functions.

Tests use temporary JSON files to verify read_metrics_from_cache(),
read_ticker_info_from_cache(), and read_metrics_from_file() handle
valid and invalid cache data correctly.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

# Add the scripts directory to the path so we can import stock_clusters
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from stock_clusters import (
    read_metrics_from_cache,
    read_metrics_from_file,
    read_ticker_info_from_cache,
    _extract_return_volatility,
)


SAMPLE_CACHE = {
    "AAPL": {
        "name": "Apple Inc.",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "returns": 0.25,
        "volatility": 0.18,
    },
    "MSFT": {
        "name": "Microsoft Corp",
        "sector": "Technology",
        "industry": "Software",
        "returns": 0.30,
        "volatility": 0.20,
    },
    "TSLA": {
        "name": "Tesla Inc.",
        "sector": "",
        "industry": "Auto Manufacturers",
        "returns": -0.05,
        "volatility": 0.55,
    },
}

CACHE_MISSING_METRICS = {
    "NODATA": {
        "name": "No Data Corp",
        "sector": "Unknown",
    },
    "PARTIAL": {
        "name": "Partial Corp",
        "returns": 0.10,
        # volatility missing
    },
}


@pytest.fixture
def cache_file(tmp_path) -> Path:
    """Create a temporary cache file with sample data."""
    cache_path = tmp_path / "tickers.json"
    cache_path.write_text(json.dumps(SAMPLE_CACHE))
    return cache_path


@pytest.fixture
def cache_with_missing_metrics(tmp_path) -> Path:
    """Create a cache file where some tickers lack metrics."""
    cache_path = tmp_path / "tickers.json"
    cache_path.write_text(json.dumps(CACHE_MISSING_METRICS))
    return cache_path


@pytest.fixture
def malformed_cache(tmp_path) -> Path:
    """Create a cache file with invalid JSON."""
    cache_path = tmp_path / "tickers.json"
    cache_path.write_text("{ invalid json content !!!")
    return cache_path


class TestReadMetricsFromCache:
    """Verify read_metrics_from_cache() extracts metrics into a DataFrame."""

    def test_reads_all_tickers(self, cache_file):
        with patch("stock_clusters.TICKER_CACHE_FILE", cache_file):
            df = read_metrics_from_cache()
        assert len(df) == 3
        assert set(df["Ticker"]) == {"AAPL", "MSFT", "TSLA"}

    def test_reads_specific_tickers(self, cache_file):
        with patch("stock_clusters.TICKER_CACHE_FILE", cache_file):
            df = read_metrics_from_cache(["AAPL", "MSFT"])
        assert len(df) == 2
        assert set(df["Ticker"]) == {"AAPL", "MSFT"}

    def test_columns_are_correct(self, cache_file):
        with patch("stock_clusters.TICKER_CACHE_FILE", cache_file):
            df = read_metrics_from_cache()
        assert list(df.columns) == ["Ticker", "Returns", "Volatility"]

    def test_values_match_cache(self, cache_file):
        with patch("stock_clusters.TICKER_CACHE_FILE", cache_file):
            df = read_metrics_from_cache(["AAPL"])
        row = df.iloc[0]
        assert row["Ticker"] == "AAPL"
        assert row["Returns"] == 0.25
        assert row["Volatility"] == 0.18

    def test_uppercases_ticker_input(self, cache_file):
        """Input tickers are uppercased before lookup."""
        with patch("stock_clusters.TICKER_CACHE_FILE", cache_file):
            df = read_metrics_from_cache(["aapl"])
        assert len(df) == 1
        assert df.iloc[0]["Ticker"] == "AAPL"

    def test_skips_tickers_without_metrics(self, cache_with_missing_metrics):
        """Tickers missing returns or volatility are excluded."""
        with patch("stock_clusters.TICKER_CACHE_FILE", cache_with_missing_metrics):
            df = read_metrics_from_cache()
        assert len(df) == 0

    def test_skips_unknown_tickers(self, cache_file):
        """Tickers not in cache are silently skipped."""
        with patch("stock_clusters.TICKER_CACHE_FILE", cache_file):
            df = read_metrics_from_cache(["AAPL", "UNKNOWN"])
        assert len(df) == 1
        assert df.iloc[0]["Ticker"] == "AAPL"

    def test_missing_file_returns_empty_dataframe(self, tmp_path):
        nonexistent = tmp_path / "nonexistent.json"
        with patch("stock_clusters.TICKER_CACHE_FILE", nonexistent):
            df = read_metrics_from_cache()
        assert len(df) == 0
        assert list(df.columns) == ["Ticker", "Returns", "Volatility"]

    def test_malformed_json_returns_empty_dataframe(self, malformed_cache):
        with patch("stock_clusters.TICKER_CACHE_FILE", malformed_cache):
            df = read_metrics_from_cache()
        assert len(df) == 0
        assert list(df.columns) == ["Ticker", "Returns", "Volatility"]

    def test_none_tickers_reads_all(self, cache_file):
        """Passing None explicitly reads all cached tickers."""
        with patch("stock_clusters.TICKER_CACHE_FILE", cache_file):
            df = read_metrics_from_cache(None)
        assert len(df) == 3

    def test_empty_tickers_list_returns_empty(self, cache_file):
        """Passing an empty list returns an empty DataFrame."""
        with patch("stock_clusters.TICKER_CACHE_FILE", cache_file):
            df = read_metrics_from_cache([])
        assert len(df) == 0


class TestReadTickerInfoFromCache:
    """Verify read_ticker_info_from_cache() extracts name and sector."""

    def test_reads_ticker_info(self, cache_file):
        with patch("stock_clusters.TICKER_CACHE_FILE", cache_file):
            info = read_ticker_info_from_cache(["AAPL"])
        assert "AAPL" in info
        assert info["AAPL"]["name"] == "Apple Inc."
        assert info["AAPL"]["sector"] == "Technology"

    def test_uppercases_ticker_input(self, cache_file):
        with patch("stock_clusters.TICKER_CACHE_FILE", cache_file):
            info = read_ticker_info_from_cache(["aapl"])
        assert "AAPL" in info

    def test_falls_back_to_industry_when_sector_empty(self, cache_file):
        """When sector is empty string, falls back to industry field."""
        with patch("stock_clusters.TICKER_CACHE_FILE", cache_file):
            info = read_ticker_info_from_cache(["TSLA"])
        # Source: data.get("sector", "") or data.get("industry", "")
        # TSLA has sector="" so it evaluates to falsy, falls back to industry
        assert info["TSLA"]["sector"] == "Auto Manufacturers"

    def test_name_defaults_to_ticker_when_missing(self, tmp_path):
        """When name key is absent, defaults to the ticker symbol."""
        cache_path = tmp_path / "tickers.json"
        cache_path.write_text(json.dumps({"NONAME": {"sector": "Test"}}))
        with patch("stock_clusters.TICKER_CACHE_FILE", cache_path):
            info = read_ticker_info_from_cache(["NONAME"])
        assert info["NONAME"]["name"] == "NONAME"

    def test_name_present_is_used(self, cache_with_missing_metrics):
        """When name key is present, its value is used."""
        with patch("stock_clusters.TICKER_CACHE_FILE", cache_with_missing_metrics):
            info = read_ticker_info_from_cache(["NODATA"])
        assert info["NODATA"]["name"] == "No Data Corp"

    def test_missing_ticker_excluded(self, cache_file):
        with patch("stock_clusters.TICKER_CACHE_FILE", cache_file):
            info = read_ticker_info_from_cache(["UNKNOWN"])
        assert len(info) == 0

    def test_missing_file_returns_empty_dict(self, tmp_path):
        nonexistent = tmp_path / "nonexistent.json"
        with patch("stock_clusters.TICKER_CACHE_FILE", nonexistent):
            info = read_ticker_info_from_cache(["AAPL"])
        assert info == {}

    def test_malformed_json_returns_empty_dict(self, malformed_cache):
        with patch("stock_clusters.TICKER_CACHE_FILE", malformed_cache):
            info = read_ticker_info_from_cache(["AAPL"])
        assert info == {}

    def test_multiple_tickers(self, cache_file):
        with patch("stock_clusters.TICKER_CACHE_FILE", cache_file):
            info = read_ticker_info_from_cache(["AAPL", "MSFT"])
        assert len(info) == 2
        assert "AAPL" in info
        assert "MSFT" in info


# --- Data for read_metrics_from_file tests ---

NESTED_FORMAT_DATA = {
    "AAPL": {
        "metrics": {
            "annualized_return": 0.25,
            "annualized_volatility": 0.18,
        },
    },
    "MSFT": {
        "metrics": {
            "annualized_return": 0.30,
            "annualized_volatility": 0.20,
        },
    },
}

FLAT_FORMAT_DATA = {
    "AAPL": {"returns": 0.25, "volatility": 0.18},
    "MSFT": {"returns": 0.30, "volatility": 0.20},
}


@pytest.fixture
def nested_data_file(tmp_path) -> Path:
    """Create a JSON file with nested metrics format."""
    path = tmp_path / "nested.json"
    path.write_text(json.dumps(NESTED_FORMAT_DATA))
    return path


@pytest.fixture
def flat_data_file(tmp_path) -> Path:
    """Create a JSON file with flat cache format."""
    path = tmp_path / "flat.json"
    path.write_text(json.dumps(FLAT_FORMAT_DATA))
    return path


@pytest.fixture
def malformed_data_file(tmp_path) -> Path:
    """Create a JSON file with invalid content."""
    path = tmp_path / "bad.json"
    path.write_text("not valid json {{{")
    return path


class TestExtractReturnVolatility:
    """Verify _extract_return_volatility handles both formats."""

    def test_nested_format(self):
        entry = {"metrics": {"annualized_return": 0.25, "annualized_volatility": 0.18}}
        ret, vol = _extract_return_volatility(entry)
        assert ret == 0.25
        assert vol == 0.18

    def test_flat_format(self):
        entry = {"returns": 0.30, "volatility": 0.20}
        ret, vol = _extract_return_volatility(entry)
        assert ret == 0.30
        assert vol == 0.20

    def test_nested_takes_precedence(self):
        """When both formats are present, nested metrics win."""
        entry = {
            "metrics": {"annualized_return": 0.10, "annualized_volatility": 0.05},
            "returns": 0.99,
            "volatility": 0.99,
        }
        ret, vol = _extract_return_volatility(entry)
        assert ret == 0.10
        assert vol == 0.05

    def test_partial_nested_falls_back_to_flat(self):
        """If nested metrics are incomplete, fall back to flat format."""
        entry = {
            "metrics": {"annualized_return": 0.10},  # missing volatility
            "returns": 0.30,
            "volatility": 0.20,
        }
        ret, vol = _extract_return_volatility(entry)
        assert ret == 0.30
        assert vol == 0.20

    def test_empty_entry_returns_none(self):
        ret, vol = _extract_return_volatility({})
        assert ret is None
        assert vol is None

    def test_partial_flat_returns_partial_none(self):
        entry = {"returns": 0.10}  # missing volatility
        ret, vol = _extract_return_volatility(entry)
        assert ret == 0.10
        assert vol is None


class TestReadMetricsFromFile:
    """Verify read_metrics_from_file() reads both JSON formats."""

    def test_reads_nested_format(self, nested_data_file):
        df = read_metrics_from_file(str(nested_data_file))
        assert len(df) == 2
        assert set(df["Ticker"]) == {"AAPL", "MSFT"}

    def test_reads_flat_format(self, flat_data_file):
        df = read_metrics_from_file(str(flat_data_file))
        assert len(df) == 2
        assert set(df["Ticker"]) == {"AAPL", "MSFT"}

    def test_columns_are_correct(self, flat_data_file):
        df = read_metrics_from_file(str(flat_data_file))
        assert list(df.columns) == ["Ticker", "Returns", "Volatility"]

    def test_values_from_nested(self, nested_data_file):
        df = read_metrics_from_file(str(nested_data_file), ["AAPL"])
        row = df.iloc[0]
        assert row["Returns"] == 0.25
        assert row["Volatility"] == 0.18

    def test_values_from_flat(self, flat_data_file):
        df = read_metrics_from_file(str(flat_data_file), ["MSFT"])
        row = df.iloc[0]
        assert row["Returns"] == 0.30
        assert row["Volatility"] == 0.20

    def test_filter_by_tickers(self, flat_data_file):
        df = read_metrics_from_file(str(flat_data_file), ["AAPL"])
        assert len(df) == 1
        assert df.iloc[0]["Ticker"] == "AAPL"

    def test_none_tickers_reads_all(self, flat_data_file):
        df = read_metrics_from_file(str(flat_data_file), None)
        assert len(df) == 2

    def test_skips_unknown_tickers(self, flat_data_file):
        df = read_metrics_from_file(str(flat_data_file), ["AAPL", "UNKNOWN"])
        assert len(df) == 1
        assert df.iloc[0]["Ticker"] == "AAPL"

    def test_uppercases_symbols(self, flat_data_file):
        df = read_metrics_from_file(str(flat_data_file), ["aapl"])
        assert len(df) == 1
        assert df.iloc[0]["Ticker"] == "AAPL"

    def test_malformed_json_returns_empty(self, malformed_data_file):
        df = read_metrics_from_file(str(malformed_data_file))
        assert len(df) == 0
        assert list(df.columns) == ["Ticker", "Returns", "Volatility"]

    def test_nonexistent_file_returns_empty(self, tmp_path):
        df = read_metrics_from_file(str(tmp_path / "does_not_exist.json"))
        assert len(df) == 0
        assert list(df.columns) == ["Ticker", "Returns", "Volatility"]

    def test_empty_tickers_list_returns_empty(self, flat_data_file):
        df = read_metrics_from_file(str(flat_data_file), [])
        assert len(df) == 0

    def test_skips_entries_without_metrics(self, tmp_path):
        """Entries without return or volatility are excluded."""
        data = {"NODATA": {"name": "No Data Corp"}}
        path = tmp_path / "nodata.json"
        path.write_text(json.dumps(data))
        df = read_metrics_from_file(str(path))
        assert len(df) == 0

    def test_stdin_reading(self, tmp_path):
        """Verify '-' reads from stdin via mock."""
        import io
        data = json.dumps(FLAT_FORMAT_DATA)
        with patch("sys.stdin", io.StringIO(data)):
            df = read_metrics_from_file("-")
        assert len(df) == 2
        assert set(df["Ticker"]) == {"AAPL", "MSFT"}
