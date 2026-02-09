"""Characterization tests: verify pure helper functions."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import MetricsData
from scraping import is_index_name as _is_index_name
from yahoo import (
    calculate_metrics as _calculate_metrics,
    extract_market_data as _extract_market_data,
    looks_like_ticker as _looks_like_ticker,
)


# ---------------------------------------------------------------------------
# _looks_like_ticker
# ---------------------------------------------------------------------------

class TestLooksLikeTicker:
    def test_valid_single_char(self):
        assert _looks_like_ticker("A") is True

    def test_valid_standard_ticker(self):
        assert _looks_like_ticker("AAPL") is True

    def test_valid_with_dot(self):
        assert _looks_like_ticker("BRK.B") is True

    def test_valid_with_dash(self):
        assert _looks_like_ticker("BF-B") is True

    def test_valid_five_chars(self):
        assert _looks_like_ticker("GOOGL") is True

    def test_invalid_too_long(self):
        """Anything over 5 chars is treated as a company name."""
        assert _looks_like_ticker("MODERNA") is False

    def test_invalid_empty(self):
        assert _looks_like_ticker("") is False

    def test_invalid_lowercase(self):
        assert _looks_like_ticker("aapl") is False

    def test_invalid_spaces(self):
        assert _looks_like_ticker("AA PL") is False

    def test_whitespace_stripped(self):
        assert _looks_like_ticker("  AAPL  ") is True

    def test_numeric_ticker(self):
        assert _looks_like_ticker("1234") is True

    def test_starts_with_number(self):
        assert _looks_like_ticker("3M") is True


# ---------------------------------------------------------------------------
# _is_index_name
# ---------------------------------------------------------------------------

class TestIsIndexName:
    def test_sp500_canonical(self):
        assert _is_index_name("sp500") is True

    def test_sp500_alias(self):
        assert _is_index_name("s&p500") is True

    def test_sp_alias(self):
        assert _is_index_name("sp") is True

    def test_nasdaq100_canonical(self):
        assert _is_index_name("nasdaq100") is True

    def test_nasdaq_alias(self):
        assert _is_index_name("nasdaq") is True

    def test_ndx_alias(self):
        assert _is_index_name("ndx") is True

    def test_dow_canonical(self):
        assert _is_index_name("dow") is True

    def test_dowjones_alias(self):
        assert _is_index_name("dowjones") is True

    def test_djia_alias(self):
        assert _is_index_name("djia") is True

    def test_case_insensitive(self):
        assert _is_index_name("SP500") is True

    def test_unknown_index(self):
        assert _is_index_name("ftse100") is False

    def test_empty_string(self):
        assert _is_index_name("") is False


# ---------------------------------------------------------------------------
# _calculate_metrics
# ---------------------------------------------------------------------------

class TestCalculateMetrics:
    def test_returns_none_for_empty_prices(self):
        assert _calculate_metrics([]) is None

    def test_returns_none_for_too_few_prices(self):
        """Needs at least 20 prices."""
        assert _calculate_metrics([100.0] * 19) is None

    def test_returns_metrics_for_sufficient_prices(self):
        """20+ prices should produce MetricsData."""
        prices = [100.0 + i * 0.5 for i in range(25)]
        result = _calculate_metrics(prices)
        assert result is not None
        assert isinstance(result, MetricsData)

    def test_constant_prices_zero_volatility(self):
        """Constant prices produce zero volatility and zero return."""
        prices = [100.0] * 25
        result = _calculate_metrics(prices)
        assert result is not None
        assert result.volatility == pytest.approx(0.0)
        assert result.returns == pytest.approx(0.0)

    def test_known_linear_series(self):
        """Verify metrics with a predictable daily-increase series."""
        # Each day increases by 1% from 100
        prices = [100.0 * (1.01 ** i) for i in range(252)]
        result = _calculate_metrics(prices)
        assert result is not None

        # Annualized return: daily mean return * 252
        # Daily return is ~1% so annualized is ~2.52 (252%)
        assert result.returns is not None
        assert result.returns > 2.0

        # Volatility should be very small (constant daily return)
        assert result.volatility is not None
        assert result.volatility < 0.01

    def test_metrics_have_updated_at(self):
        prices = [100.0 + i for i in range(25)]
        result = _calculate_metrics(prices)
        assert result is not None
        assert result.updated_at is not None

    def test_handles_zero_price_in_series(self):
        """A zero price should not cause division by zero."""
        prices = [100.0] * 10 + [0.0] + [100.0] * 15
        result = _calculate_metrics(prices)
        # Should still produce a result (zero-price entry skipped in returns)
        assert result is not None

    def test_returns_none_for_all_zeros(self):
        """All-zero prices produce no daily returns."""
        prices = [0.0] * 25
        result = _calculate_metrics(prices)
        assert result is None


# ---------------------------------------------------------------------------
# _extract_market_data
# ---------------------------------------------------------------------------

class TestExtractMarketData:
    def test_extracts_all_fields(self):
        meta = {
            "regularMarketPrice": 150.25,
            "chartPreviousClose": 149.50,
            "regularMarketVolume": 50_000_000,
            "regularMarketDayHigh": 151.00,
            "regularMarketDayLow": 148.50,
            "fiftyTwoWeekHigh": 180.00,
            "fiftyTwoWeekLow": 120.00,
        }
        md = _extract_market_data(meta)

        assert md.price == 150.25
        assert md.previous_close == 149.50
        assert md.volume == 50_000_000
        assert md.day_high == 151.00
        assert md.day_low == 148.50
        assert md.week_52_high == 180.00
        assert md.week_52_low == 120.00

    def test_sets_updated_at(self):
        md = _extract_market_data({})
        assert md.updated_at is not None

    def test_missing_fields_default_to_none(self):
        md = _extract_market_data({})
        assert md.price is None
        assert md.volume is None
        assert md.week_52_high is None

    def test_partial_metadata(self):
        meta = {"regularMarketPrice": 100.0}
        md = _extract_market_data(meta)
        assert md.price == 100.0
        assert md.previous_close is None
