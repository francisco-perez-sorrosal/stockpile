"""Characterization tests: verify Pydantic models round-trip and behavior."""

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import MarketData, MetricsData, TickerInfo


# ---------------------------------------------------------------------------
# MarketData
# ---------------------------------------------------------------------------

class TestMarketDataIsStale:
    def test_stale_when_no_updated_at(self):
        md = MarketData()
        assert md.is_stale() is True

    def test_not_stale_when_just_updated(self):
        md = MarketData(updated_at=datetime.now(timezone.utc))
        assert md.is_stale() is False

    def test_stale_after_custom_threshold(self):
        """Passing max_age_hours=0 forces staleness for any past timestamp."""
        md = MarketData(updated_at=datetime.now(timezone.utc))
        assert md.is_stale(max_age_hours=0) is True


# ---------------------------------------------------------------------------
# MetricsData
# ---------------------------------------------------------------------------

class TestMetricsDataHasData:
    def test_no_data_when_empty(self):
        md = MetricsData()
        assert md.has_data() is False

    def test_no_data_when_partial(self):
        md = MetricsData(returns=0.05)
        assert md.has_data() is False

    def test_has_data_when_both_set(self):
        md = MetricsData(returns=0.05, volatility=0.20)
        assert md.has_data() is True

    def test_has_data_with_zero_values(self):
        """Zero is valid data, not missing."""
        md = MetricsData(returns=0.0, volatility=0.0)
        assert md.has_data() is True


class TestMetricsDataIsStale:
    def test_stale_when_no_updated_at(self):
        md = MetricsData()
        assert md.is_stale() is True

    def test_not_stale_when_just_updated(self):
        md = MetricsData(updated_at=datetime.now(timezone.utc))
        assert md.is_stale() is False


# ---------------------------------------------------------------------------
# TickerInfo
# ---------------------------------------------------------------------------

class TestTickerInfoEmpty:
    def test_empty_creates_valid_instance(self):
        ti = TickerInfo.empty("TEST")
        assert ti.symbol == "TEST"
        assert ti.name == "TEST"

    def test_empty_has_no_market_data(self):
        ti = TickerInfo.empty("TEST")
        assert ti.market.price is None

    def test_empty_has_no_metrics(self):
        ti = TickerInfo.empty("TEST")
        assert ti.metrics.has_data() is False


class TestTickerInfoFlatDictRoundTrip:
    def test_round_trip_preserves_symbol(self):
        original = TickerInfo(
            symbol="AAPL",
            name="Apple Inc.",
            exchange="NMS",
            sector="Technology",
            updated_at=datetime.now(timezone.utc),
            market=MarketData(price=150.0, updated_at=datetime.now(timezone.utc)),
            metrics=MetricsData(
                returns=0.25, volatility=0.30, updated_at=datetime.now(timezone.utc)
            ),
        )
        flat = original.to_flat_dict()
        restored = TickerInfo.from_flat_dict(flat)

        assert restored.symbol == original.symbol
        assert restored.name == original.name
        assert restored.exchange == original.exchange
        assert restored.sector == original.sector

    def test_round_trip_preserves_market_price(self):
        original = TickerInfo(
            symbol="AAPL",
            market=MarketData(price=150.0, updated_at=datetime.now(timezone.utc)),
        )
        flat = original.to_flat_dict()
        restored = TickerInfo.from_flat_dict(flat)

        assert restored.market.price == original.market.price

    def test_round_trip_preserves_metrics(self):
        original = TickerInfo(
            symbol="AAPL",
            metrics=MetricsData(
                returns=0.10, volatility=0.20, updated_at=datetime.now(timezone.utc)
            ),
        )
        flat = original.to_flat_dict()
        restored = TickerInfo.from_flat_dict(flat)

        assert restored.metrics.returns == pytest.approx(original.metrics.returns)
        assert restored.metrics.volatility == pytest.approx(original.metrics.volatility)

    def test_round_trip_empty_ticker(self):
        original = TickerInfo.empty("ZZZ")
        flat = original.to_flat_dict()
        restored = TickerInfo.from_flat_dict(flat)

        assert restored.symbol == "ZZZ"
        assert restored.name == "ZZZ"

    def test_flat_dict_keys(self):
        """to_flat_dict produces expected top-level keys."""
        ti = TickerInfo(
            symbol="X",
            market=MarketData(price=1.0),
            metrics=MetricsData(returns=0.1, volatility=0.2),
        )
        flat = ti.to_flat_dict()

        expected_keys = {
            "symbol", "name", "long_name", "exchange", "type",
            "sector", "industry", "currency", "first_trade_date",
            "updated_at",
            "price", "previous_close", "volume", "day_high", "day_low",
            "week_52_high", "week_52_low", "market_updated_at",
            "returns", "volatility", "metrics_updated_at",
        }
        assert set(flat.keys()) == expected_keys


class TestTickerInfoIsStaticStale:
    def test_stale_when_no_updated_at(self):
        ti = TickerInfo(symbol="X")
        assert ti.is_static_stale() is True

    def test_not_stale_when_just_updated(self):
        ti = TickerInfo(symbol="X", updated_at=datetime.now(timezone.utc))
        assert ti.is_static_stale() is False

    def test_to_summary_keys(self):
        ti = TickerInfo(symbol="X", name="Xco", sector="Tech", industry="Software")
        summary = ti.to_summary()
        assert summary == {"name": "Xco", "sector": "Tech", "industry": "Software"}
