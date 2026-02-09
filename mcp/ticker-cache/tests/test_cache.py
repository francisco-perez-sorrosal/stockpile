"""Characterization tests: verify TickerCache with temp files."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cache import CacheOperationStats, TickerCache
from models import MarketData, MetricsData, TickerInfo


# ---------------------------------------------------------------------------
# TickerCache CRUD
# ---------------------------------------------------------------------------

class TestTickerCachePutGet:
    def test_put_get_round_trip(self, tmp_path):
        cache_file = tmp_path / "tickers.json"
        cache = TickerCache(cache_file=cache_file)

        info = TickerInfo(
            symbol="AAPL",
            name="Apple Inc.",
            exchange="NMS",
            updated_at=datetime.now(timezone.utc),
            market=MarketData(price=150.0, updated_at=datetime.now(timezone.utc)),
        )
        cache.put(info)

        retrieved = cache.get("AAPL")
        assert retrieved is not None
        assert retrieved.symbol == "AAPL"
        assert retrieved.name == "Apple Inc."
        assert retrieved.market.price == 150.0

    def test_get_missing_returns_none(self, tmp_path):
        cache_file = tmp_path / "tickers.json"
        cache = TickerCache(cache_file=cache_file)

        result = cache.get("ZZZZ")
        assert result is None

    def test_put_overwrites_existing(self, tmp_path):
        cache_file = tmp_path / "tickers.json"
        cache = TickerCache(cache_file=cache_file)

        info_v1 = TickerInfo(symbol="AAPL", name="Apple V1")
        cache.put(info_v1)

        info_v2 = TickerInfo(symbol="AAPL", name="Apple V2")
        cache.put(info_v2)

        retrieved = cache.get("AAPL")
        assert retrieved is not None
        assert retrieved.name == "Apple V2"

    def test_get_normalizes_to_uppercase(self, tmp_path):
        cache_file = tmp_path / "tickers.json"
        cache = TickerCache(cache_file=cache_file)

        info = TickerInfo(symbol="MSFT", name="Microsoft")
        cache.put(info)

        retrieved = cache.get("msft")
        assert retrieved is not None
        assert retrieved.symbol == "MSFT"

    def test_cache_file_created_on_put(self, tmp_path):
        cache_file = tmp_path / "sub" / "tickers.json"
        cache = TickerCache(cache_file=cache_file)

        info = TickerInfo(symbol="GOOG", name="Alphabet")
        cache.put(info)

        assert cache_file.exists()


class TestTickerCachePutMany:
    def test_put_many_and_all_round_trip(self, tmp_path):
        cache_file = tmp_path / "tickers.json"
        cache = TickerCache(cache_file=cache_file)

        infos = [
            TickerInfo(symbol="AAPL", name="Apple"),
            TickerInfo(symbol="MSFT", name="Microsoft"),
            TickerInfo(symbol="GOOG", name="Alphabet"),
        ]
        cache.put_many(infos)

        all_tickers = cache.all()
        assert len(all_tickers) == 3
        assert "AAPL" in all_tickers
        assert "MSFT" in all_tickers
        assert "GOOG" in all_tickers


class TestTickerCacheSymbols:
    def test_symbols_returns_correct_set(self, tmp_path):
        cache_file = tmp_path / "tickers.json"
        cache = TickerCache(cache_file=cache_file)

        infos = [
            TickerInfo(symbol="AAPL", name="Apple"),
            TickerInfo(symbol="MSFT", name="Microsoft"),
        ]
        cache.put_many(infos)

        symbols = cache.symbols()
        assert symbols == {"AAPL", "MSFT"}

    def test_symbols_empty_cache(self, tmp_path):
        cache_file = tmp_path / "tickers.json"
        cache = TickerCache(cache_file=cache_file)

        symbols = cache.symbols()
        assert symbols == set()


class TestTickerCacheStats:
    def test_stats_empty_cache(self, tmp_path):
        cache_file = tmp_path / "tickers.json"
        cache = TickerCache(cache_file=cache_file)

        stats = cache.stats()
        assert stats["cache"]["total_tickers"] == 0
        assert "operations" in stats

    def test_stats_with_data(self, tmp_path):
        cache_file = tmp_path / "tickers.json"
        cache = TickerCache(cache_file=cache_file)

        info = TickerInfo(
            symbol="AAPL",
            name="Apple",
            sector="Technology",
            updated_at=datetime.now(timezone.utc),
            market=MarketData(price=150.0, updated_at=datetime.now(timezone.utc)),
            metrics=MetricsData(
                returns=0.25,
                volatility=0.30,
                updated_at=datetime.now(timezone.utc),
            ),
        )
        cache.put(info)

        stats = cache.stats()
        assert stats["cache"]["total_tickers"] == 1
        assert "data_completeness" in stats
        assert "sectors" in stats
        assert "operations" in stats

    def test_stats_expected_keys(self, tmp_path):
        cache_file = tmp_path / "tickers.json"
        cache = TickerCache(cache_file=cache_file)

        stats = cache.stats()
        expected_top_keys = {"cache", "data_completeness", "index_coverage", "sectors", "operations"}
        assert set(stats.keys()) == expected_top_keys


class TestTickerCachePersistence:
    def test_data_persists_across_instances(self, tmp_path):
        cache_file = tmp_path / "tickers.json"

        cache1 = TickerCache(cache_file=cache_file)
        cache1.put(TickerInfo(symbol="AAPL", name="Apple"))

        cache2 = TickerCache(cache_file=cache_file)
        retrieved = cache2.get("AAPL")
        assert retrieved is not None
        assert retrieved.name == "Apple"

    def test_handles_malformed_json(self, tmp_path):
        cache_file = tmp_path / "tickers.json"
        cache_file.write_text("not valid json {{{")

        cache = TickerCache(cache_file=cache_file)
        result = cache.get("AAPL")
        assert result is None


# ---------------------------------------------------------------------------
# CacheOperationStats
# ---------------------------------------------------------------------------

class TestCacheOperationStats:
    def test_initial_state(self):
        ops = CacheOperationStats()
        assert ops.lookups == 0
        assert ops.hits == 0
        assert ops.misses == 0

    def test_record_lookup_hit(self):
        ops = CacheOperationStats()
        ops.record_lookup(hit=True)
        assert ops.lookups == 1
        assert ops.hits == 1
        assert ops.misses == 0

    def test_record_lookup_miss(self):
        ops = CacheOperationStats()
        ops.record_lookup(hit=False)
        assert ops.lookups == 1
        assert ops.hits == 0
        assert ops.misses == 1

    def test_hit_rate_pct_no_lookups(self):
        ops = CacheOperationStats()
        assert ops.hit_rate_pct == 0.0

    def test_hit_rate_pct_mixed(self):
        ops = CacheOperationStats()
        ops.record_lookup(hit=True)
        ops.record_lookup(hit=True)
        ops.record_lookup(hit=False)
        assert ops.hit_rate_pct == pytest.approx(66.7)

    def test_record_metrics_refresh(self):
        ops = CacheOperationStats()
        ops.record_metrics_refresh()
        ops.record_metrics_refresh()
        assert ops.metrics_refreshes == 2

    def test_to_dict_keys(self):
        ops = CacheOperationStats()
        d = ops.to_dict()
        expected_keys = {"lookups", "cache_hits", "cache_misses", "hit_rate_pct", "metrics_refreshes"}
        assert set(d.keys()) == expected_keys


class TestCacheStatsTracking:
    """Verify that cache get() operations update stats."""

    def test_get_hit_records_stats(self, tmp_path):
        cache_file = tmp_path / "tickers.json"
        cache = TickerCache(cache_file=cache_file)
        cache.put(TickerInfo(symbol="AAPL", name="Apple"))

        cache.get("AAPL")
        assert cache.ops.hits == 1

    def test_get_miss_records_stats(self, tmp_path):
        cache_file = tmp_path / "tickers.json"
        cache = TickerCache(cache_file=cache_file)

        cache.get("ZZZZ")
        assert cache.ops.misses == 1

    def test_get_with_track_stats_false(self, tmp_path):
        cache_file = tmp_path / "tickers.json"
        cache = TickerCache(cache_file=cache_file)

        cache.get("ZZZZ", track_stats=False)
        assert cache.ops.lookups == 0
