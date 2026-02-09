"""File-based cache for ticker information with operation statistics."""

import json
from pathlib import Path

from constants import CACHE_FILE
from models import TickerInfo


class CacheOperationStats:
    """Session-based operation statistics (resets on server restart)."""

    def __init__(self):
        self.lookups = 0
        self.hits = 0
        self.misses = 0
        self.metrics_refreshes = 0

    def record_lookup(self, hit: bool):
        self.lookups += 1
        if hit:
            self.hits += 1
        else:
            self.misses += 1

    def record_metrics_refresh(self):
        self.metrics_refreshes += 1

    @property
    def hit_rate_pct(self) -> float:
        if self.lookups == 0:
            return 0.0
        return round(self.hits / self.lookups * 100, 1)

    def to_dict(self) -> dict:
        return {
            "lookups": self.lookups,
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "hit_rate_pct": self.hit_rate_pct,
            "metrics_refreshes": self.metrics_refreshes,
        }


class TickerCache:
    """File-based cache for ticker information."""

    def __init__(self, cache_file: Path = CACHE_FILE):
        self.cache_file = cache_file
        self._cache: dict[str, dict] = {}
        self._loaded = False
        self.ops = CacheOperationStats()

    def _ensure_loaded(self):
        if self._loaded:
            return
        self._loaded = True
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    self._cache = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._cache = {}

    def _save(self):
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, "w") as f:
            json.dump(self._cache, f, indent=2)

    def get(self, symbol: str, track_stats: bool = True) -> TickerInfo | None:
        self._ensure_loaded()
        symbol = symbol.upper()
        if symbol not in self._cache:
            if track_stats:
                self.ops.record_lookup(hit=False)
            return None
        if track_stats:
            self.ops.record_lookup(hit=True)
        data = self._cache[symbol]
        data["symbol"] = symbol
        return TickerInfo.from_flat_dict(data)

    def put(self, info: TickerInfo):
        self._ensure_loaded()
        self._cache[info.symbol.upper()] = info.to_flat_dict()
        self._save()

    def put_many(self, infos: list[TickerInfo]):
        self._ensure_loaded()
        for info in infos:
            self._cache[info.symbol.upper()] = info.to_flat_dict()
        self._save()

    def all(self) -> dict[str, TickerInfo]:
        self._ensure_loaded()
        result = {}
        for sym, data in self._cache.items():
            data["symbol"] = sym
            result[sym] = TickerInfo.from_flat_dict(data)
        return result

    def symbols(self) -> set[str]:
        """Return set of all cached symbols."""
        self._ensure_loaded()
        return set(self._cache.keys())

    def stats(self, index_tickers: dict[str, list[str]] | None = None) -> dict:
        """Comprehensive cache statistics.

        Args:
            index_tickers: Optional dict of index_name -> ticker list for coverage stats
        """
        self._ensure_loaded()

        if not self._cache:
            return {
                "cache": {"total_tickers": 0},
                "data_completeness": {},
                "index_coverage": {},
                "sectors": {},
                "operations": self.ops.to_dict(),
            }

        # Basic cache info
        file_size_kb = 0.0
        if self.cache_file.exists():
            file_size_kb = round(self.cache_file.stat().st_size / 1024, 1)

        dates = []
        with_market = 0
        with_metrics = 0
        stale_static = 0
        stale_market = 0
        stale_metrics = 0
        sectors: dict[str, int] = {}

        for data in self._cache.values():
            info = TickerInfo.from_flat_dict(data)

            # Dates
            if info.updated_at:
                dates.append(info.updated_at.isoformat())

            # Data completeness
            if info.market and info.market.price is not None:
                with_market += 1
            if info.metrics.has_data():
                with_metrics += 1

            # Staleness
            if info.is_static_stale():
                stale_static += 1
            if info.market and info.market.is_stale():
                stale_market += 1
            if info.metrics.updated_at and info.metrics.is_stale():
                stale_metrics += 1

            # Sectors
            sector = info.sector or "unknown"
            sectors[sector] = sectors.get(sector, 0) + 1

        # Sort sectors by count
        sorted_sectors = dict(sorted(sectors.items(), key=lambda x: -x[1]))

        # Index coverage
        index_coverage = {}
        cached_symbols = self.symbols()

        if index_tickers:
            all_index_symbols: set[str] = set()
            symbols_in_indexes: set[str] = set()

            for index_name, tickers in index_tickers.items():
                ticker_set = set(t.upper() for t in tickers)
                all_index_symbols.update(ticker_set)
                cached_in_index = cached_symbols & ticker_set
                symbols_in_indexes.update(cached_in_index)

                index_coverage[index_name] = {
                    "cached": len(cached_in_index),
                    "total": len(ticker_set),
                    "coverage_pct": round(len(cached_in_index) / len(ticker_set) * 100, 1) if ticker_set else 0,
                }

            # Tickers in multiple indexes
            index_membership_count: dict[str, int] = {}
            for sym in cached_symbols:
                count = sum(1 for tickers in index_tickers.values() if sym in set(t.upper() for t in tickers))
                if count > 0:
                    index_membership_count[sym] = count

            in_multiple = sum(1 for c in index_membership_count.values() if c > 1)
            orphan = len(cached_symbols - all_index_symbols)

            index_coverage["in_multiple_indexes"] = in_multiple
            index_coverage["orphan"] = orphan

        return {
            "cache": {
                "total_tickers": len(self._cache),
                "file_size_kb": file_size_kb,
                "oldest_entry": min(dates) if dates else None,
                "newest_entry": max(dates) if dates else None,
            },
            "data_completeness": {
                "with_market_data": with_market,
                "with_metrics": with_metrics,
                "stale_static": stale_static,
                "stale_market": stale_market,
                "stale_metrics": stale_metrics,
            },
            "index_coverage": index_coverage,
            "sectors": sorted_sectors,
            "operations": self.ops.to_dict(),
        }
