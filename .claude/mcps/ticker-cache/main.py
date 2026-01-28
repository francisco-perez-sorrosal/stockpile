#!/usr/bin/env python3
"""MCP server for stock ticker data cache.

Provides Resources for data discovery and Tools for data access.
All Yahoo Finance API calls are handled here - skills use MCP for data.

Resources (read-only views of cached data):
    ticker://cache              - Summary of all cached tickers
    ticker://cache/stats        - Cache statistics
    ticker://ticker/{symbol}    - Full data for a cached ticker
    ticker://ticker/{symbol}/metrics - Just metrics for a cached ticker
    ticker://indexes            - List of supported index names
    ticker://indexes/{index}    - All cached ticker info for an index

Tools (actions that fetch/compute):
    lookup(query)           - Unified: ticker, comma-list, index name, or company name
    refresh_metrics(symbols) - Calculate return/volatility from 1y prices
"""

import gzip
import json
import math
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

# Cache configuration
CACHE_DIR = Path.home() / ".cache" / "ticker"
CACHE_FILE = CACHE_DIR / "tickers.json"

# Yahoo Finance API endpoints
SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"
CHART_URL = "https://query2.finance.yahoo.com/v8/finance/chart"

# Wikipedia URLs for index constituents
INDEX_URLS = {
    "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "nasdaq100": "https://en.wikipedia.org/wiki/NASDAQ-100",
    "dow": "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
}

# Index name aliases
INDEX_ALIASES = {
    "sp500": "sp500", "s&p500": "sp500", "sp": "sp500",
    "nasdaq100": "nasdaq100", "nasdaq": "nasdaq100", "ndx": "nasdaq100",
    "dow": "dow", "dowjones": "dow", "djia": "dow",
}

# Required headers to avoid 403 errors
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}

# Refresh policies (in days/hours)
STATIC_MAX_AGE_DAYS = 30
MARKET_MAX_AGE_HOURS = 24
METRICS_MAX_AGE_DAYS = 7


def _parse_args():
    """Parse CLI arguments early for server configuration."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Ticker Cache MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="Transport protocol: stdio (default), sse, or streamable-http",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for SSE transport (default: 8000)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host for SSE transport (default: 0.0.0.0)",
    )
    # Parse known args to avoid errors from other flags
    args, _ = parser.parse_known_args()
    return args


_args = _parse_args()

# Initialize MCP server with host/port for SSE transport
mcp = FastMCP("ticker-cache", host=_args.host, port=_args.port)


# =============================================================================
# Pydantic Models
# =============================================================================

class MarketData(BaseModel):
    """Market data region - refreshed every 24 hours."""

    price: float | None = None
    previous_close: float | None = None
    volume: int | None = None
    day_high: float | None = None
    day_low: float | None = None
    week_52_high: float | None = None
    week_52_low: float | None = None
    updated_at: datetime | None = None

    def is_stale(self, max_age_hours: int = MARKET_MAX_AGE_HOURS) -> bool:
        if not self.updated_at:
            return True
        age = datetime.now(timezone.utc) - self.updated_at
        return age.total_seconds() > max_age_hours * 3600


class MetricsData(BaseModel):
    """Calculated metrics region - refreshed every 7 days."""

    returns: float | None = None
    volatility: float | None = None
    updated_at: datetime | None = None

    def is_stale(self, max_age_days: int = METRICS_MAX_AGE_DAYS) -> bool:
        if not self.updated_at:
            return True
        age = datetime.now(timezone.utc) - self.updated_at
        return age.days > max_age_days

    def has_data(self) -> bool:
        return self.returns is not None and self.volatility is not None


class TickerInfo(BaseModel):
    """Stock ticker with composed data regions.

    Static fields are refreshed every 30 days.
    Market and metrics have independent refresh cycles.
    """

    # Static fields
    symbol: str
    name: str = ""
    long_name: str = ""
    exchange: str = ""
    type: str = ""
    sector: str = ""
    industry: str = ""
    currency: str = ""
    first_trade_date: str = ""
    updated_at: datetime | None = None

    # Composed data regions
    market: MarketData = Field(default_factory=MarketData)
    metrics: MetricsData = Field(default_factory=MetricsData)

    def is_static_stale(self, max_age_days: int = STATIC_MAX_AGE_DAYS) -> bool:
        if not self.updated_at:
            return True
        age = datetime.now(timezone.utc) - self.updated_at
        return age.days > max_age_days

    def to_summary(self) -> dict:
        """Minimal dict for listing (name, sector, industry)."""
        return {"name": self.name, "sector": self.sector, "industry": self.industry}

    def to_flat_dict(self) -> dict:
        """Flatten for compatibility with scripts reading cache."""
        result = {
            "symbol": self.symbol,
            "name": self.name,
            "long_name": self.long_name,
            "exchange": self.exchange,
            "type": self.type,
            "sector": self.sector,
            "industry": self.industry,
            "currency": self.currency,
            "first_trade_date": self.first_trade_date,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
        if self.market:
            result.update({
                "price": self.market.price,
                "previous_close": self.market.previous_close,
                "volume": self.market.volume,
                "day_high": self.market.day_high,
                "day_low": self.market.day_low,
                "week_52_high": self.market.week_52_high,
                "week_52_low": self.market.week_52_low,
                "market_updated_at": self.market.updated_at.isoformat() if self.market.updated_at else None,
            })
        if self.metrics:
            result.update({
                "returns": self.metrics.returns,
                "volatility": self.metrics.volatility,
                "metrics_updated_at": self.metrics.updated_at.isoformat() if self.metrics.updated_at else None,
            })
        return result

    @classmethod
    def from_flat_dict(cls, data: dict) -> "TickerInfo":
        """Create from flat dict (cache format)."""
        # Parse timestamps
        def parse_dt(val: str | None) -> datetime | None:
            if not val:
                return None
            try:
                return datetime.fromisoformat(val.replace("Z", "+00:00"))
            except ValueError:
                return None

        market = MarketData(
            price=data.get("price"),
            previous_close=data.get("previous_close"),
            volume=data.get("volume"),
            day_high=data.get("day_high"),
            day_low=data.get("day_low"),
            week_52_high=data.get("week_52_high"),
            week_52_low=data.get("week_52_low"),
            updated_at=parse_dt(data.get("market_updated_at")),
        )

        metrics = MetricsData(
            returns=data.get("returns"),
            volatility=data.get("volatility"),
            updated_at=parse_dt(data.get("metrics_updated_at")),
        )

        return cls(
            symbol=data.get("symbol", ""),
            name=data.get("name", ""),
            long_name=data.get("long_name", ""),
            exchange=data.get("exchange", ""),
            type=data.get("type", ""),
            sector=data.get("sector", ""),
            industry=data.get("industry", ""),
            currency=data.get("currency", ""),
            first_trade_date=data.get("first_trade_date", ""),
            updated_at=parse_dt(data.get("updated_at")),
            market=market,
            metrics=metrics,
        )

    @classmethod
    def empty(cls, symbol: str) -> "TickerInfo":
        """Create placeholder for unknown ticker."""
        return cls(symbol=symbol, name=symbol)


# =============================================================================
# Cache Management
# =============================================================================

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


# Global cache instance
_cache = TickerCache()


# =============================================================================
# HTTP Helpers
# =============================================================================

def _fetch_json(url: str) -> dict | None:
    try:
        request = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


def _fetch_html(url: str) -> str:
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read()
        if len(data) >= 2 and data[:2] == b"\x1f\x8b":
            data = gzip.decompress(data)
        return data.decode("utf-8")


# =============================================================================
# Wikipedia Index Scraping
# =============================================================================

def _extract_table_column(html: str, header_pattern: str, table_limit: int = 0) -> list[str]:
    table_pattern = re.compile(
        r'<table[^>]*class="[^"]*wikitable[^"]*"[^>]*>(.*?)</table>',
        re.DOTALL | re.IGNORECASE
    )
    tables = table_pattern.findall(html)

    for table_html in tables:
        header_match = re.search(r'<tr[^>]*>(.*?)</tr>', table_html, re.DOTALL | re.IGNORECASE)
        if not header_match:
            continue

        header_row = header_match.group(1)
        headers = re.findall(r'<th[^>]*>(.*?)</th>', header_row, re.DOTALL | re.IGNORECASE)

        col_index = -1
        for i, header in enumerate(headers):
            clean_header = re.sub(r'<[^>]+>', '', header).strip()
            if re.search(header_pattern, clean_header, re.IGNORECASE):
                col_index = i
                break

        if col_index < 0:
            continue

        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table_html, re.DOTALL | re.IGNORECASE)
        values = []

        for row in rows[1:]:
            cells = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', row, re.DOTALL | re.IGNORECASE)
            if col_index < len(cells):
                cell = cells[col_index]
                link_match = re.search(r'<a[^>]*>([^<]+)</a>', cell)
                value = link_match.group(1) if link_match else re.sub(r'<[^>]+>', '', cell)
                value = value.strip().replace('\n', '').replace('.', '-')
                if value and not value.startswith('—'):
                    values.append(value)

        if table_limit > 0 and len(values) >= table_limit:
            continue

        if values:
            return values

    return []


def _fetch_index_tickers(index: str) -> list[str]:
    """Fetch tickers from a market index."""
    index = INDEX_ALIASES.get(index.lower(), index.lower())
    if index not in INDEX_URLS:
        return []

    html = _fetch_html(INDEX_URLS[index])

    if index == "sp500":
        return _extract_table_column(html, r'^Symbol$')
    elif index == "nasdaq100":
        return _extract_table_column(html, r'^Ticker$')
    elif index == "dow":
        return _extract_table_column(html, r'^Symbol$', table_limit=35)

    return []


def _is_index_name(query: str) -> bool:
    return query.lower() in INDEX_ALIASES


# =============================================================================
# Yahoo Finance API
# =============================================================================

def _search_yahoo(query: str, max_results: int = 10) -> dict[str, dict]:
    params = urllib.parse.urlencode({
        "q": query, "quotesCount": max_results, "newsCount": 0,
        "listsCount": 0, "enableFuzzyQuery": "true",
    })
    data = _fetch_json(f"{SEARCH_URL}?{params}")
    if not data:
        return {}
    return {q["symbol"]: q for q in data.get("quotes", []) if q.get("symbol")}


def _get_chart_meta(symbol: str) -> dict | None:
    data = _fetch_json(f"{CHART_URL}/{symbol}?interval=1d&range=1d")
    if not data:
        return None
    result = data.get("chart", {}).get("result", [])
    return result[0].get("meta", {}) if result else None


def _extract_market_data(meta: dict) -> MarketData:
    """Extract market data from chart metadata."""
    return MarketData(
        price=meta.get("regularMarketPrice"),
        previous_close=meta.get("chartPreviousClose"),
        volume=meta.get("regularMarketVolume"),
        day_high=meta.get("regularMarketDayHigh"),
        day_low=meta.get("regularMarketDayLow"),
        week_52_high=meta.get("fiftyTwoWeekHigh"),
        week_52_low=meta.get("fiftyTwoWeekLow"),
        updated_at=datetime.now(timezone.utc),
    )


def _fetch_historical_prices(symbol: str, period: str = "1y") -> list[float] | None:
    data = _fetch_json(f"{CHART_URL}/{symbol}?interval=1d&range={period}")
    if not data:
        return None

    result = data.get("chart", {}).get("result", [])
    if not result:
        return None

    indicators = result[0].get("indicators", {})
    adjclose_list = indicators.get("adjclose", [])
    if adjclose_list and adjclose_list[0]:
        prices = adjclose_list[0].get("adjclose", [])
        return [p for p in prices if p is not None]

    quote_list = indicators.get("quote", [])
    if quote_list and quote_list[0]:
        prices = quote_list[0].get("close", [])
        return [p for p in prices if p is not None]

    return None


def _calculate_metrics(prices: list[float]) -> MetricsData | None:
    """Calculate annualized return and volatility from price series."""
    if not prices or len(prices) < 20:
        return None

    daily_returns = []
    for i in range(1, len(prices)):
        if prices[i - 1] != 0:
            daily_returns.append((prices[i] - prices[i - 1]) / prices[i - 1])

    if not daily_returns:
        return None

    n = len(daily_returns)
    mean_return = sum(daily_returns) / n
    variance = sum((r - mean_return) ** 2 for r in daily_returns) / n
    std_return = math.sqrt(variance)

    return MetricsData(
        returns=mean_return * 252,
        volatility=std_return * math.sqrt(252),
        updated_at=datetime.now(timezone.utc),
    )


def _build_ticker_info(symbol: str, quote: dict | None = None, meta: dict | None = None) -> TickerInfo:
    """Build TickerInfo from Yahoo search quote and/or chart meta."""
    quote = quote or {}
    meta = meta or {}

    # Parse first trade date
    first_trade = ""
    if meta.get("firstTradeDate"):
        try:
            first_trade = datetime.fromtimestamp(
                meta["firstTradeDate"], timezone.utc
            ).date().isoformat()
        except (ValueError, OSError):
            pass

    # Build market data if available
    market = _extract_market_data(meta) if meta else MarketData()

    return TickerInfo(
        symbol=symbol,
        name=quote.get("shortname") or quote.get("longname") or meta.get("shortName") or meta.get("longName") or "",
        long_name=quote.get("longname") or meta.get("longName") or "",
        exchange=quote.get("exchange") or meta.get("exchangeName") or "",
        type=quote.get("quoteType") or meta.get("instrumentType") or "",
        sector=quote.get("sector") or "",
        industry=quote.get("industry") or "",
        currency=meta.get("currency") or "",
        first_trade_date=first_trade,
        updated_at=datetime.now(timezone.utc),
        market=market,
        metrics=MetricsData(),
    )


def _looks_like_ticker(s: str) -> bool:
    """Heuristic: tickers are 1-5 chars, uppercase, alphanumeric.

    Real stock tickers are typically 1-4 characters (NYSE, NASDAQ).
    Some ETFs and preferred shares go up to 5. Anything longer is
    likely a company name (e.g., MODERNA, NVIDIA).
    """
    s = s.strip()
    if not s or len(s) > 5:
        return False
    return bool(re.match(r'^[A-Z0-9][A-Z0-9.\-]*$', s))


def _lookup_single(symbol: str) -> TickerInfo | None:
    """Look up a single ticker, fetching from Yahoo if not cached or stale."""
    symbol = symbol.upper()

    cached = _cache.get(symbol)
    if cached and not cached.is_static_stale():
        return cached

    quotes = _search_yahoo(symbol, max_results=20)
    quote = quotes.get(symbol, {})
    meta = _get_chart_meta(symbol)

    if quote or meta:
        info = _build_ticker_info(symbol, quote, meta)
        _cache.put(info)
        return info

    return None


# =============================================================================
# MCP Resources
# =============================================================================

@mcp.resource("ticker://cache")
def get_cache() -> str:
    """List all cached tickers with summary info."""
    all_tickers = _cache.all()
    return json.dumps({s: info.to_summary() for s, info in all_tickers.items()}, indent=2)


@mcp.resource("ticker://cache/stats")
def get_cache_stats() -> str:
    """Get comprehensive cache statistics including index coverage and operations."""
    # Fetch current index tickers for coverage calculation
    index_tickers = {}
    for index_name in INDEX_URLS.keys():
        try:
            tickers = _fetch_index_tickers(index_name)
            if tickers:
                index_tickers[index_name] = tickers
        except Exception:
            pass  # Skip index if fetch fails

    return json.dumps(_cache.stats(index_tickers), indent=2)


@mcp.resource("ticker://ticker/{symbol}")
def get_ticker(symbol: str) -> str:
    """Get full data for a cached ticker. Returns error if not cached."""
    info = _cache.get(symbol.upper())
    if info:
        return json.dumps(info.to_flat_dict(), indent=2)
    return json.dumps({"error": "not cached", "symbol": symbol.upper()})


@mcp.resource("ticker://ticker/{symbol}/metrics")
def get_ticker_metrics(symbol: str) -> str:
    """Get only metrics (returns, volatility) for a cached ticker."""
    info = _cache.get(symbol.upper())
    if info:
        if info.metrics.has_data():
            return json.dumps({
                "symbol": symbol.upper(),
                "returns": info.metrics.returns,
                "volatility": info.metrics.volatility,
                "updated_at": info.metrics.updated_at.isoformat() if info.metrics.updated_at else None,
            }, indent=2)
        return json.dumps({"error": "no metrics", "symbol": symbol.upper()})
    return json.dumps({"error": "not cached", "symbol": symbol.upper()})


@mcp.resource("ticker://indexes")
def list_indexes() -> str:
    """List available market indexes."""
    return json.dumps(list(INDEX_URLS.keys()))


@mcp.resource("ticker://indexes/{index}")
def get_index_tickers(index: str) -> str:
    """Get all cached ticker info for an index."""
    index_name = INDEX_ALIASES.get(index.lower(), index.lower())
    if index_name not in INDEX_URLS:
        return json.dumps({"error": f"Unknown index: {index}. Available: {list(INDEX_URLS.keys())}"})

    tickers = _fetch_index_tickers(index_name)
    result = {}
    for ticker in tickers:
        info = _cache.get(ticker)
        if info:
            result[ticker] = info.to_flat_dict()

    return json.dumps(result, indent=2)


# =============================================================================
# MCP Tools
# =============================================================================

@mcp.tool()
def lookup(query: str) -> str:
    """Unified lookup: ticker, comma-list, index name, or company name.

    Examples:
        lookup("AAPL")           - single ticker
        lookup("AAPL,MSFT,GOOGL") - multiple tickers
        lookup("sp500")          - all S&P 500 tickers
        lookup("Apple")          - search by name, return matches

    Always caches results. Returns dict of symbol -> info.
    """
    query = query.strip()
    results: dict[str, dict] = {}

    # Check if it's an index name
    if _is_index_name(query):
        tickers = _fetch_index_tickers(query)
        for ticker in tickers:
            info = _lookup_single(ticker)
            if info:
                results[ticker] = info.to_flat_dict()
        return json.dumps(results, indent=2)

    # Split by comma for multiple queries
    queries = [q.strip() for q in query.split(",") if q.strip()]

    for q in queries:
        q_upper = q.upper()

        # If it looks like a ticker, try direct lookup first
        if _looks_like_ticker(q_upper):
            info = _lookup_single(q_upper)
            if info:
                results[q_upper] = info.to_flat_dict()
                continue
            # Direct lookup failed - fall through to search

        # Search by company name (or ticker that wasn't found)
        matches = _search_yahoo(q, max_results=10)
        if matches:
            for symbol, quote in matches.items():
                cached = _cache.get(symbol)
                if cached and not cached.is_static_stale():
                    results[symbol] = cached.to_flat_dict()
                else:
                    meta = _get_chart_meta(symbol)
                    info = _build_ticker_info(symbol, quote, meta)
                    _cache.put(info)
                    results[symbol] = info.to_flat_dict()
        elif _looks_like_ticker(q_upper):
            # Only report "not found" for ticker-like queries with no search results
            results[q_upper] = {"error": "not found", "symbol": q_upper}

    return json.dumps(results, indent=2)


@mcp.tool()
def refresh_metrics(symbols: str) -> str:
    """Fetch 1-year prices and calculate annualized return/volatility.

    Args:
        symbols: Comma-separated tickers OR index name (sp500, nasdaq100, dow)

    Metrics are cached for 7 days. This fetches fresh data and recalculates.
    """
    symbols = symbols.strip()
    ticker_list: list[str] = []

    # Check if it's an index name
    if _is_index_name(symbols):
        ticker_list = _fetch_index_tickers(symbols)
    else:
        ticker_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]

    results: dict[str, dict] = {}

    for ticker in ticker_list:
        # Ensure ticker is cached first
        info = _cache.get(ticker)
        if not info or info.is_static_stale():
            info = _lookup_single(ticker)

        if not info:
            results[ticker] = {"error": "ticker not found"}
            continue

        # Fetch historical prices and calculate metrics
        prices = _fetch_historical_prices(ticker)
        if prices:
            metrics = _calculate_metrics(prices)
            if metrics:
                info.metrics = metrics
                _cache.put(info)
                _cache.ops.record_metrics_refresh()
                results[ticker] = {
                    "returns": metrics.returns,
                    "volatility": metrics.volatility,
                    "name": info.name,
                    "sector": info.sector,
                }
            else:
                results[ticker] = {"error": "insufficient price data"}
        else:
            results[ticker] = {"error": "could not fetch prices"}

    return json.dumps(results, indent=2)


# =============================================================================
# Main
# =============================================================================

def main():
    mcp.run(transport=_args.transport)


if __name__ == "__main__":
    main()
