#!/usr/bin/env python3
"""Stock ticker lookup tool using Yahoo Finance API directly.

No external dependencies - uses only Python standard library.
Converts company names to ticker symbols with disambiguation support.
Also fetches index constituents (S&P 500, NASDAQ-100, Dow Jones).

Caches ticker info to ~/.cache/ticker/tickers.json for performance.
"""

import argparse
import gzip
import json
import re
import sys
import urllib.request
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

# Yahoo Finance API endpoints
SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"
CHART_URL = "https://query2.finance.yahoo.com/v8/finance/chart"

# Wikipedia URLs for index constituents
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
NASDAQ100_URL = "https://en.wikipedia.org/wiki/NASDAQ-100"
DOWJONES_URL = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"

# Cache configuration
CACHE_DIR = Path.home() / ".cache" / "ticker"
CACHE_FILE = CACHE_DIR / "tickers.json"

# Required headers to avoid 403 errors
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}


class TickerInfo:
    """Stock ticker information with separate static and market data.

    Static fields (rarely change, cached 30 days):
        symbol, name, long_name, exchange, type, sector, industry, currency, first_trade_date

    Market fields (change daily, cached until next trading day):
        price, previous_close, volume, day_high, day_low, week_52_high, week_52_low

    Timestamps:
        updated_at - when static data was last fetched
        market_updated_at - when market data was last fetched
    """

    STATIC_FIELDS = frozenset({
        "symbol", "name", "long_name", "exchange", "type",
        "sector", "industry", "currency", "first_trade_date",
    })

    MARKET_FIELDS = frozenset({
        "price", "previous_close", "volume",
        "day_high", "day_low", "week_52_high", "week_52_low",
    })

    def __init__(self, data: dict | None = None):
        self._data: dict = data.copy() if data else {}

    # --- Identity ---
    @property
    def symbol(self) -> str:
        return self._data.get("symbol", "")

    @property
    def name(self) -> str:
        return self._data.get("name", "")

    @property
    def long_name(self) -> str:
        return self._data.get("long_name", "")

    # --- Classification ---
    @property
    def exchange(self) -> str:
        return self._data.get("exchange", "")

    @property
    def type(self) -> str:
        return self._data.get("type", "")

    @property
    def sector(self) -> str:
        return self._data.get("sector", "")

    @property
    def industry(self) -> str:
        return self._data.get("industry", "")

    @property
    def currency(self) -> str:
        return self._data.get("currency", "")

    @property
    def first_trade_date(self) -> str:
        return self._data.get("first_trade_date", "")

    # --- Market Data ---
    @property
    def price(self) -> float | None:
        return self._data.get("price")

    @property
    def previous_close(self) -> float | None:
        return self._data.get("previous_close")

    @property
    def volume(self) -> int | None:
        return self._data.get("volume")

    @property
    def day_high(self) -> float | None:
        return self._data.get("day_high")

    @property
    def day_low(self) -> float | None:
        return self._data.get("day_low")

    @property
    def week_52_high(self) -> float | None:
        return self._data.get("week_52_high")

    @property
    def week_52_low(self) -> float | None:
        return self._data.get("week_52_low")

    # --- Timestamps ---
    @property
    def updated_at(self) -> str:
        """When static data was last fetched."""
        return self._data.get("updated_at", "")

    @property
    def market_updated_at(self) -> str:
        """When market data was last fetched."""
        return self._data.get("market_updated_at", "")

    # --- Serialization ---
    def to_dict(self) -> dict:
        """Convert to dict for JSON/database serialization."""
        return self._data.copy()

    def to_summary(self) -> dict:
        """Return minimal dict for display (name, sector, industry)."""
        return {
            "name": self.name,
            "sector": self.sector,
            "industry": self.industry,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TickerInfo":
        """Create from dict (cache or API response)."""
        return cls(data)

    @classmethod
    def empty(cls, symbol: str) -> "TickerInfo":
        """Create placeholder for unknown ticker."""
        return cls({"symbol": symbol, "name": symbol})

    # --- Update Methods ---
    def update_static(self, data: dict):
        """Update static fields from data dict."""
        for field in self.STATIC_FIELDS:
            if field in data and data[field]:
                self._data[field] = data[field]
        self._data["updated_at"] = datetime.now(timezone.utc).isoformat()

    def update_market(self, data: dict):
        """Update market fields from data dict."""
        for field in self.MARKET_FIELDS:
            if field in data:
                self._data[field] = data[field]
        self._data["market_updated_at"] = datetime.now(timezone.utc).isoformat()

    def needs_static_refresh(self, max_age_days: int = 30) -> bool:
        """Check if static data is stale."""
        if not self.updated_at:
            return True
        try:
            updated = datetime.fromisoformat(self.updated_at.replace("Z", "+00:00"))
            age = datetime.now(timezone.utc) - updated
            return age.days > max_age_days
        except ValueError:
            return True

    def needs_market_refresh(self, max_age_hours: int = 24) -> bool:
        """Check if market data is stale."""
        if not self.market_updated_at:
            return True
        try:
            updated = datetime.fromisoformat(self.market_updated_at.replace("Z", "+00:00"))
            age = datetime.now(timezone.utc) - updated
            return age.total_seconds() > max_age_hours * 3600
        except ValueError:
            return True

    def __repr__(self) -> str:
        return f"TickerInfo({self.symbol!r}, {self.name!r})"


# --- Cache Management ---

class TickerCache:
    """File-based cache for ticker information."""

    def __init__(self, cache_file: Path = CACHE_FILE):
        self.cache_file = cache_file
        self._cache: dict[str, dict] = {}
        self._loaded = False

    def _ensure_loaded(self):
        """Load cache from file if not already loaded."""
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
        """Save cache to file."""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, "w") as f:
            json.dump(self._cache, f, indent=2)

    def get(self, symbol: str, check_static: bool = True) -> TickerInfo | None:
        """Get ticker info from cache.

        Args:
            symbol: Ticker symbol
            check_static: If True, return None if static data is stale
        """
        self._ensure_loaded()
        symbol = symbol.upper()
        if symbol not in self._cache:
            return None

        data = self._cache[symbol]
        data["symbol"] = symbol  # Ensure symbol is in data
        info = TickerInfo.from_dict(data)

        if check_static and info.needs_static_refresh():
            # DEBUG
            print(f"[cache] {symbol}: stale", file=sys.stderr)
            return None

        # DEBUG
        print(f"[cache] {symbol}: hit", file=sys.stderr)
        return info

    def get_many(self, symbols: list[str]) -> dict[str, TickerInfo]:
        """Get multiple tickers from cache. Returns only cached ones with fresh static data."""
        result = {}
        for symbol in symbols:
            info = self.get(symbol)
            if info:
                result[symbol.upper()] = info
        return result

    def put(self, info: TickerInfo):
        """Store ticker info in cache."""
        self._ensure_loaded()
        self._cache[info.symbol.upper()] = info.to_dict()
        self._save()

    def put_many(self, infos: dict[str, TickerInfo]):
        """Store multiple ticker infos in cache."""
        self._ensure_loaded()
        for symbol, info in infos.items():
            self._cache[symbol.upper()] = info.to_dict()
        self._save()

    def all(self) -> dict[str, TickerInfo]:
        """Return all cached tickers."""
        self._ensure_loaded()
        result = {}
        for sym, data in self._cache.items():
            data["symbol"] = sym
            result[sym] = TickerInfo.from_dict(data)
        return result


# Global cache instance
_cache = TickerCache()


# --- HTTP Helpers ---

def _fetch_json(url: str) -> dict | None:
    """Fetch JSON from URL with proper headers."""
    try:
        request = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


def _fetch_html(url: str) -> str:
    """Fetch HTML from URL with proper headers, handling gzip."""
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read()
        if len(data) >= 2 and data[:2] == b"\x1f\x8b":
            data = gzip.decompress(data)
        return data.decode("utf-8")


# --- Wikipedia Index Scraping ---

def _extract_table_column(html_content: str, header_pattern: str, table_limit: int = 0) -> list[str]:
    """Extract values from a table column matching the header pattern."""
    table_pattern = re.compile(r'<table[^>]*class="[^"]*wikitable[^"]*"[^>]*>(.*?)</table>', re.DOTALL | re.IGNORECASE)
    tables = table_pattern.findall(html_content)

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
                if link_match:
                    value = link_match.group(1)
                else:
                    value = re.sub(r'<[^>]+>', '', cell)
                value = value.strip().replace('\n', '').replace('.', '-')
                if value and not value.startswith('—'):
                    values.append(value)

        if table_limit > 0 and len(values) >= table_limit:
            continue

        if values:
            return values

    return []


def fetch_index_tickers(index: str) -> list[str]:
    """Fetch tickers from major market indexes."""
    index = index.lower()

    if index in ("sp500", "s&p500", "sp"):
        html = _fetch_html(SP500_URL)
        tickers = _extract_table_column(html, r'^Symbol$')
    elif index in ("nasdaq100", "nasdaq", "ndx"):
        html = _fetch_html(NASDAQ100_URL)
        tickers = _extract_table_column(html, r'^Ticker$')
    elif index in ("dow", "dowjones", "djia"):
        html = _fetch_html(DOWJONES_URL)
        tickers = _extract_table_column(html, r'^Symbol$', table_limit=35)
    else:
        raise ValueError(f"Unknown index: {index}. Use 'sp500', 'nasdaq100', or 'dow'")

    return [t for t in tickers if t]


# --- Yahoo Finance API ---

def _search_yahoo(query: str, max_results: int = 10) -> dict[str, dict]:
    """Search Yahoo Finance. Returns dict of symbol -> raw quote data."""
    params = urllib.parse.urlencode({
        "q": query,
        "quotesCount": max_results,
        "newsCount": 0,
        "listsCount": 0,
        "enableFuzzyQuery": "true",
    })
    url = f"{SEARCH_URL}?{params}"
    data = _fetch_json(url)

    if not data:
        return {}

    result = {}
    for q in data.get("quotes", []):
        symbol = q.get("symbol")
        if symbol:
            result[symbol] = q
    return result


def _get_chart_meta(symbol: str) -> dict | None:
    """Get chart metadata for a symbol."""
    url = f"{CHART_URL}/{symbol}?interval=1d&range=1d"
    data = _fetch_json(url)

    if not data:
        return None

    result = data.get("chart", {}).get("result", [])
    if not result:
        return None

    return result[0].get("meta", {})


def _extract_market_data(meta: dict) -> dict:
    """Extract market data fields from chart metadata."""
    return {
        "price": meta.get("regularMarketPrice"),
        "previous_close": meta.get("chartPreviousClose"),
        "volume": meta.get("regularMarketVolume"),
        "day_high": meta.get("regularMarketDayHigh"),
        "day_low": meta.get("regularMarketDayLow"),
        "week_52_high": meta.get("fiftyTwoWeekHigh"),
        "week_52_low": meta.get("fiftyTwoWeekLow"),
    }


def _build_ticker_info(symbol: str, quote: dict | None = None, meta: dict | None = None) -> TickerInfo:
    """Build TickerInfo from Yahoo search quote and/or chart meta.

    Args:
        symbol: Ticker symbol
        quote: Data from search endpoint (has sector/industry)
        meta: Data from chart endpoint (has market data)
    """
    quote = quote or {}
    meta = meta or {}

    # Build static data dict
    static_data = {
        "symbol": symbol,
        "name": quote.get("shortname") or quote.get("longname") or meta.get("shortName") or meta.get("longName", ""),
        "long_name": quote.get("longname") or meta.get("longName", ""),
        "exchange": quote.get("exchange") or meta.get("exchangeName", ""),
        "type": quote.get("quoteType") or meta.get("instrumentType", ""),
        "sector": quote.get("sector", ""),
        "industry": quote.get("industry", ""),
        "currency": meta.get("currency", ""),
    }

    # Convert first trade date from timestamp to ISO format
    first_trade = meta.get("firstTradeDate")
    if first_trade:
        try:
            static_data["first_trade_date"] = datetime.fromtimestamp(first_trade, timezone.utc).date().isoformat()
        except (ValueError, OSError):
            pass

    # Create TickerInfo and update with timestamps
    info = TickerInfo(static_data)
    info.update_static({})  # Set updated_at timestamp
    if meta:
        market_data = _extract_market_data(meta)
        if any(v is not None for v in market_data.values()):
            info.update_market(market_data)

    return info


# --- Lookup Functions ---

def _looks_like_ticker(s: str) -> bool:
    """Heuristic: tickers are short, uppercase, alphanumeric."""
    s = s.strip()
    if not s or len(s) > 10:
        return False
    return bool(re.match(r'^[A-Z0-9][A-Z0-9.\-]*$', s))


def lookup_ticker(symbol: str, use_cache: bool = True, refresh_market: bool = False) -> TickerInfo | None:
    """Look up a single ticker symbol. Returns None if not found.

    Args:
        symbol: Ticker symbol
        use_cache: Use cached static data if fresh
        refresh_market: Force refresh of market data even if cached

    Fetches both search (for sector/industry) and chart (for market data) endpoints
    to get the most complete information.
    """
    symbol = symbol.upper()

    # Check cache first
    if use_cache:
        cached = _cache.get(symbol)
        if cached:
            # If market refresh requested and market data is stale, update it
            if refresh_market and cached.needs_market_refresh():
                meta = _get_chart_meta(symbol)
                if meta:
                    cached.update_market(_extract_market_data(meta))
                    _cache.put(cached)
            return cached

    # Try search endpoint first (has sector/industry)
    quotes = _search_yahoo(symbol, max_results=20)
    quote = quotes.get(symbol, {})

    # Always fetch chart endpoint for market data
    meta = _get_chart_meta(symbol)

    if quote or meta:
        info = _build_ticker_info(symbol, quote, meta)
        _cache.put(info)
        return info

    return None


def lookup_name(query: str, max_results: int = 10) -> dict[str, TickerInfo]:
    """Look up by company name. Returns dict of matching symbols."""
    quotes = _search_yahoo(query, max_results)
    result = {}
    to_cache = {}

    for symbol, quote in quotes.items():
        # Check cache first
        cached = _cache.get(symbol)
        if cached:
            result[symbol] = cached
        else:
            info = _build_ticker_info(symbol, quote)
            result[symbol] = info
            to_cache[symbol] = info

    if to_cache:
        _cache.put_many(to_cache)

    return result


def lookup_many(queries: list[str], use_cache: bool = True, refresh_market: bool = False) -> dict[str, TickerInfo]:
    """Look up multiple queries (tickers or names).

    Args:
        queries: List of tickers or company names
        use_cache: Use cached static data if fresh
        refresh_market: Force refresh of market data even if cached
    """
    result: dict[str, TickerInfo] = {}

    # Separate tickers from names
    tickers = []
    names = []
    for q in queries:
        q = q.strip()
        if not q:
            continue
        if _looks_like_ticker(q):
            tickers.append(q.upper())
        else:
            names.append(q)

    # Batch lookup tickers - check cache first
    if use_cache and not refresh_market:
        cached = _cache.get_many(tickers)
        result.update(cached)
        tickers = [t for t in tickers if t not in cached]
    elif use_cache and refresh_market:
        # Still use cache but need to refresh market data
        cached = _cache.get_many(tickers)
        for symbol, info in cached.items():
            if info.needs_market_refresh():
                meta = _get_chart_meta(symbol)
                if meta:
                    info.update_market(_extract_market_data(meta))
                    _cache.put(info)
            result[symbol] = info
        tickers = [t for t in tickers if t not in cached]

    # Fetch remaining tickers
    for ticker in tickers:
        info = lookup_ticker(ticker, use_cache=False, refresh_market=refresh_market)
        if info:
            result[info.symbol] = info
        else:
            result[ticker] = TickerInfo.empty(ticker)

    # Look up names
    for name in names:
        matches = lookup_name(name)
        if matches:
            # Take first match
            first_symbol = next(iter(matches))
            result[first_symbol] = matches[first_symbol]

    return result


# --- Output Formatting ---

def format_output(results: dict[str, TickerInfo]) -> str:
    """Format results as JSON dict of symbol -> info."""
    output = {}
    for symbol, info in results.items():
        output[symbol] = {
            "name": info.name,
            "sector": info.sector,
            "industry": info.industry,
        }
    return json.dumps(output, indent=2)


# --- CLI ---

def main() -> int:
    parser = argparse.ArgumentParser(description="Look up stock ticker symbols")
    parser.add_argument(
        "query",
        nargs="?",
        help="Ticker, company name, or comma-separated list",
    )
    parser.add_argument(
        "--index", "-i",
        metavar="NAME",
        help="Index: 'sp500', 'nasdaq100', 'dow' (comma-separated for multiple)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Skip cache and fetch fresh data",
    )
    parser.add_argument(
        "--refresh-market",
        action="store_true",
        help="Refresh market data (price, volume, etc.); without query refreshes all cached",
    )
    parser.add_argument(
        "--candidates", "-c",
        type=int,
        default=5,
        help="Max ticker candidates when resolving company names (default: 5)",
    )

    args = parser.parse_args()

    all_tickers: list[str] = []

    # Collect tickers from indexes
    if args.index:
        indexes = [idx.strip() for idx in args.index.split(",") if idx.strip()]
        for idx in indexes:
            try:
                tickers = fetch_index_tickers(idx)
                all_tickers.extend(tickers)
            except ValueError as e:
                print(json.dumps({"error": str(e)}), file=sys.stderr)
                return 1

    # Collect tickers from query
    if args.query:
        queries = [q.strip() for q in args.query.split(",") if q.strip()]
        for q in queries:
            if _looks_like_ticker(q):
                all_tickers.append(q.upper())
            else:
                # Name search - get first result's ticker
                matches = lookup_name(q, max_results=args.candidates)
                if matches:
                    all_tickers.append(next(iter(matches)))

    if not all_tickers:
        if args.refresh_market:
            # Refresh all cached tickers
            all_tickers = list(_cache.all().keys())
            if not all_tickers:
                print(json.dumps({}))
                return 0
        elif not args.index and not args.query:
            parser.error("Provide a query and/or --index")
        else:
            print(json.dumps({}))
            return 0

    # Deduplicate while preserving order
    seen = set()
    unique_tickers = []
    for t in all_tickers:
        if t not in seen:
            seen.add(t)
            unique_tickers.append(t)

    # Look up all tickers (uses cache)
    results = lookup_many(unique_tickers, use_cache=not args.no_cache, refresh_market=args.refresh_market)

    print(format_output(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
