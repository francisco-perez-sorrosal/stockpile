"""Shared constants for the ticker-cache MCP server."""

from pathlib import Path

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
