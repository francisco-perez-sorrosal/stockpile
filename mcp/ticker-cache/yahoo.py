"""Yahoo Finance API: search, chart metadata, price history, and metrics calculation."""

import math
import re
import urllib.parse
from datetime import datetime, timezone

from cache import TickerCache
from constants import CHART_URL, SEARCH_URL
from http_helpers import fetch_json
from models import MarketData, MetricsData, TickerInfo


def search_yahoo(query: str, max_results: int = 10) -> dict[str, dict]:
    """Search Yahoo Finance for matching quotes."""
    params = urllib.parse.urlencode({
        "q": query, "quotesCount": max_results, "newsCount": 0,
        "listsCount": 0, "enableFuzzyQuery": "true",
    })
    data = fetch_json(f"{SEARCH_URL}?{params}")
    if not data:
        return {}
    return {q["symbol"]: q for q in data.get("quotes", []) if q.get("symbol")}


def get_chart_meta(symbol: str) -> dict | None:
    """Fetch chart metadata (current price, exchange, etc.) for a symbol."""
    data = fetch_json(f"{CHART_URL}/{symbol}?interval=1d&range=1d")
    if not data:
        return None
    result = data.get("chart", {}).get("result", [])
    return result[0].get("meta", {}) if result else None


def extract_market_data(meta: dict) -> MarketData:
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


def fetch_historical_prices(symbol: str, period: str = "1y") -> list[float] | None:
    """Fetch historical adjusted close prices for a symbol."""
    data = fetch_json(f"{CHART_URL}/{symbol}?interval=1d&range={period}")
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


MINIMUM_PRICE_POINTS = 20
TRADING_DAYS_PER_YEAR = 252


def calculate_metrics(prices: list[float]) -> MetricsData | None:
    """Calculate annualized return and volatility from price series."""
    if not prices or len(prices) < MINIMUM_PRICE_POINTS:
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
        returns=mean_return * TRADING_DAYS_PER_YEAR,
        volatility=std_return * math.sqrt(TRADING_DAYS_PER_YEAR),
        updated_at=datetime.now(timezone.utc),
    )


def build_ticker_info(symbol: str, quote: dict | None = None, meta: dict | None = None) -> TickerInfo:
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
    market = extract_market_data(meta) if meta else MarketData()

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


MAX_TICKER_LENGTH = 5


def looks_like_ticker(s: str) -> bool:
    """Heuristic: tickers are 1-5 chars, uppercase, alphanumeric.

    Real stock tickers are typically 1-4 characters (NYSE, NASDAQ).
    Some ETFs and preferred shares go up to 5. Anything longer is
    likely a company name (e.g., MODERNA, NVIDIA).
    """
    s = s.strip()
    if not s or len(s) > MAX_TICKER_LENGTH:
        return False
    return bool(re.match(r'^[A-Z0-9][A-Z0-9.\-]*$', s))


def lookup_single(symbol: str, cache: TickerCache) -> TickerInfo | None:
    """Look up a single ticker, fetching from Yahoo if not cached or stale."""
    symbol = symbol.upper()

    cached = cache.get(symbol)
    if cached and not cached.is_static_stale():
        return cached

    quotes = search_yahoo(symbol, max_results=20)
    quote = quotes.get(symbol, {})
    meta = get_chart_meta(symbol)

    if quote or meta:
        info = build_ticker_info(symbol, quote, meta)
        cache.put(info)
        return info

    return None
