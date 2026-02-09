"""MCP tool handlers for ticker lookup and metrics refresh.

Tools:
    lookup(query)           - Unified: ticker, comma-list, index name, or company name
    refresh_metrics(symbols) - Calculate return/volatility from 1y prices
"""

import json

from app import cache, mcp
from scraping import fetch_index_tickers, is_index_name
from yahoo import (
    build_ticker_info,
    calculate_metrics,
    fetch_historical_prices,
    get_chart_meta,
    lookup_single,
    looks_like_ticker,
    search_yahoo,
)


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
    if is_index_name(query):
        tickers = fetch_index_tickers(query)
        for ticker in tickers:
            info = lookup_single(ticker, cache)
            if info:
                results[ticker] = info.to_flat_dict()
        return json.dumps(results, indent=2)

    # Split by comma for multiple queries
    queries = [q.strip() for q in query.split(",") if q.strip()]

    for q in queries:
        q_upper = q.upper()

        # If it looks like a ticker, try direct lookup first
        if looks_like_ticker(q_upper):
            info = lookup_single(q_upper, cache)
            if info:
                results[q_upper] = info.to_flat_dict()
                continue
            # Direct lookup failed - fall through to search

        # Search by company name (or ticker that wasn't found)
        matches = search_yahoo(q, max_results=10)
        if matches:
            for symbol, quote in matches.items():
                cached = cache.get(symbol)
                if cached and not cached.is_static_stale():
                    results[symbol] = cached.to_flat_dict()
                else:
                    meta = get_chart_meta(symbol)
                    info = build_ticker_info(symbol, quote, meta)
                    cache.put(info)
                    results[symbol] = info.to_flat_dict()
        elif looks_like_ticker(q_upper):
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
    if is_index_name(symbols):
        ticker_list = fetch_index_tickers(symbols)
    else:
        ticker_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]

    results: dict[str, dict] = {}

    for ticker in ticker_list:
        # Ensure ticker is cached first
        info = cache.get(ticker)
        if not info or info.is_static_stale():
            info = lookup_single(ticker, cache)

        if not info:
            results[ticker] = {"error": "ticker not found"}
            continue

        # Fetch historical prices and calculate metrics
        prices = fetch_historical_prices(ticker)
        if prices:
            metrics = calculate_metrics(prices)
            if metrics:
                info.metrics = metrics
                cache.put(info)
                cache.ops.record_metrics_refresh()
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
