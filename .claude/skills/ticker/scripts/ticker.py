#!/usr/bin/env python3
"""Stock ticker lookup tool using Yahoo Finance API directly.

No external dependencies - uses only Python standard library.
Converts company names to ticker symbols with disambiguation support.
"""

import argparse
import json
import sys
import urllib.request
import urllib.parse
from dataclasses import dataclass
from typing import Literal

# Yahoo Finance API endpoints
SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"
CHART_URL = "https://query2.finance.yahoo.com/v8/finance/chart"

# Required headers to avoid 403 errors
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}


@dataclass
class TickerResult:
    symbol: str
    name: str
    exchange: str
    type: str
    sector: str | None = None
    industry: str | None = None
    market_cap: int | None = None


@dataclass
class SearchResponse:
    status: Literal["found", "ambiguous", "not_found"]
    results: list[TickerResult]
    query: str


def _fetch_json(url: str) -> dict | None:
    """Fetch JSON from URL with proper headers."""
    try:
        request = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


def _search_yahoo(query: str, max_results: int = 10) -> list[dict]:
    """Search Yahoo Finance for matching symbols."""
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
        return []

    quotes = data.get("quotes", [])
    # Filter to only items with a symbol
    return [q for q in quotes if q.get("symbol")]


def _get_chart_meta(symbol: str) -> dict | None:
    """Get chart metadata for a symbol (includes price and basic info)."""
    url = f"{CHART_URL}/{symbol}?interval=1d&range=1d"
    data = _fetch_json(url)

    if not data:
        return None

    result = data.get("chart", {}).get("result", [])
    if not result:
        return None

    return result[0].get("meta", {})


def format_market_cap(value: int | None) -> str | None:
    if value is None:
        return None
    if value >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:.2f}T"
    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    return f"${value:,}"


def search_by_name(query: str, include_details: bool = False, max_results: int = 10) -> SearchResponse:
    """Search for ticker symbols by company name."""
    quotes = _search_yahoo(query, max_results)

    if not quotes:
        return SearchResponse(status="not_found", results=[], query=query)

    results = []
    for quote in quotes:
        result = TickerResult(
            symbol=quote.get("symbol", ""),
            name=quote.get("shortname") or quote.get("longname", ""),
            exchange=quote.get("exchange", ""),
            type=quote.get("quoteType", ""),
            # Search results include sector/industry
            sector=quote.get("sector"),
            industry=quote.get("industry"),
        )

        # For market cap, we need to call the chart endpoint
        if include_details and result.symbol:
            meta = _get_chart_meta(result.symbol)
            if meta:
                result.market_cap = meta.get("marketCap")

        results.append(result)

    status = "found" if len(results) == 1 else "ambiguous"
    return SearchResponse(status=status, results=results, query=query)


def lookup_by_ticker(symbol: str) -> SearchResponse:
    """Validate a ticker and get company information."""
    # Use search to find the exact symbol - this gives us sector/industry
    quotes = _search_yahoo(symbol, max_results=20)

    # Find exact match
    exact_match = None
    for q in quotes:
        if q.get("symbol", "").upper() == symbol.upper():
            exact_match = q
            break

    if not exact_match:
        # Fallback to chart endpoint for basic info
        meta = _get_chart_meta(symbol)
        if not meta:
            return SearchResponse(status="not_found", results=[], query=symbol)

        result = TickerResult(
            symbol=symbol.upper(),
            name=meta.get("shortName") or meta.get("longName", ""),
            exchange=meta.get("exchangeName", ""),
            type=meta.get("instrumentType", ""),
            market_cap=meta.get("marketCap"),
        )
    else:
        result = TickerResult(
            symbol=exact_match.get("symbol", symbol.upper()),
            name=exact_match.get("shortname") or exact_match.get("longname", ""),
            exchange=exact_match.get("exchange", ""),
            type=exact_match.get("quoteType", ""),
            sector=exact_match.get("sector"),
            industry=exact_match.get("industry"),
        )

        # Get market cap from chart endpoint
        meta = _get_chart_meta(result.symbol)
        if meta:
            result.market_cap = meta.get("marketCap")

    return SearchResponse(status="found", results=[result], query=symbol)


def format_output(response: SearchResponse, include_details: bool = False) -> str:
    """Format response as JSON."""
    output = {
        "status": response.status,
        "query": response.query,
        "results": [],
    }

    for r in response.results:
        item = {
            "symbol": r.symbol,
            "name": r.name,
            "exchange": r.exchange,
            "type": r.type,
        }
        if include_details or r.sector or r.industry or r.market_cap:
            if r.sector:
                item["sector"] = r.sector
            if r.industry:
                item["industry"] = r.industry
            if r.market_cap:
                item["market_cap"] = format_market_cap(r.market_cap)

        output["results"].append(item)

    return json.dumps(output, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Look up stock ticker symbols from company names"
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Company name to search for",
    )
    parser.add_argument(
        "--ticker", "-t",
        metavar="SYMBOL",
        help="Look up by ticker symbol instead of company name",
    )
    parser.add_argument(
        "--details", "-d",
        action="store_true",
        help="Include additional company details (sector, industry, market cap)",
    )
    parser.add_argument(
        "--max-results", "-n",
        type=int,
        default=10,
        help="Maximum number of results to return (default: 10)",
    )

    args = parser.parse_args()

    if not args.query and not args.ticker:
        parser.error("Either a company name or --ticker SYMBOL is required")

    if args.ticker:
        response = lookup_by_ticker(args.ticker)
        include_details = True
    else:
        response = search_by_name(args.query, args.details, args.max_results)
        include_details = args.details

    print(format_output(response, include_details))
    return 0 if response.status != "not_found" else 1


if __name__ == "__main__":
    sys.exit(main())
