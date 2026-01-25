# Technical Reference

## Yahoo Finance API Endpoints

This skill uses Yahoo Finance's public (unofficial) API endpoints. No authentication required.

### Search Endpoint

```
GET https://query2.finance.yahoo.com/v1/finance/search
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `q` | Search query (company name or partial ticker) |
| `quotesCount` | Max number of quote results (default: 6) |
| `newsCount` | Max news results (set to 0 to disable) |
| `enableFuzzyQuery` | Enable fuzzy matching (true/false) |

**Example:**
```bash
curl "https://query2.finance.yahoo.com/v1/finance/search?q=apple&quotesCount=5"
```

**Response fields:**

```json
{
  "quotes": [
    {
      "symbol": "AAPL",
      "shortname": "Apple Inc.",
      "longname": "Apple Inc.",
      "exchange": "NMS",
      "quoteType": "EQUITY",
      "sector": "Technology",
      "industry": "Consumer Electronics"
    }
  ]
}
```

### Chart Endpoint

```
GET https://query2.finance.yahoo.com/v8/finance/chart/{symbol}
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `interval` | Data interval (1d, 1wk, 1mo) |
| `range` | Time range (1d, 5d, 1mo, 1y, max) |

**Example:**
```bash
curl "https://query2.finance.yahoo.com/v8/finance/chart/AAPL?interval=1d&range=1d"
```

**Response metadata:**

```json
{
  "chart": {
    "result": [{
      "meta": {
        "symbol": "AAPL",
        "shortName": "Apple Inc.",
        "longName": "Apple Inc.",
        "exchangeName": "NMS",
        "instrumentType": "EQUITY",
        "regularMarketPrice": 248.04,
        "marketCap": 3500000000000
      }
    }]
  }
}
```

## Exchange Codes

| Code | Exchange |
|------|----------|
| NMS | NASDAQ Global Select |
| NYQ | NYSE |
| PCX | NYSE Arca |
| BTS | BATS |
| PNK | Pink Sheets (OTC) |
| TOR | Toronto Stock Exchange |
| FRA | Frankfurt |
| GER | XETRA (Germany) |

## Quote Types

| Type | Description |
|------|-------------|
| EQUITY | Common stock |
| ETF | Exchange-traded fund |
| INDEX | Market index |
| MUTUALFUND | Mutual fund |
| CURRENCY | Currency pair |
| CRYPTOCURRENCY | Cryptocurrency |

## Rate Limits

Yahoo Finance does not publish official rate limits. The script includes:
- 10-second timeout per request
- User-Agent header to avoid 403 errors

For high-volume usage, consider adding delays between requests.

## Error Handling

The script silently handles errors and returns `not_found` status. Common issues:

- **403 Forbidden**: Missing or invalid User-Agent header
- **Timeout**: Network issues or Yahoo rate limiting
- **Empty results**: Invalid ticker or no matches
