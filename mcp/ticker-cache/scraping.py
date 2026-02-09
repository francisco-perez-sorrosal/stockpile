"""Wikipedia index scraping: extract stock tickers from index constituent pages."""

import re

from constants import INDEX_ALIASES, INDEX_URLS
from http_helpers import fetch_html


def extract_table_column(html: str, header_pattern: str, table_limit: int = 0) -> list[str]:
    """Extract a column from a Wikipedia-style HTML table by matching the header pattern."""
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
                if value and not value.startswith('\u2014'):
                    values.append(value)

        if table_limit > 0 and len(values) >= table_limit:
            continue

        if values:
            return values

    return []


def fetch_index_tickers(index: str) -> list[str]:
    """Fetch tickers from a market index."""
    index = INDEX_ALIASES.get(index.lower(), index.lower())
    if index not in INDEX_URLS:
        return []

    html = fetch_html(INDEX_URLS[index])

    if index == "sp500":
        return extract_table_column(html, r'^Symbol$')
    elif index == "nasdaq100":
        return extract_table_column(html, r'^Ticker$')
    elif index == "dow":
        return extract_table_column(html, r'^Symbol$', table_limit=35)

    return []


def is_index_name(query: str) -> bool:
    """Check if a query string matches a known index name or alias."""
    return query.lower() in INDEX_ALIASES
