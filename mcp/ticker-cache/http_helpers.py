"""HTTP helper functions for fetching JSON and HTML content."""

import gzip
import json
import urllib.request

from constants import HEADERS


def fetch_json(url: str) -> dict | None:
    """Fetch JSON from a URL, returning None on any error."""
    try:
        request = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


def fetch_html(url: str) -> str:
    """Fetch HTML from a URL, decompressing gzip if needed."""
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read()
        if len(data) >= 2 and data[:2] == b"\x1f\x8b":
            data = gzip.decompress(data)
        return data.decode("utf-8")
