"""
API rate limiting, monthly budget tracking, and search-result caching.

Keeps the pipeline inside provider limits at 450+ scene scale:
  - Pexels:  200 requests/hour, 20,000/month
  - Pixabay: 100 requests/60 seconds

All state that must survive a run (monthly counts, search cache) lives in
cache/ as JSON. The cache directory is gitignored.
"""

import json
import time
from collections import deque
from datetime import datetime
from pathlib import Path

CACHE_DIR = Path("cache")
SEARCH_CACHE_TTL = 24 * 3600  # seconds


class RateLimiter:
    """
    Sliding-window rate limiter: allows at most max_requests per
    window_seconds, sleeping just long enough when the window is full.
    """

    def __init__(self, name: str, max_requests: int, window_seconds: float):
        self.name = name
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._stamps: deque[float] = deque()

    def wait(self) -> None:
        """Block until a request is allowed, then record it."""
        while True:
            now = time.monotonic()
            while self._stamps and now - self._stamps[0] >= self.window_seconds:
                self._stamps.popleft()
            if len(self._stamps) < self.max_requests:
                break
            sleep_for = self.window_seconds - (now - self._stamps[0]) + 0.1
            print(f"  [{self.name}] window full "
                  f"({self.max_requests}/{self.window_seconds:.0f}s) — "
                  f"sleeping {sleep_for:.0f}s")
            time.sleep(sleep_for)
        self._stamps.append(time.monotonic())


class MonthlyBudget:
    """
    Persistent per-calendar-month request counter. Survives across runs so
    long projects don't blow through a monthly API quota unnoticed.
    """

    def __init__(self, provider: str, limit: int, path: Path = CACHE_DIR / "api_usage.json"):
        self.provider = provider
        self.limit = limit
        self.path = path

    def _month_key(self) -> str:
        return datetime.now().strftime("%Y-%m")

    def _load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def used(self) -> int:
        return self._load().get(self.provider, {}).get(self._month_key(), 0)

    def increment(self) -> None:
        """Count one request; raise if the monthly budget is exhausted."""
        data = self._load()
        month = self._month_key()
        count = data.setdefault(self.provider, {}).get(month, 0) + 1
        if count > self.limit:
            raise RuntimeError(
                f"{self.provider} monthly budget exhausted: {count - 1}/{self.limit} "
                f"requests used in {month}. Wait for the new month or raise the limit."
            )
        data[self.provider][month] = count
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self.path)


class SearchCache:
    """
    JSON-file cache of search results keyed by "provider:query", with a TTL
    (default 24h). Repeated/similar keywords across hundreds of scenes hit
    the cache instead of re-querying the API.
    """

    def __init__(self, path: Path = CACHE_DIR / "search_cache.json",
                 ttl: float = SEARCH_CACHE_TTL):
        self.path = path
        self.ttl = ttl
        self._data: dict = {}
        if path.exists():
            try:
                self._data = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def get(self, key: str) -> list | None:
        """Return cached results (possibly an empty list) or None on miss."""
        entry = self._data.get(key)
        if entry is None:
            return None
        if time.time() - entry["ts"] > self.ttl:
            del self._data[key]
            return None
        return entry["results"]

    def put(self, key: str, results: list) -> None:
        self._data[key] = {"ts": time.time(), "results": results}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data))
        tmp.replace(self.path)
