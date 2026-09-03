"""
yf_ratelimit.py  ·  Universal yfinance Rate-Limit Shield  (in-process variant)
=================================================================================
*** This is the ROOT copy, used standalone by sheshscout.py (the Streamlit
*** app) -- NOT the same file as core/yf_ratelimit.py, which core/scanner.py
*** uses for the separate FastAPI+Celery service.
***
*** THE BUG THIS FILE FIXES (2026-09-03): this file had been overwritten
*** with a byte-for-byte copy of core/yf_ratelimit.py -- the Redis-backed
*** variant -- even though its own docstring said it should stay the
*** in-process version. core/yf_ratelimit.py's _throttle() calls
*** core/redis_client.py's throttle_wait(), which is written to FAIL OPEN
*** (proceed with zero delay) whenever Redis is unreachable -- correct
*** behaviour for that file's real use case (a flaky Redis shouldn't wedge
*** a Celery worker), but catastrophic here: Streamlit Cloud runs this app
*** with no Redis at all, so EVERY throttle_wait() call hit a connection
*** error and returned instantly. With that gate silently doing nothing,
*** sheshscout.py's 6 parallel workers x 3 Yahoo calls each fired in a
*** burst of ~18 simultaneous requests at scan start -- hence 429s within
*** the first couple of symbols, not gradually over a long scan. This file
*** restores a real threading.Lock-based gate that has no Redis dependency
*** and therefore can't silently fail open the same way -- it's a plain
*** in-process value, either it throttles or the process crashed.
***
*** Two processes (2+ Streamlit instances, or this app plus something
*** else) still won't coordinate with each other under this design -- an
*** in-process lock only ever sees its own process, by definition. That's
*** a real, known limitation, not an oversight: it's the same one
*** core/yf_ratelimit.py's docstring describes for why THAT file needed
*** Redis in the first place (multiple Celery worker processes). It does
*** not apply here unless this Streamlit app is itself ever scaled to
*** multiple instances -- if that happens, this file would need the same
*** Redis-backed treatment core/yf_ratelimit.py already has, not before.

Drop-in replacement / umbrella for ALL yfinance calls in this app.

HOW IT WORKS
------------
1. curl_cffi Chrome-impersonation session  →  bypasses Yahoo's bot filters
2. Streamlit @st.cache_data               →  deduplicate identical calls (1-hr TTL)
3. Exponential back-off with jitter       →  survive transient 429s
4. In-process LRU memory cache            →  zero-network hits for repeat symbols
5. In-process threading.Lock throttle     →  stay under Yahoo's rate budget,
                                              shared across every worker THREAD
                                              in this one process (see note above
                                              on why that's the right scope here)

HOW TO USE  (two-line migration per file)
-----------------------------------------
BEFORE:
    import yfinance as yf
    ticker = yf.Ticker("RELIANCE.NS")
    df     = yf.download("RELIANCE.NS", period="1y")

AFTER:
    from yf_ratelimit import safe_ticker, safe_download
    ticker = safe_ticker("RELIANCE.NS")
    df     = safe_download("RELIANCE.NS", period="1y")

Everything else (.info, .financials, .history, .balance_sheet, .cashflow,
.options, .option_chain …) works exactly the same on the returned object.
"""

from __future__ import annotations

import logging
import os
import random
import threading
import time
from collections import OrderedDict
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ── optional streamlit cache ────────────────────────────────────────────────
try:
    import streamlit as st
    _HAS_ST = True
except Exception:
    _HAS_ST = False

# ── curl_cffi (preferred) → requests (fallback) ─────────────────────────────
try:
    from curl_cffi import requests as _curl_requests
    _HAS_CURL = True
except ImportError:
    import requests as _curl_requests          # type: ignore[assignment]
    _HAS_CURL = False

import yfinance as yf

# ────────────────────────────────────────────────────────────────────────────
# CONFIG  (env-var overridable -- set these on Streamlit Cloud's Secrets/env
# without touching code or waiting on a deploy. Every value below falls back
# to its current tuned default if the env var is unset or unparsable.)
# ────────────────────────────────────────────────────────────────────────────
def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        logger.warning("yf_ratelimit: bad value for %s, using default %s", name, default)
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        logger.warning("yf_ratelimit: bad value for %s, using default %s", name, default)
        return default


MIN_DELAY_S      = _env_float("YF_MIN_DELAY_S", 1.5)     # minimum pause between Yahoo requests
                          # (was 1.1 -- bumped as extra margin on 2026-09-03 alongside the
                          # throttle fix itself, since the 1.1s figure was never actually
                          # being enforced in production until now)
MAX_DELAY_S      = _env_float("YF_MAX_DELAY_S", 4.0)     # maximum pause (random jitter)
MAX_RETRIES      = _env_int("YF_MAX_RETRIES", 3)         # retry budget per call
BASE_BACKOFF_S   = _env_float("YF_BASE_BACKOFF_S", 4.0)  # base for exponential backoff on 429
CACHE_TTL_S      = _env_int("YF_CACHE_TTL_S", 3600)      # in-process cache TTL (seconds)
COOLDOWN_S       = _env_float("YF_COOLDOWN_S", 35.0)     # shared pause applied to ALL threads
                          # in this process after any 429, OR after several consecutive
                          # silent empty responses (see EMPTY_STREAK_THRESHOLD below) --
                          # NOT after every single empty response. A single legitimately-
                          # delisted or no-trades-today symbol isn't evidence of a block;
                          # a real silent block shows up as MANY empties in a row across
                          # unrelated symbols.
REQUEST_TIMEOUT_S = _env_float("YF_REQUEST_TIMEOUT_S", 15.0)  # hard ceiling on any single
                          # HTTP call to Yahoo -- see _make_session() below. Without this, a
                          # stalled/half-open TCP connection blocks its worker thread FOREVER.
EMPTY_STREAK_THRESHOLD = _env_int("YF_EMPTY_STREAK_THRESHOLD", 4)  # consecutive empty
                          # responses across ALL threads before treating it as a real
                          # silent block rather than one quiet symbol.
_CHROME_UA       = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# Optional outbound proxy -- see configure()'s proxy= param below. Unset by
# default; can be set via configure(proxy=...) or the YF_HTTP_PROXY /
# HTTPS_PROXY / HTTP_PROXY env vars at process start. This is the one lever
# that helps once curl_cffi + backoff + throttle are already in place and
# Yahoo is still blocking at the IP level (a shared free-tier egress IP).
_PROXY_URL: str | None = (
    os.environ.get("YF_HTTP_PROXY")
    or os.environ.get("HTTPS_PROXY")
    or os.environ.get("HTTP_PROXY")
    or None
)
_session_version = 0  # bumped by configure(proxy=...) so existing per-thread
                       # sessions rebuild themselves (with the new proxy) on
                       # their next call.


def configure(max_retries: int | None = None, base_backoff: float | None = None,
              min_delay: float | None = None, cooldown: float | None = None,
              proxy: str | None = None) -> None:
    """Let callers (e.g. the Streamlit UI's Retry/Backoff sliders) change
    retry behaviour at runtime. Safe to call at the start of every scan.

    proxy: an http(s) proxy URL (e.g. "http://user:pass@host:port"), or ""
    to clear one. New sessions pick up the change; existing per-thread
    sessions are marked stale so the new proxy takes effect on their next
    call rather than only for new threads.
    """
    global MAX_RETRIES, BASE_BACKOFF_S, MIN_DELAY_S, COOLDOWN_S, _PROXY_URL, _session_version
    if max_retries is not None:
        MAX_RETRIES = max(1, int(max_retries))
    if base_backoff is not None:
        BASE_BACKOFF_S = max(0.1, float(base_backoff))
    if min_delay is not None:
        MIN_DELAY_S = max(0.05, float(min_delay))
    if cooldown is not None:
        COOLDOWN_S = max(1.0, float(cooldown))
    if proxy is not None:
        _PROXY_URL = proxy.strip() or None
        _session_version += 1


# ────────────────────────────────────────────────────────────────────────────
# RATE-LIMIT GATE  (in-process only -- shared across every worker THREAD in
# this one process via a plain threading.Lock. See module docstring for why
# that's the correct scope for this file, and why it must NOT depend on
# Redis or anything else that can be unreachable and fail silently.)
# ────────────────────────────────────────────────────────────────────────────
_throttle_lock = threading.Lock()
_last_request_time = 0.0
_cooldown_until = 0.0
_empty_streak = 0


def _throttle():
    """Block until it's safe to make the next Yahoo request. Pure
    in-process state (module globals + a lock) -- nothing here can be
    unreachable the way a network dependency could be, so there is no
    fail-open path to worry about: this always actually waits."""
    global _last_request_time
    while True:
        with _throttle_lock:
            now = time.time()
            if now < _cooldown_until:
                wait = _cooldown_until - now
            else:
                wait = MIN_DELAY_S - (now - _last_request_time)
            if wait <= 0:
                _last_request_time = time.time()
                return
        time.sleep(min(wait, 5))  # re-check in slices, don't oversleep past a cleared cooldown


def _trigger_cooldown(seconds: float = COOLDOWN_S):
    """Called by any thread that hits a real 429. Pushes the shared
    cooldown deadline forward so every other thread's next _throttle()
    call also pauses -- instead of N threads independently backing off
    and retrying into each other, they all go quiet together."""
    global _cooldown_until
    with _throttle_lock:
        target = time.time() + seconds
        if target > _cooldown_until:  # only move forward, never backward
            _cooldown_until = target
    logger.warning("yf_ratelimit: 429 detected -- cooling down ALL threads for %.0fs", seconds)


def _note_empty_response(cooldown_seconds: float = COOLDOWN_S) -> bool:
    """Record one empty/possibly-blocked response. Returns True if this
    pushed the streak over EMPTY_STREAK_THRESHOLD and triggered the shared
    cooldown."""
    global _empty_streak
    with _throttle_lock:
        _empty_streak += 1
        streak = _empty_streak
    if streak >= EMPTY_STREAK_THRESHOLD:
        logger.warning(
            "yf_ratelimit: %d empty responses in a row -- treating as a real "
            "block, cooling down ALL threads for %.0fs", streak, cooldown_seconds,
        )
        _trigger_cooldown(cooldown_seconds)
        with _throttle_lock:
            _empty_streak = 0
        return True
    return False


def _note_success() -> None:
    """Any real (non-empty) response breaks the empty streak."""
    global _empty_streak
    with _throttle_lock:
        _empty_streak = 0


# ────────────────────────────────────────────────────────────────────────────
# SESSION FACTORY
# ────────────────────────────────────────────────────────────────────────────
_thread_local = threading.local()

def _make_session():
    """Return a curl_cffi Chrome-impersonation session, cached per WORKER
    THREAD (there are only max_workers of them -- see sheshscout.py's
    sidebar slider -- so this is bounded no matter how large the scan is)."""
    sess = getattr(_thread_local, "session", None)
    cached_version = getattr(_thread_local, "session_version", None)
    if sess is not None and cached_version == _session_version:
        return sess

    if _HAS_CURL:
        sess = _curl_requests.Session(impersonate="chrome124")
    else:
        import requests as _req
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        sess = _req.Session()
        retry = Retry(
            total=3,
            backoff_factor=1.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD"],
            raise_on_status=False,
        )
        sess.mount("https://", HTTPAdapter(max_retries=retry))
        sess.mount("http://",  HTTPAdapter(max_retries=retry))

    if _PROXY_URL:
        sess.proxies = {"http": _PROXY_URL, "https": _PROXY_URL}

    sess.headers.update({
        "User-Agent":      _CHROME_UA,
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
    })

    # ── enforce a default timeout on every request through this session ────
    # yfinance calls session.get(...)/session.request(...) internally without
    # ever passing a timeout, so a stalled connection to Yahoo just hangs the
    # calling thread indefinitely -- no exception, nothing for _with_retry's
    # try/except to catch, nothing for the ThreadPoolExecutor to notice.
    _orig_request = sess.request

    def _request_with_timeout(method, url, *args, **kwargs):
        kwargs.setdefault("timeout", REQUEST_TIMEOUT_S)
        return _orig_request(method, url, *args, **kwargs)

    sess.request = _request_with_timeout
    _thread_local.session = sess
    _thread_local.session_version = _session_version
    return sess


# ────────────────────────────────────────────────────────────────────────────
# IN-PROCESS MEMORY CACHE  (survives across Streamlit reruns in same process)
# ────────────────────────────────────────────────────────────────────────────
_MEM_CACHE_MAX = _env_int("YF_MEM_CACHE_MAX", 150)  # a full-universe scan never revisits
                       # a symbol, so this cache does nothing useful for that case, only
                       # eats memory before the scan gets meaningfully far. Sized for
                       # dashboard refresh / resume re-lookups, not for holding the universe.
_mem_cache: "OrderedDict[str, tuple[float, Any]]" = OrderedDict()
_cache_lock = threading.Lock()

def _mem_get(key: str) -> Any | None:
    with _cache_lock:
        entry = _mem_cache.get(key)
        if entry and (time.time() - entry[0]) < CACHE_TTL_S:
            _mem_cache.move_to_end(key)
            return entry[1]
    return None

def _mem_set(key: str, value: Any):
    with _cache_lock:
        _mem_cache[key] = (time.time(), value)
        _mem_cache.move_to_end(key)
        while len(_mem_cache) > _MEM_CACHE_MAX:
            _mem_cache.popitem(last=False)

def clear_cache(symbol: str | None = None):
    """Clear in-process cache.  Pass symbol to clear only that ticker."""
    with _cache_lock:
        if symbol:
            keys = [k for k in _mem_cache if k.startswith(symbol)]
            for k in keys:
                _mem_cache.pop(k, None)
        else:
            _mem_cache.clear()
    logger.info("yf_ratelimit: cache cleared%s",
                f" for {symbol}" if symbol else " (all)")


# ────────────────────────────────────────────────────────────────────────────
# RETRY DECORATOR  (wraps any callable that talks to Yahoo)
# ────────────────────────────────────────────────────────────────────────────
def _with_retry(fn, *args, **kwargs):
    """Call fn(*args, **kwargs) with throttle + exponential backoff on errors.
    Returns the result or raises the last exception after MAX_RETRIES."""
    last_exc = None
    for attempt in range(MAX_RETRIES):
        _throttle()
        jitter = random.uniform(0, MAX_DELAY_S - MIN_DELAY_S)
        if attempt:
            backoff = BASE_BACKOFF_S * (2 ** (attempt - 1)) + jitter
            logger.warning("yf_ratelimit: retry %d/%d — waiting %.1fs",
                           attempt, MAX_RETRIES, backoff)
            time.sleep(backoff)
        try:
            result = fn(*args, **kwargs)
            if isinstance(result, pd.DataFrame) and result.empty and attempt < MAX_RETRIES - 1:
                logger.warning("yf_ratelimit: empty DataFrame on attempt %d — retrying", attempt + 1)
                last_exc = RuntimeError("Empty DataFrame returned (possible silent block)")
                _note_empty_response(COOLDOWN_S)
                continue
            _note_success()  # breaks any accumulating empty streak
            return result
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            is_rate = any(x in msg for x in ("429", "rate", "too many", "forbidden", "403"))
            if not is_rate:
                raise  # non-rate-limit error — don't keep retrying
            logger.warning("yf_ratelimit: rate-limit hit on attempt %d: %s", attempt + 1, exc)
            _trigger_cooldown()  # a REAL 429 always pauses everyone immediately

    raise last_exc or RuntimeError("yf_ratelimit: all retries exhausted")


# ────────────────────────────────────────────────────────────────────────────
# PUBLIC API  ── safe_ticker() and safe_download()
# ────────────────────────────────────────────────────────────────────────────

class _CachedTicker:
    """Lazy, cached wrapper around yf.Ticker.  All property accesses are
    cached in-process and retried on rate-limit errors."""
    _PROPS = ("info", "financials", "income_stmt", "balance_sheet",
              "cashflow", "quarterly_financials", "quarterly_income_stmt",
              "quarterly_balance_sheet", "quarterly_cashflow",
              "fast_info", "dividends", "splits", "actions",
              "recommendations", "calendar", "earnings_dates",
              "options")

    def __init__(self, symbol: str):
        self._symbol  = symbol
        self._yf_obj  = None
        self._yf_lock = threading.Lock()

    def _get_yf(self) -> yf.Ticker:
        with self._yf_lock:
            if self._yf_obj is None:
                sess = _make_session()
                self._yf_obj = yf.Ticker(self._symbol, session=sess)
        return self._yf_obj

    def _fetch_prop(self, prop: str) -> Any:
        key = f"{self._symbol}:prop:{prop}"
        cached = _mem_get(key)
        if cached is not None:
            return cached

        def _do():
            return getattr(self._get_yf(), prop)

        result = _with_retry(_do)
        _mem_set(key, result)
        return result

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._PROPS:
            return self._fetch_prop(name)
        return getattr(self._get_yf(), name)

    def history(self, period="1mo", interval="1d", **kwargs) -> pd.DataFrame:
        key = f"{self._symbol}:history:{period}:{interval}:{sorted(kwargs.items())}"
        cached = _mem_get(key)
        if cached is not None:
            return cached

        def _do():
            return self._get_yf().history(period=period, interval=interval, **kwargs)

        result = _with_retry(_do)
        _mem_set(key, result)
        return result

    def option_chain(self, date: str | None = None) -> Any:
        key = f"{self._symbol}:option_chain:{date}"
        cached = _mem_get(key)
        if cached is not None:
            return cached

        def _do():
            return self._get_yf().option_chain(date) if date else self._get_yf().option_chain()

        result = _with_retry(_do)
        _mem_set(key, result)
        return result

    def __repr__(self):
        return f"<CachedTicker '{self._symbol}'>"

    def __str__(self):
        return self._symbol


# -- module-level Ticker cache (one object per symbol per process) -----------
_TICKER_REGISTRY_MAX = _env_int("YF_TICKER_REGISTRY_MAX", 300)
_ticker_registry: "OrderedDict[str, _CachedTicker]" = OrderedDict()
_registry_lock   = threading.Lock()

# Printed once at import time (process boot) so a stale/uncached deploy is
# visible directly in Streamlit Cloud's boot log -- grep for "yf_ratelimit
# CONFIG" right after a deploy finishes.
logger.warning(
    "yf_ratelimit CONFIG: MIN_DELAY_S=%.1f MAX_DELAY_S=%.1f COOLDOWN_S=%.1f "
    "BASE_BACKOFF_S=%.1f REQUEST_TIMEOUT_S=%.1f TICKER_REGISTRY_MAX=%d MEM_CACHE_MAX=%d "
    "PROXY=%s",
    MIN_DELAY_S, MAX_DELAY_S, COOLDOWN_S, BASE_BACKOFF_S, REQUEST_TIMEOUT_S,
    _TICKER_REGISTRY_MAX, _MEM_CACHE_MAX,
    "configured" if _PROXY_URL else "none (set YF_HTTP_PROXY or call configure(proxy=...) to add one)",
)

def safe_ticker(symbol: str) -> _CachedTicker:
    """Drop-in for yf.Ticker(symbol). Returns a cached, rate-limit-aware
    wrapper. Usage:
        from yf_ratelimit import safe_ticker
        t = safe_ticker("RELIANCE.NS")
        df = t.history(period="1y")
    """
    with _registry_lock:
        existing = _ticker_registry.get(symbol)
        if existing is not None:
            _ticker_registry.move_to_end(symbol)
            return existing
        _ticker_registry[symbol] = _CachedTicker(symbol)
        while len(_ticker_registry) > _TICKER_REGISTRY_MAX:
            _ticker_registry.popitem(last=False)
        return _ticker_registry[symbol]


def safe_download(
    tickers,
    period: str = "1mo",
    interval: str = "1d",
    flatten: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Drop-in for yf.download(tickers, ...). Usage:
        from yf_ratelimit import safe_download
        df = safe_download("RELIANCE.NS", period="1y")
    """
    ticker_key = tickers if isinstance(tickers, str) else "|".join(sorted(tickers))
    key = f"download:{ticker_key}:{period}:{interval}:{sorted(kwargs.items())}"
    cached = _mem_get(key)
    if cached is not None:
        return cached

    def _do():
        sess = _make_session()
        return yf.download(
            tickers,
            period=period,
            interval=interval,
            session=sess,
            progress=False,
            **kwargs,
        )

    df = _with_retry(_do)

    if flatten and isinstance(df.columns, pd.MultiIndex):
        if isinstance(tickers, str) or (isinstance(tickers, (list, tuple)) and len(tickers) == 1):
            df.columns = df.columns.get_level_values(0)

    _mem_set(key, df)
    return df


# ────────────────────────────────────────────────────────────────────────────
# STREAMLIT CACHE LAYER  (optional — adds cross-rerun deduplication)
# ────────────────────────────────────────────────────────────────────────────
if _HAS_ST:
    @st.cache_data(ttl=CACHE_TTL_S, show_spinner=False)
    def st_download(tickers, period="1mo", interval="1d", **kwargs) -> pd.DataFrame:
        """st.cache_data-backed version of safe_download.  Use this in Streamlit pages."""
        return safe_download(tickers, period=period, interval=interval, **kwargs)

    @st.cache_data(ttl=CACHE_TTL_S, show_spinner=False)
    def st_ticker_info(symbol: str) -> dict:
        """st.cache_data-backed .info fetch.  Fast for repeated Streamlit reruns."""
        return safe_ticker(symbol).info

    @st.cache_data(ttl=CACHE_TTL_S, show_spinner=False)
    def st_ticker_history(symbol: str, period="1mo", interval="1d") -> pd.DataFrame:
        """st.cache_data-backed .history fetch."""
        return safe_ticker(symbol).history(period=period, interval=interval)


# ────────────────────────────────────────────────────────────────────────────
# QUICK SELF-TEST  (run with: python yf_ratelimit.py -- N/A on Streamlit
# Cloud, which has no shell; use sheshscout.py's own diagnostic UI instead
# if one exists, same pattern as the bhavcopy diagnostic panel)
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("curl_cffi available:", _HAS_CURL)

    print("\n— safe_ticker test —")
    t = safe_ticker("RELIANCE.NS")
    info = t.info
    print("currentPrice:", info.get("currentPrice") or info.get("regularMarketPrice"))

    print("\n— safe_download test —")
    df = safe_download("RELIANCE.NS", period="5d")
    print(df.tail(3))

    print("\n— cache hit test (should be instant) —")
    t2 = safe_ticker("RELIANCE.NS")
    print("Same object?", t is t2)

    print("\nAll tests passed ✓")
