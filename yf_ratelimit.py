"""
yf_ratelimit.py  ·  Universal yfinance Rate-Limit Shield
=========================================================
Drop-in replacement / umbrella for ALL yfinance calls in this app.

HOW IT WORKS
------------
1. curl_cffi Chrome-impersonation session  →  bypasses Yahoo's bot filters
2. Streamlit @st.cache_data               →  deduplicate identical calls (1-hr TTL)
3. Exponential back-off with jitter       →  survive transient 429s
4. In-process LRU memory cache            →  zero-network hits for repeat symbols
5. Concurrency throttle (1 req/sec)       →  stay under Yahoo's rate budget

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

HUGGING FACE SPACES NOTES
--------------------------
• curl_cffi is the #1 fix for HF Spaces / Streamlit Cloud rate limits.
  Add to requirements.txt:  curl_cffi>=0.6.2
• The module auto-falls-back to requests if curl_cffi is absent.
• No secrets or env-vars required — works out of the box.
"""

from __future__ import annotations

import functools
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

logger.warning("yf_ratelimit: curl_cffi available = %s (install curl_cffi if False -- "
                "the single biggest fix for Streamlit Cloud / HF Spaces 429s)", _HAS_CURL)

# ────────────────────────────────────────────────────────────────────────────
# CONFIG  (tune here if needed)
# ────────────────────────────────────────────────────────────────────────────
MIN_DELAY_S      = 1.1    # minimum pause between Yahoo requests (bumped from 0.8 --
                          # Streamlit Cloud's free-tier egress IP is shared across many
                          # other apps, so it gets throttled harder than a dedicated IP)
MAX_DELAY_S      = 3.2    # maximum pause (random jitter)
MAX_RETRIES      = 3      # retry budget per call
BASE_BACKOFF_S   = 4.0    # base for exponential backoff on 429 (bumped from 3.0)
CACHE_TTL_S      = 3600   # in-process cache TTL (1 hour)
COOLDOWN_S       = 35.0   # shared pause applied to ALL threads after any 429 OR a
                          # silent empty-response block (bumped from 20.0 -- see the
                          # empty-DataFrame note in _with_retry: on a shared/free-tier
                          # IP, Yahoo silently blocking shows up as an empty response
                          # far more often than an actual 429 status code, so that case
                          # now triggers the same shared cooldown a real 429 does)
REQUEST_TIMEOUT_S = 15.0  # hard ceiling on any single HTTP call to Yahoo -- see
                          # _get_thread_session() below. Without this, a stalled/half-open
                          # TCP connection blocks its worker thread FOREVER. With a
                          # fixed-size ThreadPoolExecutor, enough of these pile up and
                          # every worker ends up wedged on a dead socket at once -- the
                          # scan just stops advancing mid-run, at a different, seemingly
                          # arbitrary stock count each time. This is the actual root
                          # cause of scans "getting stuck" partway through -- never
                          # about which ticker, always about how many stalled sockets
                          # happen to accumulate before every worker thread is occupied.

def configure(max_retries: int | None = None, base_backoff: float | None = None,
              min_delay: float | None = None, cooldown: float | None = None) -> None:
    """Let callers (the Streamlit UI's Retry/Backoff sliders) actually change
    retry behaviour at runtime. Previously the UI sliders for this existed
    but were never wired to anything real -- every call silently used the
    hardcoded MAX_RETRIES/BASE_BACKOFF_S above regardless of what the sidebar
    said. This makes those controls do what they claim to do.
    Safe to call at the start of every scan (cheap global reassignment).
    """
    global MAX_RETRIES, BASE_BACKOFF_S, MIN_DELAY_S, COOLDOWN_S
    if max_retries is not None:
        MAX_RETRIES = max(1, int(max_retries))
    if base_backoff is not None:
        BASE_BACKOFF_S = max(0.1, float(base_backoff))
    if min_delay is not None:
        MIN_DELAY_S = max(0.05, float(min_delay))
    if cooldown is not None:
        COOLDOWN_S = max(1.0, float(cooldown))
_CHROME_UA       = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# ────────────────────────────────────────────────────────────────────────────
# RATE-LIMIT GATE  (shared across all threads)
# ────────────────────────────────────────────────────────────────────────────
_gate_lock       = threading.Lock()
_last_request_ts = 0.0
_cooldown_until  = 0.0   # monotonic timestamp; all threads wait until this passes

# ────────────────────────────────────────────────────────────────────────────
# LIVE STATUS  (for the app UI, not just server logs)
# ────────────────────────────────────────────────────────────────────────────
# Everything below used to only reach a Python logger.warning() -- fine for
# grepping Streamlit Cloud's server log viewer after the fact, but invisible
# in the app itself while a scan is running. scanner_common's live-status
# panel polls this during a scan so 429s/cooldowns/retries show up in the UI
# in real time, not just in a log file the user has to go copy-paste out.
from collections import deque

_EVENT_LOG_MAX = 200
_event_log: "deque[dict]" = deque(maxlen=_EVENT_LOG_MAX)
_event_lock = threading.Lock()
_inflight = 0
_inflight_lock = threading.Lock()


def _log_event(message: str, level: str = "info") -> None:
    with _event_lock:
        _event_log.append({"ts": time.time(), "level": level, "message": message})


def get_recent_events(n: int = 15) -> list[dict]:
    """Most recent rate-limit-level events (429s, cooldowns, empty-response
    retries), oldest first. For display in the app's live-status panel."""
    with _event_lock:
        return list(_event_log)[-n:]


def clear_events() -> None:
    with _event_lock:
        _event_log.clear()


def get_status() -> dict:
    """Snapshot of the shared rate-limit gate, for a live status widget:
    is a cooldown active right now (and for how much longer), and how many
    Yahoo requests are actually in flight across all worker threads."""
    now = time.monotonic()
    with _gate_lock:
        cooldown_remaining = max(0.0, _cooldown_until - now)
    with _inflight_lock:
        inflight = _inflight
    return {
        "cooldown_active": cooldown_remaining > 0,
        "cooldown_remaining": cooldown_remaining,
        "inflight": inflight,
        "min_delay_s": MIN_DELAY_S,
    }


def _throttle():
    """Block until at least MIN_DELAY_S has elapsed since the last call,
    AND until any active shared cooldown (see _trigger_cooldown) has expired.
    """
    global _last_request_ts
    with _gate_lock:
        now = time.monotonic()
        if now < _cooldown_until:
            time.sleep(_cooldown_until - now)
            now = time.monotonic()
        wait = MIN_DELAY_S - (now - _last_request_ts)
        if wait > 0:
            time.sleep(wait)
        _last_request_ts = time.monotonic()


def _trigger_cooldown(seconds: float | None = None):
    """Called by any thread that hits a real 429 (or a silent empty-response
    block, see _with_retry). Pushes _cooldown_until forward so every other
    thread's next _throttle() call also pauses -- instead of 6 threads
    independently backing off and retrying into each other, they all go
    quiet together and come back once, staggered by the normal MIN_DELAY_S
    gate. This is what actually stops the retry storm that used to cascade
    into an 80+ minute stall.

    NOTE: `seconds` used to default to COOLDOWN_S at function-definition time,
    which froze in the value COOLDOWN_S had at import -- so configure()
    changing COOLDOWN_S later had no effect on this default. Reading the
    global at call time instead so configure() actually takes effect.
    """
    global _cooldown_until
    if seconds is None:
        seconds = COOLDOWN_S
    with _gate_lock:
        target = time.monotonic() + seconds
        if target > _cooldown_until:
            _cooldown_until = target
    logger.warning("yf_ratelimit: 429 detected -- cooling down ALL threads for %.0fs", seconds)
    _log_event(f"🧊 Cooling down all workers for {seconds:.0f}s", "cooldown")


# ────────────────────────────────────────────────────────────────────────────
# SESSION FACTORY  (one session per WORKER THREAD, not per symbol)
# ────────────────────────────────────────────────────────────────────────────
# Originally a fresh session was created per _CachedTicker (i.e. per symbol),
# on the theory that a new session per symbol stops Yahoo tracking connection
# state across stocks. In practice a curl_cffi session is a live libcurl
# handle with its own TLS/connection-pool buffers, and a scan covering
# hundreds-to-thousands of symbols was creating that many of them and never
# releasing the old ones (see _ticker_registry below) -- a real memory-growth
# source on a constrained free-tier instance. One session per thread (there
# are only as many of those as the "Parallel workers" slider allows) bounds
# this to a small constant no matter how large the scan is, while still
# rotating identity across the handful of worker threads.
_thread_local = threading.local()


def _get_thread_session():
    sess = getattr(_thread_local, "session", None)
    if sess is not None:
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
    # Wrapping .request() here so ANY call path (yfinance internals included)
    # gets a real ceiling, without having to touch yfinance's own code.
    _orig_request = sess.request

    def _request_with_timeout(method, url, *args, **kwargs):
        kwargs.setdefault("timeout", REQUEST_TIMEOUT_S)
        return _orig_request(method, url, *args, **kwargs)

    sess.request = _request_with_timeout
    _thread_local.session = sess
    return sess


# ────────────────────────────────────────────────────────────────────────────
# IN-PROCESS MEMORY CACHE  (survives across Streamlit reruns in same process)
# ────────────────────────────────────────────────────────────────────────────
# Previously an unbounded dict -- holds the actual DataFrames per symbol per
# property (history, financials, balance_sheet, ...), so a long full-universe
# scan grew this without limit for the rest of the process's life (only a
# restart clears it). A full-universe scan never revisits the same symbol
# twice in one run, so this cache does nothing useful for that case anyway --
# it only helps repeated lookups of the same symbol in a short window
# (dashboard refreshes, resume flows, retry-failed). Sized for that, not for
# holding the whole universe, with real LRU eviction so it can't grow past it.
_MEM_CACHE_MAX = 150
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
    """
    Call fn(*args, **kwargs) with throttle + exponential backoff on errors.
    Returns the result or raises the last exception after MAX_RETRIES.
    """
    global _inflight
    last_exc = None
    for attempt in range(MAX_RETRIES):
        _throttle()
        jitter = random.uniform(0, MAX_DELAY_S - MIN_DELAY_S)
        if attempt:
            backoff = BASE_BACKOFF_S * (2 ** (attempt - 1)) + jitter
            logger.warning("yf_ratelimit: retry %d/%d — waiting %.1fs",
                           attempt, MAX_RETRIES, backoff)
            _log_event(f"⏳ Retry {attempt}/{MAX_RETRIES} — waiting {backoff:.1f}s", "retry")
            time.sleep(backoff)
        with _inflight_lock:
            _inflight += 1
        try:
            result = fn(*args, **kwargs)
            # yf.download returns a DataFrame; empty == likely rate-limited.
            # On a shared/free-tier IP (Streamlit Cloud, HF Spaces, Render
            # free tier), Yahoo silently throttling shows up as an empty
            # response far more often than an actual 429 status -- so this
            # now triggers the same shared cooldown a real 429 does, instead
            # of only retrying locally on this one thread while every other
            # worker thread kept hammering Yahoo through the same block at
            # full speed. That mismatch is what used to turn a normal
            # rate-limit into a scan-wide crawl.
            if isinstance(result, pd.DataFrame) and result.empty and attempt < MAX_RETRIES - 1:
                logger.warning("yf_ratelimit: empty DataFrame on attempt %d — retrying", attempt + 1)
                _log_event(f"⚠️ Empty response (attempt {attempt + 1}) — possible silent throttle", "warning")
                last_exc = RuntimeError("Empty DataFrame returned (possible silent 429)")
                _trigger_cooldown()
                continue
            return result
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            is_rate = any(x in msg for x in ("429", "rate", "too many", "forbidden", "403"))
            if not is_rate:
                # Non-rate-limit error — don't keep retrying
                raise
            logger.warning("yf_ratelimit: rate-limit hit on attempt %d: %s", attempt + 1, exc)
            _log_event(f"🚫 429 rate limit (attempt {attempt + 1}/{MAX_RETRIES})", "warning")
            _trigger_cooldown()  # tell every other thread to back off too
        finally:
            with _inflight_lock:
                _inflight -= 1


    raise last_exc or RuntimeError("yf_ratelimit: all retries exhausted")


# ────────────────────────────────────────────────────────────────────────────
# PUBLIC API  ── safe_ticker() and safe_download()
# ────────────────────────────────────────────────────────────────────────────

class _CachedTicker:
    """
    Lazy, cached wrapper around yf.Ticker.  All property accesses are
    cached in-process and retried on rate-limit errors.
    """
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

    # -- lazy yf.Ticker construction -----------------------------------------
    def _get_yf(self) -> yf.Ticker:
        with self._yf_lock:
            if self._yf_obj is None:
                sess = _get_thread_session()
                self._yf_obj = yf.Ticker(self._symbol, session=sess)
        return self._yf_obj

    # -- generic cached property fetch ----------------------------------------
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

    # -- expose all standard yf.Ticker properties transparently --------------
    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._PROPS:
            return self._fetch_prop(name)
        # Pass through anything else (e.g. .ticker, .isin)
        return getattr(self._get_yf(), name)

    # -- history() — supports arbitrary kwargs --------------------------------
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

    # -- option_chain() -------------------------------------------------------
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

    # -- repr / str -----------------------------------------------------------
    def __repr__(self):
        return f"<CachedTicker '{self._symbol}'>"

    def __str__(self):
        return self._symbol


# -- module-level Ticker cache (bounded LRU, NOT one object per symbol ------
#    forever) -----------------------------------------------------------
# Previously unbounded: every symbol a scan ever touched stayed in this dict
# for the rest of the process's life (only a restart clears it), and each
# entry holds a _CachedTicker plus everything _mem_cache has cached for it.
# A full-universe scan never revisits the same symbol twice in one run, so
# none of that caching does anything useful for that case -- it's pure
# accumulation, and on a memory-constrained free-tier instance it's a real
# way to get OOM-killed partway through a large scan. Bounded with real LRU
# eviction now -- sized for "helps repeated lookups of the same symbol in a
# short window" (dashboard refreshes, resume flows), not "hold the whole
# universe in memory at once."
_TICKER_REGISTRY_MAX = 300
_ticker_registry: "OrderedDict[str, _CachedTicker]" = OrderedDict()
_registry_lock   = threading.Lock()

# Printed once at import time (process boot) so a stale/uncached deploy is
# visible directly in Streamlit Cloud's log viewer instead of having to infer
# it from scan behaviour after the fact -- grep the log for "yf_ratelimit
# CONFIG" right after a deploy finishes. If these numbers don't match this
# file, the platform is still running an old build.
logger.warning(
    "yf_ratelimit CONFIG: MIN_DELAY_S=%.1f MAX_DELAY_S=%.1f COOLDOWN_S=%.1f "
    "BASE_BACKOFF_S=%.1f REQUEST_TIMEOUT_S=%.1f TICKER_REGISTRY_MAX=%d MEM_CACHE_MAX=%d",
    MIN_DELAY_S, MAX_DELAY_S, COOLDOWN_S, BASE_BACKOFF_S, REQUEST_TIMEOUT_S,
    _TICKER_REGISTRY_MAX, _MEM_CACHE_MAX,
)

def safe_ticker(symbol: str) -> _CachedTicker:
    """
    Drop-in for yf.Ticker(symbol).

    Returns a cached, rate-limit-aware wrapper.  The same object is reused
    across calls with the same symbol within a short window; least-recently-
    used symbols are evicted once _TICKER_REGISTRY_MAX is exceeded so a long
    scan can't grow this without bound.

    Usage:
        from yf_ratelimit import safe_ticker
        t = safe_ticker("RELIANCE.NS")
        print(t.info["currentPrice"])
        df = t.history(period="1y")
    """
    with _registry_lock:
        existing = _ticker_registry.get(symbol)
        if existing is not None:
            _ticker_registry.move_to_end(symbol)
            return existing
        ticker = _CachedTicker(symbol)
        _ticker_registry[symbol] = ticker
        while len(_ticker_registry) > _TICKER_REGISTRY_MAX:
            _ticker_registry.popitem(last=False)
        return ticker


def safe_download(
    tickers,
    period: str = "1mo",
    interval: str = "1d",
    flatten: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Drop-in for yf.download(tickers, ...).

    Extra args vs yf.download:
        flatten (bool): If True (default), flatten MultiIndex columns to
                        single-level "Close", "Open" … instead of
                        ("Close", "AAPL") etc.  Matches pre-0.2 behaviour.

    Usage:
        from yf_ratelimit import safe_download
        df = safe_download("RELIANCE.NS", period="1y")
        df = safe_download(["RELIANCE.NS", "TCS.NS"], start="2023-01-01")
    """
    # Build a stable cache key
    ticker_key = tickers if isinstance(tickers, str) else "|".join(sorted(tickers))
    key = f"download:{ticker_key}:{period}:{interval}:{sorted(kwargs.items())}"
    cached = _mem_get(key)
    if cached is not None:
        return cached

    def _do():
        sess = _get_thread_session()
        return yf.download(
            tickers,
            period=period,
            interval=interval,
            session=sess,
            progress=False,
            **kwargs,
        )

    df = _with_retry(_do)

    # Flatten MultiIndex columns (yfinance >= 0.2 wraps single-ticker downloads too)
    if flatten and isinstance(df.columns, pd.MultiIndex):
        if isinstance(tickers, str) or (isinstance(tickers, (list, tuple)) and len(tickers) == 1):
            df.columns = df.columns.get_level_values(0)
        # For multi-ticker downloads keep MultiIndex — caller can handle it

    _mem_set(key, df)
    return df


# ────────────────────────────────────────────────────────────────────────────
# STREAMLIT CACHE LAYER  (optional — adds cross-rerun deduplication)
# Only activated when Streamlit is present (i.e. inside a Streamlit app)
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
# QUICK SELF-TEST  (run with: python yf_ratelimit.py)
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
