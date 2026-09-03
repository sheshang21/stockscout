"""
scanner_common.py — Shared infrastructure for all 3 SheshScout modes
=======================================================================
Positional (Ultra-Strict long-term), Intraday Short, Intraday Long all
import from here so the three modes look, behave, and get maintained
identically outside of their own core stock-scoring logic.

Provides:
  • Mode-namespaced session_state / widget keys (fixes state bleeding
    between modes when the user switches the mode dropdown — this was
    the source of the "Streamlit auto-reset" symptom: two modes quietly
    sharing session_state keys like "scan_results").
  • A single global Yahoo concurrency gate + bulletproof_fetch, shared
    across modes because it's really about total simultaneous
    connections to Yahoo from this process, not about any one mode.
  • Disk-backed dead-symbol skip list, shared across modes (a delisted
    symbol is delisted no matter which scanner asks).
  • Disk-backed scan checkpoint/resume, one file per mode, so a killed
    process (Streamlit Cloud health-check restarts, OOM, etc.) never
    loses more than the in-flight batch — for all 3 modes, not just one.
  • Reusable sidebar widgets: exchange picker, scan-mode picker
    (Quick / Full / Slot-wise / Range / Custom — identical across all
    3 modes), rate-limiting controls, checkpoint/resume controls.
  • A generic concurrent scan runner: every mode supplies only its own
    "fetch + analyze one symbol" callable; progress bar, stats, batch
    pausing, checkpointing, retry-failed and the non-blocking auto
    refresh loop are implemented exactly once here.
  • CSV export helper that always includes the company Name column.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Callable, Iterable

import pandas as pd
import streamlit as st

import yf_ratelimit
from yf_ratelimit import safe_ticker as _rl_ticker, safe_download as _rl_download

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODE_POSITIONAL = "positional"
MODE_SHORT = "intraday_short"
MODE_LONG = "intraday_long"

MODE_LABELS = {
    MODE_POSITIONAL: "📊 Positional Scanner — Long-Term Value Investing",
    MODE_SHORT: "📉 Intraday Short Screener",
    MODE_LONG: "📈 Intraday Long (Buy) Screener",
}


# ── yfinance shim (rate-limit-safe, shared object for every mode) ───────────
class _YFShim:
    """Thin shim so `yf.Ticker(...)` / `yf.download(...)` calls in mode
    modules transparently go through yf_ratelimit's Chrome-impersonation +
    shared-cooldown + retry machinery."""

    @staticmethod
    def Ticker(symbol, **_):
        return _rl_ticker(symbol)

    @staticmethod
    def download(tickers_, **kwargs):
        return _rl_download(tickers_, **kwargs)


yf = _YFShim()


# ── Mode-namespaced keys ─────────────────────────────────────────────────
def sskey(mode_key: str, name: str) -> str:
    """Namespace a session_state / widget key by mode so switching the mode
    dropdown never lets one mode read or clobber another mode's state."""
    return f"{mode_key}__{name}"


def get_state(mode_key: str, name: str, default=None):
    return st.session_state.get(sskey(mode_key, name), default)


def set_state(mode_key: str, name: str, value) -> None:
    st.session_state[sskey(mode_key, name)] = value


def pop_state(mode_key: str, name: str, default=None):
    return st.session_state.pop(sskey(mode_key, name), default)


# ── Global concurrency gate (shared across all modes/threads) ───────────────
_YF_SEMAPHORE_COUNT = 6
_YF_SEMAPHORE = threading.Semaphore(_YF_SEMAPHORE_COUNT)
_SEM_LOCK = threading.Lock()


def _set_semaphore(count: int) -> None:
    global _YF_SEMAPHORE, _YF_SEMAPHORE_COUNT
    with _SEM_LOCK:
        if count != _YF_SEMAPHORE_COUNT:
            _YF_SEMAPHORE = threading.Semaphore(count)
            _YF_SEMAPHORE_COUNT = count


def bulletproof_fetch(func, *args, **kwargs):
    """Single-shot, semaphore-gated call. yf_ratelimit's _CachedTicker already
    retries every individual Yahoo call internally with its own exponential
    backoff before raising — retrying the *whole* fetch again here on top of
    that multiplies delays (outer x inner) while holding a worker slot the
    entire time, which is what used to stall scans for 80+ minutes under real
    rate-limiting. So: call once, catch, bail. The semaphore is only held for
    the single attempt, never across a sleep/backoff, so one slow symbol
    can't starve the other workers.
    """
    with _YF_SEMAPHORE:
        try:
            return func(*args, **kwargs)
        except Exception:
            return None


# ── Known-dead symbol cache (shared across modes, disk-backed) ──────────────
# A delisted symbol still triggers yf_ratelimit's full internal retry ladder
# on every fresh scan/restart unless we short-circuit it before any network
# call. Two independent empty results, at least an hour apart, are required
# before a symbol is treated as dead — a one-off rate-limit burst (many
# symbols empty at once, all within seconds of each other) never triggers a
# blacklist entry; only a symbol that's *still* empty on a later, separate
# scan will. Entries expire after 30 days in case a halted stock relists.
DEAD_SYMBOLS_PATH = os.path.join(_BASE_DIR, ".dead_symbols.json")
_DEAD_SYMBOLS_TTL = 30 * 24 * 3600
_DEAD_STRIKE_MIN_GAP = 3600
_DEAD_STRIKE_THRESHOLD = 2
_DEAD_SYMBOLS_CACHE: dict | None = None
_DEAD_SYMBOLS_LOCK = threading.Lock()


def _load_dead_symbols() -> dict:
    try:
        with open(DEAD_SYMBOLS_PATH, "r") as f:
            data = json.load(f)
        now = time.time()
        cleaned = {}
        for s, strikes in data.items():
            strikes = [t for t in strikes if now - t < _DEAD_SYMBOLS_TTL]
            if strikes:
                cleaned[s] = strikes
        return cleaned
    except Exception:
        return {}


def is_known_dead(symbol: str) -> bool:
    global _DEAD_SYMBOLS_CACHE
    with _DEAD_SYMBOLS_LOCK:
        if _DEAD_SYMBOLS_CACHE is None:
            _DEAD_SYMBOLS_CACHE = _load_dead_symbols()
        return len(_DEAD_SYMBOLS_CACHE.get(symbol, [])) >= _DEAD_STRIKE_THRESHOLD


def mark_dead_symbol(symbol: str) -> None:
    global _DEAD_SYMBOLS_CACHE
    with _DEAD_SYMBOLS_LOCK:
        if _DEAD_SYMBOLS_CACHE is None:
            _DEAD_SYMBOLS_CACHE = _load_dead_symbols()
        strikes = _DEAD_SYMBOLS_CACHE.get(symbol, [])
        now = time.time()
        if not strikes or (now - strikes[-1]) >= _DEAD_STRIKE_MIN_GAP:
            strikes.append(now)
            _DEAD_SYMBOLS_CACHE[symbol] = strikes
            try:
                with open(DEAD_SYMBOLS_PATH, "w") as f:
                    json.dump(_DEAD_SYMBOLS_CACHE, f)
            except Exception:
                pass


def clear_dead_symbols() -> None:
    global _DEAD_SYMBOLS_CACHE
    with _DEAD_SYMBOLS_LOCK:
        _DEAD_SYMBOLS_CACHE = {}
        try:
            os.remove(DEAD_SYMBOLS_PATH)
        except Exception:
            pass


def count_dead_symbols() -> int:
    return len(_load_dead_symbols())


# ── Generic thread-safe TTL data cache (mode + symbol + extra) ─────────────
_DATA_CACHE: dict = {}
_DATA_CACHE_LOCK = threading.Lock()


def cache_get(key: str, ttl_seconds: float):
    now = time.time()
    with _DATA_CACHE_LOCK:
        entry = _DATA_CACHE.get(key)
        if entry and (now - entry["ts"]) < ttl_seconds:
            return entry["data"]
    return None


def cache_set(key: str, data) -> None:
    with _DATA_CACHE_LOCK:
        _DATA_CACHE[key] = {"ts": time.time(), "data": data}


# ── Disk-based scan checkpoint (one file per mode, survives restarts) ──────
def _checkpoint_path(mode_key: str) -> str:
    return os.path.join(_BASE_DIR, f".scan_checkpoint_{mode_key}.json")


def universe_signature(yf_symbols: Iterable[str]) -> str:
    return hashlib.sha256(",".join(sorted(yf_symbols)).encode()).hexdigest()


def load_checkpoint(mode_key: str):
    try:
        with open(_checkpoint_path(mode_key), "r") as f:
            return json.load(f)
    except Exception:
        return None


def save_checkpoint(mode_key: str, signature: str, stocks_to_scan: list,
                     results: list, failed_symbols: list, scanned_symbols: list) -> None:
    try:
        with open(_checkpoint_path(mode_key), "w") as f:
            json.dump({
                "signature": signature,
                "stocks_to_scan": stocks_to_scan,
                "results": results,
                "failed_symbols": failed_symbols,
                "scanned_symbols": scanned_symbols,
            }, f)
    except Exception:
        pass  # e.g. read-only filesystem -- resume just won't be available


def clear_checkpoint(mode_key: str) -> None:
    try:
        os.remove(_checkpoint_path(mode_key))
    except Exception:
        pass


# ── Sidebar: exchange + universe picker ─────────────────────────────────────
def render_exchange_selector(mode_key: str):
    """Returns (scan_nse, scan_bse, universe) where universe is the combined
    list[TickerRecord] (with company names) for whatever's selected."""
    import tickers as _tickers

    st.sidebar.subheader("📈 Select Exchanges to Scan")
    scan_nse = st.sidebar.checkbox("✅ Scan NSE Stocks", value=True,
                                    help="Loads tickers + names from nse_tickers.csv, adds .NS suffix",
                                    key=sskey(mode_key, "scan_nse"))
    scan_bse = st.sidebar.checkbox("✅ Scan BSE Stocks", value=True,
                                    help="Loads BSE codes + names from bse_codes.csv, adds .BO suffix",
                                    key=sskey(mode_key, "scan_bse"))

    if not scan_nse and not scan_bse:
        st.sidebar.error("⚠️ Please select at least one exchange!")
        return scan_nse, scan_bse, []

    universe = _tickers.load_universe(scan_nse, scan_bse)
    nse_count = sum(1 for r in universe if r["exchange"] == "NSE")
    bse_count = sum(1 for r in universe if r["exchange"] == "BSE")

    if universe:
        parts = []
        if scan_nse:
            parts.append(f"NSE: {nse_count}")
        if scan_bse:
            parts.append(f"BSE: {bse_count}")
        st.sidebar.success(f"✅ Loaded {len(universe)} stocks\n" + " | ".join(parts))
    else:
        st.sidebar.error("❌ No stocks loaded — check nse_tickers.csv / bse_codes.csv are present")

    return scan_nse, scan_bse, universe


# ── Sidebar: scan-mode picker (Quick / Full / Slot-wise / Range / Custom) ──
def render_scan_mode_selector(mode_key: str, universe: list) -> list:
    """Returns stocks_to_scan: list[TickerRecord]. Identical mechanics across
    every mode — this is the "replicate everything except the core logic"
    piece the positional scanner already had."""
    import tickers as _tickers

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔎 Scan Mode")

    scan_mode = st.sidebar.radio(
        "Scan Mode",
        ["Quick Scan (50 stocks)", "Full Scan (All stocks)", "Slot-wise Scan", "Range Scan", "Custom List"],
        key=sskey(mode_key, "scan_mode"),
    )

    if scan_mode == "Quick Scan (50 stocks)":
        return universe[:50]

    if scan_mode == "Full Scan (All stocks)":
        return universe

    if scan_mode == "Range Scan":
        st.sidebar.subheader("📐 Range Scan Settings")
        st.sidebar.info("Enter the row range (1-based) from the loaded NSE/BSE list to scan.")

        nse_records = [r for r in universe if r["exchange"] == "NSE"]
        bse_records = [r for r in universe if r["exchange"] == "BSE"]
        range_stocks = []

        if nse_records:
            st.sidebar.markdown(f"**NSE** — {len(nse_records)} stocks available")
            c1, c2 = st.sidebar.columns(2)
            with c1:
                nse_from = st.number_input("NSE From", min_value=1, max_value=len(nse_records),
                                            value=1, step=1, key=sskey(mode_key, "range_nse_from"))
            with c2:
                nse_to = st.number_input("NSE To", min_value=1, max_value=len(nse_records),
                                          value=min(100, len(nse_records)), step=1,
                                          key=sskey(mode_key, "range_nse_to"))
            if nse_from > nse_to:
                st.sidebar.error("NSE 'From' must be ≤ 'To'")
            else:
                sl = nse_records[(nse_from - 1):nse_to]
                range_stocks.extend(sl)
                st.sidebar.success(f"NSE: rows {nse_from}–{nse_to} → {len(sl)} stocks")

        if bse_records:
            st.sidebar.markdown(f"**BSE** — {len(bse_records)} stocks available")
            c3, c4 = st.sidebar.columns(2)
            with c3:
                bse_from = st.number_input("BSE From", min_value=1, max_value=len(bse_records),
                                            value=1, step=1, key=sskey(mode_key, "range_bse_from"))
            with c4:
                bse_to = st.number_input("BSE To", min_value=1, max_value=len(bse_records),
                                          value=min(100, len(bse_records)), step=1,
                                          key=sskey(mode_key, "range_bse_to"))
            if bse_from > bse_to:
                st.sidebar.error("BSE 'From' must be ≤ 'To'")
            else:
                sl = bse_records[(bse_from - 1):bse_to]
                range_stocks.extend(sl)
                st.sidebar.success(f"BSE: rows {bse_from}–{bse_to} → {len(sl)} stocks")

        if not range_stocks:
            st.sidebar.warning("⚠️ No stocks in selected range. Check exchange selection above.")
        return range_stocks

    if scan_mode == "Slot-wise Scan":
        st.sidebar.subheader("📦 Select Slots to Scan")
        total = len(universe)
        slot_size = 1000
        num_slots = (total + slot_size - 1) // slot_size if total else 0
        st.sidebar.info(f"📊 Total stocks: {total}\n💼 Slot size: 1000 stocks\n📦 Total slots: {num_slots}")

        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("✅ Select All", use_container_width=True, key=sskey(mode_key, "slots_select_all")):
                for slot_num in range(num_slots):
                    st.session_state[sskey(mode_key, f"slot_{slot_num}")] = True
                st.rerun()
        with col2:
            if st.button("❌ Deselect All", use_container_width=True, key=sskey(mode_key, "slots_deselect_all")):
                for slot_num in range(num_slots):
                    st.session_state[sskey(mode_key, f"slot_{slot_num}")] = False
                st.rerun()

        st.sidebar.markdown("---")
        selected_slots = []
        for slot_num in range(num_slots):
            start_idx = slot_num * slot_size
            end_idx = min((slot_num + 1) * slot_size, total)
            slot_stocks = universe[start_idx:end_idx]
            nse_in_slot = sum(1 for s in slot_stocks if s["exchange"] == "NSE")
            bse_in_slot = sum(1 for s in slot_stocks if s["exchange"] == "BSE")
            label = f"Slot {slot_num + 1}: {start_idx + 1}-{end_idx}"
            detail = f"({len(slot_stocks)}: {nse_in_slot} NSE, {bse_in_slot} BSE)"
            if st.sidebar.checkbox(f"{label} {detail}", key=sskey(mode_key, f"slot_{slot_num}")):
                selected_slots.append(slot_num)

        stocks_to_scan = []
        for slot_num in selected_slots:
            start_idx = slot_num * slot_size
            end_idx = min((slot_num + 1) * slot_size, total)
            stocks_to_scan.extend(universe[start_idx:end_idx])

        if not selected_slots:
            st.sidebar.warning("⚠️ Please select at least one slot to scan")
            return []
        nse_sel = sum(1 for s in stocks_to_scan if s["exchange"] == "NSE")
        bse_sel = sum(1 for s in stocks_to_scan if s["exchange"] == "BSE")
        st.sidebar.success(f"✅ {len(selected_slots)} slot(s) selected\n📊 Total: {len(stocks_to_scan)} stocks\n"
                            f"🔵 NSE: {nse_sel} | 🟠 BSE: {bse_sel}")
        return stocks_to_scan

    # Custom List
    custom_input = st.sidebar.text_area(
        "Enter symbols (one per line)",
        "Stock names with exchange suffix:\nRELIANCE.NS\n500002.BO\nINFY.NS\n\n"
        "Or without (defaults to NSE, bare numbers default to BSE):\nRELIANCE\nTCS",
        height=150, key=sskey(mode_key, "custom_list"),
    )
    universe_by_yf = {r["yf_symbol"]: r for r in universe}
    raw_symbols = [s.strip() for s in custom_input.split("\n") if s.strip()]
    # Skip the placeholder helper lines if the user never replaced them
    raw_symbols = [s for s in raw_symbols if not s.lower().startswith(("stock names", "or without"))]
    return [_tickers.record_for_custom_symbol(s, universe_by_yf) for s in raw_symbols]


# ── Sidebar: rate-limiting controls (workers / batching / retry) ───────────
def render_rate_limit_controls(mode_key: str) -> dict:
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ Rate Limiting Controls")
    st.sidebar.info(
        "**⚡ Concurrent scan, shared Yahoo connection budget across all 3 modes.**\n"
        "Recommended: **4–6 workers**. Each worker uses the same global semaphore so "
        "Yahoo never sees more than (workers × calls-per-stock) simultaneous connections.\n"
        "Reduce to 2–3 only if you still see 429s."
    )

    max_workers_ui = st.sidebar.slider(
        "Parallel workers", min_value=1, max_value=16, value=6, step=1,
        help="How many stocks to fetch simultaneously. Lower if hitting 429s.",
        key=sskey(mode_key, "max_workers"),
    )
    _set_semaphore(max_workers_ui)

    batch_size = st.sidebar.number_input(
        "Batch size (0 = no batching)", min_value=0, max_value=1000, value=0, step=10,
        help="Pause after every N stocks. 0 disables. Use 50–100 if heavy rate limiting.",
        key=sskey(mode_key, "batch_size"),
    )
    batch_pause = st.sidebar.number_input(
        "Batch pause (sec)", min_value=5, max_value=300, value=30, step=5,
        help="How long to pause after each batch. Only used if batch size > 0.",
        key=sskey(mode_key, "batch_pause"),
    )

    with st.sidebar.expander("🔧 Retry / Backoff Settings"):
        retry_max = st.number_input(
            "Max retries per stock", min_value=1, max_value=10, value=3, step=1,
            help="How many times yf_ratelimit retries a failed fetch before giving up.",
            key=sskey(mode_key, "retry_max"),
        )
        retry_initial_delay = st.number_input(
            "Retry base backoff (sec)", min_value=0.5, max_value=30.0, value=3.0, step=0.5,
            help="Base delay for exponential backoff on retries. Doubles each retry.",
            key=sskey(mode_key, "retry_delay"),
        )
        stats_interval = st.number_input(
            "Stats update every N stocks", min_value=1, max_value=100, value=10, step=1,
            help="How often the stats bar refreshes during scan.",
            key=sskey(mode_key, "stats_interval"),
        )

    # These sliders now actually take effect (previously dead controls that
    # were accepted by bulletproof_fetch's signature but never used).
    yf_ratelimit.configure(max_retries=retry_max, base_backoff=retry_initial_delay)

    dead_count = count_dead_symbols()
    if dead_count > 0:
        if st.sidebar.button(f"🧹 Clear skip-list ({dead_count} symbols)", use_container_width=True,
                              help="Symbols currently being skipped as 'dead' (shared across all 3 modes). "
                                   "Clear this if results look too low — a rate-limit burst can wrongly "
                                   "flag valid symbols.",
                              key=sskey(mode_key, "clear_dead")):
            clear_dead_symbols()
            st.sidebar.success("Skip-list cleared")
            st.rerun()

    return {
        "max_workers": max_workers_ui,
        "batch_size": batch_size,
        "batch_pause": batch_pause,
        "retry_max": retry_max,
        "retry_delay": retry_initial_delay,
        "stats_interval": stats_interval,
    }


# ── Checkpoint / resume UI + scan-trigger buttons ───────────────────────────
def render_scan_trigger(mode_key: str, stocks_to_scan: list, button_label: str):
    """Returns (do_scan, resume_scan, checkpoint_or_None, signature)."""
    signature = universe_signature([s["yf_symbol"] for s in stocks_to_scan])
    checkpoint = load_checkpoint(mode_key)
    resumable = (
        checkpoint is not None
        and checkpoint.get("signature") == signature
        and len(checkpoint.get("scanned_symbols", [])) < len(stocks_to_scan)
    )

    do_scan = False
    resume_scan = False

    if resumable:
        remaining = len(stocks_to_scan) - len(checkpoint.get("scanned_symbols", []))
        st.sidebar.info(f"⏸ Interrupted scan found: {remaining} stock(s) left to go")
        c1, c2 = st.sidebar.columns(2)
        if c1.button("▶️ RESUME SCAN", type="primary", use_container_width=True,
                      key=sskey(mode_key, "resume_btn")):
            do_scan, resume_scan = True, True
        if c2.button("🔄 START FRESH", use_container_width=True, key=sskey(mode_key, "fresh_btn")):
            clear_checkpoint(mode_key)
            do_scan, resume_scan = True, False
    else:
        if st.sidebar.button(button_label, type="primary", use_container_width=True,
                              key=sskey(mode_key, "scan_btn")):
            do_scan, resume_scan = True, False

    return do_scan, resume_scan, checkpoint, signature


# ── Generic concurrent scan runner ──────────────────────────────────────────
def run_scan(mode_key: str, stocks_to_scan: list,
             fetch_and_analyze: Callable[[dict], tuple[str, dict | None]],
             rate_cfg: dict, resume_scan: bool, checkpoint) -> None:
    """Runs the scan and writes results into mode-namespaced session_state:
        results, timestamp, failed_records

    `fetch_and_analyze(record) -> (status, analysis)` where status is one of
    'ok' | 'filtered' | 'failed'. `record` is a TickerRecord (symbol/name/
    yf_symbol/exchange) so every mode's core logic gets the company name for
    free without re-deriving it. This function owns everything that must
    behave identically across modes: progress bar, batching, checkpointing,
    stats, and non-blocking behaviour (no long blocking sleeps that would
    make the whole app feel frozen — "no lagging").
    """
    set_state(mode_key, "results", None)
    pop_state(mode_key, "timestamp")
    pop_state(mode_key, "failed_records")

    st.markdown("---")
    if resume_scan and checkpoint:
        st.subheader("📊 Resuming scan...")
        results = list(checkpoint.get("results", []))
        failed_records = list(checkpoint.get("failed_symbols", []))
        already_scanned = set(checkpoint.get("scanned_symbols", []))
        scan_universe = [s for s in stocks_to_scan if s["yf_symbol"] not in already_scanned]
    else:
        st.subheader("📊 Scanning...")
        results = []
        failed_records = []
        already_scanned = set()
        scan_universe = list(stocks_to_scan)
        clear_checkpoint(mode_key)

    signature = universe_signature([s["yf_symbol"] for s in stocks_to_scan])

    progress_bar = st.progress(0)
    status_text = st.empty()
    stats_placeholder = st.empty()
    live_status = st.empty()

    total = len(stocks_to_scan)
    failed = len(failed_records)
    completed = len(already_scanned)
    start_time = time.time()

    max_workers = min(rate_cfg["max_workers"], len(scan_universe)) if scan_universe else 1
    status_text.info(f"⚡ Concurrent scan: {max_workers} workers × {len(scan_universe)} stocks remaining")

    # Checkpointing after *every* stock used to mean writing the whole
    # (ever-growing) results/scanned-symbols JSON back to disk once per
    # stock — fine for a 50-stock Quick Scan, but O(n²) total bytes written
    # on a ~8,000-stock Full Scan, which is exactly the kind of "lagging"
    # this rewrite is meant to kill. Checkpoint on a time/count throttle
    # instead — worst case a crash loses the last few seconds of progress,
    # which resume already tolerates fine.
    _checkpoint_every = max(1, total // 100 if total else 1)
    _last_checkpoint_ts = time.time()

    # Live-status panel below. Previously the only visibility into a running
    # scan was the aggregate progress bar + a periodic (every stats_interval-th
    # stock) stats line -- every 429 / cooldown / retry yf_ratelimit hits only
    # went to the server log (invisible in the app itself, and on Streamlit
    # Cloud only reachable by opening the separate log viewer). This surfaces
    # yf_ratelimit's rate-limit events inside the app, updating live as the
    # scan runs -- kept to a single current-status line rather than a growing
    # per-symbol list, since most stocks get filtered out by design and an
    # itemized log of that is just noise, not "live status."
    _last_live_update = 0.0
    _last_symbol = {"symbol": None, "icon": "", "label": ""}

    def _render_live_status(force: bool = False):
        nonlocal _last_live_update
        now = time.time()
        if not force and (now - _last_live_update) < 0.35:
            return
        _last_live_update = now

        yf_status = yf_ratelimit.get_status()
        lines = []
        if yf_status["cooldown_active"]:
            lines.append(f"🧊 **Cooling down** — all workers paused for another "
                         f"{yf_status['cooldown_remaining']:.0f}s (Yahoo rate limit)")
        lines.append(f"🔌 {yf_status['inflight']} request(s) in flight · {max_workers} workers configured")

        if _last_symbol["symbol"]:
            lines.append(f"{_last_symbol['icon']} Last: `{_last_symbol['symbol']}` — {_last_symbol['label']}")

        events = yf_ratelimit.get_recent_events(4)
        if events:
            lines.append("")
            lines.append("**Rate-limit activity:**")
            for ev in reversed(events):
                age = now - ev["ts"]
                lines.append(f"&nbsp;&nbsp;{ev['message']} _{age:.0f}s ago_")

        live_status.markdown("\n\n".join(lines))

    _render_live_status(force=True)

    scan_interrupted = False
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_rec = {executor.submit(fetch_and_analyze, rec): rec for rec in scan_universe}

            for future in as_completed(future_to_rec):
                rec = future_to_rec[future]
                try:
                    status, analysis = future.result()
                except Exception:
                    status, analysis = "failed", None

                completed += 1
                already_scanned.add(rec["yf_symbol"])

                if status == "ok" and analysis is not None:
                    results.append(analysis)
                    _last_symbol.update(symbol=rec["symbol"], icon="✅", label="qualified")
                elif status == "failed":
                    failed += 1
                    failed_records.append(rec["yf_symbol"])
                    _last_symbol.update(symbol=rec["symbol"], icon="❌", label="failed")
                else:
                    _last_symbol.update(symbol=rec["symbol"], icon="⏭️", label="filtered out")
                # 'filtered' -> counted implicitly (completed - len(results) - failed)

                _now = time.time()
                if (completed % _checkpoint_every == 0 or completed == total
                        or (_now - _last_checkpoint_ts) >= 5):
                    save_checkpoint(mode_key, signature, stocks_to_scan, results, failed_records, list(already_scanned))
                    _last_checkpoint_ts = _now

                _render_live_status(force=(completed == total))

                if rate_cfg["batch_size"] > 0 and completed % rate_cfg["batch_size"] == 0 and completed < total:
                    status_text.warning(f"⏸ Batch pause {rate_cfg['batch_pause']}s after {completed} stocks...")
                    time.sleep(rate_cfg["batch_pause"])

                progress_bar.progress(completed / total)

                if completed % rate_cfg["stats_interval"] == 0 or completed == total:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (total - completed) / rate if rate > 0 else 0
                    status_text.info(
                        f"📊 {completed}/{total} done · ✅ {len(results)} valid · ⏱ ETA {eta:.0f}s"
                    )
                    filtered_out = completed - len(results) - failed
                    stats_placeholder.info(
                        f"✅ Valid: {len(results)} | Filtered: {max(0, filtered_out)} | Failed: {failed}"
                    )
    except Exception as e:
        scan_interrupted = True
        st.error(f"⚠️ Scan interrupted: {e}. Progress up to this point is saved — click ▶️ RESUME SCAN to continue.")

    if scan_interrupted:
        st.stop()

    clear_checkpoint(mode_key)
    results = [r for r in results if r is not None]

    set_state(mode_key, "results", results)
    set_state(mode_key, "timestamp", datetime.now())
    set_state(mode_key, "failed_records", failed_records)

    elapsed_time = (time.time() - start_time) / 60
    st.success(f"✅ Scan complete! Found {len(results)} stocks meeting criteria")

    c1, c2, c3 = st.columns(3)
    c1.metric("✅ Successfully Processed", len(results))
    c2.metric("❌ Failed", failed)
    c3.metric("⏱️ Time Taken", f"{elapsed_time:.1f} min")

    if failed > 0 and failed_records:
        with st.expander(f"⚠️ Failed Tickers ({failed})", expanded=False):
            st.write(", ".join(failed_records[:20]))
            if len(failed_records) > 20:
                st.caption(f"...and {len(failed_records) - 20} more")

            if st.button("🔄 Retry Failed Tickers", key=sskey(mode_key, "retry_failed_btn")):
                records_by_symbol = {r["yf_symbol"]: r for r in stocks_to_scan}
                with st.spinner("Retrying failed tickers..."):
                    retry_results = []
                    for yf_symbol in failed_records:
                        rec = records_by_symbol.get(yf_symbol)
                        if not rec:
                            continue
                        try:
                            status, analysis = fetch_and_analyze(rec)
                            if status == "ok" and analysis is not None:
                                retry_results.append(analysis)
                        except Exception:
                            pass

                    if retry_results:
                        current = get_state(mode_key, "results", [])
                        set_state(mode_key, "results", current + retry_results)
                        st.success(f"✅ Recovered {len(retry_results)} additional stocks!")
                        time.sleep(1)
                    else:
                        st.warning("No additional stocks recovered")

    time.sleep(0.3)
    status_text.empty()
    stats_placeholder.empty()
    progress_bar.empty()
    st.rerun()


# ── CSV export (always includes company Name) ───────────────────────────────
def download_buttons(mode_key: str, filtered_df: pd.DataFrame, full_df: pd.DataFrame, file_prefix: str) -> None:
    st.markdown("---")
    st.subheader("💾 Download Results")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📥 Download Filtered CSV",
            filtered_df.to_csv(index=False),
            f"{file_prefix}_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            use_container_width=True,
            key=sskey(mode_key, "dl_filtered"),
        )
    with col2:
        st.download_button(
            "📥 Download All Results CSV",
            full_df.to_csv(index=False),
            f"{file_prefix}_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            use_container_width=True,
            key=sskey(mode_key, "dl_all"),
        )


# ── Shared CSS + page chrome (call once, from the top-level entrypoint) ────
def inject_base_css() -> None:
    st.markdown("""<style>
.main-header{font-size:2.5rem;font-weight:700;color:#1f77b4;text-align:center;margin-bottom:1rem}
.sub-header{font-size:1.5rem;font-weight:600;color:#333;margin:1rem 0}
.metric-card{background:#f8f9fb;padding:0.8rem;border-radius:8px;border-left:4px solid #1f77b4;margin:0.5rem 0}
.stDataFrame{font-size:0.9rem}
div[data-testid="stDataFrame"] > div{background:#f8f9fb}
.price-up{color:#00c853;font-weight:bold}
.price-down{color:#ff1744;font-weight:bold}
.price-neutral{color:#666}
</style>""", unsafe_allow_html=True)


def footer(text: str) -> None:
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align:center;color:#666;'>
    <p>{text}</p>
    <p style='font-size:0.85rem;'>⚠ Educational purposes only. Not financial advice.</p>
    </div>
    """, unsafe_allow_html=True)
