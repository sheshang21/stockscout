"""
tickers.py — Shared NSE/BSE ticker universe, loaded once from CSV.
====================================================================
Replaces the old nse.txt / bse.txt / nse_tickers.txt (symbol-only,
one per line) with two CSVs that also carry the company name:

    nse_tickers.csv   columns: "NSE Ticker", "Name"
    bse_codes.csv     columns: "BSE Code",   "Name"

BSE stocks on Yahoo Finance are addressed by their numeric BSE scrip
code (e.g. "500002.BO" for ABB India), not the old alpha symbols that
lived in bse.txt — those never reliably resolved on Yahoo for most
BSE-only names. The new bse_codes.csv fixes that at the source.

Every ticker is returned as a small dict:
    {"symbol": "RELIANCE", "name": "Reliance Industries...",
     "yf_symbol": "RELIANCE.NS", "exchange": "NSE"}

Loaded once per process via st.cache_data (falls back to a plain
in-memory cache if Streamlit isn't importable, e.g. unit testing).
"""

from __future__ import annotations

import os
from typing import TypedDict

import pandas as pd

try:
    import streamlit as st
    _cache_data = st.cache_data
except Exception:  # pragma: no cover - non-Streamlit contexts (tests)
    def _cache_data(func=None, **_kwargs):
        if func is None:
            return lambda f: f
        return func

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NSE_CSV_PATH = os.path.join(_BASE_DIR, "nse_tickers.csv")
BSE_CSV_PATH = os.path.join(_BASE_DIR, "bse_codes.csv")


class TickerRecord(TypedDict):
    symbol: str
    name: str
    yf_symbol: str
    exchange: str


def _clean_name(raw) -> str:
    if raw is None:
        return ""
    return " ".join(str(raw).split())  # collapse whitespace, strip trailing pad


@_cache_data(show_spinner=False)
def load_nse_universe() -> list[TickerRecord]:
    try:
        df = pd.read_csv(NSE_CSV_PATH, encoding="utf-8-sig", dtype=str)
    except FileNotFoundError:
        return []
    except Exception:
        return []
    df = df.dropna(subset=[df.columns[0]])
    records: list[TickerRecord] = []
    seen = set()
    for _, row in df.iterrows():
        sym = str(row.iloc[0]).strip().upper()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        name = _clean_name(row.iloc[1]) if len(row) > 1 else ""
        records.append({
            "symbol": sym,
            "name": name,
            "yf_symbol": f"{sym}.NS",
            "exchange": "NSE",
        })
    return records


@_cache_data(show_spinner=False)
def load_bse_universe() -> list[TickerRecord]:
    try:
        df = pd.read_csv(BSE_CSV_PATH, encoding="utf-8-sig", dtype=str)
    except FileNotFoundError:
        return []
    except Exception:
        return []
    df = df.dropna(subset=[df.columns[0]])
    records: list[TickerRecord] = []
    seen = set()
    for _, row in df.iterrows():
        code = str(row.iloc[0]).strip()
        # BSE scrip codes are numeric (6 digits); guard against stray header/blank rows
        if not code or not code.isdigit() or code in seen:
            continue
        seen.add(code)
        name = _clean_name(row.iloc[1]) if len(row) > 1 else ""
        records.append({
            "symbol": code,
            "name": name,
            "yf_symbol": f"{code}.BO",
            "exchange": "BSE",
        })
    return records


@_cache_data(show_spinner=False)
def load_universe(include_nse: bool, include_bse: bool) -> list[TickerRecord]:
    """Combined universe in a stable order: NSE first, then BSE, de-duplicated
    on yf_symbol (defensive — the two source files shouldn't overlap)."""
    records: list[TickerRecord] = []
    if include_nse:
        records.extend(load_nse_universe())
    if include_bse:
        records.extend(load_bse_universe())
    seen = set()
    deduped = []
    for r in records:
        if r["yf_symbol"] in seen:
            continue
        seen.add(r["yf_symbol"])
        deduped.append(r)
    return deduped


def name_lookup_map(records: list[TickerRecord]) -> dict[str, str]:
    """yf_symbol -> company name, for quick lookup when only the symbol is on hand."""
    return {r["yf_symbol"]: r["name"] for r in records}


def record_for_custom_symbol(raw_symbol: str, universe_by_yf: dict[str, "TickerRecord"]) -> TickerRecord:
    """Build a TickerRecord for a symbol typed into the Custom List box.
    Looks up the name from the known universe if we recognise it; otherwise
    ships with an empty name rather than guessing.
    """
    raw = raw_symbol.strip().upper()
    if raw.endswith(".NS") or raw.endswith(".BO"):
        yf_symbol = raw
        symbol = raw.rsplit(".", 1)[0]
        exchange = "NSE" if raw.endswith(".NS") else "BSE"
    else:
        # Bare BSE numeric codes are unambiguous; anything else defaults to NSE
        if raw.isdigit():
            symbol, exchange, yf_symbol = raw, "BSE", f"{raw}.BO"
        else:
            symbol, exchange, yf_symbol = raw, "NSE", f"{raw}.NS"
    known = universe_by_yf.get(yf_symbol)
    name = known["name"] if known else ""
    return {"symbol": symbol, "name": name, "yf_symbol": yf_symbol, "exchange": exchange}
