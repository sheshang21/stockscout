# Indian Stock Scout

NSE & BSE stock scanner with 3 modes, all in one Streamlit app.

## Modes

1. **Positional Scanner** — long-term value investing. Ultra-strict 250-point
   fundamentals + technicals scoring (`mode_positional.py`).
2. **Intraday Short Screener** — flags short-selling setups (`mode_intraday_short.py`).
3. **Intraday Long (Buy) Screener** — flags intraday buy setups (`mode_intraday_long.py`).

Pick a mode from the dropdown at the top of the app. Every mode walks through
the same sidebar shape — Exchange → Scan Mode (Quick / Full / Slot-wise /
Range / Custom List) → Rate Limiting → mode-specific filters → Scan button —
and only the core stock-scoring logic differs between them.

## Files

| File | Purpose |
|---|---|
| `sheshscout.py` | Entry point — page config, mode dropdown, dispatch |
| `scanner_common.py` | Shared infra: rate limiting, checkpoint/resume, dead-symbol skip-list, sidebar widgets, scan orchestration, CSV export |
| `tickers.py` | Loads `nse_tickers.csv` / `bse_codes.csv` (symbol **and** company name) |
| `indicators.py` | Shared technical-indicator math (RSI, MACD, ATR, Bollinger, operator/pump detection) used by all 3 modes |
| `intraday_data.py` | Shared raw-data fetch for the 2 intraday modes (identical Yahoo calls; only scoring differs) |
| `mode_positional.py` | Positional scanner core logic + UI |
| `mode_intraday_short.py` | Short screener core logic + UI |
| `mode_intraday_long.py` | Long screener core logic + UI |
| `yf_ratelimit.py` | Chrome-impersonation session + shared cooldown + retry ladder around every Yahoo call |
| `nse_tickers.csv`, `bse_codes.csv` | Ticker universes with company names (replaces the old `.txt` files) |

## Data files

- `nse_tickers.csv` — columns `NSE Ticker, Name`. Suffix `.NS` for Yahoo.
- `bse_codes.csv` — columns `BSE Code, Name`. Suffix `.BO` for Yahoo. BSE
  stocks are addressed by their **numeric scrip code** (e.g. `500002.BO` for
  ABB India) — the old `bse.txt` used alpha symbols that didn't reliably
  resolve on Yahoo for most BSE-only names.

To refresh either universe, replace the CSV (same 2 columns, header row) —
no code changes needed.

## Session state / state isolation

Every session_state key and widget key is namespaced by mode
(`scanner_common.sskey`), so switching the mode dropdown never lets one
mode's scan results or filter selections leak into another.

## Rate limiting & resiliency

- A single global semaphore caps simultaneous Yahoo connections across
  whatever mode is currently scanning.
- `yf_ratelimit.py` wraps every Yahoo call in a Chrome-impersonation session,
  a shared cooldown (one 429 pauses every worker, not just the one that hit
  it), and a retry ladder — now actually configurable from the sidebar's
  Retry/Backoff expander (previously those sliders existed but did nothing).
- A disk-backed checkpoint (one file per mode) survives process restarts —
  interrupted scans resume instead of starting over. Checkpoints are written
  on a throttle (every ~1% of the universe or 5s, whichever comes first) so
  a full ~8,000-stock scan doesn't spend most of its time re-writing an
  ever-growing JSON file to disk after every single symbol.
- A disk-backed dead-symbol skip-list (shared across all 3 modes) avoids
  re-hitting Yahoo for delisted symbols on every scan.

## Local dev

```bash
pip install -r requirements.txt
streamlit run sheshscout.py
```
