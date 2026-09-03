"""
sheshscout.py — Indian Stock Scout: entry point.

Three scanner modes share one app:
  1. Positional Scanner   — long-term value investing (ultra-strict fundamentals + technicals)
  2. Intraday Short        — short-selling setups
  3. Intraday Long          — buy/long setups

Pick a mode from the dropdown below; every mode then walks through the same
shape of sidebar (exchange → scan mode → rate limiting → mode-specific
filters → scan button) before showing results. Switching the dropdown never
loses another mode's in-progress config or last scan — each mode's
session_state is fully namespaced (see scanner_common.sskey).
"""

import warnings
import logging

import streamlit as st

import scanner_common as sc

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

st.set_page_config(page_title="Indian Stock Scout", page_icon="🎯", layout="wide")
sc.inject_base_css()

st.markdown('<p class="main-header">🎯 Indian Stock Scout</p>', unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#666;'>NSE & BSE stock scanner — pick a mode to get started</p>",
    unsafe_allow_html=True,
)

_mode_options = [sc.MODE_POSITIONAL, sc.MODE_SHORT, sc.MODE_LONG]
_default_mode = st.session_state.get("active_mode", sc.MODE_POSITIONAL)
_default_index = _mode_options.index(_default_mode) if _default_mode in _mode_options else 0

selected_mode = st.selectbox(
    "🔀 Scanner Mode",
    _mode_options,
    index=_default_index,
    format_func=lambda m: sc.MODE_LABELS[m],
    key="active_mode",
)

st.markdown("---")

if selected_mode == sc.MODE_POSITIONAL:
    import mode_positional
    mode_positional.render()
elif selected_mode == sc.MODE_SHORT:
    import mode_intraday_short
    mode_intraday_short.render()
elif selected_mode == sc.MODE_LONG:
    import mode_intraday_long
    mode_intraday_long.render()
