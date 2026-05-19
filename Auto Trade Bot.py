"""
Nifty Pilot — Upstox Live Trading Dashboard
============================================
Token:  stored in .streamlit/secrets.toml  →  [upstox] access_token
Run:    streamlit run nifty_pilot_upstox.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import streamlit.components.v1 as components
from datetime import date, timedelta
from collections import deque

# ─────────────────────────────────────────────
# TOKEN — read from secrets.toml
# ─────────────────────────────────────────────
try:
    ACCESS_TOKEN = st.secrets["upstox"]["access_token"]
except Exception:
    ACCESS_TOKEN = "YOUR_ACCESS_TOKEN_HERE"

MOCK_MODE = ACCESS_TOKEN in ("", "YOUR_ACCESS_TOKEN_HERE")

# ─────────────────────────────────────────────
# INDEX CATALOGUE
# ─────────────────────────────────────────────
# Lot sizes effective Jan 2026 per NSE/BSE circulars
# Nifty50=65, BankNifty=30, FinNifty=40, MidcapNifty=120, Sensex=20
INDICES = {
    "Nifty 50":        {"key": "NSE_INDEX|Nifty 50",         "lot": 65},
    "Nifty Bank":      {"key": "NSE_INDEX|Nifty Bank",       "lot": 30},
    "Nifty Fin Svc":   {"key": "NSE_INDEX|Nifty Fin Service","lot": 40},
    "Nifty Midcap 50": {"key": "NSE_INDEX|Nifty Midcap 50",  "lot": 120},
    "Sensex":          {"key": "BSE_INDEX|SENSEX",           "lot": 20},
}

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
ADX_PERIOD      = 14
CANDLE_INTERVAL = "1minute"
OI_PERCENTILE   = 70
TOP_N_STRIKES   = 10
HISTORY_LEN     = 15
MONITOR_SECS    = 300
MIN_HOLD_SECS   = 600   # 10 minutes — used globally for mockup trade hold

# ─────────────────────────────────────────────
# MULTI-TIMEFRAME TREND CONSTANTS
# ─────────────────────────────────────────────
# Trend is established on 15m and 5m; trade executes on 3m
TF_15M  = "15minute"
TF_5M   = "5minute"
TF_3M   = "3minute"
# EMA periods for trend detection on each timeframe
TF_EMA_FAST = 9
TF_EMA_SLOW = 21

# ─────────────────────────────────────────────
# PER-INDEX DI SCENARIO TARGET TABLE
# ─────────────────────────────────────────────
INDEX_TARGETS = {
    "Nifty 50": {
        "crossover": {"immediate": 60,   "trend": 150,  "label": "+DI cross"},
        "strong":    {"immediate": 100,  "trend": 325,  "label": "DI > 25"},
    },
    "Nifty Bank": {
        "crossover": {"immediate": 200,  "trend": 500,  "label": "+DI cross"},
        "strong":    {"immediate": 375,  "trend": 1000, "label": "DI > 25"},
    },
    "Nifty Fin Svc": {
        "crossover": {"immediate": 95,   "trend": 240,  "label": "+DI cross"},
        "strong":    {"immediate": 180,  "trend": 475,  "label": "DI > 25"},
    },
    "Nifty Midcap 50": {
        "crossover": {"immediate": 60,   "trend": 150,  "label": "+DI cross"},
        "strong":    {"immediate": 100,  "trend": 325,  "label": "DI > 25"},
    },
    "Sensex": {
        "crossover": {"immediate": 275,  "trend": 800,  "label": "+DI cross"},
        "strong":    {"immediate": 500,  "trend": 1600, "label": "DI > 25"},
    },
}

# Legacy fallback (used only for display gear bars — kept for backward compat)
GEAR_PTS = {4: 200, 3: 150, 2: 100, 1: 50}

# ── Option selection scoring weights (must sum to 100) ──
W_ADX       = 25   # option's own trend strength
W_DELTA     = 15   # proximity to ideal delta (0.4–0.6)
W_OI        = 15   # OI strength vs chain average
W_VOLUME    = 10   # volume (fresh activity)
W_SPREAD    = 10   # bid-ask tightness (lower = better)
W_MATRIX    = 15   # OI momentum decision matrix alignment
W_MTF       = 10   # multi-timeframe trend alignment

# ── Filters (hard cutoffs before scoring) ──
DELTA_MIN   = 0.20
DELTA_MAX   = 0.80
MAX_SPREAD_PCT = 5.0
IV_LOOKBACK = 30
ORDER_URL       = "https://api-hft.upstox.com/v2/order/place"
BASE_URL        = "https://api.upstox.com/v2"

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
def _init():
    defaults = dict(
        selected_index  = "Nifty 50",
        trade_active    = False,
        trade_mode      = "mockup",
        entry_price     = 0.0,
        entry_adx       = 0.0,
        entry_pdi       = 0.0,
        entry_ndi       = 0.0,
        entry_side      = "",
        entry_signal    = "NONE",
        entry_strike    = 0,
        entry_expiry    = "",
        entry_ikey      = "",
        entry_opt_ltp   = 0.0,
        target_pts      = 0,
        trailing_sl_pct = 30.0,
        exit_price      = 0.0,
        sl_price        = 0.0,
        highest_pnl     = 0.0,
        last_signal     = None,
        live_order_id   = None,
        monitor_start   = None,
        signal_log      = [],
        act_log         = [],
        h_times         = deque(maxlen=HISTORY_LEN),
        h_opt_ltp       = deque(maxlen=HISTORY_LEN),
        h_opt_adx       = deque(maxlen=HISTORY_LEN),
        h_idx_adx       = deque(maxlen=HISTORY_LEN),
        h_opt_pdi       = deque(maxlen=HISTORY_LEN),
        h_opt_ndi       = deque(maxlen=HISTORY_LEN),
        h_oi            = deque(maxlen=HISTORY_LEN),
        # live-trade order panel state
        lo_strike       = None,
        lo_side         = None,
        lo_ikey         = None,
        lo_qty          = 1,
        lo_target_pts   = 50,
        lo_tsl_pct      = 30.0,
        # sv_ = shared view-state written by _fetch_data, read by display fragments
        sv_live_px      = None,
        sv_dmi          = None,
        sv_best_opt     = None,
        sv_score        = 0,
        sv_checks       = [],
        sv_idx_g        = 1,
        sv_opt_g        = 1,
        sv_bullish      = True,
        sv_oi_chg       = 0,
        sv_now_ts       = "",
        sv_fetch_error  = "",
        sv_mock_banner  = False,
        sv_cpr          = None,
        sv_entry_sig    = None,
        sv_di_targets   = None,
        sv_trend_3d     = None,       # multi-timeframe trend analysis result (renamed from 3-day)
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

HEADERS = {
    "Accept":        "application/json",
    "Content-Type":  "application/json",
    "Authorization": f"Bearer {ACCESS_TOKEN}",
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def browser_alert(title, msg):
    components.html(f"""<script>
    if(Notification.permission==="granted"){{new Notification("{title}",{{body:"{msg}"}})}}
    else if(Notification.permission!=="denied"){{Notification.requestPermission().then(p=>{{
    if(p==="granted")new Notification("{title}",{{body:"{msg}"}});}});}}
    </script>""", height=0)

def add_log(store_key, entry):
    st.session_state[store_key].insert(0, entry)
    if len(st.session_state[store_key]) > 30:
        st.session_state[store_key].pop()

def ts():
    return pd.Timestamp.now(tz="Asia/Kolkata").strftime("%H:%M:%S")

def gear(adx):
    """Legacy gear for visual gear-bar display only."""
    return 4 if adx >= 40 else 3 if adx >= 35 else 2 if adx >= 30 else 1


def get_di_targets(index_name: str, dmi: dict) -> dict:
    """
    Returns target dictionary for the current index and DI scenario.
    """
    idx_tbl = INDEX_TARGETS.get(index_name, INDEX_TARGETS["Nifty 50"])
    pdi     = dmi.get("pdi", 0)
    ndi     = dmi.get("ndi", 0)
    adx     = dmi.get("adx", 0)

    dominant_di = max(pdi, ndi)
    scenario = "strong" if dominant_di >= 25 else "crossover"
    tbl      = idx_tbl[scenario]
    target = tbl["trend"] if adx >= 30 else tbl["immediate"]
    sl     = round(target * 0.50)

    return {
        "scenario":  scenario,
        "label":     tbl["label"],
        "immediate": tbl["immediate"],
        "trend":     tbl["trend"],
        "target":    target,
        "sl":        sl,
        "adx":       adx,
        "pdi":       pdi,
        "ndi":       ndi,
    }

def ha(d):
    return float(np.mean(list(d))) if d else 0.0

# ─────────────────────────────────────────────
# ADX / DMI — pure numpy
# ─────────────────────────────────────────────
def compute_dmi(df, period=14):
    if len(df) < period + 1:
        return None
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    tr   = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
    up   = h[1:]-h[:-1]; dn = l[:-1]-l[1:]
    dm_p = np.where((up>dn)&(up>0), up, 0.0)
    dm_n = np.where((dn>up)&(dn>0), dn, 0.0)

    def rma(a, p):
        return pd.Series(a).ewm(alpha=1/p, min_periods=p, adjust=False).mean().values

    atr  = rma(tr,   period)
    sdmp = rma(dm_p, period)
    sdmn = rma(dm_n, period)
    safe = np.where(atr == 0, 1e-10, atr)
    pdi  = np.clip(100 * sdmp / safe, 0, 100)
    ndi  = np.clip(100 * sdmn / safe, 0, 100)
    dx   = 100 * np.abs(pdi - ndi) / np.where((pdi + ndi) == 0, 1e-10, pdi + ndi)
    adx  = np.clip(rma(dx, period), 0, 100)
    return {
        "adx": round(float(adx[-1]), 2),
        "pdi": round(float(pdi[-1]), 2),
        "ndi": round(float(ndi[-1]), 2),
    }

# ─────────────────────────────────────────────
# TECHNICAL INDICATORS
# ─────────────────────────────────────────────

def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def compute_supertrend(df: pd.DataFrame, period: int, multiplier: float):
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)

    tr = np.maximum(
        h - l,
        np.maximum(np.abs(h - c.shift(1)), np.abs(l - c.shift(1)))
    )
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    hl2        = (h + l) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    final_upper = upper_band.copy()
    final_lower = lower_band.copy()
    for i in range(1, len(df)):
        final_upper.iloc[i] = (upper_band.iloc[i]
                               if (upper_band.iloc[i] < final_upper.iloc[i-1]
                                   or c.iloc[i-1] > final_upper.iloc[i-1])
                               else final_upper.iloc[i-1])
        final_lower.iloc[i] = (lower_band.iloc[i]
                               if (lower_band.iloc[i] > final_lower.iloc[i-1]
                                   or c.iloc[i-1] < final_lower.iloc[i-1])
                               else final_lower.iloc[i-1])

    trend = pd.Series(index=df.index, dtype=float)
    trend.iloc[0] = 1
    for i in range(1, len(df)):
        if c.iloc[i] > final_upper.iloc[i-1]:
            trend.iloc[i] = 1
        elif c.iloc[i] < final_lower.iloc[i-1]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i-1]

    return trend


def compute_cpr(df_daily: pd.DataFrame):
    if len(df_daily) < 2:
        return None
    prev  = df_daily.iloc[-2]
    pivot = (float(prev["high"]) + float(prev["low"]) + float(prev["close"])) / 3
    bc    = (float(prev["high"]) + float(prev["low"])) / 2
    tc    = (2 * pivot) - bc
    if tc < bc:
        tc, bc = bc, tc
    width_pct = abs(tc - bc) / pivot * 100
    return {
        "pivot":     round(pivot, 2),
        "bc":        round(bc, 2),
        "tc":        round(tc, 2),
        "width_pct": round(width_pct, 3),
        "narrow":    width_pct < 0.20,
    }


def get_daily_candles(ikey: str, days: int = 5) -> pd.DataFrame:
    """Fetch recent daily OHLC for CPR and trend calculation."""
    enc   = ikey.replace("|", "%7C").replace(" ", "%20")
    to_dt = date.today().isoformat()
    fr_dt = (date.today() - timedelta(days=days + 5)).isoformat()
    r     = requests.get(
        f"{BASE_URL}/historical-candle/{enc}/1day/{fr_dt}/{to_dt}",
        headers=HEADERS, timeout=5)
    r.raise_for_status()
    candles = r.json()["data"]["candles"]
    df = pd.DataFrame(candles, columns=["ts","open","high","low","close","vol","oi"])
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").reset_index(drop=True)
    for col in ["open","high","low","close"]: df[col] = df[col].astype(float)
    return df


# ─────────────────────────────────────────────
# MULTI-TIMEFRAME TREND ANALYSIS
# ─────────────────────────────────────────────
def compute_tf_trend(df: pd.DataFrame, label: str = "") -> dict:
    """
    Analyse a single timeframe's candles to determine trend direction.

    Uses 4 signals (majority vote):
      1. EMA 9/21 crossover       — is EMA(9) > EMA(21)?
      2. EMA 9/21 slope           — is EMA(21) rising (bullish) or falling (bearish)?
      3. DMI bias                 — is +DI > -DI?
      4. Supertrend(9, 1×ATR)     — is trend = +1 (up)?

    Returns:
      {
        "direction":  "UPTREND" | "DOWNTREND" | "SIDEWAYS",
        "bias":       "CE" | "PE" | "NEUTRAL",
        "strength":   0–4  (how many of the 4 signals agree),
        "signals": {
          "ema_cross":    True/False  (EMA9 > EMA21),
          "ema_slope":    True/False  (EMA21 rising),
          "dmi_bullish":  True/False  (+DI > -DI),
          "supertrend":   True/False  (ST uptrend),
        },
        "ema9":  float, "ema21": float,
        "adx": float, "pdi": float, "ndi": float,
        "label": str,
      }
    """
    MIN_BARS = 30
    fallback = {
        "direction": "SIDEWAYS", "bias": "NEUTRAL", "strength": 0,
        "signals": {"ema_cross": False, "ema_slope": False,
                    "dmi_bullish": False, "supertrend": False},
        "ema9": 0, "ema21": 0, "adx": 0, "pdi": 0, "ndi": 0,
        "label": label,
    }
    if df is None or len(df) < MIN_BARS:
        return fallback

    close = df["close"].astype(float)

    # ── Signal 1: EMA 9/21 crossover ──
    ema9  = compute_ema(close, TF_EMA_FAST)
    ema21 = compute_ema(close, TF_EMA_SLOW)
    ema_cross_bull = bool(ema9.iloc[-1] > ema21.iloc[-1])
    ema_cross_bear = not ema_cross_bull

    # ── Signal 2: EMA 21 slope (rising vs falling) ──
    # Compare current EMA21 to 3 bars ago
    if len(ema21) >= 4:
        ema_slope_bull = bool(ema21.iloc[-1] > ema21.iloc[-4])
        ema_slope_bear = bool(ema21.iloc[-1] < ema21.iloc[-4])
    else:
        ema_slope_bull = False
        ema_slope_bear = False

    # ── Signal 3: DMI bias ──
    dmi = compute_dmi(df, ADX_PERIOD)
    if dmi:
        dmi_bullish = dmi["pdi"] > dmi["ndi"]
        dmi_bearish = dmi["ndi"] > dmi["pdi"]
    else:
        dmi_bullish = dmi_bearish = False
        dmi = {"adx": 0, "pdi": 0, "ndi": 0}

    # ── Signal 4: Supertrend(9, 1×ATR) ──
    st_series = compute_supertrend(df, period=9, multiplier=1.0)
    st_bull = bool(st_series.iloc[-1] == 1)
    st_bear = bool(st_series.iloc[-1] == -1)

    # ── Majority vote ──
    bull_count = sum([ema_cross_bull, ema_slope_bull, dmi_bullish, st_bull])
    bear_count = sum([ema_cross_bear, ema_slope_bear, dmi_bearish, st_bear])

    if bull_count >= 3:
        direction, bias, strength = "UPTREND", "CE", bull_count
    elif bear_count >= 3:
        direction, bias, strength = "DOWNTREND", "PE", bear_count
    elif bull_count >= 2 and bear_count <= 1:
        direction, bias, strength = "UPTREND", "CE", bull_count
    elif bear_count >= 2 and bull_count <= 1:
        direction, bias, strength = "DOWNTREND", "PE", bear_count
    else:
        direction, bias, strength = "SIDEWAYS", "NEUTRAL", max(bull_count, bear_count)

    return {
        "direction": direction,
        "bias":      bias,
        "strength":  strength,
        "signals": {
            "ema_cross":   ema_cross_bull if direction != "DOWNTREND" else not ema_cross_bear,
            "ema_slope":   ema_slope_bull if direction != "DOWNTREND" else not ema_slope_bear,
            "dmi_bullish": dmi_bullish    if direction != "DOWNTREND" else not dmi_bearish,
            "supertrend":  st_bull        if direction != "DOWNTREND" else not st_bear,
        },
        "ema9":  round(float(ema9.iloc[-1]), 2),
        "ema21": round(float(ema21.iloc[-1]), 2),
        "adx":   dmi["adx"],
        "pdi":   dmi["pdi"],
        "ndi":   dmi["ndi"],
        "label": label,
    }


def compute_mtf_trend(df_15m: pd.DataFrame, df_5m: pd.DataFrame) -> dict:
    """
    Multi-timeframe trend: 15m establishes direction, 5m confirms.

    Trade only when both 15m and 5m agree on direction.
    Strength = combined signal count (0–8).

    Returns:
      {
        "tf_15m":     compute_tf_trend result for 15m,
        "tf_5m":      compute_tf_trend result for 5m,
        "aligned":    True if 15m and 5m agree (or both neutral),
        "direction":  combined direction,
        "bias":       "CE" | "PE" | "NEUTRAL",
        "strength":   0–8  (sum of both timeframe signal counts),
        "trade_ok":   True if alignment is strong enough to trade,
      }
    """
    t15 = compute_tf_trend(df_15m, "15m")
    t5  = compute_tf_trend(df_5m,  "5m")

    # Alignment: both must agree on direction (or at least one neutral + other trending)
    both_bull = (t15["bias"] == "CE" and t5["bias"] == "CE")
    both_bear = (t15["bias"] == "PE" and t5["bias"] == "PE")
    # Partial alignment: 15m trending + 5m neutral (or vice versa)
    partial_bull = (t15["bias"] == "CE" and t5["bias"] == "NEUTRAL") or \
                   (t15["bias"] == "NEUTRAL" and t5["bias"] == "CE")
    partial_bear = (t15["bias"] == "PE" and t5["bias"] == "NEUTRAL") or \
                   (t15["bias"] == "NEUTRAL" and t5["bias"] == "PE")

    combined_strength = t15["strength"] + t5["strength"]

    if both_bull:
        direction, bias = "UPTREND", "CE"
        aligned, trade_ok = True, True
    elif both_bear:
        direction, bias = "DOWNTREND", "PE"
        aligned, trade_ok = True, True
    elif partial_bull:
        direction, bias = "UPTREND", "CE"
        aligned, trade_ok = True, (combined_strength >= 5)
    elif partial_bear:
        direction, bias = "DOWNTREND", "PE"
        aligned, trade_ok = True, (combined_strength >= 5)
    else:
        # Conflict (15m bull + 5m bear or vice versa) or both neutral
        direction, bias = "SIDEWAYS", "NEUTRAL"
        aligned, trade_ok = False, False

    return {
        "tf_15m":    t15,
        "tf_5m":     t5,
        "aligned":   aligned,
        "direction": direction,
        "bias":      bias,
        "strength":  combined_strength,
        "trade_ok":  trade_ok,
    }


def _mtf_score(side: str, mtf: dict) -> float:
    """
    Score 0–1 for how well the option side aligns with the MTF trend.
      CE + both-TF-UPTREND   = 1.0    PE + both-TF-DOWNTREND = 1.0
      Partial alignment       = 0.7
      CE + SIDEWAYS           = 0.4    PE + SIDEWAYS          = 0.4
      CE + DOWNTREND          = 0.05   PE + UPTREND           = 0.05
    Combined strength modulates the score.
    """
    bias     = mtf.get("bias", "NEUTRAL")
    strength = mtf.get("strength", 0)
    aligned  = mtf.get("aligned", False)
    str_factor = min(strength / 8.0, 1.0)  # 0..1 over 8 max signals

    if bias == side:
        # Aligned: full score × strength
        base = 1.0 if aligned else 0.7
        return base * max(str_factor, 0.3)
    elif bias == "NEUTRAL":
        return 0.4
    else:
        # Against the trend
        return max(0.05, 0.15 * (1 - str_factor))


def evaluate_entry_signal(df: pd.DataFrame, cpr: dict, spot: float,
                          mtf_trend: dict | None = None) -> dict:
    """
    Evaluate all entry conditions on 3-min candle data.

    MULTI-TIMEFRAME GATE (conditions 1–2):
      1. 15-min trend established (UPTREND for CE, DOWNTREND for PE)
      2. 5-min trend confirms same direction

    EXECUTION CONDITIONS on 3-min candles (conditions 3–9):
      3. 9-EMA crosses above 21-EMA (FRESH crossover)
      4. ADX >= 25
      5. +DI >= 25 (CE) or -DI >= 25 (PE)
      6. spot > CPR tc (CE) or spot < CPR bc (PE)
      7. CPR is narrow
      8. Supertrend(9, ATR×1) = uptrend (CE) / downtrend (PE)
      9. Supertrend(21, ATR×2) = uptrend (CE) / downtrend (PE)

    ALL 9 conditions must be true for a signal.
    """
    MIN_BARS = 30
    if len(df) < MIN_BARS or cpr is None:
        return {"signal": "NONE", "reason": "Insufficient data"}

    close = df["close"].astype(float)

    # ── MTF trend gate ──
    mtf = mtf_trend or {}
    tf_15m = mtf.get("tf_15m", {})
    tf_5m  = mtf.get("tf_5m", {})
    mtf_bias    = mtf.get("bias", "NEUTRAL")
    mtf_aligned = mtf.get("aligned", False)
    mtf_trade_ok = mtf.get("trade_ok", False)

    # 15m trend direction
    tf15_bull = tf_15m.get("bias") == "CE"
    tf15_bear = tf_15m.get("bias") == "PE"
    # 5m confirms
    tf5_bull  = tf_5m.get("bias") == "CE"
    tf5_bear  = tf_5m.get("bias") == "PE"

    # ── MAs (3-min candles) ──
    ema9  = compute_ema(close, 9)
    ema21 = compute_ema(close, 21)
    ma9_above_21      = bool(ema9.iloc[-1] > ema21.iloc[-1])
    ma9_prev_below_21 = bool(ema9.iloc[-2] <= ema21.iloc[-2])
    ma9_prev_above_21 = bool(ema9.iloc[-2] >= ema21.iloc[-2])
    # FIX: require fresh crossover, not just current position
    ma_cross_up   = ma9_above_21 and ma9_prev_below_21
    ma_cross_down = (not ma9_above_21) and ma9_prev_above_21

    # ── DMI (3-min candles) ──
    dmi = compute_dmi(df, 14)
    if dmi is None:
        return {"signal": "NONE", "reason": "ADX unavailable"}
    adx_ok  = dmi["adx"] >= 25
    pdi_ok  = dmi["pdi"] >= 25
    ndi_ok  = dmi["ndi"] >= 25

    # ── CPR ──
    cpr_narrow     = cpr.get("narrow", False)
    above_cpr      = spot > cpr["tc"]
    below_cpr      = spot < cpr["bc"]

    # ── Supertrends (3-min candles) ──
    st9  = compute_supertrend(df, period=9,  multiplier=1.0)
    st21 = compute_supertrend(df, period=21, multiplier=2.0)
    st9_up   = bool(st9.iloc[-1]  == 1)
    st21_up  = bool(st21.iloc[-1] == 1)
    st9_dn   = bool(st9.iloc[-1]  == -1)
    st21_dn  = bool(st21.iloc[-1] == -1)

    # ── CE signal (all 9 must be true) ──
    ce_conditions = {
        "15m trend UPTREND":        tf15_bull,
        "5m trend confirms UP":     tf5_bull,
        "9-MA crosses above 21-MA": ma_cross_up,
        "ADX ≥ 25":                 adx_ok,
        "+DI ≥ 25":                 pdi_ok,
        "Price > CPR upper":        above_cpr,
        "Narrow CPR":               cpr_narrow,
        "ST(9,1) uptrend":          st9_up,
        "ST(21,2) uptrend":         st21_up,
    }
    ce_signal = all(ce_conditions.values())

    # ── PE signal (all 9 must be true) ──
    pe_conditions = {
        "15m trend DOWNTREND":      tf15_bear,
        "5m trend confirms DOWN":   tf5_bear,
        "9-MA crosses below 21-MA": ma_cross_down,
        "ADX ≥ 25":                 adx_ok,
        "-DI ≥ 25":                 ndi_ok,
        "Price < CPR lower":        below_cpr,
        "Narrow CPR":               cpr_narrow,
        "ST(9,1) downtrend":        st9_dn,
        "ST(21,2) downtrend":       st21_dn,
    }
    pe_signal = all(pe_conditions.values())

    signal = "CE" if ce_signal else "PE" if pe_signal else "NONE"

    return {
        "signal":        signal,
        "ce_conditions": ce_conditions,
        "pe_conditions": pe_conditions,
        "ema9":          round(float(ema9.iloc[-1]), 2),
        "ema21":         round(float(ema21.iloc[-1]), 2),
        "adx":           dmi["adx"],
        "pdi":           dmi["pdi"],
        "ndi":           dmi["ndi"],
        "st9_trend":     "up" if st9_up else "down",
        "st21_trend":    "up" if st21_up else "down",
        "cpr":           cpr,
        "ce_count":      sum(ce_conditions.values()),
        "pe_count":      sum(pe_conditions.values()),
        "mtf_bias":      mtf_bias,
        "mtf_aligned":   mtf_aligned,
        "exec_tf":       "3m",
    }


def find_atm_option(spot: float, ikey: str, side: str) -> dict | None:
    """Find the ATM option (strike closest to spot) for the given side."""
    expiry = get_nearest_expiry(ikey)
    chain  = get_chain(ikey, expiry)
    best   = None
    min_dist = float("inf")
    for strike_data in chain:
        sp   = float(strike_data["strike_price"])
        dist = abs(sp - spot)
        opt_key = "call_options" if side == "CE" else "put_options"
        opt     = strike_data.get(opt_key, {})
        md      = opt.get("market_data", {})
        ltp     = md.get("ltp", 0) or 0
        oi      = md.get("oi", 0) or 0
        ikey_opt = opt.get("instrument_key", "")
        if ltp > 0 and oi > 0 and ikey_opt and dist < min_dist:
            min_dist = dist
            best = {
                "strike":    sp,
                "side":      side,
                "ltp":       ltp,
                "oi":        oi,
                "prev_oi":   md.get("prev_oi", 0) or 0,
                "iv":        opt.get("option_greeks",{}).get("iv",0) or 0,
                "delta":     opt.get("option_greeks",{}).get("delta",0) or 0,
                "bid_price": md.get("bid_price", 0) or 0,
                "ask_price": md.get("ask_price", 0) or 0,
                "volume":    md.get("volume", 0) or 0,
                "ikey":      ikey_opt,
                "expiry":    expiry,
                "dist":      dist,
            }
    return best


# ─────────────────────────────────────────────
# UPSTOX API CALLS
# ─────────────────────────────────────────────
def get_live_price(ikey):
    r = requests.get(f"{BASE_URL}/market-quote/ohlc", headers=HEADERS,
                     params={"instrument_key": ikey, "interval": "1d"}, timeout=5)
    r.raise_for_status()
    feeds = r.json()["data"]
    return float(feeds[list(feeds.keys())[0]]["last_price"])

def get_intraday_candles(ikey, interval="1minute"):
    """Fetch intraday candles for any supported interval (1minute, 3minute, 5minute, 15minute, etc.)."""
    enc = ikey.replace("|", "%7C").replace(" ", "%20")
    r   = requests.get(f"{BASE_URL}/historical-candle/intraday/{enc}/{interval}",
                       headers=HEADERS, timeout=5)
    r.raise_for_status()
    c   = r.json()["data"]["candles"]
    df  = pd.DataFrame(c, columns=["ts","open","high","low","close","vol","oi"])
    df["ts"] = pd.to_datetime(df["ts"])
    df  = df.sort_values("ts").reset_index(drop=True)
    for col in ["open","high","low","close"]: df[col] = df[col].astype(float)
    return df

def get_candles(ikey):
    """Legacy 1-minute candle fetch (used by per-option ADX scoring)."""
    return get_intraday_candles(ikey, CANDLE_INTERVAL)

def get_nearest_expiry(ikey):
    r = requests.get(f"{BASE_URL}/option/contract", headers=HEADERS,
                     params={"instrument_key": ikey}, timeout=5)
    r.raise_for_status()
    exp    = sorted({i["expiry"] for i in r.json()["data"] if i.get("expiry")})
    today  = date.today().isoformat()
    future = [e for e in exp if e >= today]
    return future[0] if future else exp[-1]

def get_chain(ikey, expiry):
    r = requests.get(f"{BASE_URL}/option/chain", headers=HEADERS,
                     params={"instrument_key": ikey, "expiry_date": expiry}, timeout=8)
    r.raise_for_status()
    return r.json()["data"]

def _matrix_score(side: str, price_rising: bool, oi_rising: bool) -> float:
    if price_rising and oi_rising:
        return 1.0 if side == "CE" else 0.0
    if not price_rising and oi_rising:
        return 0.0 if side == "CE" else 1.0
    if price_rising and not oi_rising:
        return 0.5 if side == "CE" else 0.0
    return 0.0 if side == "CE" else 0.5


def _score_option(row: dict, chain_iv_min: float, chain_iv_max: float,
                   chain_oi_max: float, chain_vol_max: float,
                   price_rising: bool, oi_rising: bool,
                   mtf_trend: dict | None = None) -> float:
    """
    Composite 0-100 score:
      ADX(25) + Delta(15) + OI(15) + Volume(10) + Spread(10) + Matrix(15) + MTF(10)
    All weights sum to 100.
    """
    # ── ADX score (0-1) ──
    adx_score = min(row.get("adx", 0) / 60.0, 1.0)

    # ── Delta score (0-1) ──
    delta_abs = abs(row.get("delta", 0.5))
    if delta_abs <= 0.5:
        delta_score = (delta_abs - DELTA_MIN) / (0.5 - DELTA_MIN)
    else:
        delta_score = (DELTA_MAX - delta_abs) / (DELTA_MAX - 0.5)
    delta_score = max(0.0, min(delta_score, 1.0))

    # ── OI score (0-1) ──
    oi_score = row.get("oi", 0) / max(chain_oi_max, 1)

    # ── Volume score (0-1) ──
    vol_score = row.get("volume", 0) / max(chain_vol_max, 1)

    # ── Spread score (0-1) ──
    bid = row.get("bid_price", 0)
    ask = row.get("ask_price", 0)
    ltp = row.get("ltp", 1)
    if ask > bid > 0 and ltp > 0:
        spread_score = max(0.0, 1.0 - (ask - bid) / ltp * 100 / MAX_SPREAD_PCT)
    else:
        spread_score = 0.5

    # ── OI Momentum Matrix score (0-1) ──
    mat_score = _matrix_score(row.get("side", "CE"), price_rising, oi_rising)

    # ── Multi-timeframe trend alignment score (0-1) ──
    if mtf_trend:
        mtf_sc = _mtf_score(row.get("side", "CE"), mtf_trend)
    else:
        mtf_sc = 0.4  # neutral if no trend data

    composite = (
        W_ADX       * adx_score     +
        W_DELTA     * delta_score   +
        W_OI        * oi_score      +
        W_VOLUME    * vol_score     +
        W_SPREAD    * spread_score  +
        W_MATRIX    * mat_score     +
        W_MTF       * mtf_sc
    ) / 100.0

    return round(composite * 100, 2)


@st.cache_data(ttl=60, show_spinner=False)
def find_best_option(spot_bucket, ikey, h_opt_ltp_avg: float = 0.0,
                     mtf_bias: str = "NEUTRAL",
                     mtf_strength: int = 0,
                     mtf_aligned: bool = False):
    """
    Cached for 60 s — returns instantly on cache hit.

    Selection pipeline (updated with multi-timeframe trend):
      1. Fetch full option chain for nearest expiry
      2. Hard filter: ltp>0, oi>0, |delta| in [DELTA_MIN, DELTA_MAX],
                      bid-ask spread <= MAX_SPREAD_PCT
      3. If MTF trend is aligned and strong (strength >= 5), pre-filter to keep only
         the trend-aligned side (CE for uptrend, PE for downtrend)
      4. OI percentile filter: keep top OI_PERCENTILE% by OI
      5. Distance filter: TOP_N_STRIKES closest to spot
      6. Fetch 1-min candles → compute ADX/+DI/-DI for each survivor
      7. Composite score: ADX(25) + Delta(15) + OI(15) + Volume(10)
                          + Spread(10) + Matrix(15) + MTF(10)
      8. Return highest scorer with all metadata
    """
    # Reconstruct a minimal MTF dict for scoring (cache-friendly — no dict arg)
    mtf_trend = {
        "bias": mtf_bias,
        "strength": mtf_strength,
        "aligned": mtf_aligned,
        "direction": "UPTREND" if mtf_bias == "CE"
                     else "DOWNTREND" if mtf_bias == "PE"
                     else "SIDEWAYS",
    }

    expiry = get_nearest_expiry(ikey)
    chain  = get_chain(ikey, expiry)
    rows   = []

    for strike in chain:
        sp = strike["strike_price"]
        for side, opt_key in [("CE","call_options"), ("PE","put_options")]:
            opt      = strike.get(opt_key, {})
            md       = opt.get("market_data", {})
            greeks   = opt.get("option_greeks", {})
            oi       = md.get("oi", 0)       or 0
            ltp      = md.get("ltp", 0)      or 0
            prev_oi  = md.get("prev_oi", 0)  or 0
            volume   = md.get("volume", 0)   or 0
            bid      = md.get("bid_price", 0) or 0
            ask      = md.get("ask_price", 0) or 0
            iv       = greeks.get("iv", 0)    or 0
            delta    = greeks.get("delta", 0) or 0
            o_ikey   = opt.get("instrument_key", "")

            if ltp <= 0 or oi <= 0 or not o_ikey:
                continue

            if not (DELTA_MIN <= abs(delta) <= DELTA_MAX):
                continue

            if ask > bid > 0 and ltp > 0:
                if (ask - bid) / ltp * 100 > MAX_SPREAD_PCT:
                    continue

            rows.append({
                "strike":    sp,
                "side":      side,
                "ltp":       ltp,
                "oi":        oi,
                "prev_oi":   prev_oi,
                "volume":    volume,
                "bid_price": bid,
                "ask_price": ask,
                "iv":        iv,
                "delta":     delta,
                "ikey":      o_ikey,
                "dist":      abs(sp - spot_bucket),
                "expiry":    expiry,
            })

    if not rows:
        return None

    df = pd.DataFrame(rows)

    # ── MTF trend pre-filter: strong aligned trend → keep only aligned side ──
    if mtf_aligned and mtf_strength >= 5 and mtf_bias in ("CE", "PE"):
        df_filtered = df[df["side"] == mtf_bias]
        # Fall back to full set if filter leaves nothing
        if len(df_filtered) > 0:
            df = df_filtered

    # ── Chain-level stats for normalisation ──
    chain_iv_min  = df["iv"].min()
    chain_iv_max  = df["iv"].max()
    chain_oi_max  = df["oi"].max()
    chain_vol_max = df["volume"].max() if df["volume"].max() > 0 else 1

    # ── OI percentile filter ──
    oi_thr = np.percentile(df["oi"], OI_PERCENTILE)
    df     = df[df["oi"] >= oi_thr]

    # ── Distance filter ──
    df_top = df.nsmallest(TOP_N_STRIKES, "dist")

    # ── Fetch ADX for each candidate, score ──
    results = []
    for _, row in df_top.iterrows():
        try:
            dmi = compute_dmi(get_candles(row["ikey"]), ADX_PERIOD)
            if not dmi:
                continue
            candidate = {**row.to_dict(), **dmi}

            _oi_rising    = (candidate["oi"] - candidate.get("prev_oi", candidate["oi"])) > 0
            _price_rising = (candidate["ltp"] > h_opt_ltp_avg) if h_opt_ltp_avg > 0 else True

            candidate["composite_score"] = _score_option(
                candidate, chain_iv_min, chain_iv_max, chain_oi_max, chain_vol_max,
                _price_rising, _oi_rising, mtf_trend
            )
            candidate["matrix_price_rising"] = _price_rising
            candidate["matrix_oi_rising"]    = _oi_rising
            results.append(candidate)
        except Exception:
            pass

    if not results:
        return None

    best = max(results, key=lambda x: x["composite_score"])

    # ── IV rank ──
    iv_range        = max(chain_iv_max - chain_iv_min, 1e-6)
    best["iv_rank"] = round((best["iv"] - chain_iv_min) / iv_range * 100, 1)

    # ── Matrix quadrant ──
    _pr = best.get("matrix_price_rising", True)
    _oir = best.get("matrix_oi_rising", True)
    if _pr and _oir:
        best["matrix_signal"] = "Long Buildup"
        best["matrix_rec"]    = "Buy CE"
    elif not _pr and _oir:
        best["matrix_signal"] = "Short Buildup"
        best["matrix_rec"]    = "Buy PE"
    elif _pr and not _oir:
        best["matrix_signal"] = "Short Covering"
        best["matrix_rec"]    = "Caution — Exit PE"
    else:
        best["matrix_signal"] = "Long Unwinding"
        best["matrix_rec"]    = "Caution — Exit CE"

    best["matrix_aligned"] = (
        (best["matrix_rec"] == "Buy CE" and best["side"] == "CE") or
        (best["matrix_rec"] == "Buy PE" and best["side"] == "PE")
    )

    # ── MTF alignment flag ──
    best["mtf_aligned"] = (
        (mtf_bias == "CE" and best["side"] == "CE") or
        (mtf_bias == "PE" and best["side"] == "PE") or
        mtf_bias == "NEUTRAL"
    )
    best["mtf_bias"]     = mtf_bias
    best["mtf_strength"] = mtf_strength

    # ── Score breakdown ──
    _delta_abs = abs(best.get("delta", 0.5))
    _d_score = max(0, min(
        (_delta_abs - DELTA_MIN) / (0.5 - DELTA_MIN) if _delta_abs <= 0.5
        else (DELTA_MAX - _delta_abs) / (DELTA_MAX - 0.5), 1))
    _mtf_sc = _mtf_score(best["side"], mtf_trend)
    best["score_breakdown"] = {
        "ADX":     round(W_ADX    * min(best.get("adx", 0) / 60, 1), 1),
        "Delta":   round(W_DELTA  * _d_score, 1),
        "OI":      round(W_OI     * best.get("oi", 0) / max(chain_oi_max, 1), 1),
        "Volume":  round(W_VOLUME * best.get("volume", 0) / max(chain_vol_max, 1), 1),
        "Spread":  round(W_SPREAD * max(0, 1 - (best.get("ask_price", 0) - best.get("bid_price", 0))
                        / max(best.get("ltp", 1), 1) * 100 / MAX_SPREAD_PCT), 1),
        "Matrix":  round(W_MATRIX * _matrix_score(best["side"], _pr, _oir), 1),
        "MTF":     round(W_MTF * _mtf_sc, 1),
    }
    return best

# ─────────────────────────────────────────────
# LIVE ORDER PLACEMENT
# ─────────────────────────────────────────────
def place_market_order(instrument_token, qty, transaction_type="BUY"):
    payload = {
        "quantity":           qty,
        "product":            "D",
        "validity":           "DAY",
        "price":              0,
        "instrument_token":   instrument_token,
        "order_type":         "MARKET",
        "transaction_type":   transaction_type,
        "disclosed_quantity": 0,
        "trigger_price":      0,
        "is_amo":             False,
        "tag":                "nifty-pilot",
    }
    r = requests.post(ORDER_URL, json=payload, headers=HEADERS, timeout=8)
    r.raise_for_status()
    return r.json()["data"]["order_id"]

def place_sl_order(instrument_token, qty, trigger_price, limit_price, transaction_type="SELL"):
    payload = {
        "quantity":           qty,
        "product":            "D",
        "validity":           "DAY",
        "price":              limit_price,
        "instrument_token":   instrument_token,
        "order_type":         "SL",
        "transaction_type":   transaction_type,
        "disclosed_quantity": 0,
        "trigger_price":      trigger_price,
        "is_amo":             False,
        "tag":                "nifty-pilot-sl",
    }
    r = requests.post(ORDER_URL, json=payload, headers=HEADERS, timeout=8)
    r.raise_for_status()
    return r.json()["data"]["order_id"]

# ─────────────────────────────────────────────
# SCORING
# ─────────────────────────────────────────────
def buy_confidence(dmi, opt):
    h_opt_ltp = ha(st.session_state.h_opt_ltp)
    h_opt_adx = ha(st.session_state.h_opt_adx)
    h_oi      = ha(st.session_state.h_oi)
    checks = [
        ("Option ADX vs 15m avg",  opt["adx"] > h_opt_adx and opt["adx"] > 30,           25),
        ("Index ADX strength",     dmi["adx"] >= 30,                                      20),
        ("+DI/-DI direction",
            (opt["side"]=="CE" and opt["pdi"] > opt["ndi"]) or
            (opt["side"]=="PE" and opt["ndi"] > opt["pdi"]),                               25),
        ("OI vs 15m avg",          opt["oi"] >= h_oi if h_oi else True,                   15),
        ("Option price rising",    opt["ltp"] > h_opt_ltp if h_opt_ltp else True,         15),
    ]
    score = sum(p for _, passed, p in checks if passed)
    return score, checks

# ─────────────────────────────────────────────
# MOCK DATA
# ─────────────────────────────────────────────
def mock_data(idx_name):
    bases = {"Nifty 50":24500,"Nifty Bank":51000,"Nifty Fin Svc":22000,
             "Nifty Midcap 50":12000,"Sensex":80000}
    base = bases.get(idx_name, 24500)
    px   = round(base + np.random.uniform(-80, 80), 2)
    adx  = round(30 + np.random.uniform(-3, 15), 2)
    pdi  = round(20 + np.random.uniform(-4, 10), 2)
    ndi  = round(10 + np.random.uniform(-4, 10), 2)
    bullish = pdi > ndi; side = "CE" if bullish else "PE"
    strike  = round(px/50)*50 + (50 if bullish else -50)
    oi      = int(np.random.uniform(1e5, 5e5))
    opt_ltp = round(np.random.uniform(80, 300), 2)
    bid     = round(opt_ltp * 0.99, 2)
    ask     = round(opt_ltp * 1.01, 2)
    delta   = round(np.random.uniform(0.3, 0.7) * (1 if side == "CE" else -1), 3)
    volume  = int(np.random.uniform(5e4, 5e5))
    iv      = round(np.random.uniform(12, 30), 2)
    cs      = round(np.random.uniform(55, 90), 1)
    best = {
        "strike":    strike, "side": side,
        "ltp":       opt_ltp,
        "oi":        oi, "prev_oi": int(oi * np.random.uniform(.85, 1.15)),
        "volume":    volume,
        "bid_price": bid, "ask_price": ask,
        "iv":        iv,  "iv_rank": round(np.random.uniform(20, 70), 1),
        "delta":     delta,
        "adx":       round(adx + np.random.uniform(-5, 10), 2),
        "pdi":       pdi, "ndi": ndi,
        "expiry":    (date.today() + timedelta(days=3)).isoformat(),
        "ikey":      f"NSE_FO|MOCK{strike}{side}",
        "composite_score":      cs,
        "matrix_price_rising": bullish,
        "matrix_oi_rising":    oi > int(np.random.uniform(1e5, 5e5)) * 0.9,
        "matrix_signal":       "Long Buildup" if bullish else "Short Buildup",
        "matrix_rec":          "Buy CE" if bullish else "Buy PE",
        "matrix_aligned":      True,
        "mtf_aligned":         True,
        "mtf_bias":            "CE" if bullish else "PE",
        "mtf_strength":        6,
        "score_breakdown": {
            "ADX":     round(cs * 0.25, 1), "Delta":  round(cs * 0.15, 1),
            "OI":      round(cs * 0.15, 1), "Volume": round(cs * 0.10, 1),
            "Spread":  round(cs * 0.10, 1), "Matrix": round(cs * 0.15, 1),
            "MTF":     round(cs * 0.10, 1),
        },
    }
    return px, {"adx":adx,"pdi":pdi,"ndi":ndi}, best

# ─────────────────────────────────────────────
# TRAILING SL LOGIC
# ─────────────────────────────────────────────
def update_trailing_sl(current_ltp, entry_price, tsl_pct):
    pnl = current_ltp - entry_price
    if pnl > st.session_state.highest_pnl:
        st.session_state.highest_pnl = pnl
    trail_lock  = st.session_state.highest_pnl * (1 - tsl_pct / 100)
    sl_abs      = entry_price + max(0, trail_lock)
    return round(sl_abs, 2)

# ─────────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────────
st.set_page_config(page_title="Nifty Pilot", layout="wide", page_icon="")

# ── TOP BAR ──────────────────────────────────
hdr_l, hdr_idx, hdr_px, hdr_mode = st.columns([2, 2, 1.5, 1])
hdr_l.title("Nifty Pilot")

selected = hdr_idx.selectbox(
    "Index",
    options=list(INDICES.keys()),
    index=list(INDICES.keys()).index(st.session_state.selected_index),
    label_visibility="collapsed",
)
if selected != st.session_state.selected_index:
    for k in ["h_times","h_opt_ltp","h_opt_adx","h_idx_adx",
              "h_opt_pdi","h_opt_ndi","h_oi"]:
        st.session_state[k] = deque(maxlen=HISTORY_LEN)
    st.session_state.selected_index = selected
    st.session_state.trade_active   = False
    # Clear trend cache on index change
    for k in list(st.session_state.keys()):
        if k.startswith("_cpr_cache_") or k.startswith("_mtf_cache_"):
            del st.session_state[k]

ikey     = INDICES[selected]["key"]
lot_size = INDICES[selected]["lot"]

_spot_placeholder = hdr_px.empty()
_mode_placeholder = hdr_mode.empty()

# ── LIVE FRAGMENT — reruns every 5 s

@st.fragment(run_every=5)
def _fetch_data():
    # Live spot price
    if MOCK_MODE:
        spot_display = round({"Nifty 50":24500,"Nifty Bank":51000,"Nifty Fin Svc":22000,
                              "Nifty Midcap 50":12000,"Sensex":80000}.get(selected,24500)
                             + np.random.uniform(-30,30), 2)
    else:
        try:
            spot_display = get_live_price(ikey)
        except Exception:
            spot_display = 0.0

    _spot_placeholder.metric("Spot", f"₹{spot_display:,.2f}")
    if MOCK_MODE:
        _mode_placeholder.warning("Mock")
    else:
        _mode_placeholder.success("Live")

    st.session_state.sv_mock_banner = MOCK_MODE

    # ── FETCH MARKET DATA ──
    try:
        if MOCK_MODE:
            live_px, dmi, best_opt = mock_data(selected)
        else:
            live_px = get_live_price(ikey)
            df_idx  = get_candles(ikey)          # 1-min for index DMI
            dmi     = compute_dmi(df_idx, ADX_PERIOD)
            if not dmi:
                st.session_state.sv_fetch_error = "Waiting for candles — market may have just opened."
                return

            # ── Multi-timeframe trend (refreshed every cycle) ──
            _mtf_key = f"_mtf_cache_{ikey}"
            try:
                df_15m = get_intraday_candles(ikey, TF_15M)
                df_5m  = get_intraday_candles(ikey, TF_5M)
                mtf_trend = compute_mtf_trend(df_15m, df_5m)
                st.session_state[_mtf_key] = mtf_trend
            except Exception:
                mtf_trend = st.session_state.get(_mtf_key, {
                    "tf_15m": {"direction":"SIDEWAYS","bias":"NEUTRAL","strength":0,
                               "signals":{},"ema9":0,"ema21":0,"adx":0,"pdi":0,"ndi":0,"label":"15m"},
                    "tf_5m":  {"direction":"SIDEWAYS","bias":"NEUTRAL","strength":0,
                               "signals":{},"ema9":0,"ema21":0,"adx":0,"pdi":0,"ndi":0,"label":"5m"},
                    "aligned": False, "direction": "SIDEWAYS",
                    "bias": "NEUTRAL", "strength": 0, "trade_ok": False,
                })

            spot_bucket    = round(live_px, -2)
            _h_ltp_avg     = ha(st.session_state.h_opt_ltp)
            best_opt       = find_best_option(
                spot_bucket, ikey, _h_ltp_avg,
                mtf_bias=mtf_trend["bias"],
                mtf_strength=mtf_trend["strength"],
                mtf_aligned=mtf_trend["aligned"]
            )
    except Exception as e:
        st.session_state.sv_fetch_error = f"API error: {e}"
        return

    if not best_opt:
        st.session_state.sv_fetch_error = "No option signal — chain may be empty or market just opened."
        return
    st.session_state.sv_fetch_error = ""

    # Patch cached option LTP with fresh quote
    if not MOCK_MODE:
        try:
            best_opt = {**best_opt, "ltp": get_live_price(best_opt["ikey"])}
        except Exception:
            pass

    # Push rolling history
    now_ts = ts()
    st.session_state.h_times.append(now_ts)
    st.session_state.h_opt_ltp.append(best_opt["ltp"])
    st.session_state.h_opt_adx.append(best_opt["adx"])
    st.session_state.h_idx_adx.append(dmi["adx"])
    st.session_state.h_opt_pdi.append(best_opt["pdi"])
    st.session_state.h_opt_ndi.append(best_opt["ndi"])
    st.session_state.h_oi.append(best_opt["oi"])

    score, checks    = buy_confidence(dmi, best_opt)
    idx_g            = gear(dmi["adx"])
    opt_g            = gear(best_opt["adx"])
    bullish          = dmi["pdi"] > dmi["ndi"]
    oi_chg           = best_opt["oi"] - best_opt.get("prev_oi", best_opt["oi"])

    # ── CPR + entry signal (entry on 3-min candles with MTF gate) ──
    if not MOCK_MODE:
        try:
            _daily_key = f"_cpr_cache_{ikey}"
            if _daily_key not in st.session_state:
                df_daily = get_daily_candles(ikey, days=5)
                st.session_state[_daily_key] = compute_cpr(df_daily)
            cpr = st.session_state[_daily_key]
        except Exception:
            cpr = None
        try:
            # Fetch 3-min candles for execution timeframe
            df_3m = get_intraday_candles(ikey, TF_3M)
            entry_sig = evaluate_entry_signal(df_3m, cpr, live_px, mtf_trend=mtf_trend)
        except Exception:
            entry_sig = {"signal": "NONE", "reason": "Indicator error"}
    else:
        _mock_spot = live_px
        cpr = {
            "pivot": round(_mock_spot, 2),
            "bc":    round(_mock_spot * 0.999, 2),
            "tc":    round(_mock_spot * 1.001, 2),
            "width_pct": round(abs(_mock_spot*1.001 - _mock_spot*0.999) / _mock_spot * 100, 3),
            "narrow": True,
        }
        _mock_df = pd.DataFrame({
            "open":  [live_px]*50,
            "high":  [live_px + i*0.3 + 5 for i in range(50)],
            "low":   [live_px + i*0.3 - 5 for i in range(50)],
            "close": [live_px + i*0.5     for i in range(50)],
        })
        # Mock MTF trend
        _mock_tf = {
            "direction": "UPTREND" if bullish else "DOWNTREND",
            "bias": "CE" if bullish else "PE",
            "strength": 3,
            "signals": {"ema_cross": bullish, "ema_slope": bullish,
                        "dmi_bullish": bullish, "supertrend": bullish},
            "ema9": live_px, "ema21": live_px,
            "adx": dmi["adx"], "pdi": dmi["pdi"], "ndi": dmi["ndi"],
            "label": "",
        }
        mtf_trend = {
            "tf_15m": {**_mock_tf, "label": "15m"},
            "tf_5m":  {**_mock_tf, "label": "5m"},
            "aligned": True,
            "direction": "UPTREND" if bullish else "DOWNTREND",
            "bias": "CE" if bullish else "PE",
            "strength": 6,
            "trade_ok": True,
        }
        entry_sig = evaluate_entry_signal(_mock_df, cpr, live_px, mtf_trend=mtf_trend)

    # Cache all derived vars
    st.session_state.sv_live_px    = live_px
    st.session_state.sv_dmi        = dmi
    st.session_state.sv_best_opt   = best_opt
    st.session_state.sv_score      = score
    st.session_state.sv_checks     = checks
    st.session_state.sv_idx_g      = idx_g
    st.session_state.sv_opt_g      = opt_g
    st.session_state.sv_bullish    = bullish
    st.session_state.sv_oi_chg     = oi_chg
    st.session_state.sv_now_ts     = now_ts
    st.session_state.sv_cpr        = cpr
    st.session_state.sv_entry_sig  = entry_sig
    st.session_state.sv_di_targets = get_di_targets(selected, dmi)
    st.session_state.sv_trend_3d   = mtf_trend


@st.fragment(run_every=5)
def _frag_index():
    if st.session_state.get("sv_mock_banner"):
        st.info("Mock mode — add Upstox token to `.streamlit/secrets.toml` to go live.")
    err = st.session_state.get("sv_fetch_error", "")
    if err:
        st.warning(err)
        return
    if "sv_live_px" not in st.session_state or st.session_state.sv_live_px is None:
        st.caption("Loading…")
        return
    live_px = st.session_state.sv_live_px
    dmi     = st.session_state.sv_dmi
    idx_g   = st.session_state.sv_idx_g
    bullish = st.session_state.sv_bullish
    sig     = "BULLISH" if bullish else "BEARISH"
    if st.session_state.last_signal and st.session_state.last_signal != sig:
        browser_alert("SIGNAL CHANGE", f"{selected} is now {sig}")
    st.session_state.last_signal = sig
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("LTP",       f"₹{live_px:,.2f}")
    _di_t = st.session_state.get("sv_di_targets") or get_di_targets(selected, dmi)
    c2.metric("ADX", f"{dmi['adx']:.2f}",
              f"{_di_t['label']} → target {_di_t['target']} pts")
    c3.metric("+DI / -DI", f"{dmi['pdi']:.2f} / {dmi['ndi']:.2f}")
    c4.metric("Signal",    sig)

    # ── MULTI-TIMEFRAME TREND CARD ──
    trend = st.session_state.get("sv_trend_3d")
    if trend and trend.get("tf_15m"):
        _td = trend["direction"]
        _tb = trend["bias"]
        _ts = trend["strength"]
        _aligned = trend.get("aligned", False)
        _trade_ok = trend.get("trade_ok", False)
        _t15 = trend["tf_15m"]
        _t5  = trend["tf_5m"]

        if _td == "UPTREND":
            _t_col, _t_bg, _t_bdr, _t_icon = "#27500A", "#EAF3DE", "#3B6D11", "▲"
        elif _td == "DOWNTREND":
            _t_col, _t_bg, _t_bdr, _t_icon = "#791F1F", "#FCEBEB", "#A32D2D", "▼"
        else:
            _t_col, _t_bg, _t_bdr, _t_icon = "#633806", "#FAEEDA", "#854F0B", "◆"

        _strength_bar = "●" * _ts + "○" * (8 - _ts)
        _align_label = "ALIGNED ✓" if _aligned else "CONFLICT ✗"
        _trade_label = "Trade OK" if _trade_ok else "No Trade"
        _trade_bg    = "#3B6D11" if _trade_ok else "#A32D2D"

        # Per-timeframe signal checkmarks
        def _tf_signals_html(tf_data, label):
            sigs = tf_data.get("signals", {})
            d = tf_data.get("direction", "SIDEWAYS")
            b = tf_data.get("bias", "NEUTRAL")
            s = tf_data.get("strength", 0)
            if d == "UPTREND":
                _c, _i = "#27500A", "▲"
            elif d == "DOWNTREND":
                _c, _i = "#791F1F", "▼"
            else:
                _c, _i = "#633806", "◆"
            sig_items = "".join(
                f'<span style="color:{"#27500A" if v else "#888"};margin-right:8px">'
                f'{"✓" if v else "✗"} {k}</span>'
                for k, v in sigs.items()
            )
            return (
                f'<div style="margin-top:6px">'
                f'<div style="display:flex;align-items:center;gap:8px">'
                f'<span style="font-size:12px;font-weight:600;color:{_c}">{_i} {label} {d}</span>'
                f'<span style="font-size:10px;color:{_c};opacity:.7">({s}/4)</span>'
                f'</div>'
                f'<div style="font-size:10px;display:flex;flex-wrap:wrap;margin-top:2px">{sig_items}</div>'
                f'</div>'
            )

        _tf15_html = _tf_signals_html(_t15, "15m")
        _tf5_html  = _tf_signals_html(_t5, "5m")

        st.markdown(f"""
<div style="border:1.5px solid {_t_bdr};border-radius:10px;
            background:{_t_bg};padding:11px 16px;margin:10px 0">
  <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px">
    <div>
      <div style="font-size:10px;font-weight:500;color:{_t_col};
                  letter-spacing:.06em;margin-bottom:3px">MULTI-TIMEFRAME TREND (15m + 5m → 3m)</div>
      <div style="display:flex;align-items:center;gap:10px">
        <span style="font-size:20px;font-weight:500;color:{_t_col}">{_t_icon} {_td}</span>
        <span style="background:{_t_bdr};color:#fff;border-radius:5px;
                     padding:2px 9px;font-size:11px;font-weight:500">Bias: {_tb}</span>
        <span style="background:{_trade_bg};color:#fff;border-radius:5px;
                     padding:2px 9px;font-size:11px;font-weight:500">{_trade_label}</span>
        <span style="font-size:14px;color:{_t_col};letter-spacing:2px">{_strength_bar}</span>
      </div>
    </div>
    <div style="text-align:right">
      <div style="font-size:10px;color:{_t_col};opacity:.7">Alignment</div>
      <div style="font-size:16px;font-weight:500;color:{_t_col}">{_align_label}</div>
      <div style="font-size:10px;color:{_t_col};opacity:.7">Strength {_ts}/8</div>
    </div>
  </div>
  {_tf15_html}
  {_tf5_html}
</div>
""", unsafe_allow_html=True)


@st.fragment(run_every=5)
def _frag_buy_trade():
    if "sv_best_opt" not in st.session_state:
        st.caption("Loading option data…")
        return
    best_opt   = st.session_state.sv_best_opt
    dmi        = st.session_state.sv_dmi
    score      = st.session_state.sv_score
    opt_g      = st.session_state.sv_opt_g
    oi_chg     = st.session_state.sv_oi_chg
    bullish    = st.session_state.sv_bullish
    now_ts     = st.session_state.sv_now_ts
    checks     = st.session_state.sv_checks
    di_tgt     = st.session_state.get("sv_di_targets") or get_di_targets(selected, dmi)
    trend_3d   = st.session_state.get("sv_trend_3d")
    st.divider()

    # Best option buy box
    st.subheader("Best option to buy")
    bdr = "#3B6D11" if best_opt["side"] == "CE" else "#854F0B"
    bbg = "#f0faf4" if best_opt["side"] == "CE" else "#fff8f0"

    # Trend alignment indicator for the buy box
    _t3d_aligned = best_opt.get("trend3d_aligned", True)
    _t3d_bias    = best_opt.get("trend3d_bias", "NEUTRAL")
    _t3d_str     = best_opt.get("trend3d_strength", 0)
    _t3d_tag     = ""
    if _t3d_bias != "NEUTRAL":
        if _t3d_aligned:
            _t3d_tag = (f'<span style="background:#3B6D11;color:#fff;border-radius:4px;'
                        f'padding:1px 7px;font-size:10px;margin-left:8px">'
                        f'3D trend ✓ {_t3d_bias} ({_t3d_str}/4)</span>')
        else:
            _t3d_tag = (f'<span style="background:#A32D2D;color:#fff;border-radius:4px;'
                        f'padding:1px 7px;font-size:10px;margin-left:8px">'
                        f'Against 3D trend ({_t3d_bias} {_t3d_str}/4)</span>')

    st.markdown(f"""
    <div style="border:2px solid {bdr};border-radius:12px;padding:14px 18px;background:{bbg};margin-bottom:14px">
      <div style="display:flex;justify-content:space-between;align-items:flex-start">
        <div>
          <div style="font-size:11px;font-weight:500;color:{bdr};letter-spacing:.05em;margin-bottom:3px">BUY SIGNAL{_t3d_tag}</div>
          <div style="display:flex;align-items:baseline;gap:8px">
            <span style="font-size:26px;font-weight:500;color:{bdr}">{best_opt['side']} {int(best_opt['strike'])}</span>
            <span style="font-size:12px;color:gray">exp {best_opt['expiry']}</span>
          </div>
          <div style="font-size:12px;color:gray;margin-top:2px">LTP &nbsp;
            <span style="font-size:20px;font-weight:500;color:#111">₹{best_opt['ltp']:.2f}</span>
          </div>
        </div>
        <div style="text-align:right">
          <div style="font-size:10px;color:gray">Option ADX</div>
          <div style="font-size:30px;font-weight:500;color:{bdr}">{best_opt['adx']:.2f}</div>
          <div style="font-size:11px;color:gray">Gear {opt_g}</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Row 1: core metrics
    d1,d2,d3,d4,d5,d6 = st.columns(6)
    d1.metric("OI",           f"{best_opt['oi']:,}")
    d2.metric("OI change",    f"{oi_chg:+,}", f"{oi_chg/max(best_opt.get('prev_oi',1),1)*100:+.1f}%")
    d3.metric("IV",           f"{best_opt['iv']:.1f}%")
    d4.metric("IV rank",      f"{best_opt.get('iv_rank', 0):.0f}/100",
              help="0=cheapest, 100=most expensive vs today's chain")
    d5.metric("|Delta|",      f"{abs(best_opt.get('delta', 0)):.2f}",
              help="0.4–0.6 = near ATM ideal")
    d6.metric("Volume",       f"{best_opt.get('volume', 0):,}")

    # Row 2: composite score breakdown (now includes Trend3D)
    sb = best_opt.get("score_breakdown", {})
    cs = best_opt.get("composite_score", 0)
    score_color = "#27500A" if cs >= 70 else "#633806" if cs >= 50 else "#791F1F"
    score_bg    = "#EAF3DE" if cs >= 70 else "#FAEEDA" if cs >= 50 else "#FCEBEB"
    _sb_html = "".join(
        f'<div style="font-size:11px;color:{score_color}">'
        f'<span style="opacity:.65">{k}</span>&nbsp;<strong>{v}</strong></div>'
        for k, v in sb.items()
    )
    st.markdown(
        f'<div style="border-radius:8px;background:{score_bg};padding:10px 14px;'
        f'margin:8px 0;display:flex;align-items:center;gap:16px;flex-wrap:wrap">'
        f'<div style="font-size:12px;color:{score_color}">'
        f'<span style="font-size:20px;font-weight:500">{cs:.0f}</span>/100 composite score'
        f'</div>{_sb_html}</div>',
        unsafe_allow_html=True
    )

    # Row 3: bid-ask spread
    bid = best_opt.get("bid_price", 0); ask = best_opt.get("ask_price", 0)
    if bid and ask:
        spread_pct = (ask - bid) / max(best_opt["ltp"], 1) * 100
        spread_col = "#3B6D11" if spread_pct < 1.5 else "#BA7517" if spread_pct < 3 else "#A32D2D"
        st.markdown(
            f"<span style='font-size:12px;color:{spread_col}'>"
            f"Bid ₹{bid:.2f} &nbsp;/&nbsp; Ask ₹{ask:.2f} &nbsp;|&nbsp; "
            f"Spread {spread_pct:.1f}%"
            f"{'&nbsp;✓ tight' if spread_pct < 1.5 else '&nbsp;⚠ wide' if spread_pct > 3 else ''}"
            f"</span>",
            unsafe_allow_html=True
        )

    # ── OI MOMENTUM DECISION MATRIX ──
    _price_rising  = best_opt.get("matrix_price_rising", oi_chg > 0)
    _oi_rising     = best_opt.get("matrix_oi_rising",    oi_chg > 0)
    _matrix_signal = best_opt.get("matrix_signal", "Long Buildup")
    _matrix_rec    = best_opt.get("matrix_rec",    "Buy CE")
    _aligned       = best_opt.get("matrix_aligned", True)

    _MATRIX_STYLES = {
        "Long Buildup":   ("#27500A","#EAF3DE","#3B6D11","▲",
                           "New buyers entering — trend confirmed"),
        "Short Buildup":  ("#791F1F","#FCEBEB","#A32D2D","▼",
                           "New sellers aggressive — bearish pressure"),
        "Short Covering": ("#633806","#FAEEDA","#854F0B","△",
                           "Sellers exiting; rally may be weak or unsustained"),
        "Long Unwinding": ("#633806","#FAEEDA","#854F0B","▽",
                           "Buyers exiting; trend weakening or reversing"),
    }
    _matrix_col, _matrix_bg, _matrix_border, _matrix_icon, _matrix_detail = \
        _MATRIX_STYLES.get(_matrix_signal, _MATRIX_STYLES["Long Buildup"])

    _align_txt = "Matrix aligned — included in score" if _aligned \
                 else "Matrix conflicts — score penalised"
    _align_col = "#27500A" if _aligned else "#A32D2D"

    st.markdown(
        f'''<div style="border:1.5px solid {_matrix_border};border-radius:10px;
                padding:12px 16px;background:{_matrix_bg};margin:10px 0">
      <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px">
    <div>
      <div style="font-size:10px;font-weight:500;letter-spacing:.06em;
                  color:{_matrix_col};margin-bottom:2px">OI MOMENTUM MATRIX</div>
      <div style="display:flex;align-items:center;gap:8px">
        <span style="font-size:18px;font-weight:500;color:{_matrix_col}">{_matrix_icon} {_matrix_signal}</span>
        <span style="background:{_matrix_border};color:#fff;border-radius:5px;
                     padding:2px 9px;font-size:11px;font-weight:500">{_matrix_rec}</span>
      </div>
      <div style="font-size:12px;color:{_matrix_col};margin-top:3px;opacity:.85">{_matrix_detail}</div>
    </div>
    <div style="text-align:right">
      <div style="font-size:11px;color:{_align_col};font-weight:500">{_align_txt}</div>
      <div style="font-size:10px;color:{_matrix_col};margin-top:2px;opacity:.7">
        Price {'rising' if _price_rising else 'falling'} vs 15m avg
        &nbsp;|&nbsp; OI {'increasing' if _oi_rising else 'decreasing'}
        &nbsp;|&nbsp; Chg {oi_chg:+,}
      </div>
    </div>
      </div>
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-top:10px;
              font-size:10px;border-top:0.5px solid {_matrix_border};padding-top:8px">
    <div style="text-align:center;padding:4px;border-radius:4px;background:#EAF3DE">
      <div style="color:#3B6D11;font-weight:500">Price ▲ OI ▲</div>
      <div style="color:#27500A">Long Buildup</div>
      <div style="color:#3B6D11;font-weight:500">Buy CE</div>
    </div>
    <div style="text-align:center;padding:4px;border-radius:4px;background:#FCEBEB">
      <div style="color:#A32D2D;font-weight:500">Price ▼ OI ▲</div>
      <div style="color:#791F1F">Short Buildup</div>
      <div style="color:#A32D2D;font-weight:500">Buy PE</div>
    </div>
    <div style="text-align:center;padding:4px;border-radius:4px;background:#FAEEDA">
      <div style="color:#854F0B;font-weight:500">Price ▲ OI ▼</div>
      <div style="color:#633806">Short Covering</div>
      <div style="color:#854F0B;font-weight:500">Exit PE</div>
    </div>
    <div style="text-align:center;padding:4px;border-radius:4px;background:#FAEEDA">
      <div style="color:#854F0B;font-weight:500">Price ▼ OI ▼</div>
      <div style="color:#633806">Long Unwinding</div>
      <div style="color:#854F0B;font-weight:500">Exit CE</div>
    </div>
      </div>
    </div>''',
        unsafe_allow_html=True
    )
    st.caption(f"OI insight: {_matrix_detail}   |   `{best_opt['ikey']}`")

    st.divider()

    # ── DI SCENARIO TARGET CARD ──
    _dt = di_tgt
    _dt_col  = "#27500A" if _dt["scenario"] == "strong" else "#185FA5"
    _dt_bg   = "#EAF3DE" if _dt["scenario"] == "strong" else "#E6F1FB"
    _dt_bdr  = "#3B6D11" if _dt["scenario"] == "strong" else "#185FA5"
    st.markdown(f"""
<div style="border:1.5px solid {_dt_bdr};border-radius:10px;
            background:{_dt_bg};padding:11px 16px;margin-bottom:12px">
  <div style="font-size:10px;font-weight:500;color:{_dt_col};
              letter-spacing:.06em;margin-bottom:6px">
    DI SCENARIO — {selected.upper()}
  </div>
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px">
    <div>
      <div style="font-size:10px;color:{_dt_col};opacity:.7">Scenario</div>
      <div style="font-size:14px;font-weight:500;color:{_dt_col}">{_dt["label"]}</div>
    </div>
    <div>
      <div style="font-size:10px;color:{_dt_col};opacity:.7">Active target</div>
      <div style="font-size:18px;font-weight:500;color:{_dt_col}">{_dt["target"]} pts</div>
    </div>
    <div>
      <div style="font-size:10px;color:{_dt_col};opacity:.7">Immediate move</div>
      <div style="font-size:14px;font-weight:500;color:{_dt_col}">{_dt["immediate"]} pts</div>
    </div>
    <div>
      <div style="font-size:10px;color:{_dt_col};opacity:.7">Trend potential</div>
      <div style="font-size:14px;font-weight:500;color:{_dt_col}">{_dt["trend"]} pts</div>
    </div>
  </div>
  <div style="display:flex;gap:16px;margin-top:8px;font-size:11px;color:{_dt_col};opacity:.8">
    <span>ADX {_dt["adx"]:.1f}</span>
    <span>+DI {_dt["pdi"]:.1f}</span>
    <span>-DI {_dt["ndi"]:.1f}</span>
    <span>SL = {_dt["sl"]} pts (50% of target)</span>
    <span>{"Using trend target — ADX ≥ 30" if _dt["adx"] >= 30 else "Using immediate target — ADX building"}</span>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── MOCKUP TRADE CONTROL ──
    entry_sig = st.session_state.get("sv_entry_sig") or {"signal":"NONE","reason":"Loading…"}
    cpr       = st.session_state.get("sv_cpr") or {}
    sig       = entry_sig.get("signal", "NONE")

    st.subheader("Trade control — mockup")
    st.caption(f"SL ≈ 50% of target | Min hold {MIN_HOLD_SECS//60} min | ATM option | Signal-gated | No real order sent.")

    # ── Entry Signal Dashboard ──
    _ce_cond = entry_sig.get("ce_conditions", {})
    _pe_cond = entry_sig.get("pe_conditions", {})
    _ce_cnt  = entry_sig.get("ce_count", 0)
    _pe_cnt  = entry_sig.get("pe_count", 0)
    _sig_col = "#3B6D11" if sig=="CE" else "#A32D2D" if sig=="PE" else "#854F0B"
    _sig_bg  = "#EAF3DE" if sig=="CE" else "#FCEBEB" if sig=="PE" else "#FAEEDA"
    _sig_lbl = (f"BUY CE signal active — {_ce_cnt}/7 conditions met" if sig=="CE"
                else f"BUY PE signal active — {_pe_cnt}/7 conditions met" if sig=="PE"
                else f"No signal — CE:{_ce_cnt}/7  PE:{_pe_cnt}/7 conditions met")

    st.markdown(
        f'''<div style="border-radius:8px;background:{_sig_bg};border:1.5px solid {_sig_col};
                padding:10px 14px;margin-bottom:10px">
  <div style="font-size:13px;font-weight:500;color:{_sig_col};margin-bottom:8px">{_sig_lbl}</div>
  <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:4px 16px;font-size:11px">
    {" ".join(
        f'<div style="color:{"#27500A" if v else "#888"}">' +
        ("✓ " if v else "✗ ") + k + "</div>"
        for k,v in (_ce_cond if sig=="CE" else _pe_cond if sig=="PE"
                    else {**{k: v for k,v in _ce_cond.items()}, **{k: v for k,v in _pe_cond.items()}}).items()
    )}
  </div>
  <div style="margin-top:8px;font-size:10px;color:{_sig_col};opacity:.8">
    EMA9={entry_sig.get("ema9","—")}  EMA21={entry_sig.get("ema21","—")}
    ADX={entry_sig.get("adx","—")}  +DI={entry_sig.get("pdi","—")}  -DI={entry_sig.get("ndi","—")}
    &nbsp;|&nbsp; ST(9,1)={entry_sig.get("st9_trend","—")}  ST(21,2)={entry_sig.get("st21_trend","—")}
    &nbsp;|&nbsp; CPR {cpr.get("bc","—")}–{cpr.get("tc","—")} ({"narrow ✓" if cpr.get("narrow") else "wide ✗"})
  </div>
</div>''',
        unsafe_allow_html=True
    )

    if not st.session_state.trade_active or st.session_state.trade_mode != "mockup":
        if sig == "NONE":
            st.info("Waiting for a valid signal — all 7 conditions must align for CE or PE entry.")
        _btn_label = (f"Enter ATM {sig} trade (mockup)" if sig != "NONE"
                      else "Enter trade — override (no signal)")
        _btn_type  = "primary" if sig != "NONE" else "secondary"
        if st.button(_btn_label, use_container_width=True, type=_btn_type):
            _trade_side = sig if sig != "NONE" else best_opt["side"]
            if not MOCK_MODE and sig != "NONE":
                _atm = find_atm_option(
                    st.session_state.get("sv_live_px", best_opt["ltp"]),
                    ikey, _trade_side)
                _opt = _atm if _atm else best_opt
            else:
                _opt = best_opt
                _opt = {**_opt, "side": _trade_side}
            _tgt   = di_tgt["target"]
            _sl_pt = di_tgt["sl"]
            st.session_state.trade_active  = True
            st.session_state.trade_mode    = "mockup"
            st.session_state.entry_price   = _opt["ltp"]
            st.session_state.entry_adx     = _opt.get("adx", best_opt.get("adx", 0))
            st.session_state.entry_pdi     = _opt.get("pdi", best_opt.get("pdi", 0))
            st.session_state.entry_ndi     = _opt.get("ndi", best_opt.get("ndi", 0))
            st.session_state.entry_side    = _opt["side"]
            st.session_state.entry_strike  = int(_opt["strike"])
            st.session_state.entry_expiry  = _opt["expiry"]
            st.session_state.entry_ikey    = _opt["ikey"]
            st.session_state.entry_opt_ltp = _opt["ltp"]
            st.session_state.entry_signal  = sig
            st.session_state.target_pts    = _tgt
            st.session_state.exit_price    = round(_opt["ltp"] + _tgt, 2)
            st.session_state.sl_price      = round(_opt["ltp"] - _sl_pt, 2)
            st.session_state.highest_pnl   = 0.0
            st.session_state.monitor_start = time.time()
            add_log("act_log", {
                "time": now_ts,
                "event": (f"[MOCK] {'ATM ' if sig!='NONE' else ''}Entered {_opt['side']} "
                          f"{int(_opt['strike'])} @ ₹{_opt['ltp']:.2f} "
                          f"| Target +{_tgt} | SL −{_sl_pt} "
                          f"| Signal:{sig}")
            })
            st.rerun()

    elif st.session_state.trade_active and st.session_state.trade_mode == "mockup":
        _entry     = st.session_state.entry_price
        _sl_fixed  = st.session_state.sl_price
        _target    = st.session_state.exit_price
        _tgt_pts   = st.session_state.target_pts
        _sl_pts    = round(_entry - _sl_fixed, 2)

        pnl        = round(best_opt["ltp"] - _entry, 2)
        elapsed    = int(time.time() - st.session_state.monitor_start) if st.session_state.monitor_start else 0
        min_held   = elapsed >= MIN_HOLD_SECS
        remaining  = max(0, MIN_HOLD_SECS - elapsed)
        rm, rs     = divmod(remaining, 60)

        if min_held:
            tsl = update_trailing_sl(best_opt["ltp"], _entry,
                                     st.session_state.trailing_sl_pct)
            effective_sl = max(tsl, _sl_fixed)
        else:
            effective_sl = _sl_fixed

        st.session_state.sl_price = effective_sl

        _side    = st.session_state.entry_side
        _strike  = st.session_state.entry_strike
        _expiry  = st.session_state.entry_expiry
        _ikey    = st.session_state.entry_ikey
        _eltp    = st.session_state.entry_opt_ltp
        _opt_bdr = "#3B6D11" if _side == "CE" else "#854F0B"
        _opt_bg  = "#EAF3DE" if _side == "CE" else "#FAEEDA"
        _opt_tc  = "#27500A" if _side == "CE" else "#633806"
        _cur_ltp = best_opt["ltp"] if best_opt.get("ikey") == _ikey else _eltp
        _ltp_chg = round(_cur_ltp - _eltp, 2)
        _chg_col = "#3B6D11" if _ltp_chg >= 0 else "#A32D2D"
        st.markdown(f"""
<div style="border:2px solid {_opt_bdr};border-radius:10px;padding:11px 16px;
        background:{_opt_bg};margin-bottom:12px;
        display:flex;align-items:center;justify-content:space-between">
  <div style="display:flex;align-items:center;gap:10px">
    <span style="background:{_opt_bdr};color:#fff;border-radius:5px;
                 padding:2px 9px;font-size:11px;font-weight:500">{_side}</span>
    <span style="font-size:20px;font-weight:500;color:{_opt_tc}">{_strike}</span>
    <span style="font-size:12px;color:{_opt_tc};opacity:.75">exp {_expiry}</span>
  </div>
  <div style="text-align:right">
    <span style="font-size:18px;font-weight:500;color:{_opt_tc}">₹{_cur_ltp:.2f}</span>
    <span style="font-size:12px;color:{_chg_col};margin-left:6px">
      {'+' if _ltp_chg >= 0 else ''}{_ltp_chg:.2f} from entry</span>
  </div>
</div>
""", unsafe_allow_html=True)

        # ── Hold timer bar ──
        if not min_held:
            hold_pct = int(elapsed / MIN_HOLD_SECS * 100)
            hold_col = "#185FA5" if hold_pct < 50 else "#BA7517" if hold_pct < 85 else "#3B6D11"
            st.markdown(f"""
<div style="margin-bottom:10px">
  <div style="display:flex;justify-content:space-between;font-size:11px;
              color:var(--color-text-secondary);margin-bottom:3px">
    <span>Min hold — {rm}:{rs:02d} remaining</span>
    <span>{hold_pct}%</span>
  </div>
  <div style="height:6px;border-radius:3px;background:var(--color-background-secondary)">
    <div style="height:100%;width:{hold_pct}%;border-radius:3px;
                background:{hold_col};transition:width 1s"></div>
  </div>
  <div style="font-size:10px;color:var(--color-text-tertiary);margin-top:2px">
    SL &amp; exit locked until {MIN_HOLD_SECS//60}-min minimum hold completes
  </div>
</div>
""", unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div style="font-size:11px;color:#3B6D11;margin-bottom:8px">'
                f'✓ {MIN_HOLD_SECS//60}-min minimum hold complete — SL and exit active</div>',
                unsafe_allow_html=True
            )

        _warn_msg = (f"Trade live — Target +{_tgt_pts} pts  |  "
                     f"SL −{_sl_pts} pts  |  "
                     f"{'TSL active' if min_held else f'Hold {rm}:{rs:02d}'}")
        st.warning(_warn_msg)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Entry",        f"₹{_entry:,.2f}")
        m2.metric("Target",       f"₹{_target:,.2f}", f"+{_tgt_pts} pts")
        m3.metric("Stop loss",    f"₹{effective_sl:,.2f}",
                  f"−{_sl_pts} pts{'  (fixed)' if not min_held else '  (trailing)'}")
        m4.metric("Live P&L",     f"₹{pnl:+,.2f}")
        m5.metric("Hold time",    f"{elapsed//60}m {elapsed%60:02d}s",
                  "✓ free to exit" if min_held else f"locked {rm}:{rs:02d}")

        if best_opt["ltp"] >= _target:
            browser_alert("TARGET HIT", f"[MOCK] P&L: ₹{pnl:+.2f}")
            st.balloons()
            st.success(f"Target hit! P&L: ₹{pnl:+,.2f}")
            add_log("act_log", {"time":now_ts,
                "event":f"[MOCK] Target ₹{_target} hit @ ₹{best_opt['ltp']:.2f} | P&L ₹{pnl:+.2f}"})
            st.session_state.trade_active = False
            st.rerun()

        if min_held and best_opt["ltp"] <= effective_sl:
            _sl_type = "TSL" if tsl > _sl_fixed else "SL"
            browser_alert(f"{_sl_type} HIT", f"[MOCK] P&L: ₹{pnl:+.2f}")
            st.error(f"{_sl_type} hit at ₹{effective_sl:,.2f} | P&L: ₹{pnl:+,.2f}")
            add_log("act_log", {"time":now_ts,
                "event":f"[MOCK] {_sl_type} ₹{effective_sl:.2f} hit @ ₹{best_opt['ltp']:.2f} | P&L ₹{pnl:+.2f}"})
            st.session_state.trade_active = False
            st.rerun()

        if not min_held:
            st.caption(f"⚠ Emergency exit available but min-hold ({rm}:{rs:02d} left) not complete.")
        if st.button("Emergency exit (mockup)", use_container_width=True,
                     type="secondary", disabled=False):
            add_log("act_log", {"time":now_ts,
                "event":f"[MOCK] Emergency exit @ ₹{best_opt['ltp']:.2f} | P&L ₹{pnl:+.2f} | held {elapsed}s"})
            st.session_state.trade_active = False
            st.rerun()

    if st.session_state.act_log:
        with st.expander("Activity log", expanded=False):
            st.dataframe(pd.DataFrame(st.session_state.act_log[:20]),
                         hide_index=True, use_container_width=True)


@st.fragment(run_every=5)
def _frag_analysis():
    if "sv_best_opt" not in st.session_state:
        st.caption("Loading…")
        return
    best_opt = st.session_state.sv_best_opt
    dmi      = st.session_state.sv_dmi
    score    = st.session_state.sv_score
    checks   = st.session_state.sv_checks
    now_ts   = st.session_state.sv_now_ts
    st.markdown("---")
    st.subheader(f"Signal analysis — {MIN_HOLD_SECS//60}-min comparison & trade monitor")

    left, right = st.columns([1.2, 1])
    with left:
        def pct(cur, ref): return round((cur-ref)/abs(ref)*100,1) if ref else 0.0
        h15 = {k: ha(st.session_state[f"h_{k}"]) for k in
               ["opt_ltp","opt_adx","idx_adx","opt_pdi","opt_ndi","oi"]}
        cmp = {
            "Metric":  ["Option LTP","Option ADX","Index ADX","+DI","-DI","OI"],
            "Now":     [f"₹{best_opt['ltp']:.2f}",f"{best_opt['adx']:.2f}",
                        f"{dmi['adx']:.2f}",f"{best_opt['pdi']:.2f}",
                        f"{best_opt['ndi']:.2f}",f"{best_opt['oi']:,}"],
            "15m avg": [f"₹{h15['opt_ltp']:.2f}",f"{h15['opt_adx']:.2f}",
                        f"{h15['idx_adx']:.2f}",f"{h15['opt_pdi']:.2f}",
                        f"{h15['opt_ndi']:.2f}",f"{h15['oi']:,.0f}"],
            "Change":  [f"{pct(best_opt['ltp'],h15['opt_ltp']):+.1f}%",
                        f"{pct(best_opt['adx'],h15['opt_adx']):+.1f}%",
                        f"{pct(dmi['adx'],h15['idx_adx']):+.1f}%",
                        f"{pct(best_opt['pdi'],h15['opt_pdi']):+.1f}%",
                        f"{pct(best_opt['ndi'],h15['opt_ndi']):+.1f}%",
                        f"{pct(best_opt['oi'],h15['oi']):+.1f}%"],
        }
        st.dataframe(pd.DataFrame(cmp), hide_index=True, use_container_width=True)

        if len(st.session_state.h_opt_ltp) > 1:
            chart_df = pd.DataFrame({
                "Option LTP": list(st.session_state.h_opt_ltp),
                "Option ADX": list(st.session_state.h_opt_adx),
            }, index=list(st.session_state.h_times))
            st.line_chart(chart_df, height=160)

    with right:
        verdict_lbl = "BUY" if score>=75 else "WAIT" if score>=45 else "AVOID"
        if score >= 75:   st.success(f"**BUY** — Confidence {score}/100")
        elif score >= 45: st.warning(f"**WAIT** — Confidence {score}/100")
        else:             st.error(f"**AVOID** — Confidence {score}/100")
        for label, passed, pts in checks:
            st.markdown(f"{'✅' if passed else '❌'} {label} &nbsp; `+{pts if passed else 0}/{pts}`")

    st.divider()

    # 5-min monitor
    st.subheader("5-min trade monitor")
    if st.session_state.monitor_start:
        elapsed   = int(time.time() - st.session_state.monitor_start)
        remaining = MONITOR_SECS - elapsed % MONITOR_SECS
        mm, ss    = divmod(remaining, 60)
        st.markdown(f"Next 5-min update in: **{mm}:{ss:02d}**")

    n1,n2,n3,n4 = st.columns(4)
    n1.metric("Entry",       f"₹{st.session_state.entry_price:,.2f}" if st.session_state.trade_active else "—")
    n2.metric("Current LTP", f"₹{best_opt['ltp']:.2f}")

    if st.session_state.trade_active:
        pnl2      = round(best_opt["ltp"] - st.session_state.entry_price, 2)
        adx_fall  = best_opt["adx"] < st.session_state.entry_adx - 5
        dir_flip  = ((best_opt["side"]=="CE" and best_opt["ndi"]>best_opt["pdi"]) or
                     (best_opt["side"]=="PE" and best_opt["pdi"]>best_opt["ndi"]))
        if   pnl2 > 20:    rec, rmsg = "EXIT", "Target reached — book profit"
        elif pnl2 < -15:   rec, rmsg = "EXIT", "Stop-loss zone"
        elif adx_fall:     rec, rmsg = "WEAK", "ADX falling — trend weakening"
        elif dir_flip:     rec, rmsg = "FLIP", "DI crossover — direction reversed"
        else:              rec, rmsg = "HOLD", f"Conditions holding (score {score}/100)"
        n3.metric("P&L", f"₹{pnl2:+,.2f}")
        n4.metric("Recommendation", rec)
        add_log("signal_log", {"time":now_ts,"type":rec,"msg":rmsg})
        if rec == "EXIT":   st.error(f"**{rec}** — {rmsg}")
        elif rec == "HOLD": st.success(f"**HOLD** — {rmsg}")
        else:               st.warning(f"**{rec}** — {rmsg}")
    else:
        n3.metric("P&L","—")
        n4.metric("Status","Ready" if score>=75 else "Wait" if score>=45 else "Avoid")

    if st.session_state.signal_log:
        st.dataframe(pd.DataFrame(st.session_state.signal_log[:10]),
                     hide_index=True, use_container_width=True)


@st.fragment(run_every=10)
def _frag_orders():
    if "sv_best_opt" not in st.session_state:
        st.caption("Loading…")
        return
    best_opt = st.session_state.sv_best_opt
    live_px  = st.session_state.sv_live_px
    opt_g    = st.session_state.sv_opt_g
    now_ts   = st.session_state.sv_now_ts
    st.markdown("---")
    st.subheader("Live order placement")

    if MOCK_MODE:
        st.warning(
            "Live order placement is **disabled in mock mode**. "
            "Add your Upstox access token to `.streamlit/secrets.toml` to enable real trading."
        )
    else:
        st.error(
            "Real money trading. Every order below is sent live to Upstox. "
            "Double-check all fields before confirming."
        )

        with st.expander("Configure & place live order", expanded=True):
            st.markdown("**Step 1 — Select the option to trade**")

            try:
                expiry_sel = get_nearest_expiry(ikey)
                chain_sel  = get_chain(ikey, expiry_sel)
            except Exception as e:
                st.error(f"Could not load chain: {e}"); chain_sel = []

            strikes = sorted({int(s["strike_price"]) for s in chain_sel})
            sides   = ["CE", "PE"]

            col_s, col_side, col_exp = st.columns(3)
            lo_strike_val = col_s.selectbox(
                "Strike", strikes,
                index=strikes.index(int(best_opt["strike"])) if int(best_opt["strike"]) in strikes else 0
            )
            lo_side_val   = col_side.selectbox(
                "Side", sides,
                index=sides.index(best_opt["side"])
            )
            col_exp.text_input("Expiry", value=expiry_sel, disabled=True)

            lo_ikey_val = None
            for row in chain_sel:
                if int(row["strike_price"]) == lo_strike_val:
                    opt_key = "call_options" if lo_side_val == "CE" else "put_options"
                    lo_ikey_val = row.get(opt_key, {}).get("instrument_key")
                    break

            lo_ltp = 0.0
            if lo_ikey_val:
                for row in chain_sel:
                    if int(row["strike_price"]) == lo_strike_val:
                        opt_key = "call_options" if lo_side_val == "CE" else "put_options"
                        lo_ltp  = row.get(opt_key,{}).get("market_data",{}).get("ltp",0) or 0
                        break
            if lo_ltp:
                st.caption(f"Selected option LTP: ₹{lo_ltp:.2f} | Key: `{lo_ikey_val}`")

            st.markdown("**Step 2 — Set quantity, target & trailing SL**")
            col_q, col_t, col_tsl = st.columns(3)
            lo_qty_val    = col_q.number_input("Lots", min_value=1, max_value=50, value=1, step=1)
            _live_di_tgt = st.session_state.get("sv_di_targets")
            _live_default = _live_di_tgt["target"] if _live_di_tgt else GEAR_PTS[opt_g]
            lo_target_val = col_t.number_input("Target (pts)", min_value=10, max_value=2000,
                                                value=_live_default, step=10)
            lo_tsl_val    = col_tsl.number_input("Trailing SL (%)", min_value=5.0, max_value=80.0,
                                                  value=30.0, step=5.0)

            qty_units     = lo_qty_val * lot_size
            target_px_val = round(lo_ltp + lo_target_val, 2) if lo_ltp else 0.0
            sl_px_val     = round(lo_ltp * (1 - lo_tsl_val / 100), 2) if lo_ltp else 0.0

            st.markdown("**Step 3 — Review & confirm**")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Entry LTP",    f"₹{lo_ltp:.2f}")
            r2.metric("Target price", f"₹{target_px_val:.2f}")
            r3.metric("Init SL",      f"₹{sl_px_val:.2f}")
            r4.metric("Total qty",    f"{qty_units} units ({lo_qty_val} lot{'s' if lo_qty_val>1 else ''})")

            st.markdown("**Step 4 — Place order**")
            col_place, col_cancel = st.columns(2)

            if not st.session_state.trade_active:
                confirm = col_place.checkbox("I confirm this is a live order with real money")
                if confirm:
                    if col_place.button("Place BUY order (LIVE)", use_container_width=True, type="primary"):
                        if not lo_ikey_val:
                            st.error("Could not resolve instrument key for the selected strike.")
                        else:
                            try:
                                order_id = place_market_order(lo_ikey_val, qty_units, "BUY")
                                sl_trigger = sl_px_val
                                sl_limit   = round(sl_trigger * 0.995, 2)
                                sl_oid     = place_sl_order(lo_ikey_val, qty_units, sl_trigger, sl_limit, "SELL")

                                st.session_state.trade_active   = True
                                st.session_state.trade_mode     = "live"
                                st.session_state.entry_price    = lo_ltp
                                st.session_state.entry_side     = lo_side_val
                                st.session_state.target_pts     = lo_target_val
                                st.session_state.exit_price     = target_px_val
                                st.session_state.sl_price       = sl_px_val
                                st.session_state.trailing_sl_pct= lo_tsl_val
                                st.session_state.highest_pnl    = 0.0
                                st.session_state.live_order_id  = order_id
                                st.session_state.monitor_start  = time.time()
                                st.session_state.lo_ikey        = lo_ikey_val
                                st.session_state.lo_qty         = qty_units

                                add_log("act_log", {
                                    "time": now_ts,
                                    "event": f"[LIVE] BUY {lo_side_val} {lo_strike_val} "
                                             f"qty={qty_units} order_id={order_id} | "
                                             f"SL order={sl_oid}"
                                })
                                st.success(f"Order placed! ID: {order_id}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Order failed: {e}")
            else:
                if st.session_state.trade_mode == "live":
                    pnl_live  = round(best_opt["ltp"] - st.session_state.entry_price, 2)
                    tsl_live  = update_trailing_sl(best_opt["ltp"], st.session_state.entry_price,
                                                   st.session_state.trailing_sl_pct)
                    st.session_state.sl_price = tsl_live
                    l1,l2,l3,l4 = st.columns(4)
                    l1.metric("Entry",          f"₹{st.session_state.entry_price:,.2f}")
                    l2.metric("Target",         f"₹{st.session_state.exit_price:,.2f}")
                    l3.metric("Trailing SL",    f"₹{tsl_live:,.2f}")
                    l4.metric("Live P&L",       f"₹{pnl_live:+,.2f}")

                    if best_opt["ltp"] >= st.session_state.exit_price:
                        browser_alert("TARGET HIT", f"[LIVE] P&L: +{pnl_live}")
                        st.success(f"[LIVE] Target hit! P&L: ₹{pnl_live:+,.2f}")
                        try:
                            exit_id = place_market_order(st.session_state.lo_ikey,
                                                         st.session_state.lo_qty, "SELL")
                            add_log("act_log", {"time":now_ts,
                                "event":f"[LIVE] Target exit order={exit_id} P&L ₹{pnl_live:+.2f}"})
                        except Exception as e:
                            st.error(f"Exit order failed: {e}")
                        st.session_state.trade_active = False; st.rerun()

                    if best_opt["ltp"] <= tsl_live and st.session_state.highest_pnl > 0:
                        browser_alert("TSL HIT", f"[LIVE] Trailing SL triggered. P&L: {pnl_live:+}")
                        st.error(f"[LIVE] Trailing SL hit! P&L: ₹{pnl_live:+,.2f}")
                        try:
                            exit_id = place_market_order(st.session_state.lo_ikey,
                                                         st.session_state.lo_qty, "SELL")
                            add_log("act_log", {"time":now_ts,
                                "event":f"[LIVE] TSL exit order={exit_id} P&L ₹{pnl_live:+.2f}"})
                        except Exception as e:
                            st.error(f"TSL exit order failed: {e}")
                        st.session_state.trade_active = False; st.rerun()

                    if col_cancel.button("Exit position (LIVE)", use_container_width=True):
                        try:
                            exit_id = place_market_order(st.session_state.lo_ikey,
                                                         st.session_state.lo_qty, "SELL")
                            add_log("act_log", {"time":now_ts,
                                "event":f"[LIVE] Manual exit order={exit_id} P&L ₹{pnl_live:+.2f}"})
                            browser_alert("EXIT", f"[LIVE] Manual exit | P&L ₹{pnl_live:+.2f}")
                        except Exception as e:
                            st.error(f"Exit failed: {e}")
                        st.session_state.trade_active = False; st.rerun()

# ── Page layout ──
_fetch_data()

st.subheader(f"{selected} — index")
_frag_index()

st.divider()
st.subheader("Best option to buy")
_frag_buy_trade()

st.markdown("---")
st.subheader(f"Signal analysis — {MIN_HOLD_SECS//60}-min comparison & trade monitor")
_frag_analysis()

st.markdown("---")
st.subheader("Live order placement")
_frag_orders()

st.caption(
    f"Index: `{ikey}` | Lot size: {lot_size} | ADX period: {ADX_PERIOD} | "
    f"History: {len(st.session_state.h_opt_ltp)}/{HISTORY_LEN} bars"
)
