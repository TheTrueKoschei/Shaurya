"""
=====================================================================
 ai_insight.py  —  AI Insight Briefing panel for PyClimaExplorer
=====================================================================
 Drop this file NEXT TO ULTIMATE.py.

 Design rule that keeps this defensible in front of judges:
   * Every NUMBER on screen is computed here in Python (numpy/xarray).
   * The language model NEVER computes anything. It receives the
     finished statistics and only turns them into plain sentences.
   * If there is no API key, no internet, or the API fails, a
     deterministic template writes the briefing instead. The panel
     never goes blank during a demo.

 API key — put ONE of these in .streamlit/secrets.toml or your env:
     GEMINI_API_KEY = "..."      (free tier: aistudio.google.com)
     GROQ_API_KEY   = "..."      (free tier: console.groq.com)
     OPENAI_API_KEY = "..."
=====================================================================
"""
from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
import streamlit as st

try:
    import requests
except Exception:          # requests missing -> offline mode only
    requests = None


# =====================================================================
# 0. CONFIG
# =====================================================================
GEMINI_MODEL = "gemini-2.5-flash"          # if this 404s, try "gemini-2.0-flash"
GROQ_MODEL   = "llama-3.3-70b-versatile"
OPENAI_MODEL = "gpt-4o-mini"
LLM_TIMEOUT  = 20                          # seconds — fail fast on venue wifi

LANGUAGES = {"English": "English", "हिंदी (Hindi)": "Hindi"}

CATEGORY_WORD = {
    "temperature": "temperature",
    "rainfall":    "rainfall",
    "wind":        "wind speed",
    "humidity":    "humidity",
    "snow":        "snow cover",
    "general":     "climate variable",
}


def _secret(name: str):
    """Look in st.secrets first, then environment variables."""
    try:
        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass                                # no secrets.toml present
    v = os.environ.get(name)
    return str(v).strip() if v else None


def _provider():
    """Return (provider_name, api_key) for whichever key is configured."""
    for prov, env in (("gemini", "GEMINI_API_KEY"),
                      ("groq",   "GROQ_API_KEY"),
                      ("openai", "OPENAI_API_KEY")):
        k = _secret(env)
        if k:
            return prov, k
    return None, None


# =====================================================================
# 1. STATS CORE  —  pure numpy / pandas, no xarray, no streamlit
#    (kept separate so it is easy to test and impossible to break
#     by swapping the front end)
# =====================================================================
def _unit_transform(sample_mean, var_type, units_raw):
    """
    CMIP/ERA5 files ship SI units that look absurd to a non-scientist
    (300 K, 3e-5 kg m-2 s-1). Convert once, consistently, and remember
    that we did so we can show it on screen.

    Returns (offset, scale, display_units, note). Apply as (v+offset)*scale.
    """
    u = (units_raw or "").strip()
    ul = u.lower()

    if var_type == "temperature" and np.isfinite(sample_mean) and sample_mean > 150:
        return -273.15, 1.0, "°C", "converted from Kelvin"

    if var_type == "rainfall" and "kg" in ul and ("s-1" in ul or "s^-1" in ul or "/s" in ul):
        return 0.0, 86400.0, "mm/day", "converted from kg m⁻² s⁻¹"

    pretty = {"degc": "°C", "deg_c": "°C", "celsius": "°C", "c": "°C",
              "degk": "K", "percent": "%", "m s-1": "m/s", "m s**-1": "m/s"}
    return 0.0, 1.0, pretty.get(ul, u if u else "units"), None


def _slope(x, y):
    """Least-squares slope of y on x, ignoring NaNs. None if too few points."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return None
    try:
        return float(np.polyfit(x[m], y[m], 1)[0])
    except Exception:
        return None


def _pct_rank(value, population):
    """Percentile rank of `value` within `population` (0-100)."""
    pop = np.asarray(population, float)
    pop = pop[np.isfinite(pop)]
    if pop.size == 0 or not np.isfinite(value):
        return None
    return float((pop < value).mean() * 100.0)


def core_stats(series, years, is_dates, grid=None, lats=None, lons=None,
               slice_index=0, units="units", unit_note=None,
               variable="variable", var_type="general"):
    """
    `series` : 1-D area-averaged time series, already unit-converted
    `years`  : same length; calendar years if is_dates else step numbers
    `grid`   : optional 2-D (lat, lon) slice, already unit-converted
    """
    series = np.asarray(series, float).ravel()
    finite = series[np.isfinite(series)]
    n = series.size

    s = {
        "variable":      variable,
        "category":      var_type,
        "units":         units,
        "unit_note":     unit_note,
        "n_time_steps":  int(n),
        "time_is_dates": bool(is_dates),
    }

    if finite.size == 0:
        return s

    s["mean"] = round(float(np.nanmean(finite)), 3)
    s["min"]  = round(float(np.nanmin(finite)), 3)
    s["max"]  = round(float(np.nanmax(finite)), 3)

    # ---- record period -------------------------------------------------
    if is_dates and np.isfinite(years).any():
        y0, y1 = int(np.nanmin(years)), int(np.nanmax(years))
        s["period"] = f"{y0}-{y1}"
        s["period_years"] = y1 - y0
    else:
        s["period"] = f"{n} time steps"
        s["period_years"] = None

    # ---- baseline (first third) vs recent (last third) -----------------
    if n >= 6:
        k = max(1, n // 3)
        base   = series[:k]
        recent = series[-k:]
        bm = float(np.nanmean(base))
        rm = float(np.nanmean(recent))
        if np.isfinite(bm) and np.isfinite(rm):
            s["baseline_mean"] = round(bm, 3)
            s["recent_mean"]   = round(rm, 3)
            s["anomaly"]       = round(rm - bm, 3)
            s["direction"]     = "higher" if rm > bm else "lower"
            if abs(bm) > 1e-9:
                s["percent_change"] = round((rm - bm) / abs(bm) * 100.0, 1)

    # ---- trend per decade (only meaningful with real dates) ------------
    if is_dates and np.isfinite(years).any():
        dfy = pd.DataFrame({"year": years, "v": series}).dropna()
        if not dfy.empty:
            annual = dfy.groupby("year")["v"].mean()
            if annual.size >= 3:
                sl = _slope(annual.index.to_numpy(float), annual.to_numpy(float))
                if sl is not None:
                    s["trend_per_decade"] = round(sl * 10.0, 3)
                    s["trend_word"] = ("rising" if sl > 0 else
                                       "falling" if sl < 0 else "flat")

    # ---- the currently displayed slice ---------------------------------
    idx = int(np.clip(slice_index, 0, n - 1))
    cur = float(series[idx])
    if np.isfinite(cur):
        s["current_slice_index"] = idx
        s["current_slice_value"] = round(cur, 3)
        if is_dates and np.isfinite(years[idx]):
            s["current_slice_year"] = int(years[idx])
        pr = _pct_rank(cur, series)
        if pr is not None:
            s["current_slice_percentile"] = round(pr, 1)

    # ---- spatial spread + hotspot --------------------------------------
    if grid is not None and lats is not None and lons is not None:
        g = np.asarray(grid, float)
        while g.ndim > 2:                    # drop stray level/ensemble axes
            g = g[0]
        if g.ndim == 2 and np.isfinite(g).any():
            s["spatial_min"] = round(float(np.nanmin(g)), 3)
            s["spatial_max"] = round(float(np.nanmax(g)), 3)
            s["spatial_spread"] = round(s["spatial_max"] - s["spatial_min"], 3)
            try:
                la = np.asarray(lats, float)
                lo = np.asarray(lons, float)
                j, i = np.unravel_index(np.nanargmax(g), g.shape)
                if j < la.size and i < lo.size:
                    s["hotspot_lat"] = round(float(la[j]), 2)
                    s["hotspot_lon"] = round(float(lo[i]), 2)
                s["region_bounds"] = {
                    "lat_min": round(float(np.nanmin(la)), 2),
                    "lat_max": round(float(np.nanmax(la)), 2),
                    "lon_min": round(float(np.nanmin(lo)), 2),
                    "lon_max": round(float(np.nanmax(lo)), 2),
                }
            except Exception:
                pass

    return s


# =====================================================================
# 2. XARRAY ADAPTER  —  turns the dashboard's selection into stats
# =====================================================================
MONTHS_EN = ["January","February","March","April","May","June","July",
             "August","September","October","November","December"]
MONTHS_HI = ["जनवरी","फ़रवरी","मार्च","अप्रैल","मई","जून","जुलाई",
             "अगस्त","सितंबर","अक्टूबर","नवंबर","दिसंबर"]


def _refine_category(var_type, units, variable):
    """`air` and friends don't match the keyword lists. Units give it away."""
    if var_type != "general":
        return var_type
    u = (units or "").strip().lower()
    if u in ("k", "kelvin", "degc", "deg_c", "c", "celsius", "°c"):
        return "temperature"
    if "mm" in u or ("kg" in u and "s" in u):
        return "rainfall"
    if u in ("%", "percent") or "humid" in variable.lower():
        return "humidity"
    if u in ("m/s", "m s-1", "m s**-1"):
        return "wind"
    return var_type


def seasonal_and_decadal(data, ds, off=0.0, scale=1.0):
    """Month-of-year cycle, decade averages and record years."""
    out = {}
    try:
        dims = data.dims
        tdim = "time" if "time" in dims else ("TIME" if "TIME" in dims else None)
        if tdim is None or tdim not in ds.coords:
            return out
        series = (data.mean(dim=["lat", "lon"], skipna=True)
                  if ("lat" in dims and "lon" in dims) else data)
        vals = (np.asarray(series.values, float).ravel() + off) * scale

        arr = np.asarray(ds[tdim].values)
        if np.issubdtype(arr.dtype, np.datetime64):
            idx = pd.to_datetime(arr)
        elif arr.dtype == object:
            idx = pd.to_datetime([f"{t.year}-{t.month:02d}-01" for t in arr])
        else:
            return out
        if idx.size != vals.size:
            return out

        df = pd.DataFrame({"v": vals, "year": idx.year, "month": idx.month}).dropna()
        if df.empty:
            return out

        mon = df.groupby("month")["v"].mean()
        if mon.size >= 6:
            out["peak_month"] = int(mon.idxmax())
            out["peak_month_value"] = round(float(mon.max()), 2)
            out["low_month"] = int(mon.idxmin())
            out["low_month_value"] = round(float(mon.min()), 2)
            out["seasonal_swing"] = round(float(mon.max() - mon.min()), 2)

        yr = df.groupby("year")["v"].mean()
        if yr.size >= 5:
            out["record_high_year"] = int(yr.idxmax())
            out["record_high_value"] = round(float(yr.max()), 2)
            out["record_low_year"] = int(yr.idxmin())
            out["record_low_value"] = round(float(yr.min()), 2)
            dec = yr.groupby((yr.index // 10) * 10).mean()
            if dec.size >= 2:
                out["decade_average"] = {f"{int(d)}s": round(float(v), 2)
                                         for d, v in dec.items()}
                out["first_decade"] = f"{int(dec.index[0])}s"
                out["last_decade"] = f"{int(dec.index[-1])}s"
                out["decade_change"] = round(float(dec.iloc[-1] - dec.iloc[0]), 2)
            # how unusual the last 10 years are against the first 30
            if yr.size >= 40:
                base = yr.iloc[:30]
                recent10 = yr.iloc[-10:]
                above = int((recent10 > base.mean()).sum())
                out["recent_years_above_baseline"] = f"{above} of the last 10"
    except Exception:
        pass
    return out


def compute_briefing_stats(data, ds, variable, var_type, time_index=0):
    """Returns the stats dict, or None if this variable can't be summarised."""
    try:
        dims = data.dims
        tdim = "time" if "time" in dims else ("TIME" if "TIME" in dims else None)
        has_grid = ("lat" in dims and "lon" in dims)

        # --- area-average time series (lazy reduction, then realise) ----
        series_da = data.mean(dim=["lat", "lon"], skipna=True) if has_grid else data
        series_raw = np.asarray(series_da.values, dtype=float).ravel()
        if series_raw.size == 0 or not np.isfinite(series_raw).any():
            return None

        # --- units --------------------------------------------------------
        off, scale, units, note = _unit_transform(
            float(np.nanmean(series_raw)), var_type, data.attrs.get("units", "")
        )
        series = (series_raw + off) * scale

        # --- time axis ----------------------------------------------------
        years, is_dates = None, False
        if tdim is not None and tdim in ds.coords:
            arr = np.asarray(ds[tdim].values)
            if np.issubdtype(arr.dtype, np.datetime64):
                years = pd.to_datetime(arr).year.to_numpy().astype(float)
                is_dates = True
            elif arr.dtype == object:
                # cftime objects expose .year; raw numbers do not
                try:
                    years = np.array([float(t.year) for t in arr])
                    is_dates = True
                except Exception:
                    years = None
        if years is None or years.size != series.size:
            years = np.arange(series.size, dtype=float)
            is_dates = False

        # --- one spatial slice for the hotspot ----------------------------
        grid = lats = lons = None
        if has_grid:
            try:
                idx = int(np.clip(time_index, 0, data.sizes.get(tdim, 1) - 1)) if tdim else 0
                gvals = data.isel({tdim: idx}).values if tdim else data.values
                grid = (np.asarray(gvals, float) + off) * scale
                lats = ds["lat"].values
                lons = ds["lon"].values
            except Exception:
                grid = lats = lons = None

        var_type = _refine_category(var_type, data.attrs.get("units", ""), variable)
        s = core_stats(
            series=series, years=years, is_dates=is_dates,
            grid=grid, lats=lats, lons=lons,
            slice_index=int(time_index), units=units, unit_note=note,
            variable=variable, var_type=var_type,
        )
        s["category"] = var_type
        s.update(seasonal_and_decadal(data, ds, off, scale))

        # Simple linear extrapolation, clearly labelled. Computed here in
        # Python so the model never has to produce a projected figure.
        if "trend_per_decade" in s and is_dates:
            try:
                last = int(np.nanmax(years))
                horizon = 2050 if last < 2050 else last + 25
                s["projection_year"] = horizon
                s["projection_change"] = round(
                    s["trend_per_decade"] * (horizon - last) / 10.0, 2)
            except Exception:
                pass
        return s
    except Exception:
        return None


# =====================================================================
# 3. NARRATIVE LAYER
# =====================================================================
_PROMPT = """You are a climate communicator writing for district officials and \
farmers in India. You explain climate data to people who have never opened a \
scientific dataset.

Below are statistics ALREADY COMPUTED from a NetCDF climate file. These numbers \
are final and correct.

RULES — follow exactly:
1. Use ONLY the numbers given below. Never calculate, estimate, adjust or invent \
any figure. If a number is not listed, do not mention it.
2. Do not mention variable codes, file formats or statistics jargon. Write the \
way a radio broadcaster would.
3. Write in {language}.
4. Reply with RAW JSON only — no markdown fences, no preamble — with exactly \
these four keys:
   "observation" : 2-3 sentences on what the data shows, including the trend and \
how recent years compare with earlier ones.
   "seasonal"    : 2-3 sentences on the shape of the year — which months run \
highest and lowest, how big the swing is, and which years stand out.
   "impact"      : 2-3 sentences on who this affects and how (farmers, water \
supply, health, power demand — pick what fits the variable).
   "action"      : 2-3 sentences of concrete advice a district administration \
could act on this year.

STATISTICS:
{stats}
"""

KEYS = ("observation", "seasonal", "impact", "action")

HEADINGS = {
    "English": [("🔍", "What the data shows"), ("🗓️", "The shape of the year"),
                ("👥", "Who this affects"),    ("✅", "What can be done")],
    "Hindi":   [("🔍", "आँकड़े क्या कहते हैं"), ("🗓️", "साल का स्वरूप"),
                ("👥", "किस पर असर पड़ेगा"),   ("✅", "क्या किया जा सकता है")],
}


def _call_llm(prompt: str):
    """Returns (text, provider_label) or (None, reason)."""
    if requests is None:
        return None, "requests not installed"

    prov, key = _provider()
    if not prov:
        return None, "no API key configured"

    try:
        if prov == "gemini":
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{GEMINI_MODEL}:generateContent",
                headers={"Content-Type": "application/json", "x-goog-api-key": key},
                json={"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                      "generationConfig": {"temperature": 0.4, "maxOutputTokens": 1100}},
                timeout=LLM_TIMEOUT,
            )
            r.raise_for_status()
            txt = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            return txt, f"Gemini · {GEMINI_MODEL}"

        url, model = (("https://api.groq.com/openai/v1/chat/completions", GROQ_MODEL)
                      if prov == "groq" else
                      ("https://api.openai.com/v1/chat/completions", OPENAI_MODEL))
        r = requests.post(
            url,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": model, "temperature": 0.4, "max_tokens": 1100,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=LLM_TIMEOUT,
        )
        r.raise_for_status()
        txt = r.json()["choices"][0]["message"]["content"]
        return txt, f"{prov.capitalize()} · {model}"

    except Exception as e:
        return None, f"{type(e).__name__}"


def _parse_json(txt):
    """Models sometimes wrap JSON in fences or prose. Dig it out."""
    if not txt:
        return None
    t = txt.strip()
    if t.startswith("```"):
        t = t.split("```")[1]
        if t.lstrip().lower().startswith("json"):
            t = t.lstrip()[4:]
    a, b = t.find("{"), t.rfind("}")
    if a == -1 or b == -1:
        return None
    try:
        d = json.loads(t[a:b + 1])
        if all(k in d for k in ("observation", "impact", "action")):
            return {k: str(d.get(k, "")).strip() for k in KEYS}
    except Exception:
        pass
    return None


# =====================================================================
#  OFFLINE TEMPLATE — bilingual, deterministic, numbers-only
# =====================================================================
WORD_HI = {"temperature": "तापमान", "rainfall": "वर्षा", "humidity": "आर्द्रता",
           "wind": "हवा की गति", "snow": "बर्फ़ की चादर",
           "general": "यह जलवायु चर"}

_IMPACT = {
 "temperature": {
  "up": {"en": "Hotter conditions raise irrigation demand, heat stress for outdoor "
               "workers and livestock, and peak electricity load in summer. Health "
               "services see more heat-related admissions in the worst weeks.",
         "hi": "बढ़ती गर्मी से सिंचाई की माँग, खुले में काम करने वालों और पशुओं पर "
               "गर्मी का दबाव, और गर्मियों में बिजली की खपत बढ़ती है। सबसे कठिन "
               "हफ़्तों में गर्मी से जुड़े मरीज़ भी बढ़ जाते हैं।"},
  "down": {"en": "Cooler conditions shift crop calendars and can delay sowing and "
                 "ripening, which changes when labour and market access are needed.",
           "hi": "ठंडी परिस्थितियाँ फ़सल कैलेंडर बदल देती हैं और बुवाई तथा पकने में "
                 "देरी कर सकती हैं, जिससे मज़दूरी और मंडी की ज़रूरत का समय बदलता है।"}},
 "rainfall": {
  "up": {"en": "Wetter conditions increase flood and waterlogging risk, and can "
               "damage standing crops at harvest. Low-lying blocks and poor drainage "
               "areas carry most of that risk.",
         "hi": "अधिक वर्षा से बाढ़ और जलभराव का ख़तरा बढ़ता है और कटाई के समय खड़ी "
               "फ़सल को नुक़सान हो सकता है। निचले इलाक़ों और कमज़ोर जल-निकासी वाले "
               "क्षेत्रों पर सबसे अधिक जोखिम रहता है।"},
  "down": {"en": "Drier conditions strain groundwater, drinking water supply and "
                 "rain-fed agriculture, where farmers have the least buffer.",
           "hi": "कम वर्षा से भूजल, पेयजल आपूर्ति और वर्षा-आधारित खेती पर दबाव पड़ता "
                 "है, जहाँ किसानों के पास सबसे कम सहारा होता है।"}},
 "humidity": {
  "up": {"en": "Higher humidity raises fungal disease pressure in crops and makes "
               "heat far more dangerous to the body, since sweat cools less well.",
         "hi": "अधिक आर्द्रता से फ़सलों में फफूँद रोग बढ़ते हैं और गर्मी शरीर के लिए "
               "अधिक ख़तरनाक हो जाती है, क्योंकि पसीना कम असर करता है।"},
  "down": {"en": "Lower humidity increases evaporation losses from soil and storage, "
                 "and raises fire risk in dry months.",
           "hi": "कम आर्द्रता से मिट्टी और भंडारण से वाष्पीकरण बढ़ता है और सूखे महीनों "
                 "में आग का ख़तरा बढ़ जाता है।"}},
 "wind": {
  "up": {"en": "Stronger winds raise storm-damage risk to crops, roofs and power "
               "lines, though wind energy yield improves.",
         "hi": "तेज़ हवाओं से फ़सल, छतों और बिजली लाइनों को तूफ़ान से नुक़सान का "
               "ख़तरा बढ़ता है, हालाँकि पवन ऊर्जा का उत्पादन बेहतर होता है।"},
  "down": {"en": "Weaker winds reduce wind-power output and worsen air stagnation "
                 "in towns and cities.",
           "hi": "कमज़ोर हवाओं से पवन ऊर्जा घटती है और शहरों में हवा का ठहराव "
                 "बढ़ता है।"}},
 "snow": {
  "up": {"en": "More snow cover means a larger meltwater store feeding rivers later "
               "in the year.",
         "hi": "अधिक बर्फ़ का अर्थ है नदियों के लिए साल में आगे चलकर अधिक पिघला पानी।"},
  "down": {"en": "Less snow cover means less meltwater feeding rivers in the dry "
                 "season, when it matters most.",
           "hi": "कम बर्फ़ का अर्थ है सूखे मौसम में नदियों को कम पिघला पानी मिलना, "
                 "जब उसकी सबसे अधिक ज़रूरत होती है।"}},
 "general": {
  "up": {"en": "This shift changes the conditions local planning has been built around.",
         "hi": "यह बदलाव उन परिस्थितियों को बदल देता है जिन पर स्थानीय योजनाएँ बनी हैं।"},
  "down": {"en": "This shift changes the conditions local planning has been built around.",
           "hi": "यह बदलाव उन परिस्थितियों को बदल देता है जिन पर स्थानीय योजनाएँ बनी हैं।"}},
}

_ACTION = {
 "temperature": {
  "up": {"en": "Issue heat advisories before the peak month, shift outdoor work to "
               "early morning, keep drinking water points stocked along worksites and "
               "markets, and budget for higher summer power demand.",
         "hi": "सबसे गर्म महीने से पहले गर्मी की चेतावनी जारी करें, खुले में काम सुबह "
               "जल्दी कराएँ, कार्यस्थलों और बाज़ारों पर पेयजल की व्यवस्था रखें, और "
               "गर्मियों में अधिक बिजली माँग के लिए बजट रखें।"},
  "down": {"en": "Review sowing calendars with the agriculture department and plan "
                 "for later harvest dates.",
           "hi": "कृषि विभाग के साथ बुवाई कैलेंडर की समीक्षा करें और कटाई की तारीख़ें "
                 "आगे खिसकने की तैयारी रखें।"}},
 "rainfall": {
  "up": {"en": "Clear drainage before the peak month, review reservoir release rules, "
               "pre-position relief stock in low-lying blocks, and map which villages "
               "lose road access first.",
         "hi": "सबसे अधिक वर्षा वाले महीने से पहले नालियाँ साफ़ कराएँ, जलाशय से पानी "
               "छोड़ने के नियम देखें, निचले ब्लॉकों में राहत सामग्री पहले से रखें, और "
               "चिन्हित करें कि किन गाँवों का सड़क संपर्क पहले टूटता है।"},
  "down": {"en": "Prioritise water harvesting and recharge structures, distribute "
                 "drought-tolerant seed ahead of sowing, and monitor borewell levels "
                 "through the dry months.",
           "hi": "जल संचयन और भूजल रिचार्ज ढाँचों को प्राथमिकता दें, बुवाई से पहले "
                 "सूखा-सहनशील बीज बाँटें, और सूखे महीनों में बोरवेल के जलस्तर पर "
                 "नज़र रखें।"}},
 "humidity": {
  "up": {"en": "Issue crop fungal-disease advisories before the humid months and fold "
               "humidity into local heat warnings.",
         "hi": "आर्द्र महीनों से पहले फ़सलों में फफूँद रोग की सलाह जारी करें और गर्मी "
               "की चेतावनी में आर्द्रता को भी शामिल करें।"},
  "down": {"en": "Promote mulching and drip irrigation to cut evaporation losses.",
           "hi": "वाष्पीकरण घटाने के लिए मल्चिंग और ड्रिप सिंचाई को बढ़ावा दें।"}},
 "wind": {
  "up": {"en": "Inspect power lines and rooftops before the storm season and review "
                "shelter readiness.",
         "hi": "तूफ़ानी मौसम से पहले बिजली लाइनों और छतों की जाँच कराएँ और आश्रय "
               "स्थलों की तैयारी देखें।"},
  "down": {"en": "Factor lower wind output into renewable energy planning.",
           "hi": "अक्षय ऊर्जा योजना में कम पवन उत्पादन को ध्यान में रखें।"}},
 "snow": {
  "up": {"en": "Plan for larger and earlier meltwater flows downstream.",
         "hi": "नीचे की ओर अधिक और जल्दी आने वाले पिघले पानी की तैयारी रखें।"},
  "down": {"en": "Plan for reduced dry-season river flow and secure alternate water "
                 "sources.",
           "hi": "सूखे मौसम में नदियों का बहाव घटने की तैयारी रखें और वैकल्पिक जल "
                 "स्रोत सुनिश्चित करें।"}},
 "general": {
  "up": {"en": "Review local plans against this trend.",
         "hi": "इस रुझान के अनुसार स्थानीय योजनाओं की समीक्षा करें।"},
  "down": {"en": "Review local plans against this trend.",
           "hi": "इस रुझान के अनुसार स्थानीय योजनाओं की समीक्षा करें।"}},
}


def _template_briefing(s, language="English"):
    """Offline briefing. Deterministic, built only from computed numbers."""
    hi = str(language).lower().startswith("hi")
    L = "hi" if hi else "en"
    u = s.get("units", "")
    cat = s.get("category", "general")
    word = (WORD_HI.get(cat, WORD_HI["general"]) if hi
            else CATEGORY_WORD.get(cat, "climate variable"))
    months = MONTHS_HI if hi else MONTHS_EN
    period = s.get("period", "")

    up = True
    if "trend_per_decade" in s:
        up = s["trend_per_decade"] > 0
    elif "anomaly" in s:
        up = s["anomaly"] > 0

    # ---------- observation ----------
    o = []
    if hi:
        o.append(f"इस क्षेत्र में {period} के दौरान औसत {word} {s.get('mean','—')} {u} रहा है।")
        if "anomaly" in s:
            d = "अधिक" if s["anomaly"] > 0 else "कम"
            o.append(f"रिकॉर्ड के सबसे हाल के एक-तिहाई हिस्से का औसत {s['recent_mean']} {u} है, "
                     f"जो शुरुआती एक-तिहाई से {abs(s['anomaly'])} {u} {d} है।")
        if "trend_per_decade" in s:
            d = "बढ़" if up else "घट"
            o.append(f"लंबी अवधि में यह हर दशक {abs(s['trend_per_decade'])} {u} की दर से {d} रहा है।")
        if "decade_change" in s:
            o.append(f"{s['first_decade']} से {s['last_decade']} तक कुल बदलाव "
                     f"{s['decade_change']:+} {u} रहा है।")
        if "projection_change" in s:
            d = "बढ़" if s["projection_change"] > 0 else "घट"
            o.append(f"यदि यही रुझान जारी रहा, तो {s['projection_year']} तक यह लगभग "
                     f"{abs(s['projection_change'])} {u} और {d} सकता है।")
    else:
        o.append(f"Across the mapped region, {word} over {period} averages "
                 f"{s.get('mean','—')} {u}.")
        if "anomaly" in s:
            o.append(f"The most recent third of the record averages {s['recent_mean']} {u}, "
                     f"{abs(s['anomaly'])} {u} {s['direction']} than the earliest third.")
        if "trend_per_decade" in s:
            o.append(f"The long-term trend is {s['trend_word']} at "
                     f"{abs(s['trend_per_decade'])} {u} per decade.")
        if "decade_change" in s:
            o.append(f"From the {s['first_decade']} to the {s['last_decade']} the change "
                     f"totals {s['decade_change']:+} {u}.")
        if "recent_years_above_baseline" in s:
            o.append(f"{s['recent_years_above_baseline']} years sat above the "
                     f"long-run average.")
        if "projection_change" in s:
            d = "higher" if s["projection_change"] > 0 else "lower"
            o.append(f"If that rate continues, by {s['projection_year']} it would be "
                     f"about {abs(s['projection_change'])} {u} {d} than today.")

    # ---------- seasonal ----------
    sea = []
    if "peak_month" in s:
        pm = months[s["peak_month"] - 1]; lm = months[s["low_month"] - 1]
        if hi:
            sea.append(f"साल में सबसे ऊँचा महीना {pm} है ({s['peak_month_value']} {u}) "
                       f"और सबसे नीचा {lm} ({s['low_month_value']} {u})।")
            sea.append(f"इन दोनों के बीच का अंतर {s['seasonal_swing']} {u} है।")
        else:
            sea.append(f"{pm} is the peak month at {s['peak_month_value']} {u}, and "
                       f"{lm} the lowest at {s['low_month_value']} {u}.")
            sea.append(f"The swing across the year is {s['seasonal_swing']} {u}.")
    if "record_high_year" in s:
        if hi:
            sea.append(f"रिकॉर्ड का सबसे ऊँचा साल {s['record_high_year']} रहा "
                       f"({s['record_high_value']} {u}) और सबसे नीचा "
                       f"{s['record_low_year']} ({s['record_low_value']} {u})।")
        else:
            sea.append(f"The highest year on record is {s['record_high_year']} at "
                       f"{s['record_high_value']} {u}; the lowest is "
                       f"{s['record_low_year']} at {s['record_low_value']} {u}.")
    if not sea:
        sea = ["इस फ़ाइल में महीनेवार आँकड़े नहीं हैं, इसलिए मौसमी चक्र नहीं बताया जा सकता।"] if hi               else ["This file has no usable monthly calendar, so the seasonal cycle "
                    "cannot be described."]

    # ---------- impact ----------
    imp = [_IMPACT.get(cat, _IMPACT["general"])["up" if up else "down"][L]]
    if "current_slice_percentile" in s:
        p = s["current_slice_percentile"]
        if hi:
            wh = "सबसे ऊँचे" if p >= 80 else "सबसे नीचे" if p <= 20 else "सामान्य"
            imp.insert(0, f"अभी स्क्रीन पर दिख रहा समय-बिंदु इस रिकॉर्ड में {wh} हिस्से में आता है।")
        else:
            wh = ("among the highest" if p >= 80 else
                  "among the lowest" if p <= 20 else "close to typical")
            imp.insert(0, f"The time step currently on screen sits {wh} in this record.")

    act = [_ACTION.get(cat, _ACTION["general"])["up" if up else "down"][L]]
    if "peak_month" in s and not hi:
        act.append(f"Time these measures to land before {months[s['peak_month']-1]}, "
                   f"the peak month in this record.")
    elif "peak_month" in s:
        act.append(f"इन क़दमों को {months[s['peak_month']-1]} से पहले पूरा करें, जो इस "
                   f"रिकॉर्ड का सबसे तीव्र महीना है।")

    return {"observation": " ".join(o), "seasonal": " ".join(sea),
            "impact": " ".join(imp), "action": " ".join(act)}


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_briefing(stats_json: str, language: str):
    """Cached so repeat clicks and Streamlit reruns cost nothing."""
    stats = json.loads(stats_json)
    txt, label = _call_llm(_PROMPT.format(language=language, stats=stats_json))
    parsed = _parse_json(txt)
    if parsed:
        return parsed, f"AI narrative — {label}"
    return _template_briefing(stats, language), f"Offline narrative ({label})"


def generate_briefing(stats: dict, language: str = "English"):
    return _cached_briefing(json.dumps(stats, sort_keys=True), language)


# =====================================================================
# 4. STREAMLIT RENDERER
# =====================================================================
def _time_variables(ds, exclude=None, limit=6):
    """Variables in this file that actually have a usable time series."""
    found = []
    try:
        for v in ds.data_vars:
            if v == exclude:
                continue
            d = ds[v].dims
            tdim = "time" if "time" in d else ("TIME" if "TIME" in d else None)
            if tdim and "lat" in d and "lon" in d:
                n = int(ds[v].sizes.get(tdim, 0))
                if n >= 3:
                    found.append((v, n))
    except Exception:
        return []
    found.sort(key=lambda x: -x[1])
    return found[:limit]


def _metric(label, value, sub, cls=""):
    return (f'<div class="metric-card {cls}">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div>'
            f'<div class="metric-sub">{sub}</div></div>')


def render_insight_briefing(data, ds, variable, var_type, time_index, card_key):
    with st.container(border=True):
        st.markdown(
            '<div class="card-header-block">'
            '<div class="card-header-label">AI Layer</div>'
            '<div class="card-header-title">🧠 Insight Briefing</div></div>',
            unsafe_allow_html=True)

        stats = compute_briefing_stats(data, ds, variable, var_type, time_index)
        if not stats or "mean" not in stats:
            st.info("Not enough valid data in this variable to build a briefing.")
            return

        u = stats.get("units", "")

        # ---- the numbers, computed in Python --------------------------------
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(_metric("📅 Record", stats.get("period", "—"),
                                f'{stats["n_time_steps"]} steps'), unsafe_allow_html=True)
        with c2:
            st.markdown(_metric("📊 Average", f'{stats["mean"]} {u}',
                                "whole record, area-averaged"), unsafe_allow_html=True)
        with c3:
            if "anomaly" in stats:
                sign = "+" if stats["anomaly"] > 0 else ""
                cls = "metric-hot" if stats["anomaly"] > 0 else "metric-cold"
                st.markdown(_metric("🔀 Recent vs baseline", f'{sign}{stats["anomaly"]} {u}',
                                    "last third vs first third", cls), unsafe_allow_html=True)
            else:
                st.markdown(_metric("🔀 Recent vs baseline", "—", "record too short"),
                            unsafe_allow_html=True)
        with c4:
            if "trend_per_decade" in stats:
                sign = "+" if stats["trend_per_decade"] > 0 else ""
                cls = "metric-hot" if stats["trend_per_decade"] > 0 else "metric-cold"
                st.markdown(_metric("📈 Trend", f'{sign}{stats["trend_per_decade"]} {u}',
                                    "per decade", cls), unsafe_allow_html=True)
            else:
                st.markdown(_metric("📈 Trend", "—", "needs dated time axis"),
                            unsafe_allow_html=True)

        if stats.get("unit_note"):
            st.caption(f"Units {stats['unit_note']} for readability.")

        # ---- guard: nothing to narrate --------------------------------------
        if "anomaly" not in stats and "trend_per_decade" not in stats:
            if stats["n_time_steps"] < 3:
                why = (f"`{variable}` has only {stats['n_time_steps']} time step"
                       f"{'' if stats['n_time_steps'] == 1 else 's'}, so there is no "
                       "change over time to describe.")
            else:
                why = (f"`{variable}` has no dated time axis, so trends cannot be "
                       "computed from it.")
            cands = _time_variables(ds, exclude=variable)
            if cands:
                lst = ", ".join(f"**{v}** ({n} steps)" for v, n in cands)
                st.warning(f"{why}\n\nFor a briefing with a real trend in it, switch "
                           f"the variable selector above to one of these: {lst}")
            else:
                st.warning(f"{why}\n\nNo variable in this file has a long enough "
                           "time axis, so the briefing will describe the spatial "
                           "pattern only.")

        st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)

        # ---- controls --------------------------------------------------------
        cc1, cc2 = st.columns([1, 1.4])
        with cc1:
            lang_label = st.selectbox("Briefing language", list(LANGUAGES.keys()),
                                      key=f"lang_{card_key}")
        with cc2:
            st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
            go_btn = st.button("🧠  Generate Insight Briefing",
                               key=f"brief_btn_{card_key}", use_container_width=True)

        # Keyed on the statistics too: two files (or two states) that share a
        # variable name must not show each other's briefing.
        _sig = abs(hash(json.dumps(stats, sort_keys=True, default=str))) % 10**10
        state_key = f"_brief_{card_key}_{variable}_{lang_label}_{_sig}"
        if go_btn:
            with st.spinner("Reading the statistics and writing the briefing…"):
                st.session_state[state_key] = generate_briefing(stats, LANGUAGES[lang_label])

        # ---- output ----------------------------------------------------------
        if state_key in st.session_state:
            brief, source = st.session_state[state_key]
            heads = HEADINGS.get(LANGUAGES.get(lang_label, "English"), HEADINGS["English"])
            for (icon, head), key in zip(heads, KEYS):
                body = brief.get(key, "").strip()
                if not body:
                    continue
                st.markdown(
                    f'<div class="metric-card" style="margin-bottom:10px;">'
                    f'<div class="metric-label">{icon} {head}</div>'
                    f'<div style="font-size:15px; line-height:1.68; margin-top:6px;">'
                    f'{body}</div></div>', unsafe_allow_html=True)
            st.caption(f"Numbers computed in Python from the NetCDF file. {source}.")
        else:
            st.caption("Every figure above is computed directly from your file. "
                       "Press the button to turn them into a plain-language briefing.")

        with st.expander("🔬 Exact numbers passed to the model"):
            st.json(stats)

