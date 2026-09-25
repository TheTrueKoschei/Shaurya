"""
=====================================================================
 local_brain.py  —  offline answerer for PyClimaExplorer
=====================================================================
 No API key. No model download. No internet. No dependencies beyond
 the standard library.

 It matches a question against a set of intents and answers from the
 fact sheet the dashboard already computes. Because every answer is a
 template filled with numbers taken straight from the file, it is
 incapable of hallucinating — the worst it can do is say it doesn't
 understand the question.

 The pitch line: this runs on a district office machine with no
 internet and no GPU.
=====================================================================
"""
from __future__ import annotations

import re

MONTH_FULL = {
    "Jan": "January", "Feb": "February", "Mar": "March", "Apr": "April",
    "May": "May", "Jun": "June", "Jul": "July", "Aug": "August",
    "Sep": "September", "Oct": "October", "Nov": "November", "Dec": "December",
}

ADVICE = {
    "temperature": {
        "up": "Plan heat advisories for the hottest months, shift outdoor work "
              "to early morning, and budget for higher summer power demand.",
        "down": "Review sowing calendars with the agriculture department, since "
                "cooler conditions delay ripening.",
    },
    "rainfall": {
        "up": "Clear drainage before the season, review reservoir release rules, "
              "and pre-position relief stock in low-lying blocks.",
        "down": "Prioritise water harvesting, recharge structures and "
                "drought-tolerant seed distribution.",
    },
    "humidity": {
        "up": "Issue crop fungal-disease advisories and fold humidity into local "
              "heat warnings.",
        "down": "Promote mulching and drip irrigation to cut evaporation losses.",
    },
    "wind": {
        "up": "Inspect power lines and rooftops before the storm season; wind "
              "energy yield should improve.",
        "down": "Factor lower wind output into renewable energy planning.",
    },
    "general": {
        "up": "Review local plans against this trend.",
        "down": "Review local plans against this trend.",
    },
}


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _s(sheet):
    return sheet.get("statistics", {}) or {}


def _u(sheet):
    return _s(sheet).get("units", "")


def _what(sheet):
    return sheet.get("what_it_measures", "this variable")


def _direction(sheet):
    st = _s(sheet)
    if "trend_per_decade" in st:
        return "up" if st["trend_per_decade"] > 0 else "down"
    if "anomaly" in st:
        return "up" if st["anomaly"] > 0 else "down"
    return "up"


def _category(sheet):
    return _s(sheet).get("category", "general")


def _missing(what):
    return (f"That isn't in what the dashboard computed from this file. "
            f"To answer it I would need {what}.")


# ---------------------------------------------------------------------
# intent handlers
# ---------------------------------------------------------------------
def _a_trend(sheet):
    st, u = _s(sheet), _u(sheet)
    bits = []
    if "trend_per_decade" in st:
        t = st["trend_per_decade"]
        bits.append(f"Yes — {_what(sheet)} is {st.get('trend_word','changing')} "
                    f"by {abs(t)} {u} every decade across {st.get('period','the record')}.")
    if "anomaly" in st:
        bits.append(f"The most recent third of the record averages "
                    f"{st['recent_mean']} {u}, which is {abs(st['anomaly'])} {u} "
                    f"{st['direction']} than the earliest third.")
    if "decade_average" in sheet:
        d = sheet["decade_average"]
        ks = list(d)
        if len(ks) >= 2:
            bits.append(f"By decade it runs from {d[ks[0]]} {u} in the {ks[0]} "
                        f"to {d[ks[-1]]} {u} in the {ks[-1]}.")
    return " ".join(bits) if bits else _missing("a longer time series in this file")


def _a_high_month(sheet):
    m = sheet.get("highest_month")
    if not m:
        return _missing("monthly data — this file has no usable calendar")
    v = sheet.get("monthly_average", {}).get(m)
    return (f"{MONTH_FULL.get(m, m)} is the peak month, averaging {v} {_u(sheet)}. "
            f"The lowest is {MONTH_FULL.get(sheet.get('lowest_month',''), '—')}.")


def _a_low_month(sheet):
    m = sheet.get("lowest_month")
    if not m:
        return _missing("monthly data — this file has no usable calendar")
    v = sheet.get("monthly_average", {}).get(m)
    return (f"{MONTH_FULL.get(m, m)} is the lowest month, averaging {v} {_u(sheet)}. "
            f"The peak is {MONTH_FULL.get(sheet.get('highest_month',''), '—')}.")


def _a_extreme_year(sheet):
    hi, lo, u = sheet.get("highest_year"), sheet.get("lowest_year"), _u(sheet)
    if not hi:
        return _missing("at least five years of data in this file")
    out = f"The highest year on record is {hi['year']} at {hi['value']} {u}."
    if lo:
        out += f" The lowest is {lo['year']} at {lo['value']} {u}."
    return out


def _a_decades(sheet):
    d, u = sheet.get("decade_average"), _u(sheet)
    if not d:
        return _missing("a dated time axis spanning several decades")
    parts = ", ".join(f"{k} {v} {u}" for k, v in list(d.items())[:9])
    ks = list(d)
    chg = round(d[ks[-1]] - d[ks[0]], 2)
    return (f"Decade averages: {parts}. That is a change of {chg:+} {u} "
            f"from the {ks[0]} to the {ks[-1]}.")


def _a_average(sheet):
    st, u = _s(sheet), _u(sheet)
    if "mean" not in st:
        return _missing("valid values in this variable")
    return (f"Across {st.get('period','the record')}, {_what(sheet)} averages "
            f"{st['mean']} {u}, ranging from {st.get('min')} to {st.get('max')} {u}.")


def _a_spatial(sheet):
    st, u = _s(sheet), _u(sheet)
    if "spatial_max" not in st:
        return _missing("a lat/lon grid in this file")
    out = (f"On the time step currently displayed, values run from "
           f"{st['spatial_min']} to {st['spatial_max']} {u} across the region.")
    if "hotspot_lat" in st:
        out += (f" The highest reading sits near {st['hotspot_lat']}°N, "
                f"{st['hotspot_lon']}°E.")
    return out


def _a_area(sheet):
    a = sheet.get("area_covered")
    return (f"This file covers {a}." if a
            else _missing("latitude and longitude coordinates in this file"))


def _a_period(sheet):
    st = _s(sheet)
    return (f"The record covers {st.get('period','an unknown period')}, "
            f"which is {st.get('n_time_steps','?')} time steps.")


def _a_advice(sheet):
    cat = _category(sheet)
    tip = ADVICE.get(cat, ADVICE["general"])[_direction(sheet)]
    st, u = _s(sheet), _u(sheet)
    lead = ""
    if "trend_per_decade" in st:
        lead = (f"Given {_what(sheet)} is {st.get('trend_word','changing')} by "
                f"{abs(st['trend_per_decade'])} {u} per decade: ")
    return lead + tip


def _a_variables(sheet):
    v = sheet.get("file_variables", [])
    return (f"This file holds {len(v)} variable(s): {', '.join(v[:12])}. "
            f"Currently selected: {sheet.get('selected_variable','—')}.")


def _a_help(sheet):
    return ("I answer from the statistics computed off your loaded file. Try: "
            "is it getting worse, which month is highest, which year was the "
            "hottest, compare the decades, what is the average, what area does "
            "this cover, or what should be done about it.")


# ---------------------------------------------------------------------
# intent table — first match on a weighted keyword score wins
# ---------------------------------------------------------------------
INTENTS = [
    (_a_help,         {"help", "what can you", "what do you", "commands", "options"}),
    (_a_advice,       {"should", "advice", "do about", "farmer", "farmers", "action",
                       "recommend", "prepare", "mitigate", "policy", "plan for"}),
    (_a_decades,      {"decade", "decades", "1990s", "2000s", "2010s", "compare period"}),
    (_a_extreme_year, {"which year", "what year", "hottest year", "wettest year",
                       "driest year", "coldest year", "record year", "worst year",
                       "extreme year", "highest year", "lowest year"}),
    (_a_low_month,    {"lowest month", "coldest month", "driest month",
                       "least", "minimum month", "weakest month"}),
    (_a_high_month,   {"month", "months", "season", "seasonal", "monsoon",
                       "hottest month", "wettest month", "peak"}),
    (_a_trend,        {"trend", "worse", "increasing", "decreasing", "rising",
                       "falling", "changing", "change over", "getting", "faster",
                       "how fast", "warming", "drying"}),
    (_a_spatial,      {"where", "hotspot", "region is", "highest reading",
                       "spatial", "across the", "which part", "map"}),
    (_a_area,         {"area", "cover", "covers", "extent", "bounds", "boundary",
                       "which region", "location"}),
    (_a_period,       {"how long", "period", "record", "years of data", "since when",
                       "time steps", "range of years", "from when"}),
    (_a_variables,    {"variable", "variables", "what data", "what file",
                       "dataset", "columns", "fields"}),
    (_a_average,      {"average", "mean", "typical", "normal", "usual", "overall",
                       "range", "minimum", "maximum", "highest value", "lowest value"}),
]


def answer_locally(sheet: dict, question: str) -> str:
    """Match the question to an intent and fill a template from the fact sheet."""
    if not question or not question.strip():
        return _a_help(sheet)

    q = " " + re.sub(r"[^a-z0-9 ]+", " ", question.lower()) + " "
    q = re.sub(r"\s+", " ", q)

    best, best_score = None, 0
    for handler, keys in INTENTS:
        score = 0
        for k in keys:
            if k in q:
                # multi-word matches are far more specific than single words
                score += 3 + 2 * k.count(" ")
        if score > best_score:
            best, best_score = handler, score

    if best is None:
        return ("I couldn't match that to anything I can compute from this file. "
                + _a_help(sheet))
    try:
        return best(sheet)
    except Exception:
        return _missing("more complete statistics from this file")
