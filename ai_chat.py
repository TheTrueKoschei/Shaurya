"""
=====================================================================
 ai_chat.py  —  Dataset-grounded chat for PyClimaExplorer
=====================================================================
 This is deliberately NOT a general-purpose chatbot.

 Before every question, the dashboard computes a fact sheet from the
 NetCDF file that is currently loaded — record length, trend, monthly
 cycle, decade averages, extreme years, spatial extent. The model sees
 only that fact sheet and the user's question. It is instructed to
 answer from those numbers alone and to say plainly when the answer is
 not in them.

 So the demo line is: this does not know anything about climate in
 general. It knows YOUR file.

 Reuses the provider, key handling and offline fallback in ai_insight.py.
=====================================================================
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import streamlit as st

from ai_insight import (
    compute_briefing_stats, _call_llm, _provider,
    CATEGORY_WORD, LANGUAGES,
)
from local_brain import answer_locally

MAX_HISTORY = 6          # turns kept in the prompt
MAX_ANSWER_CHARS = 700

STARTERS = [
    "Is it getting worse, and how fast?",
    "Which months are the most extreme?",
    "What should a farmer do about this?",
]


# =====================================================================
# 1. FACT SHEET — everything the model is allowed to know
# =====================================================================
def _monthly_and_decadal(data, ds, variable, off=0.0, scale=1.0):
    """Monthly climatology, decade averages and extreme years."""
    facts = {}
    try:
        dims = data.dims
        tdim = "time" if "time" in dims else ("TIME" if "TIME" in dims else None)
        if tdim is None or tdim not in ds.coords:
            return facts

        series = (data.mean(dim=["lat", "lon"], skipna=True)
                  if ("lat" in dims and "lon" in dims) else data)
        vals = (np.asarray(series.values, float).ravel() + off) * scale

        arr = np.asarray(ds[tdim].values)
        if np.issubdtype(arr.dtype, np.datetime64):
            idx = pd.to_datetime(arr)
        elif arr.dtype == object:
            try:
                idx = pd.to_datetime([f"{t.year}-{t.month:02d}-01" for t in arr])
            except Exception:
                return facts
        else:
            return facts

        if idx.size != vals.size:
            return facts

        df = pd.DataFrame({"v": vals, "year": idx.year, "month": idx.month}).dropna()
        if df.empty:
            return facts

        names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        mon = df.groupby("month")["v"].mean()
        if mon.size >= 6:
            facts["monthly_average"] = {names[int(m) - 1]: round(float(v), 2)
                                        for m, v in mon.items()}
            facts["highest_month"] = names[int(mon.idxmax()) - 1]
            facts["lowest_month"] = names[int(mon.idxmin()) - 1]

        yr = df.groupby("year")["v"].mean()
        if yr.size >= 5:
            facts["highest_year"] = {"year": int(yr.idxmax()), "value": round(float(yr.max()), 2)}
            facts["lowest_year"] = {"year": int(yr.idxmin()), "value": round(float(yr.min()), 2)}
            dec = yr.groupby((yr.index // 10) * 10).mean()
            facts["decade_average"] = {f"{int(d)}s": round(float(v), 2)
                                       for d, v in dec.items()}
    except Exception:
        pass
    return facts


@st.cache_data(show_spinner=False, ttl=1800)
def _fact_sheet_cached(_key: str, _payload: str):
    return _payload


def build_fact_sheet(ds, variable, var_type, time_index=0):
    """The complete, and only, knowledge the model gets."""
    data = ds[variable]
    stats = compute_briefing_stats(data, ds, variable, var_type, time_index) or {}

    off, scale = 0.0, 1.0
    try:
        from ai_insight import _unit_transform
        raw = float(np.nanmean(np.asarray(data.values[:1], float)))
        off, scale, _u, _n = _unit_transform(raw, var_type, data.attrs.get("units", ""))
    except Exception:
        pass

    sheet = {
        "file_variables": list(ds.data_vars)[:20],
        "selected_variable": variable,
        "what_it_measures": CATEGORY_WORD.get(var_type, "climate variable"),
        "statistics": stats,
    }
    sheet.update(_monthly_and_decadal(data, ds, variable, off, scale))

    rb = stats.get("region_bounds")
    if rb:
        sheet["area_covered"] = (f"{rb['lat_min']} to {rb['lat_max']} N, "
                                 f"{rb['lon_min']} to {rb['lon_max']} E")
    return sheet


# =====================================================================
# 2. PROMPT
# =====================================================================
_SYSTEM = """You are the assistant built into a climate dashboard. A user has \
loaded a NetCDF climate file and is asking about it.

You know NOTHING except the FACT SHEET below. It was computed in Python \
directly from their file.

RULES:
1. Answer only from the fact sheet. Never calculate a new figure, never \
estimate, never draw on general climate knowledge for specific numbers.
2. If the fact sheet does not contain the answer, say so in one sentence and \
name what the dashboard would need in order to answer it. Do not guess.
3. Keep answers under 90 words. Plain language, no jargon, no variable codes.
4. You may give practical advice (farming, water, health, planning) as long as \
the reasoning rests on figures that are in the fact sheet.
5. Answer in {language}.

FACT SHEET:
{sheet}
"""


def _ask(sheet: dict, history, question: str, language="English"):
    prompt = _SYSTEM.format(language=language,
                            sheet=json.dumps(sheet, sort_keys=True))
    convo = ""
    for role, text in history[-MAX_HISTORY:]:
        convo += f"\n{'User' if role == 'user' else 'Assistant'}: {text}"
    prompt += f"\nCONVERSATION SO FAR:{convo}\n\nUser: {question}\nAssistant:"

    txt, label = _call_llm(prompt)
    if txt:
        return txt.strip()[:MAX_ANSWER_CHARS], label
    return None, label


# =====================================================================
# 3. SIDEBAR UI
# =====================================================================
def _sheet(ds, variable, var_type, time_index):
    """
    Fact sheet, computed once per (file, variable, time step) and kept in
    session state. Previously it was rebuilt twice for every question.
    Session state is used rather than st.cache_data because an xarray
    Dataset can't be hashed as a cache key.
    """
    try:
        sig = (variable, var_type, int(time_index),
               tuple(sorted((str(k), int(v)) for k, v in ds.sizes.items())),
               str(ds.attrs.get("region", "")),
               str(ds.attrs.get("source_files", "")))
    except Exception:
        return build_fact_sheet(ds, variable, var_type, time_index)
    memo = st.session_state.setdefault("_pce_sheet_memo", {})
    if sig not in memo:
        if len(memo) > 16:
            memo.clear()
        memo[sig] = build_fact_sheet(ds, variable, var_type, time_index)
    return memo[sig]


def _submit(question, ds, variable, var_type, time_index, hist_key, box_key):
    """
    Runs as a button callback, i.e. BEFORE the rerun. The answer is already
    in history when the panel redraws, so no explicit st.rerun() is needed,
    and the text box can be cleared (widget state is only writable here).
    """
    if box_key is not None:
        question = (st.session_state.get(box_key) or "").strip()
    if not question:
        if box_key is not None:
            st.session_state[box_key] = ""
        return
    history = st.session_state.get(hist_key, [])
    sheet = _sheet(ds, variable, var_type, time_index)

    offline = st.session_state.get(f"chat_offline_{variable}", True)
    if offline or not _provider()[0]:
        answer = answer_locally(sheet, question)
    else:
        answer, _label = _ask(sheet, history, question)
        if not answer:                       # network died: never go blank
            answer = answer_locally(sheet, question)

    history = history + [("user", question), ("assistant", answer)]
    st.session_state[hist_key] = history
    if box_key is not None:
        st.session_state[box_key] = ""


def _clear(hist_key):
    st.session_state[hist_key] = []


def _chat_body(ds, variable, var_type, time_index=0):
    prov, _key = _provider()

    st.toggle(
        "\u26a1 Offline mode", value=(not prov),
        key=f"chat_offline_{variable}", disabled=(not prov),
        help="Answers from templates filled with numbers computed off your "
             "file. No internet, no API, no model download.")
    offline = st.session_state.get(f"chat_offline_{variable}", True) or not prov

    if offline:
        st.caption(f"**Offline.** Answering about **{variable}** from "
                   "statistics computed on this machine.")
    else:
        st.caption(f"Answers come only from **{variable}** in the file you "
                   "loaded. Nothing else.")

    hist_key = f"_chat_{variable}"
    box_key = f"chatbox_{variable}"
    history = st.session_state.setdefault(hist_key, [])
    args = (ds, variable, var_type, time_index, hist_key)

    # ---- transcript -----------------------------------------------------
    for role, text in history[-8:]:
        if role == "user":
            st.markdown(
                f'<div style="background:rgba(79,255,210,0.09);'
                f'border-left:2px solid #4fffd2;padding:7px 10px;'
                f'border-radius:6px;margin:6px 0;font-size:13px;">{text}</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div style="background:rgba(255,255,255,0.04);'
                f'padding:7px 10px;border-radius:6px;margin:6px 0;'
                f'font-size:13px;line-height:1.55;">{text}</div>',
                unsafe_allow_html=True)

    # ---- one-tap questions ----------------------------------------------
    st.caption("Try:")
    for i, s in enumerate(STARTERS):
        st.button(s, key=f"starter_{i}_{variable}", use_container_width=True,
                  on_click=_submit, args=(s,) + args + (None,))

    st.text_input("Your question", key=box_key,
                  placeholder="e.g. which decade was driest?",
                  label_visibility="collapsed")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.button("Ask", key=f"send_{variable}", use_container_width=True,
                  on_click=_submit, args=("",) + args + (box_key,))
    with c2:
        st.button("Clear", key=f"clear_{variable}", use_container_width=True,
                  on_click=_clear, args=(hist_key,))

    if history:
        with st.expander("🔬 What the model was given"):
            st.json(_sheet(ds, variable, var_type, time_index))


# =====================================================================
# 4b. FLOATING LAUNCHER  —  fixed bottom-right, survives scrolling
# =====================================================================
_FLOAT_CSS = """
<style>
/* st.container(key="pce_chat") gets the class .st-key-pce_chat.
   That class is Streamlit's supported hook for styling a container,
   so this does not depend on internal test ids that change between
   versions. */
.st-key-pce_chat{
    position:fixed !important;
    bottom:22px !important; right:22px !important;
    left:auto !important; top:auto !important;
    width:min(400px, calc(100vw - 44px)) !important;
    max-height:76vh !important; overflow-y:auto !important;
    z-index:999999 !important;
    background:#0f1d2c !important;
    border:1px solid rgba(79,255,210,0.34) !important;
    border-radius:16px !important;
    padding:16px 16px 12px 16px !important;
    box-shadow:0 12px 44px rgba(0,0,0,0.55) !important;
    animation:pce-pop 0.18s ease-out;
}
.st-key-pce_launch{
    position:fixed !important;
    bottom:22px !important; right:22px !important;
    left:auto !important; top:auto !important;
    width:auto !important; z-index:999999 !important;
    background:transparent !important; border:none !important;
    box-shadow:none !important; padding:0 !important;
}
.st-key-pce_launch button{
    border-radius:999px !important;
    padding:0.8rem 1.45rem !important;
    font-size:0.95rem !important; font-weight:600 !important;
    background:#4fffd2 !important; color:#06131f !important;
    border:none !important;
    box-shadow:0 8px 26px rgba(79,255,210,0.36) !important;
    transition:transform 0.18s ease, box-shadow 0.18s ease !important;
}
.st-key-pce_launch button:hover{
    transform:translateY(-3px) scale(1.03);
    box-shadow:0 12px 32px rgba(79,255,210,0.52) !important;
}
.st-key-pce_launch button p{ color:#06131f !important; }
@keyframes pce-pop{
    from{ opacity:0; transform:translateY(14px) scale(0.97); }
    to  { opacity:1; transform:none; }
}
@media (max-width:640px){
    .st-key-pce_chat{
        bottom:12px !important; right:12px !important; left:12px !important;
        width:auto !important; max-height:72vh !important;
    }
}
</style>
"""

OPEN_KEY = "_pce_chat_open"


def _keyed(key):
    """st.container(key=...) where supported; plain container otherwise."""
    try:
        return st.container(key=key)
    except TypeError:
        return st.container()


# st.fragment reruns only the decorated function on interaction, instead
# of the whole script. That is what makes the chat stop rebuilding every
# chart on the page each time a button inside it is pressed.
_FRAGMENT = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None)


def _as_fragment(fn):
    return _FRAGMENT(fn) if _FRAGMENT else fn


def _set_open(value):
    st.session_state[OPEN_KEY] = value


@_as_fragment
def _floating_fragment(ds, variable, var_type, time_index):
    if not st.session_state.get(OPEN_KEY, False):
        with _keyed("pce_launch"):
            st.button("💬  Ask this dataset", key="pce_open_btn",
                      on_click=_set_open, args=(True,))
        return

    with _keyed("pce_chat"):
        head, shut = st.columns([5, 1])
        with head:
            st.markdown("##### 💬 Ask this dataset")
        with shut:
            st.button("✕", key="pce_close_btn", on_click=_set_open, args=(False,))
        _chat_body(ds, variable, var_type, time_index)


def render_floating_chat(ds, variable, var_type, time_index=0):
    """A launcher pinned bottom-right that opens the chat in place."""
    st.markdown(_FLOAT_CSS, unsafe_allow_html=True)
    _floating_fragment(ds, variable, var_type, time_index)


def render_sidebar_chat(ds, variable, var_type, time_index=0):
    """Kept so an older call site still works."""
    with st.expander("💬  Ask this dataset", expanded=False):
        _chat_body(ds, variable, var_type, time_index)
