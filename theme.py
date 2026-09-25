"""
=====================================================================
 theme.py  —  visual polish pass for PyClimaExplorer
=====================================================================
 Pure CSS. Loads after the app's own stylesheet and overrides it.
 Touches no logic, so removing the two lines that call it returns the
 app exactly to how it looked before.

 What it fixes, in order of how much it matters:

   1. Real cards. The render functions now use st.container(border=True),
      which emits stVerticalBlockBorderWrapper. This styles that wrapper
      with the project's teal-on-dark card look, so panels genuinely
      contain their contents instead of floating next to them.

   2. Vertical rhythm. Streamlit's default block gap is uniform, which
      makes a heading sit as far from its own chart as from the previous
      one. Headings are pulled tight to what they label and pushed away
      from what they follow — the single biggest reason a Streamlit page
      reads as unpolished.

   3. Dead space. The default top padding wastes most of a laptop's
      first screen. Trimmed, so content starts where the eye lands.

   4. Widget consistency. Buttons, toggles, selects, expanders, tabs and
      the file uploader are given one shared treatment.
=====================================================================
"""
import streamlit as st

_CSS = """
<style>
/* ============================================================
   1. PAGE FRAME
   ============================================================ */
.block-container{
    padding-top:2.1rem !important;
    padding-bottom:4rem !important;
    max-width:1500px;
}
[data-testid="stDecoration"]{ display:none !important; }

html{ scroll-behavior:smooth; }
*,*::before,*::after{ -webkit-font-smoothing:antialiased; }

/* ============================================================
   2. REAL CARDS  (st.container(border=True))
   ============================================================ */
[data-testid="stVerticalBlockBorderWrapper"]{
    background:var(--card-bg, rgba(255,255,255,0.025));
    border:1px solid var(--card-border, rgba(255,255,255,0.09));
    border-radius:16px;
    padding:20px 22px 18px 22px !important;
    box-shadow:0 2px 20px rgba(0,0,0,0.22);
    transition:border-color var(--anim-fast,0.22s) ease,
               box-shadow  var(--anim-fast,0.22s) ease,
               transform   var(--anim-fast,0.22s) ease;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover{
    border-color:rgba(var(--teal-rgb,79,255,210),0.28);
    box-shadow:0 6px 28px rgba(0,0,0,0.30);
}
/* nested containers must not look like cards inside cards */
[data-testid="stVerticalBlockBorderWrapper"]
    [data-testid="stVerticalBlockBorderWrapper"]{
    background:transparent; border:none; box-shadow:none;
    padding:0 !important;
}
/* leftovers from the old markdown-div approach */
.card:empty{ display:none !important; }

/* ============================================================
   3. VERTICAL RHYTHM — headings belong to what follows them
   ============================================================ */
[data-testid="stVerticalBlock"]{ gap:0.6rem; }

h1,h2,h3,h4{
    font-family:var(--font-head,'Syne',sans-serif) !important;
    letter-spacing:-0.012em;
    line-height:1.25 !important;
}
h1{ font-size:2.05rem !important; margin:0 0 0.35rem 0 !important; }
h2{ font-size:1.45rem !important; margin:1.5rem 0 0.15rem 0 !important; }
h3{ font-size:1.13rem !important; margin:0 0 0.55rem 0 !important; }

/* a card's own heading should hug the card's top edge */
[data-testid="stVerticalBlockBorderWrapper"] h3:first-of-type,
[data-testid="stVerticalBlockBorderWrapper"] h2:first-of-type{
    margin-top:0 !important;
}
.section-label,.card-header-block{ margin-bottom:0.45rem !important; }

/* charts carry their own padding; stop double-spacing them */
[data-testid="stPlotlyChart"]{ margin:-2px 0 -6px 0; }
[data-testid="stPlotlyChart"] > div{ border-radius:12px; overflow:hidden; }

hr{ margin:1.5rem 0 !important; opacity:0.16; }

/* ============================================================
   4. METRIC CARDS
   ============================================================ */
.metric-card{
    border-radius:13px !important;
    padding:13px 15px !important;
    transition:transform var(--anim-fast,0.22s) ease,
               border-color var(--anim-fast,0.22s) ease;
    height:100%;
}
.metric-card:hover{
    transform:translateY(-2px);
    border-color:rgba(var(--teal-rgb,79,255,210),0.34);
}
.metric-label{
    font-size:0.70rem !important; letter-spacing:0.07em;
    text-transform:uppercase; opacity:0.72;
}
.metric-value{
    font-family:var(--font-head,'Syne',sans-serif);
    font-size:1.42rem !important; line-height:1.2 !important;
    margin-top:3px;
}
.metric-sub{ font-size:0.72rem !important; opacity:0.58; margin-top:2px; }

/* ============================================================
   5. WIDGETS
   ============================================================ */
.stButton > button{
    border-radius:10px !important;
    border:1px solid rgba(var(--teal-rgb,79,255,210),0.26) !important;
    background:rgba(var(--teal-rgb,79,255,210),0.07) !important;
    font-weight:500 !important; font-size:0.86rem !important;
    padding:0.45rem 0.9rem !important;
    transition:all var(--anim-fast,0.22s) ease;
}
.stButton > button:hover{
    background:rgba(var(--teal-rgb,79,255,210),0.15) !important;
    border-color:rgba(var(--teal-rgb,79,255,210),0.55) !important;
    transform:translateY(-1px);
}
.stButton > button:active{ transform:translateY(0); }

[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input{
    border-radius:9px !important;
    border-color:rgba(255,255,255,0.11) !important;
}
[data-testid="stTextInput"] input:focus{
    border-color:rgba(var(--teal-rgb,79,255,210),0.5) !important;
    box-shadow:0 0 0 2px rgba(var(--teal-rgb,79,255,210),0.12) !important;
}

[data-testid="stExpander"]{
    border-radius:12px !important;
    border:1px solid rgba(255,255,255,0.08) !important;
    overflow:hidden;
}
[data-testid="stExpander"] summary{
    font-size:0.86rem !important; font-weight:500 !important;
    padding:0.6rem 0.85rem !important;
}
[data-testid="stExpander"] summary:hover{
    color:var(--teal,#4fffd2) !important;
}

.stTabs [data-baseweb="tab-list"]{ gap:4px; border-bottom:1px solid rgba(255,255,255,0.07); }
.stTabs [data-baseweb="tab"]{
    border-radius:9px 9px 0 0; padding:0.5rem 1rem; font-size:0.87rem;
}
.stTabs [aria-selected="true"]{
    background:rgba(var(--teal-rgb,79,255,210),0.09) !important;
    color:var(--teal,#4fffd2) !important;
}

[data-testid="stRadio"] label,[data-testid="stCheckbox"] label{ font-size:0.85rem !important; }

/* ============================================================
   6. SIDEBAR
   ============================================================ */
[data-testid="stSidebar"]{
    border-right:1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] .block-container{ padding-top:1.4rem !important; }
[data-testid="stSidebar"] [data-testid="stExpander"]{ margin-bottom:0.55rem; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{ gap:0.4rem; }
[data-testid="stFileUploaderDropzone"]{
    border-radius:12px !important;
    border:1.5px dashed rgba(var(--teal-rgb,79,255,210),0.28) !important;
    background:rgba(var(--teal-rgb,79,255,210),0.035) !important;
    transition:all var(--anim-fast,0.22s) ease;
}
[data-testid="stFileUploaderDropzone"]:hover{
    border-color:rgba(var(--teal-rgb,79,255,210),0.55) !important;
    background:rgba(var(--teal-rgb,79,255,210),0.07) !important;
}

/* ============================================================
   7. MESSAGES + SCROLLBAR
   ============================================================ */
[data-testid="stAlert"]{ border-radius:11px !important; font-size:0.86rem; }
[data-testid="stCaptionContainer"] p{ font-size:0.76rem !important; opacity:0.64; }

::-webkit-scrollbar{ width:9px; height:9px; }
::-webkit-scrollbar-track{ background:transparent; }
::-webkit-scrollbar-thumb{
    background:rgba(var(--teal-rgb,79,255,210),0.20); border-radius:9px;
}
::-webkit-scrollbar-thumb:hover{ background:rgba(var(--teal-rgb,79,255,210),0.38); }

/* ============================================================
   8. NARROW SCREENS
   ============================================================ */
@media (max-width:820px){
    .block-container{ padding-top:1.2rem !important; }
    [data-testid="stVerticalBlockBorderWrapper"]{ padding:15px 14px !important; }
    h1{ font-size:1.6rem !important; }
    .metric-value{ font-size:1.18rem !important; }
}

/* ============================================================
   9. SIDEBAR TOGGLE — always visible, not just on hover
   ============================================================ */
/* close button, inside the sidebar */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarHeader"] [data-testid="stSidebarCollapseButton"]{
    opacity:1 !important; visibility:visible !important;
    display:flex !important;
}
[data-testid="stSidebarCollapseButton"] button{
    background:rgba(var(--teal-rgb,79,255,210),0.10) !important;
    border:1px solid rgba(var(--teal-rgb,79,255,210),0.32) !important;
    border-radius:10px !important;
    transition:background .2s ease, transform .2s ease !important;
}
[data-testid="stSidebarCollapseButton"] button:hover{
    background:rgba(var(--teal-rgb,79,255,210),0.22) !important;
    transform:translateX(-2px);
}
/* reopen button, shown in the header once the sidebar is closed */
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapsedControl"] button,
button[data-testid="collapsedControl"]{
    opacity:1 !important; visibility:visible !important;
    background:rgba(var(--teal-rgb,79,255,210),0.16) !important;
    border:1px solid rgba(var(--teal-rgb,79,255,210),0.45) !important;
    border-radius:10px !important;
    box-shadow:0 0 18px rgba(var(--teal-rgb,79,255,210),0.25) !important;
    transition:background .2s ease, transform .2s ease !important;
}
[data-testid="stExpandSidebarButton"]:hover,
[data-testid="stSidebarCollapsedControl"] button:hover,
button[data-testid="collapsedControl"]:hover{
    background:rgba(var(--teal-rgb,79,255,210),0.28) !important;
    transform:translateX(2px);
}
[data-testid="stSidebar"]{
    transition:transform .32s cubic-bezier(.4,0,.2,1),
               margin-left .32s cubic-bezier(.4,0,.2,1),
               width .32s cubic-bezier(.4,0,.2,1) !important;
}

/* ============================================================
   10. NAV — segmented pill bar with an active state
   ============================================================ */
.st-key-pce_nav{
    position:sticky; top:50px; z-index:60;
    background:__TRACK__ !important;
    backdrop-filter:blur(16px); -webkit-backdrop-filter:blur(16px);
    border:1px solid rgba(var(--teal-rgb,79,255,210),0.16) !important;
    border-radius:999px !important;
    padding:6px !important;
    margin:2px 0 8px 0;
    box-shadow:0 8px 28px rgba(0,0,0,__SHADOW__);
}
.st-key-pce_nav [data-testid="stHorizontalBlock"]{ gap:6px !important; }
div.st-key-pce_nav [data-testid="stHorizontalBlock"] button{
    position:relative; overflow:hidden;
    border-radius:999px !important;
    border:1px solid transparent !important;
    background:transparent !important;
    box-shadow:none !important;
    min-height:42px;
    padding:0.5rem 0.9rem !important;
    transition:background .28s ease, color .28s ease,
               transform .2s ease, box-shadow .28s ease,
               border-color .28s ease !important;
}
div.st-key-pce_nav [data-testid="stHorizontalBlock"] button p{
    font-size:0.9rem !important; font-weight:600 !important;
    letter-spacing:0.015em; white-space:nowrap;
    color:__IDLE__ !important;
    transition:color .28s ease;
}
div.st-key-pce_nav [data-testid="stHorizontalBlock"] button:hover{
    background:rgba(var(--teal-rgb,79,255,210),0.10) !important;
    border-color:rgba(var(--teal-rgb,79,255,210),0.28) !important;
    transform:translateY(-1px);
}
div.st-key-pce_nav [data-testid="stHorizontalBlock"] button:hover p{
    color:rgb(var(--teal-rgb,79,255,210)) !important;
}
div.st-key-pce_nav [data-testid="stHorizontalBlock"] button:active{
    transform:translateY(0) scale(0.98);
}
/* the page you are on */
div.st-key-pce_nav [data-testid="stHorizontalBlock"] button[kind="primary"],
div.st-key-pce_nav [data-testid="stHorizontalBlock"] [data-testid="stBaseButton-primary"]{
    background:linear-gradient(135deg,
        rgba(var(--teal-rgb,79,255,210),1) 0%,
        rgba(var(--teal-rgb,79,255,210),0.78) 100%) !important;
    border-color:transparent !important;
    box-shadow:0 4px 20px rgba(var(--teal-rgb,79,255,210),0.38),
               inset 0 1px 0 rgba(255,255,255,0.35) !important;
}
div.st-key-pce_nav [data-testid="stHorizontalBlock"] button[kind="primary"] p,
div.st-key-pce_nav [data-testid="stHorizontalBlock"] [data-testid="stBaseButton-primary"] p{
    color:__ACTIVE_TEXT__ !important;
}
div.st-key-pce_nav [data-testid="stHorizontalBlock"] button[kind="primary"]:hover,
div.st-key-pce_nav [data-testid="stHorizontalBlock"] [data-testid="stBaseButton-primary"]:hover{
    transform:translateY(-1px);
    box-shadow:0 6px 26px rgba(var(--teal-rgb,79,255,210),0.5),
               inset 0 1px 0 rgba(255,255,255,0.35) !important;
}
/* light sweep across a button on hover */
div.st-key-pce_nav [data-testid="stHorizontalBlock"] button::after{
    content:""; position:absolute; top:0; left:-130%;
    width:60%; height:100%; pointer-events:none;
    background:linear-gradient(100deg, transparent,
               rgba(255,255,255,0.22), transparent);
    transition:left .6s ease;
}
div.st-key-pce_nav [data-testid="stHorizontalBlock"] button:hover::after{ left:140%; }

@media (max-width:820px){
    .st-key-pce_nav{ border-radius:18px !important; position:static; }
    div.st-key-pce_nav [data-testid="stHorizontalBlock"] button p{ font-size:0.78rem !important; }
}

/* ============================================================
   11. HEADER — transparent, so the page runs to the top edge
   ============================================================ */
[data-testid="stHeader"]{
    background:transparent !important;
    backdrop-filter:none !important; -webkit-backdrop-filter:none !important;
    border-bottom:none !important;
    box-shadow:none !important;
}
[data-testid="stToolbar"]{ background:transparent !important; }
</style>
"""



# Light mode only. Streamlit's upload widget keeps its dark styling even when
# the app switches to light, leaving a black button with unreadable text.
_LIGHT_FIXES = """
<style>
[data-testid="stFileUploaderDropzone"],
section[data-testid="stFileUploadDropzone"]{
    background:rgba(0,122,110,0.04) !important;
    border:1.5px dashed rgba(0,122,110,0.38) !important;
}
[data-testid="stFileUploaderDropzone"]:hover,
section[data-testid="stFileUploadDropzone"]:hover{
    background:rgba(0,122,110,0.08) !important;
    border-color:rgba(0,122,110,0.60) !important;
}
/* the Upload / Browse button */
[data-testid="stFileUploaderDropzone"] button,
section[data-testid="stFileUploadDropzone"] button{
    background:#ffffff !important;
    color:#007a6e !important;
    border:1px solid rgba(0,122,110,0.45) !important;
    box-shadow:0 2px 8px rgba(12,40,70,0.08) !important;
}
[data-testid="stFileUploaderDropzone"] button *,
section[data-testid="stFileUploadDropzone"] button *{
    color:#007a6e !important; fill:#007a6e !important;
}
[data-testid="stFileUploaderDropzone"] button:hover,
section[data-testid="stFileUploadDropzone"] button:hover{
    background:rgba(0,122,110,0.08) !important;
    border-color:#007a6e !important;
}
/* "Drag and drop" and "200MB per file" hints */
[data-testid="stFileUploaderDropzoneInstructions"],
[data-testid="stFileUploaderDropzoneInstructions"] *,
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span:not(button span){
    color:rgba(12,40,70,0.62) !important;
}
[data-testid="stFileUploaderDropzone"] svg:not(button svg){
    color:rgba(12,40,70,0.55) !important; fill:rgba(12,40,70,0.55) !important;
}
/* the file row that appears after uploading */
[data-testid="stFileUploaderFile"],
[data-testid="stFileUploaderFile"] *,
[data-testid="stFileUploaderFileName"]{
    color:#0c2846 !important;
}
[data-testid="stFileUploaderFile"] small{ color:rgba(12,40,70,0.55) !important; }
[data-testid="stFileUploaderDeleteBtn"] button,
[data-testid="stFileUploaderDeleteBtn"] button *{
    background:transparent !important; color:rgba(12,40,70,0.60) !important;
    fill:rgba(12,40,70,0.60) !important; border:none !important;
}
</style>
"""

def inject_theme(dark=True):
    """Call once, immediately after the app's own stylesheet."""
    css = (_CSS
           .replace("__TRACK__", "rgba(8,18,30,0.62)" if dark else "rgba(255,255,255,0.72)")
           .replace("__SHADOW__", "0.30" if dark else "0.08")
           .replace("__IDLE__", "rgba(255,255,255,0.78)" if dark else "rgba(12,40,70,0.78)")
           .replace("__ACTIVE_TEXT__", "#04121c" if dark else "#ffffff"))
    if not dark:
        css += _LIGHT_FIXES
    st.markdown(css, unsafe_allow_html=True)
