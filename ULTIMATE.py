import streamlit as st
import numpy as np
import pandas as pd
import xarray as xr
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import tempfile
import base64
import os
import geopandas as gpd

# --- AI Insight Briefing (ai_insight.py must sit next to this file) ---
try:
    from ai_insight import render_insight_briefing
    AI_INSIGHT_OK = True
    _AI_INSIGHT_ERR = ""
except Exception as _e:
    AI_INSIGHT_OK = False
    _AI_INSIGHT_ERR = str(_e)


# =================================================================
# SHAPEFILE PATH  — relative path for deployment
# =================================================================
_script_dir = os.path.dirname(os.path.abspath(__file__))

# Natural Earth ships de facto boundaries by default. For an Indian audience
# that draws Kashmir on the Line of Control rather than on India's official
# boundary. If the India point-of-view file is present, prefer it.
_NE_CANDIDATES = [
    "ne_10m_admin_0_countries_ind.shp",   # India POV (de jure) - preferred
    "ne_50m_admin_0_countries_ind.shp",
    "ne_110m_admin_0_countries.shp",      # de facto fallback
    "ne_10m_admin_0_countries.shp",
]
NE_COUNTRIES_PATH = next(
    (os.path.join(_script_dir, f) for f in _NE_CANDIDATES
     if os.path.exists(os.path.join(_script_dir, f))),
    os.path.join(_script_dir, "ne_110m_admin_0_countries.shp"),
)
NE_IS_INDIA_POV = NE_COUNTRIES_PATH.endswith("_ind.shp")

# India's state boundaries, from geoBoundaries (CC BY 4.0). Far more detailed
# than the 1:110m world file, and the union of the states becomes India's
# outline for clipping. Optional: without it, Natural Earth is used.
_STATE_CANDIDATES = [
    "geoBoundaries-IND-ADM1_simplified.geojson",
    "india_states.geojson",
    "geoBoundaries-IND-ADM1.geojson",
]
INDIA_STATES_PATH = next(
    (os.path.join(_script_dir, f) for f in _STATE_CANDIDATES
     if os.path.exists(os.path.join(_script_dir, f))), "")


@st.cache_resource(show_spinner=False)
def _load_world(path):
    try:
        return gpd.read_file(path).to_crs("EPSG:4326")
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _load_india_states(path):
    if not path:
        return None
    try:
        g = gpd.read_file(path).to_crs("EPSG:4326")
        # ~500 m tolerance: invisible at state zoom, far lighter to draw
        g["geometry"] = g.geometry.simplify(0.005, preserve_topology=True)
        return g
    except Exception:
        return None


WORLD_GDF = _load_world(NE_COUNTRIES_PATH)
INDIA_STATES_GDF = _load_india_states(INDIA_STATES_PATH)
_BOUNDARY_SIG = f"{NE_COUNTRIES_PATH}|{INDIA_STATES_PATH}"

# Set each run by the Region Filter. Maps read these to draw a selected state
# in context, from the full file, rather than as a handful of cropped cells.
ACTIVE_REGION = None      # dict(name, box, geom) or None
DS_FULL = None            # the dataset before the Region Filter cut it
# =================================================================
# CUSTOM PLOTLY THEME  — Code A (rebuilt each render, dark/light aware)
# =================================================================
_CLIMATE_PALETTE = ["#4fffd2","#60b4ff","#ffa040","#ff6b6b","#c89bff","#7effa0","#ffe066"]

def _build_plotly_template(dark: bool):
    """Rebuilds the pyclima Plotly template for dark or light mode."""
    if dark:
        font_color = "rgba(255,255,255,0.88)";  grid = "rgba(255,255,255,0.07)"
        line_c     = "rgba(255,255,255,0.09)";  tick  = "rgba(255,255,255,0.18)"
        tick_f     = "rgba(255,255,255,0.52)";  titf  = "rgba(255,255,255,0.68)"
        leg_bg     = "rgba(0,0,0,0.28)";        leg_b = "rgba(255,255,255,0.08)"
        hov_bg     = "rgba(6,20,32,0.94)";      hov_b = "#4fffd2"
        hov_fc     = "white"
    else:
        font_color = "rgba(12,30,48,0.90)";     grid = "rgba(0,80,120,0.07)"
        line_c     = "rgba(0,80,120,0.12)";     tick  = "rgba(0,80,120,0.20)"
        tick_f     = "rgba(12,40,70,0.62)";     titf  = "rgba(12,40,70,0.78)"
        leg_bg     = "rgba(225,240,255,0.80)";  leg_b = "rgba(0,140,120,0.18)"
        hov_bg     = "rgba(238,250,255,0.97)";  hov_b = "#009688"
        hov_fc     = "rgba(12,30,48,0.92)"

    pio.templates["pyclima"] = go.layout.Template(
        layout=go.Layout(
            font=dict(family="'DM Sans','Segoe UI',sans-serif", color=font_color, size=13),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            colorway=_CLIMATE_PALETTE,
            title=dict(font=dict(size=15, color="#4fffd2" if dark else "#009688"), x=0.01),
            xaxis=dict(gridcolor=grid, linecolor=line_c, tickcolor=tick,
                       tickfont=dict(color=tick_f, size=11),
                       title_font=dict(color=titf, size=12), zeroline=False),
            yaxis=dict(gridcolor=grid, linecolor=line_c, tickcolor=tick,
                       tickfont=dict(color=tick_f, size=11),
                       title_font=dict(color=titf, size=12), zeroline=False),
            legend=dict(bgcolor=leg_bg, bordercolor=leg_b, borderwidth=1,
                        font=dict(color=font_color, size=12)),
            hoverlabel=dict(bgcolor=hov_bg, bordercolor=hov_b,
                            font=dict(color=hov_fc, size=12,
                                      family="'DM Sans','Segoe UI',sans-serif"),
                            namelength=-1),
            margin=dict(l=10, r=10, t=44, b=10),
        )
    )
    pio.templates.default = "pyclima"

# =================================================================
# 3D GLOBE  — Code B backend (lighting) + Code A styling
# =================================================================
INDIA_CENTER   = (22.5, 79.0)          # lat, lon — roughly Bhopal
INDIA_BOUNDS   = (5.0, 38.5, 66.5, 98.5)  # lat_min, lat_max, lon_min, lon_max
_GLOBE_ACCENT  = "#4fffd2"
_GLOBE_BASE    = "#0e1c2b"


def _sphere_xyz(lat_deg, lon_deg, R=1.0):
    """Lat/lon in degrees -> cartesian coordinates on a sphere of radius R."""
    la = np.deg2rad(np.asarray(lat_deg, dtype=float))
    lo = np.deg2rad(np.asarray(lon_deg, dtype=float))
    return (R * np.cos(la) * np.cos(lo),
            R * np.cos(la) * np.sin(lo),
            R * np.sin(la))


def _camera_over(lat, lon, dist=1.75):
    """Put the viewer directly above a given lat/lon."""
    x, y, z = _sphere_xyz(lat, lon, dist)
    return dict(eye=dict(x=float(x), y=float(y), z=float(z)),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1))


def _find_india(gdf):
    """
    Natural Earth vintages name the country column differently, and some
    shipped copies have odd casing or padding. Scan every text column.
    """
    for col in ("ADMIN", "NAME", "NAME_LONG", "SOVEREIGNT", "NAME_EN",
                "name", "admin", "COUNTRY", "CNTRY_NAME"):
        if col in gdf.columns:
            m = gdf[col].astype(str).str.strip().str.lower() == "india"
            if m.any():
                return m
    for col in gdf.columns:
        if col == "geometry":
            continue
        try:
            m = gdf[col].astype(str).str.strip().str.lower() == "india"
            if m.any():
                return m
        except Exception:
            continue
    return None


_INDIA_GEOM = None


@st.cache_resource(show_spinner=False)
def _india_geom_cached(sig):
    from shapely.ops import unary_union
    from shapely.geometry import Polygon, MultiPolygon
    geom = None
    try:
        if INDIA_STATES_GDF is not None:
            geom = unary_union(INDIA_STATES_GDF.geometry.tolist())
        elif WORLD_GDF is not None:
            m = _find_india(WORLD_GDF)
            if m is not None and m.any():
                geom = unary_union(WORLD_GDF.loc[m, "geometry"].tolist())
    except Exception:
        return None
    if geom is None:
        return None
    # Keep outer rings only. Slivers where simplified state edges don't meet
    # exactly would otherwise punch holes through the middle of the country.
    parts = [geom] if geom.geom_type == "Polygon" else list(getattr(geom, "geoms", []))
    parts = [Polygon(p.exterior) for p in parts
             if p.geom_type == "Polygon" and p.area > 1e-4]
    if not parts:
        return None
    return MultiPolygon(parts) if len(parts) > 1 else parts[0]


def _india_geom():
    """India's outline as one shapely geometry (states union if available)."""
    return _india_geom_cached(_BOUNDARY_SIG)


def _mask_inside(geom, lat_grid, lon_grid):
    """Boolean array: True where the grid point falls inside `geom`."""
    if geom is None:
        return None
    lon_w = np.where(lon_grid > 180.0, lon_grid - 360.0, lon_grid)  # 0..360 -> -180..180
    try:
        import shapely
        if hasattr(shapely, "contains_xy"):
            return np.asarray(shapely.contains_xy(geom, lon_w, lat_grid), dtype=bool)
    except Exception:
        pass
    try:
        from shapely.geometry import Point
        from shapely.prepared import prep
        pg = prep(geom)
        out = np.zeros(lat_grid.shape, dtype=bool)
        for idx in np.ndindex(lat_grid.shape):
            out[idx] = pg.contains(Point(float(lon_w[idx]), float(lat_grid[idx])))
        return out
    except Exception:
        return None


def _upsample(vals, lat, lon, target=150):
    """
    Bilinear refinement so a coarse 2.5-degree grid still traces a
    recognisable coastline once it is clipped. Pure numpy, no scipy.
    """
    vals = np.asarray(vals, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if vals.ndim != 2:
        return vals, lat, lon

    if lat.size > 1 and lat[0] > lat[-1]:
        lat, vals = lat[::-1], vals[::-1, :]
    if lon.size > 1 and lon[0] > lon[-1]:
        lon, vals = lon[::-1], vals[:, ::-1]

    ny, nx = vals.shape
    if ny < 2 or nx < 2:
        return vals, lat, lon
    fy = int(np.ceil(target / ny)); fx = int(np.ceil(target / nx))
    if fy <= 1 and fx <= 1:
        return vals, lat, lon

    new_lat = np.linspace(lat[0], lat[-1], min(ny * max(fy, 1), 400))
    new_lon = np.linspace(lon[0], lon[-1], min(nx * max(fx, 1), 400))

    tmp = np.empty((ny, new_lon.size))
    for i in range(ny):
        tmp[i] = np.interp(new_lon, lon, vals[i])
    out = np.empty((new_lat.size, new_lon.size))
    for j in range(new_lon.size):
        out[:, j] = np.interp(new_lat, lat, tmp[:, j])
    return out, new_lat, new_lon


def _boundary_trace(rows, R, color, width, name=None, simplify=None):
    """
    All polygons collapsed into ONE Scatter3d using None as a pen-lift.
    177 separate traces make the globe stutter when rotating; one does not.
    """
    xs, ys, zs = [], [], []
    for geom in rows:
        if geom is None:
            continue
        if simplify:
            try:
                geom = geom.simplify(simplify)
            except Exception:
                pass
        polys = ([geom] if geom.geom_type == "Polygon"
                 else list(geom.geoms) if geom.geom_type == "MultiPolygon" else [])
        for poly in polys:
            lo, la = poly.exterior.coords.xy
            px, py, pz = _sphere_xyz(np.array(la), np.array(lo), R)
            xs.extend(px.tolist() + [None])
            ys.extend(py.tolist() + [None])
            zs.extend(pz.tolist() + [None])
    if not xs:
        return None
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                        line=dict(color=color, width=width),
                        name=name or "", showlegend=False, hoverinfo="skip")


def make_globe_figure(lon, lat, values, title="3D Climate Globe", focus="india",
                      label="Value", units="", clip_to_india=True):
    """
    Three stacked layers so the globe survives an India-only dataset:

      R = 1.000  a full-world base sphere, drawn independently of the data.
                 Without this, an India-only file renders as a curved patch
                 floating in empty space rather than a globe.
      R = 1.004  the climate data, laid on the sphere like a decal. It covers
                 whatever region the file covers and no more.
      R = 1.008  coastlines — the world dim, India picked out in accent.

    focus: "india" | "data" | "world"
    """
    fig = go.Figure()

    # ---- layer 1: the base sphere (always the whole planet) -----------
    blat = np.linspace(-90, 90, 73)
    blon = np.linspace(-180, 180, 145)
    BLON, BLAT = np.meshgrid(blon, blat)
    bx, by, bz = _sphere_xyz(BLAT, BLON, 1.0)
    fig.add_trace(go.Surface(
        x=bx, y=by, z=bz,
        surfacecolor=np.zeros_like(bx),
        colorscale=[[0, _GLOBE_BASE], [1, _GLOBE_BASE]],
        showscale=False, opacity=1.0,
        lighting=dict(ambient=0.62, diffuse=0.45, specular=0.08, roughness=0.9),
        lightposition=dict(x=200, y=0, z=150),
        hoverinfo="skip",
    ))

    # ---- layer 2: the data decal, clipped to India's coastline --------
    vals = np.asarray(values, dtype=float)
    la = np.asarray(lat, dtype=float)
    lo = np.asarray(lon, dtype=float)

    # Refine first so the clipped edge follows the coast rather than
    # stair-stepping across 2.5-degree cells.
    span = (float(np.nanmax(la) - np.nanmin(la)),
            float(np.nanmax(lo) - np.nanmin(lo)))
    regional = span[0] < 70 and span[1] < 70
    if regional:
        vals, la, lo = _upsample(vals, la, lo, target=220)

    lon_grid, lat_grid = np.meshgrid(lo, la)
    dx, dy, dz = _sphere_xyz(lat_grid, lon_grid, 1.004)

    # Punching NaN through the coordinates (not just the colour) is what
    # removes the rectangle — Plotly draws no facet where a vertex is NaN.
    clipped = False
    if regional and clip_to_india:
        inside = _india_mask_grid(lo, la, _BOUNDARY_SIG)
        if inside is not None and inside.any():
            dx = np.where(inside, dx, np.nan)
            dy = np.where(inside, dy, np.nan)
            dz = np.where(inside, dz, np.nan)
            vals = np.where(inside, vals, np.nan)
            clipped = True

    vmin = float(np.nanmin(vals)); vmax_v = float(np.nanmax(vals))
    if not np.isfinite(vmin) or not np.isfinite(vmax_v) or vmin == vmax_v:
        vmin, vmax_v = (vmin - 0.5, vmin + 0.5) if np.isfinite(vmin) else (0.0, 1.0)

    cb_title = f"{label} ({units})" if units else str(label)
    fig.add_trace(go.Surface(
        x=dx, y=dy, z=dz,
        surfacecolor=vals,                 # real values, not 0-1
        colorscale="RdBu_r", cmin=vmin, cmax=vmax_v,
        showscale=True,
        colorbar=dict(
            title=dict(text=cb_title, font=dict(color="rgba(255,255,255,0.70)", size=12)),
            tickfont=dict(color="rgba(255,255,255,0.58)", size=10),
            thickness=13, len=0.72,
        ),
        opacity=1.0,
        lighting=dict(ambient=0.45, diffuse=0.85, specular=0.35,
                      roughness=0.55, fresnel=0.2),
        lightposition=dict(x=200, y=0, z=150),
        hovertemplate=(f"<b>{label}</b>: %{{surfacecolor:.2f}} {units}"
                       "<extra></extra>"),
    ))

    # ---- layer 3: borders — world dim, Indian states faint, India bright
    if WORLD_GDF is not None:
        try:
            mask = _find_india(WORLD_GDF)
            rest = (WORLD_GDF.loc[~mask, "geometry"].tolist() if mask is not None
                    else WORLD_GDF["geometry"].tolist())
            t = _boundary_trace(rest, 1.008, "rgba(255,255,255,0.20)", 1.0,
                                simplify=0.25)
            if t is not None:
                fig.add_trace(t)
        except Exception:
            pass
    if INDIA_STATES_GDF is not None:
        try:
            t = _boundary_trace(INDIA_STATES_GDF.geometry.tolist(), 1.009,
                                "rgba(255,255,255,0.42)", 1.1, simplify=0.02)
            if t is not None:
                fig.add_trace(t)
        except Exception:
            pass
    try:
        ig = _india_geom()
        if ig is not None:
            t = _boundary_trace([ig], 1.010, _GLOBE_ACCENT, 3.2, name="India",
                                simplify=0.01)
            if t is not None:
                fig.add_trace(t)
    except Exception:
        pass

    # ---- outline the analysed extent -----------------------------------
    # An inland region has no coastline to clip against, so the data stays
    # rectangular. Drawing its edge makes that read as "this is the area
    # under analysis" rather than as a rendering fault.
    try:
        span_deg = max(float(np.nanmax(la) - np.nanmin(la)),
                       float(np.nanmax(lo) - np.nanmin(lo)))
        # A coastal or national extent already reads as a shape once clipped.
        # Only an inland box (almost every cell inside India, and small)
        # needs an outline to look deliberate rather than accidental.
        if regional and clipped and inside_frac > 0.97 and span_deg < 15:
            la0, la1 = float(np.nanmin(la)), float(np.nanmax(la))
            lo0, lo1 = float(np.nanmin(lo)), float(np.nanmax(lo))
            ring_lat, ring_lon = [], []
            steps = 40
            for a, b, c, d in ((la0, lo0, la0, lo1), (la0, lo1, la1, lo1),
                               (la1, lo1, la1, lo0), (la1, lo0, la0, lo0)):
                ring_lat += np.linspace(a, c, steps).tolist()
                ring_lon += np.linspace(b, d, steps).tolist()
            fx, fy, fz = _sphere_xyz(np.array(ring_lat), np.array(ring_lon), 1.012)
            fig.add_trace(go.Scatter3d(
                x=fx, y=fy, z=fz, mode="lines",
                line=dict(color="rgba(255,255,255,0.55)", width=2),
                showlegend=False, hoverinfo="skip"))
    except Exception:
        pass

    # ---- camera --------------------------------------------------------
    if focus == "world":
        cam = _camera_over(15.0, 60.0, dist=2.15)
    elif focus == "data":
        try:
            cam = _camera_over(float(np.nanmean(lat)), float(np.nanmean(lon)), dist=1.95)
        except Exception:
            cam = _camera_over(*INDIA_CENTER, dist=1.95)
    else:
        cam = _camera_over(*INDIA_CENTER, dist=1.95)

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode="data",
            camera=cam,
            xaxis_backgroundcolor="rgba(0,0,0,0)",
            yaxis_backgroundcolor="rgba(0,0,0,0)",
            zaxis_backgroundcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig

# =================================================================
# IMAGE LOAD  — Code B path-relative approach + Code A absolute path
# =================================================================
def _get_img_b64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""


img = (
    _get_img_b64(os.path.join(_script_dir, "photo.jpg"))
)

# =================================================================
# PAGE CONFIG
# =================================================================
st.set_page_config(
    page_title="PyClimaExplorer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =================================================================
# SESSION STATE
# =================================================================
if "page"           not in st.session_state: st.session_state.page           = "Explore"
if "dataset_loaded" not in st.session_state: st.session_state.dataset_loaded = False
if "dark_mode"      not in st.session_state: st.session_state.dark_mode      = True
if "story_step"     not in st.session_state: st.session_state.story_step     = 0  # Code B

_build_plotly_template(st.session_state.dark_mode)
DK = st.session_state.dark_mode

# =================================================================
# THEME TOKENS  — Code A, recomputed each render
# =================================================================
if DK:
    _bg_overlay   = "rgba(0,0,0,0.52), rgba(0,0,0,0.84)"
    _sidebar_bg   = "#060f18"
    _card_bg      = "rgba(6,18,32,0.68)"
    _card_border  = "rgba(79,255,210,0.12)"
    _text_main    = "rgba(255,255,255,0.90)"
    _text_muted   = "rgba(255,255,255,0.45)"
    _teal         = "#4fffd2"
    _teal_rgb     = "79,255,210"
    _section_bg   = "rgba(79,255,210,0.05)"
    _section_bdr  = "rgba(79,255,210,0.28)"
    _metric_bg    = "rgba(6,22,36,0.84)"
    _metric_bdr   = "rgba(79,255,210,0.16)"
    _metric_val   = "#4fffd2"
    _metric_lbl   = "rgba(255,255,255,0.38)"
    _metric_sub   = "rgba(255,255,255,0.38)"
    _btn_bg       = "rgba(6,18,32,0.62)"
    _btn_bdr      = "rgba(79,255,210,0.14)"
    _btn_color    = "rgba(255,255,255,0.72)"
    _glass_bg     = "rgba(4,14,22,0.62)"
    _glass_bdr    = "rgba(79,255,210,0.14)"
    _sb_text      = "rgba(255,255,255,0.82)"
    _crumb_color  = "rgba(255,255,255,0.32)"
    _hr_color     = "rgba(79,255,210,0.18)"
    _header_bg    = "rgba(2,5,14,0.92)"
else:
    _bg_overlay   = "rgba(240,244,248,0.72), rgba(230,238,245,0.88)"
    _sidebar_bg   = "#eef4f8"
    _card_bg      = "rgba(240,250,255,0.82)"
    _card_border  = "rgba(0,148,130,0.22)"
    _text_main    = "rgba(12,32,52,0.92)"
    _text_muted   = "rgba(12,40,70,0.55)"
    _teal         = "#007a6e"
    _teal_rgb     = "0,122,110"
    _section_bg   = "rgba(0,148,130,0.07)"
    _section_bdr  = "rgba(0,148,130,0.32)"
    _metric_bg    = "rgba(228,245,252,0.88)"
    _metric_bdr   = "rgba(0,148,130,0.22)"
    _metric_val   = "#006b60"
    _metric_lbl   = "rgba(12,40,70,0.48)"
    _metric_sub   = "rgba(12,40,70,0.44)"
    _btn_bg       = "rgba(210,238,252,0.70)"
    _btn_bdr      = "rgba(0,148,130,0.22)"
    _btn_color    = "rgba(12,32,52,0.82)"
    _glass_bg     = "rgba(220,242,255,0.78)"
    _glass_bdr    = "rgba(0,148,130,0.20)"
    _sb_text      = "rgba(12,32,52,0.90)"
    _crumb_color  = "rgba(12,40,70,0.45)"
    _hr_color     = "rgba(0,148,130,0.20)"
    _header_bg    = "rgba(255,255,255,1.0)"

# (…rest of your file remains exactly as in paste.txt, unchanged…)


# =================================================================
# CSS  — Code A full GUI  +  Deploy/Run buttons TOP-LEFT fix
# =================================================================
st.markdown(f"""
<style>
/* ── GOOGLE FONTS ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Syne:wght@600;700;800&display=swap');

/* ── ROOT TOKENS ── */
:root {{
    --teal:        {_teal};
    --teal-rgb:    {_teal_rgb};
    --card-bg:     {_card_bg};
    --card-border: {_card_border};
    --text-main:   {_text_main};
    --text-muted:  {_text_muted};
    --font-body:   'DM Sans','Segoe UI',sans-serif;
    --font-head:   'Syne','Segoe UI',sans-serif;
    --anim-fast:   0.22s;
    --anim-med:    0.42s;
}}

/* ── SCROLLBAR ── */
::-webkit-scrollbar {{ width:5px; }}
::-webkit-scrollbar-track {{ background:rgba(0,0,0,0.10); }}
::-webkit-scrollbar-thumb {{ background:rgba(var(--teal-rgb),0.28); border-radius:3px; }}

/* ══════════════════════════════════════════
   ANIMATED BACKGROUND (Code A)
   ══════════════════════════════════════════ */
.stApp {{
    background-image:
        linear-gradient({_bg_overlay}),
        url("data:image/jpg;base64,{img}");
    background-size: 115% 115%;
    background-position: 50% 50%;
    background-attachment: fixed;
    color: var(--text-main);
    font-family: var(--font-body);
    animation: bgDrift 22s ease-in-out infinite;
}}
@keyframes bgDrift {{
    0%   {{ background-size:115% 115%; background-position:50% 50%; }}
    25%  {{ background-size:120% 120%; background-position:52% 48%; }}
    50%  {{ background-size:118% 118%; background-position:50% 52%; }}
    75%  {{ background-size:122% 122%; background-position:48% 50%; }}
    100% {{ background-size:115% 115%; background-position:50% 50%; }}
}}

/* ══════════════════════════════════════════
   DEPLOY / RUN BUTTONS  →  TOP-LEFT FIX
   ══════════════════════════════════════════
   Streamlit renders the toolbar (which contains
   Deploy & Run/Stop) inside [data-testid="stHeader"].
   By default Streamlit right-aligns it.
   We fix the header as a flex row and push the
   toolbar to the LEFT edge.
   ══════════════════════════════════════════ */
[data-testid="stHeader"] {{
    position: fixed !important;
    top: 0 !important; left: 0 !important; right: 0 !important;
    z-index: 9999 !important;
    height: 46px !important;
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    justify-content: flex-start !important;   /* ← key: children align LEFT */
    padding: 0 10px !important;
    background: {_header_bg} !important;
    backdrop-filter: blur(14px) !important;
    -webkit-backdrop-filter: blur(14px) !important;
    border-bottom: 1px solid {'rgba(79,255,210,0.18)' if DK else 'rgba(0,0,0,0.08)'} !important;
    box-shadow: 0 2px 18px rgba(0,0,0,{'0.40' if DK else '0.06'}) !important;
}}

/* Toolbar sits at far LEFT */
[data-testid="stToolbar"] {{
    order: 0 !important;
    position: static !important;
    display: flex !important;
    align-items: center !important;
    gap: 6px !important;
    margin-left: 0 !important;
    margin-right: auto !important;  /* pushes any other header children right */
    background: transparent !important;
}}

/* Style each toolbar action button */
[data-testid="stToolbar"] button,
[data-testid="stToolbarActions"] button {{
    background: rgba(var(--teal-rgb),{'0.10' if DK else '0.06'}) !important;
    border: 1px solid rgba(var(--teal-rgb),{'0.30' if DK else '0.28'}) !important;
    border-radius: 7px !important;
    color: {'var(--teal)' if DK else 'rgba(12,40,70,0.85)'} !important;
    font-family: var(--font-body) !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    padding: 4px 12px !important;
    letter-spacing: 0.4px !important;
    transition: background 0.18s ease, border-color 0.18s ease !important;
    cursor: pointer !important;
}}
[data-testid="stToolbar"] button:hover,
[data-testid="stToolbarActions"] button:hover {{
    background: rgba(var(--teal-rgb),0.18) !important;
    border-color: var(--teal) !important;
    color: {'#ffffff' if DK else 'rgba(12,32,52,0.95)'} !important;
    box-shadow: 0 2px 12px rgba(var(--teal-rgb),0.18) !important;
}}
/* any SVG icons or spans inside toolbar buttons */
[data-testid="stToolbar"] button span,
[data-testid="stToolbar"] button svg,
[data-testid="stToolbarActions"] button span,
[data-testid="stToolbarActions"] button svg {{
    color: {'var(--teal)' if DK else 'rgba(12,40,70,0.85)'} !important;
    fill:  {'var(--teal)' if DK else 'rgba(12,40,70,0.85)'} !important;
}}

/* ── LIGHT MODE: Streamlit default header buttons (Deploy etc) ── */
{'/* dark — no override */' if DK else '''
header[data-testid="stHeader"] button,
header[data-testid="stHeader"] a,
[data-testid="stHeader"] [data-testid="stToolbar"] > * > button,
[data-testid="stDeployButton"] {
    background: rgba(255,255,255,0.92) !important;
    border: 1px solid rgba(0,148,130,0.30) !important;
    color: rgba(12,40,70,0.88) !important;
    border-radius: 7px !important;
}
[data-testid="stDeployButton"]:hover {
    background: rgba(0,148,130,0.10) !important;
    border-color: rgba(0,148,130,0.55) !important;
    color: rgba(0,90,80,0.95) !important;
}
'''}

/* Push main content below fixed header */
[data-testid="stAppViewContainer"] > .main {{
    padding-top: 54px !important;
    transition: margin-left .25s ease;
}}

/* ── SIDEBAR — natural Streamlit collapse/expand behaviour ── */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg,
        {_sidebar_bg} 0%,
        {'rgba(2,8,16,0.98)' if DK else 'rgba(230,238,245,0.98)'} 100%) !important;
    border-right: 1px solid {_card_border} !important;
    transition: margin-left 0.3s ease;
}}
/* collapse arrow — let Streamlit place it naturally, just style it */
button[data-testid="collapsedControl"] {{
    background: rgba(var(--teal-rgb),0.08) !important;
    border: 1px solid rgba(var(--teal-rgb),0.22) !important;
    border-radius: 0 8px 8px 0 !important;
    color: var(--teal) !important;
}}
section[data-testid="stSidebar"] * {{ color: {_sb_text} !important; }}
section[data-testid="stSidebar"] .stExpander {{
    border: 1px solid {_card_border} !important;
    border-radius: 12px !important;
    background: {'rgba(255,255,255,0.04)' if DK else 'rgba(255,255,255,0.80)'} !important;
    margin-bottom: 8px;
    transition: border-color var(--anim-fast) ease;
}}
section[data-testid="stSidebar"] .stExpander:hover {{
    border-color: rgba(var(--teal-rgb), 0.28) !important;
}}

/* ── LIGHT MODE: force sidebar internals to stay light ── */
{'/* dark — no overrides needed */' if DK else '''
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background: rgba(220,235,245,0.90) !important;
    border: 1.5px dashed rgba(0,148,130,0.45) !important;
    border-radius: 10px !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] span,
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] div,
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] small {
    color: rgba(12,40,70,0.80) !important;
    background: transparent !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button,
section[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
    background: rgba(255,255,255,0.90) !important;
    border: 1px solid rgba(0,148,130,0.40) !important;
    color: rgba(12,40,70,0.85) !important;
    border-radius: 8px !important;
}
section[data-testid="stSidebar"] details > summary,
section[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    background: rgba(255,255,255,0.75) !important;
    border: 1px solid rgba(0,148,130,0.20) !important;
    color: rgba(12,40,70,0.85) !important;
}
section[data-testid="stSidebar"] .stButton > button {
    background: rgba(220,240,250,0.90) !important;
    border: 1px solid rgba(0,148,130,0.30) !important;
    color: rgba(12,40,70,0.88) !important;
    border-radius: 10px !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(0,148,130,0.12) !important;
    border-color: rgba(0,148,130,0.55) !important;
    color: rgba(0,90,80,0.95) !important;
}
'''}

/* ── TOPBAR TITLE ── */
.topbar-title {{
    font-family: var(--font-head);
    font-size: 24px; font-weight: 700;
    background: linear-gradient(90deg, {_teal}, {'#60b4ff' if DK else '#00a896'});
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-align: center; padding: 8px 0 4px 0; letter-spacing: 1px;
}}

/* ── NAV BUTTONS ── */
div[data-testid="stHorizontalBlock"] div[data-testid="column"] button {{
    background: {_btn_bg} !important;
    border: 1px solid {_btn_bdr} !important;
    border-radius: 11px !important;
    color: {_btn_color} !important;
    font-family: var(--font-body) !important;
    font-size: 13px !important; font-weight: 500 !important;
    width: 100% !important; padding: 10px 4px !important;
    letter-spacing: 0.35px; backdrop-filter: blur(8px);
    transition: background var(--anim-fast) ease,
                border-color var(--anim-fast) ease,
                color var(--anim-fast) ease,
                transform var(--anim-fast) ease,
                box-shadow var(--anim-fast) ease !important;
}}
div[data-testid="stHorizontalBlock"] div[data-testid="column"] button:hover {{
    background: rgba(var(--teal-rgb), 0.14) !important;
    border-color: var(--teal) !important;
    color: var(--teal) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 5px 18px rgba(var(--teal-rgb), 0.18) !important;
}}
div[data-testid="stHorizontalBlock"] div[data-testid="column"] button:active {{
    transform: translateY(0px) !important; box-shadow: none !important;
}}

/* ── VARIABLE TYPE BADGES ── */
.type-badge {{
    display: inline-block; font-size: 11px; font-weight: 700;
    padding: 3px 12px; border-radius: 20px; margin-bottom: 4px;
    letter-spacing: 0.8px; vertical-align: middle; text-transform: uppercase;
}}
.badge-temperature {{ background:rgba(255,107,107,0.14); color:#ff8c8c; border:1px solid rgba(255,107,107,0.28); }}
.badge-rainfall    {{ background:rgba(96,180,255,0.14);  color:#7ac6ff; border:1px solid rgba(96,180,255,0.28);  }}
.badge-wind        {{ background:rgba(var(--teal-rgb),0.10); color:var(--teal); border:1px solid rgba(var(--teal-rgb),0.25); }}
.badge-humidity    {{ background:rgba(200,155,255,0.14); color:#d4aaff; border:1px solid rgba(200,155,255,0.28); }}
.badge-general     {{ background:rgba(180,180,180,0.10); color:#aaa;    border:1px solid rgba(180,180,180,0.18); }}

/* ── VAR HEADER ── */
.var-header {{
    font-family: var(--font-head);
    font-size: 20px; font-weight: 700; color: var(--teal);
    margin: 28px 0 12px 0; padding: 12px 20px;
    border-left: 4px solid var(--teal);
    background: linear-gradient(90deg, {_section_bg} 0%, transparent 100%);
    border-radius: 0 12px 12px 0;
    animation: fadeInLeft 0.45s ease both;
}}

/* ── SECTION DIVIDER ── */
.section-divider {{
    font-family: var(--font-body);
    font-size: 13px; font-weight: 600; color: {_text_muted};
    margin: 30px 0 10px 0; padding: 7px 16px;
    border-left: 3px solid {_section_bdr};
    background: {_section_bg};
    border-radius: 0 8px 8px 0;
    letter-spacing: 0.9px; text-transform: uppercase;
    animation: fadeInUp 0.35s ease both;
}}

/* ── CHART CARDS ── */
.card {{
    background: var(--card-bg);
    padding: 22px; border-radius: 18px;
    border: 1px solid var(--card-border);
    box-shadow: 0 6px 28px rgba(0,0,0,{'0.30' if DK else '0.08'}),
                inset 0 1px 0 rgba(255,255,255,{'0.05' if DK else '0.55'});
    backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
    transition: box-shadow var(--anim-med) ease,
                border-color var(--anim-med) ease,
                transform var(--anim-fast) ease;
    animation: fadeInUp 0.45s ease both;
    color: var(--text-main);
}}
.card:hover {{
    box-shadow: 0 12px 44px rgba(0,0,0,{'0.42' if DK else '0.12'}),
                0 0 0 1px rgba(var(--teal-rgb),0.16);
    border-color: rgba(var(--teal-rgb), 0.26);
    transform: translateY(-2px);
}}
.card h2, .card h3, .card p, .card span, .card label, .card div {{
    color: var(--text-main) !important;
}}

/* ── METRIC CARDS ── */
.metric-card {{
    background: {_metric_bg};
    border: 1px solid {_metric_bdr};
    border-radius: 13px; padding: 15px 18px; margin: 7px 0;
    backdrop-filter: blur(10px);
    animation: fadeInUp 0.5s ease both;
    transition: transform var(--anim-fast) ease, box-shadow var(--anim-fast) ease;
}}
.metric-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 8px 22px rgba(var(--teal-rgb),0.12);
}}
.metric-label {{
    font-size: 10px; font-weight: 700; color: {_metric_lbl};
    letter-spacing: 1px; text-transform: uppercase; margin-bottom: 4px;
}}
.metric-value {{
    font-family: var(--font-head);
    font-size: 22px; font-weight: 700;
    color: {_metric_val}; line-height: 1.2;
}}
.metric-sub {{ font-size: 11px; color: {_metric_sub}; margin-top: 3px; }}
.metric-hot   {{ border-left: 4px solid #e05050 !important; }}
.metric-cold  {{ border-left: 4px solid #5090d0 !important; }}
.metric-rain  {{ border-left: 4px solid #5090d0 !important; }}
.metric-wind  {{ border-left: 4px solid var(--teal) !important; }}
.metric-humid {{ border-left: 4px solid #9988cc !important; }}

/* ── WIDGET POLISH ── */
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stTextInput"]  > div > div > input {{
    background: {'rgba(4,16,30,0.74)' if DK else 'rgba(222,242,255,0.88)'} !important;
    border: 1px solid {_card_border} !important;
    border-radius: 10px !important;
    color: {_text_main} !important;
    font-family: var(--font-body) !important;
    transition: border-color var(--anim-fast) ease !important;
}}
div[data-testid="stNumberInput"] > div > div > input {{
    background: {'rgba(4,16,30,0.74)' if DK else 'rgba(222,242,255,0.88)'} !important;
    border: 1px solid {_card_border} !important;
    border-radius: 10px !important; color: {_text_main} !important;
}}
div[data-testid="stAlert"] {{ border-radius: 12px !important; }}

/* ── LANDING PAGE ── */
.landing-wrapper {{
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    min-height: 72vh; text-align: center; padding-top: 40px;
    animation: fadeInUp 0.9s ease both;
}}
.landing-logo {{
    font-size: 72px; margin-bottom: 10px;
    filter: drop-shadow(0 0 32px rgba(var(--teal-rgb),0.40));
    animation: floatLogo 5s ease-in-out infinite;
}}
@keyframes floatLogo {{
    0%,100% {{ transform: translateY(0);   }}
    50%      {{ transform: translateY(-10px); }}
}}
.landing-title {{
    font-family: var(--font-head);
    font-size: 62px; font-weight: 800;
    background: linear-gradient(130deg, {_teal} 0%, {'#60b4ff' if DK else '#006db3'} 60%, {'#c89bff' if DK else '#00b4a8'} 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: 2px; margin-bottom: 16px; line-height: 1.1;
}}
.landing-sub {{
    font-size: 17px; color: {_text_muted};
    max-width: 500px; line-height: 1.75; margin-bottom: 8px;
}}
.landing-hint {{
    font-size: 13px; color: {'rgba(255,255,255,0.28)' if DK else 'rgba(12,40,70,0.38)'};
    margin-top: 28px; letter-spacing: 0.3px;
    animation: pulse 2.5s ease-in-out infinite;
}}

/* ── BREADCRUMB ── */
.breadcrumb {{
    font-size: 12px; color: {_crumb_color};
    padding: 2px 0 14px 2px; letter-spacing: 0.3px;
    animation: fadeInUp 0.3s ease both;
}}
.breadcrumb-sep    {{ color: {'rgba(255,255,255,0.18)' if DK else 'rgba(12,40,70,0.22)'}; margin: 0 6px; }}
.breadcrumb-active {{ color: var(--teal); font-weight: 600; }}

/* ── GLASS OVERLAY ── */
.glass-overlay {{
    background: {_glass_bg};
    border: 1px solid {_glass_bdr};
    border-radius: 22px; padding: 70px 42px 54px 42px; margin-top: 16px;
    backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
    box-shadow: 0 12px 48px rgba(0,0,0,{'0.54' if DK else '0.10'});
    min-height: 420px; text-align: center;
    animation: fadeInUp 0.6s ease both;
}}
.glass-icon  {{ font-size: 60px; margin-bottom: 20px;
                filter: drop-shadow(0 0 18px rgba(var(--teal-rgb),0.30)); }}
.glass-title {{ font-family: var(--font-head); font-size: 28px; font-weight: 700;
                color: var(--teal); margin-bottom: 12px; }}
.glass-sub   {{ font-size: 15px; color: {_text_muted};
                max-width: 500px; margin: 0 auto 36px auto; line-height: 1.75; }}

/* ── STATUS PILL (Code B) ── */
.status-pill {{
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 12px; border-radius: 99px;
    font-family: var(--font-body); font-size: 0.70rem; font-weight: 500; letter-spacing: 0.06em;
}}
.status-ok {{
    background: rgba(var(--teal-rgb),0.10);
    border: 1px solid rgba(var(--teal-rgb),0.30);
    color: var(--teal);
}}
.status-ok::before {{ content:'●'; font-size:0.5rem; animation:pulse 2s ease-in-out infinite; }}

/* ── CARD HEADER BLOCK (Code B) ── */
.card-header-block {{
    padding-bottom: 0.7rem; margin-bottom: 0.5rem;
    border-bottom: 1px solid {_card_border};
}}
.card-header-label {{
    font-family: var(--font-head); font-size: 0.70rem; font-weight: 700;
    letter-spacing: 0.16em; text-transform: uppercase;
    color: var(--teal); opacity: 0.80; margin-bottom: 0.15rem;
}}
.card-header-title {{
    font-family: var(--font-head); font-size: 1.05rem; font-weight: 700;
    color: {_text_main}; margin: 0; line-height: 1.25;
}}

/* ── STORY MODE PROGRESS BAR (Code B) ── */
.story-progress-bar {{ display:flex; gap:8px; align-items:center; margin-bottom:1rem; }}

/* ── DIVIDER ── */
hr {{ border: none; height: 1px;
      background: linear-gradient(90deg, transparent, {_hr_color}, {_hr_color}, transparent);
      margin: 14px 0; }}

/* ── KEYFRAMES ── */
@keyframes pulse    {{ 0%,100% {{ opacity:.25; }} 50% {{ opacity:.80; }} }}
@keyframes fadeInUp {{ from {{ opacity:0; transform:translateY(18px); }} to {{ opacity:1; transform:translateY(0); }} }}
@keyframes fadeInLeft {{ from {{ opacity:0; transform:translateX(-16px); }} to {{ opacity:1; transform:translateX(0); }} }}

/* ── HIDE Streamlit footer/menu ── */
#MainMenu {{ visibility: hidden; }}
footer     {{ visibility: hidden; }}

/* ── LIGHT MODE TEXT FIXES ── */
.card, .card * {{ color: var(--text-main) !important; }}
div[data-testid="stSelectbox"] * {{ color: var(--text-main) !important; }}
div[data-testid="stTextInput"] input {{ color: var(--text-main) !important; }}

/* ── EXPORT DATAFRAME ── */
[data-testid="stDataFrame"] {{ border-radius: 12px !important; overflow: hidden !important; }}

</style>
""", unsafe_allow_html=True)

# --- visual polish pass (theme.py). Safe to delete these five lines. ---
try:
    from theme import inject_theme
    inject_theme(dark=DK)
except Exception:
    pass

# =================================================================
# VARIABLE CLASSIFIER
# =================================================================
TEMPERATURE_KEYS = ("temp","tas","t2m","tmax","tmin","tasmax","tasmin","air_temp","t_ref","2m_temperature")
RAINFALL_KEYS    = ("pr","precip","rain","prc","prcp","tp","precipitation")
WIND_KEYS        = ("uas","vas","u10","v10","wind","wnd","sfcwind","u_wind","v_wind","ws")
HUMIDITY_KEYS    = ("hurs","huss","rh","humid","q2m","specific_humidity","relative_humidity")
SNOW_KEYS        = ("snw","snd","snowfall","snow_depth","snc","snowcover","snow")

def classify_variable(name):
    n = name.lower().strip()
    if any(k in n for k in TEMPERATURE_KEYS): return "temperature"
    if any(k in n for k in RAINFALL_KEYS):    return "rainfall"
    if any(k in n for k in WIND_KEYS):        return "wind"
    if any(k in n for k in HUMIDITY_KEYS):    return "humidity"
    if any(k in n for k in SNOW_KEYS):        return "snow"
    return "general"

CATEGORY_META = {
    "temperature": ("🌡️","Temperature","badge-temperature"),
    "rainfall":    ("🌧️","Rainfall",   "badge-rainfall"),
    "wind":        ("💨","Wind",        "badge-wind"),
    "humidity":    ("💧","Humidity",    "badge-humidity"),
    "snow":        ("❄️","Snow",        "badge-general"),
    "general":     ("📊","General",     "badge-general"),
}

def auto_find(ds_vars, keys):
    for v in ds_vars:
        if any(k in v.lower() for k in keys): return v
    return None

# =================================================================
# HELPERS
# =================================================================
def _spatial_mean(xr_da):
    if "lat" in xr_da.dims and "lon" in xr_da.dims:
        return xr_da.mean(dim=["lat","lon"])
    return xr_da

def section_label(text):
    st.markdown(f'<div class="section-divider">{text}</div>', unsafe_allow_html=True)

def render_breadcrumb(current_page):
    crumbs = ["Home","Explore","Compare","Story Mode","Export"]
    sep = '<span class="breadcrumb-sep">›</span>'
    idx = crumbs.index(current_page) if current_page in crumbs else 1
    parts = [f'<span style="color:{_crumb_color};">🌍 PyClimaExplorer</span>']
    for c in crumbs[:idx+1]:
        parts.append(
            f'<span class="breadcrumb-active">● {c}</span>'
            if c == current_page
            else f'<span style="color:{_crumb_color};">{c}</span>'
        )
    st.markdown(f'<div class="breadcrumb">{sep.join(parts)}</div>', unsafe_allow_html=True)

def show_glass_placeholder(icon, title, subtitle):
    st.markdown(f"""
    <div class="glass-overlay">
        <div class="glass-icon">{icon}</div>
        <div class="glass-title">{title}</div>
        <div class="glass-sub">{subtitle}</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    c = st.columns([2.5,1,2.5])
    with c[1]:
        if st.button("← Back to Explore", key=f"back_{title}"):
            st.session_state.page = "Explore"; st.rerun()

# Code B helper
def card_header(label, title):
    st.markdown(
        f'<div class="card-header-block">'
        f'<div class="card-header-label">{label}</div>'
        f'<div class="card-header-title">{title}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

# =================================================================
# CHART RENDERERS — Code A layout (.card divs) + Code B logic/polish
# =================================================================

# =================================================================
# SHARED MAP BUILDER — used by Explore, Compare and Story Mode, so the
# three pages can never drift apart again.
# =================================================================
_MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


@st.cache_data(show_spinner=False, max_entries=48)
def _india_mask_grid(x, y, sig):
    """Inside-India mask for a lon (x) / lat (y) grid. Cached: point-in-polygon
    against a detailed outline is the costly step, and grids repeat."""
    LON, LAT = np.meshgrid(np.asarray(x, float), np.asarray(y, float))
    return _mask_inside(_india_geom(), LAT, LON)


@st.cache_resource(show_spinner=False)
def _outline_sets(sig, shift360):
    """Pen-lifted line sets for 2-D maps: (rest of world, states, India)."""
    def rings(geoms):
        xs, ys = [], []
        for g in geoms:
            if g is None:
                continue
            polys = ([g] if g.geom_type == "Polygon" else
                     list(g.geoms) if g.geom_type == "MultiPolygon" else [])
            for poly in polys:
                x, y = poly.exterior.coords.xy
                x = np.asarray(x, float); y = np.asarray(y, float)
                if shift360:
                    x = np.where(x < 0, x + 360.0, x)
                xo = x.astype(object); yo = y.astype(object)
                jumps = np.where(np.abs(np.diff(x)) > 180)[0]
                if jumps.size:                      # don't streak across the seam
                    xo = np.insert(xo, jumps + 1, None)
                    yo = np.insert(yo, jumps + 1, None)
                xs.extend(list(xo) + [None]); ys.extend(list(yo) + [None])
        return xs, ys

    rest = states = india = ([], [])
    try:
        if WORLD_GDF is not None:
            m = _find_india(WORLD_GDF)
            rest = rings(WORLD_GDF.loc[~m, "geometry"].tolist() if m is not None
                         else WORLD_GDF["geometry"].tolist())
    except Exception:
        pass
    try:
        if INDIA_STATES_GDF is not None:
            states = rings(INDIA_STATES_GDF.geometry.simplify(0.01).tolist())
    except Exception:
        pass
    try:
        ig = _india_geom()
        if ig is not None:
            india = rings([ig])
    except Exception:
        pass
    return rest, states, india


def display_values(raw, variable, units_raw):
    """2-D array in display units (Kelvin -> degC, kg m-2 s-1 -> mm/day)."""
    vals = np.asarray(raw, dtype=float)
    while vals.ndim > 2:
        vals = vals[0]
    units = str(units_raw or "")
    try:
        from ai_insight import _unit_transform
        off, scale, units, _n = _unit_transform(
            float(np.nanmean(vals)), classify_variable(variable), units)
        vals = (vals + off) * scale
    except Exception:
        pass
    return vals, units


def _cell_km(lat):
    lat = np.asarray(lat, float)
    return float(np.median(np.abs(np.diff(lat)))) * 111 if lat.size > 1 else 0.0


def prepare_map(vals, lat, lon, clip=True):
    """
    Whole-file map: refine so the coast can be traced, then clip to India.
    Coarse sources are still refined, and the caption says how coarse.
    """
    vals = np.asarray(vals, float)
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    ny, nx = vals.shape
    base = dict(ny=ny, nx=nx, lat=lat, lon=lon, focus=False,
                coarse_source=(ny < 6 or nx < 6))
    if ny < 2 or nx < 2:                  # nothing to interpolate between
        return dict(base, z=vals, x=lon, y=lat, coarse=True, clipped=False)

    biggest = max(ny, nx)
    target = 300 if biggest <= 100 else min(300, 2 * biggest)
    z, y, x = _upsample(vals, lat, lon, target=target)
    clipped = False
    if clip:
        try:
            inside = _india_mask_grid(x, y, _BOUNDARY_SIG)
            if inside is not None and inside.any() and not inside.all():
                z = np.where(inside, z, np.nan)
                clipped = True
        except Exception:
            pass
    return dict(base, z=z, x=x, y=y, coarse=False, clipped=clipped)


@st.cache_data(show_spinner=False, max_entries=48)
def _region_mask_grid(x, y, name, box, sig):
    """Inside-the-selected-region mask on a refined grid."""
    LON, LAT = np.meshgrid(np.asarray(x, float), np.asarray(y, float))
    geom = _state_geom(sig, name) if name else None
    if geom is not None:
        m = _mask_inside(geom, LAT, LON)
        if m is not None and m.any():
            return m
    la0, la1, lo0, lo1 = box
    lonw = np.where(LON > 180.0, LON - 360.0, LON)
    return (LAT >= la0) & (LAT <= la1) & (lonw >= lo0) & (lonw <= lo1)


def prepare_focus(vals, lat, lon, region):
    """
    A selected state drawn in context: crop the FULL field to a window around
    the state (plus a cell of margin so there is something to interpolate
    from), refine that window finely, and keep the surroundings for a faded
    backdrop. This is what stops a small state becoming four flat blocks.
    """
    vals = np.asarray(vals, float)
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    la0, la1, lo0, lo1 = region["box"]
    if np.nanmax(lon) > 180.0:
        lo0, lo1 = lo0 % 360.0, lo1 % 360.0
    span = max(la1 - la0, lo1 - lo0)
    pad = max(0.6, 0.25 * span)
    w = (la0 - pad, la1 + pad, lo0 - pad, lo1 + pad)

    rla = float(np.median(np.abs(np.diff(lat)))) if lat.size > 1 else 1.0
    rlo = float(np.median(np.abs(np.diff(lon)))) if lon.size > 1 else 1.0
    iy = np.where((lat >= w[0] - rla) & (lat <= w[1] + rla))[0]
    ix = np.where((lon >= w[2] - rlo) & (lon <= w[3] + rlo))[0]
    if iy.size < 2 or ix.size < 2:
        return prepare_map(vals, lat, lon)
    vc = vals[np.ix_(iy, ix)]
    lat_c, lon_c = lat[iy], lon[ix]

    z, y, x = _upsample(vc, lat_c, lon_c, target=300)
    try:
        india = _india_mask_grid(x, y, _BOUNDARY_SIG)
        if india is not None and india.any():
            z = np.where(india, z, np.nan)
    except Exception:
        pass
    box = tuple(float(b) for b in region["box"])
    inside = _region_mask_grid(x, y, region.get("name") if region.get("geom") is not None else "",
                               box, _BOUNDARY_SIG)
    z_focus = np.where(inside, z, np.nan)
    if not np.isfinite(z_focus).any():       # region fell between cells
        z_focus = z

    src_ny = int(((lat >= la0) & (lat <= la1)).sum())
    src_nx = int(((lon >= lo0) & (lon <= lo1)).sum())
    return dict(z=z_focus, z_ctx=z, x=x, y=y, lat=lat, lon=lon,
                ny=max(src_ny, 1), nx=max(src_nx, 1), src_cells=src_ny * src_nx,
                coarse=False, clipped=True, focus=True, region=region,
                coarse_source=(src_ny < 6 or src_nx < 6),
                window=(max(w[0], float(y.min())), min(w[1], float(y.max())),
                        max(w[2], float(x.min())), min(w[3], float(x.max()))))


def map_source(ds_sub):
    """What to draw from: the unfiltered file when a region is selected."""
    if ACTIVE_REGION is not None and DS_FULL is not None:
        return DS_FULL, ACTIVE_REGION
    return ds_sub, None


def prep_for(vals, lat, lon, region):
    return prepare_map(vals, lat, lon) if region is None else prepare_focus(vals, lat, lon, region)


def map_range(*zs, symmetric=None):
    """One colour range across one or more grids (shared scales for Compare)."""
    v = np.concatenate([np.asarray(z, float).ravel() for z in zs])
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    lo, hi = float(v.min()), float(v.max())
    sym = (lo < 0 < hi) if symmetric is None else symmetric
    if sym:
        m = max(abs(lo), abs(hi)) or 1.0
        return -m, m
    if lo == hi:
        return lo - 0.5, hi + 0.5
    return lo, hi


def _geom_xy(geom, shift360):
    xs, ys = [], []
    polys = ([geom] if geom.geom_type == "Polygon" else
             list(geom.geoms) if geom.geom_type == "MultiPolygon" else [])
    for p in polys:
        x, y = p.exterior.coords.xy
        x = np.asarray(x, float)
        if shift360:
            x = np.where(x < 0, x + 360.0, x)
        xs.extend(list(x) + [None]); ys.extend(list(y) + [None])
    return xs, ys


def map_figure(prep, variable, units, palette, vmin, vmax, height=440):
    ink = "rgba(255,255,255,0.65)" if DK else "rgba(12,40,70,0.70)"
    cb = dict(title=dict(text=f"{variable} ({units})" if units else variable,
                         font=dict(color=ink, size=12)),
              tickfont=dict(color=ink, size=10), thickness=13, len=0.85)
    hov = (f"<b>Lat:</b> %{{y:.2f}}\u00b0<br><b>Lon:</b> %{{x:.2f}}\u00b0"
           f"<br><b>{variable}:</b> %{{z:.2f}} {units}<extra></extra>")
    lon = prep["lon"]
    shift360 = float(np.nanmax(lon)) > 180.0

    fig = go.Figure()
    if prep["coarse"]:
        fig.add_trace(go.Heatmap(z=prep["z"], x=prep["x"], y=prep["y"],
                                 colorscale=palette, zmin=vmin, zmax=vmax,
                                 zsmooth=False, xgap=1, ygap=1,
                                 colorbar=cb, hovertemplate=hov))
    else:
        if prep.get("focus"):
            # surroundings, faded, so the state sits in its real context
            fig.add_trace(go.Contour(z=prep["z_ctx"], x=prep["x"], y=prep["y"],
                                     colorscale=palette, zmin=vmin, zmax=vmax,
                                     contours=dict(coloring="heatmap", showlines=False),
                                     line=dict(width=0), connectgaps=False,
                                     showscale=False, opacity=0.28,
                                     hoverinfo="skip"))
        fig.add_trace(go.Contour(z=prep["z"], x=prep["x"], y=prep["y"],
                                 colorscale=palette, zmin=vmin, zmax=vmax,
                                 contours=dict(coloring="heatmap", showlines=False),
                                 line=dict(width=0), connectgaps=False,
                                 colorbar=cb, hovertemplate=hov))

    rest, states, india = _outline_sets(_BOUNDARY_SIG, shift360)
    for (xs, ys), colour, width in (
            (rest,   "rgba(255,255,255,0.30)" if DK else "rgba(12,40,70,0.28)", 0.8),
            (states, "rgba(255,255,255,0.40)" if DK else "rgba(12,40,70,0.38)", 0.7),
            (india,  _teal, 2.0)):
        if xs:
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="skip",
                                     showlegend=False,
                                     line=dict(color=colour, width=width)))

    if prep.get("focus"):
        reg = prep["region"]
        try:
            if reg.get("geom") is not None:
                rx, ry = _geom_xy(reg["geom"], shift360)
            else:
                a0, a1, b0, b1 = reg["box"]
                rx, ry = [b0, b1, b1, b0, b0], [a0, a0, a1, a1, a0]
            fig.add_trace(go.Scatter(
                x=rx, y=ry, mode="lines", hoverinfo="skip", showlegend=False,
                line=dict(color="rgba(255,255,255,0.95)" if DK else "rgba(12,40,70,0.9)",
                          width=2.4)))
        except Exception:
            pass
        la0, la1, lo0, lo1 = prep["window"]
    else:
        lat = prep["lat"]
        la0, la1 = float(np.nanmin(lat)), float(np.nanmax(lat))
        lo0, lo1 = float(np.nanmin(lon)), float(np.nanmax(lon))
        if prep["coarse"]:
            dla = float(np.median(np.abs(np.diff(lat)))) / 2 if lat.size > 1 else 0.5
            dlo = float(np.median(np.abs(np.diff(lon)))) / 2 if lon.size > 1 else 0.5
            la0, la1, lo0, lo1 = la0 - dla, la1 + dla, lo0 - dlo, lo1 + dlo

    ratio = 1.0 / max(np.cos(np.deg2rad((la0 + la1) / 2)), 0.2)
    fig.update_layout(
        height=height, showlegend=False,
        xaxis=dict(title="Longitude (\u00b0E)", range=[lo0, lo1],
                   constrain="domain", showgrid=False, zeroline=False),
        yaxis=dict(title="Latitude (\u00b0N)", range=[la0, la1],
                   scaleanchor="x", scaleratio=ratio,
                   constrain="domain", showgrid=False, zeroline=False),
    )
    return fig


def map_caption(prep):
    km = _cell_km(prep["lat"])
    size = f"{prep['ny']}\u00d7{prep['nx']}"
    cell = f" of about {km:.0f} km" if km else ""
    if prep["coarse"]:
        st.caption(f"Only {size} grid cells{cell} \u2014 too few to smooth, so each "
                   "block is one real cell.")
    elif prep.get("focus"):
        name = prep["region"]["name"]
        if prep["coarse_source"]:
            n = prep.get("src_cells", 0)
            lead = ("No grid cell centre falls inside " + name if n == 0 else
                    f"Only {n} grid cell{'s' if n != 1 else ''}{cell} "
                    f"fall{'s' if n == 1 else ''} inside {name}")
            st.caption(f"{lead}, so the map is smoothed from the cells around it. "
                       "It shows the broad regional pattern, not local detail \u2014 "
                       "the 0.25\u00b0 IMD files resolve states properly.")
        else:
            st.caption(f"{name} at full colour, with its surroundings faded for "
                       "context.")
    elif prep["coarse_source"]:
        st.caption(f"Smoothed from only {size} cells{cell}. The colours show the "
                   "broad pattern, not local detail.")
    elif prep["clipped"]:
        src = ("geoBoundaries (CC BY 4.0)" if INDIA_STATES_GDF is not None
               else "Natural Earth")
        st.caption(f"Clipped to India's boundary. Borders: {src}.")


def time_labels(tvals):
    """Readable labels for a time axis: 'Jan 1948', or '03 Jan 2020' if daily."""
    arr = np.asarray(tvals)
    try:
        if np.issubdtype(arr.dtype, np.datetime64):
            ts = pd.to_datetime(arr)
            lab = [t.strftime("%b %Y") for t in ts]
            if len(set(lab)) < len(lab):
                lab = [t.strftime("%d %b %Y") for t in ts]
            return lab
        if arr.dtype == object:
            lab = [f"{_MONTH_ABBR[t.month - 1]} {t.year}" for t in arr]
            if len(set(lab)) < len(lab):
                lab = [f"{t.day:02d} {_MONTH_ABBR[t.month - 1]} {t.year}" for t in arr]
            return lab
    except Exception:
        pass
    return [f"step {i}" for i in range(arr.size)]


def render_heatmap(data, ds, variable, palette, time_index, card_key):
    dims = data.dims
    with st.container(border=True):
        st.subheader("\U0001F5FA\ufe0f Spatial Climate Heatmap")
        if "lat" in dims and "lon" in dims:
            try:
                t_idx = int(np.clip(time_index, 0, data.sizes.get("time", 0) - 1))
                src_ds, reg = map_source(ds)
                da = src_ds[variable]
                raw = da.isel(time=t_idx).values if "time" in da.dims else da.values
                vals, units = display_values(raw, variable, da.attrs.get("units", ""))
                prep = prep_for(vals, src_ds["lat"].values, src_ds["lon"].values, reg)
                vmin, vmax = map_range(prep["z"])
                fig = map_figure(prep, variable, units, palette, vmin, vmax)
                st.plotly_chart(fig, use_container_width=True, key=f"heatmap_{card_key}")
                map_caption(prep)
            except Exception as e:
                st.info(f"Could not render heatmap: {e}")
        else:
            st.info("No lat/lon dimensions found for this variable.")


def render_timeseries(data, variable, card_key):
    dims = data.dims
    with st.container(border=True):
        st.subheader("\U0001F4C8 Climate Time Series")
        time_dim = "time" if "time" in dims else ("TIME" if "TIME" in dims else None)
        if time_dim:
            try:
                # The original plotted every grid cell on top of each other,
                # which turns 900 months x 9 cells into an unreadable band.
                # Average over space first so there is one line per date.
                series = (data.mean(dim=["lat", "lon"], skipna=True)
                          if ("lat" in dims and "lon" in dims) else data)
                y = np.asarray(series.values, dtype=float).ravel()
                x = np.asarray(data[time_dim].values)

                units = str(data.attrs.get("units", ""))
                try:
                    from ai_insight import _unit_transform
                    off, scale, units, _n = _unit_transform(
                        float(np.nanmean(y)), classify_variable(variable), units)
                    y = (y + off) * scale
                except Exception:
                    pass

                n = min(len(x), len(y))
                x, y = x[:n], y[:n]
                s = pd.Series(y)
                win = 12 if n >= 48 else max(3, n // 8)
                smooth = s.rolling(win, center=True, min_periods=max(2, win // 2)).mean()

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=x, y=y, mode="lines", name="monthly",
                    line=dict(width=0.9, color=f"rgba({_teal_rgb},0.30)"),
                    hovertemplate=f"<b>%{{x}}</b><br>{variable}: %{{y:.2f}} {units}<extra></extra>"))
                fig.add_trace(go.Scatter(
                    x=x, y=smooth, mode="lines", name=f"{win}-month average",
                    line=dict(width=2.6, color=_teal),
                    hovertemplate=f"<b>%{{x}}</b><br>average: %{{y:.2f}} {units}<extra></extra>"))

                # straight-line trend across the record
                try:
                    xi = np.arange(n, dtype=float)
                    m = np.isfinite(y)
                    if m.sum() >= 3:
                        k, c = np.polyfit(xi[m], np.asarray(y)[m], 1)
                        fig.add_trace(go.Scatter(
                            x=x, y=k * xi + c, mode="lines", name="trend",
                            line=dict(width=1.6, color="rgba(255,140,120,0.85)", dash="dash"),
                            hoverinfo="skip"))
                except Exception:
                    pass

                # Let the axis frame the data instead of stretching to zero.
                lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
                pad = (hi - lo) * 0.08 or 0.5
                fig.update_layout(
                    xaxis_title="Time",
                    yaxis_title=f"{variable} ({units})" if units else variable,
                    yaxis=dict(range=[lo - pad, hi + pad]),
                    height=400, hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.01,
                                x=0, font=dict(size=11)),
                )
                st.plotly_chart(fig, use_container_width=True, key=f"ts_{card_key}")
            except Exception as e:
                st.info(f"Could not render time series: {e}")
        else:
            st.info("No time dimension found.")


def render_globe(data, ds, variable, time_index, card_key):
    dims = data.dims
    with st.container(border=True):
        st.subheader("\U0001F30F 3D Globe View")
        if "lat" in dims and "lon" in dims:
            try:
                cA, cB = st.columns([2.2, 1])
                with cA:
                    focus_label = st.radio(
                        "Camera", ["\U0001F1EE\U0001F1F3 India", "\U0001F4CD Data region", "\U0001F30D Whole world"],
                        horizontal=True, key=f"globe_focus_{card_key}",
                        help="The globe is always the whole planet. This only moves the camera.")
                with cB:
                    clip = st.checkbox("Clip to India outline", value=True,
                                       key=f"globe_clip_{card_key}")
                focus = {"\U0001F1EE\U0001F1F3 India": "india",
                         "\U0001F4CD Data region": "data",
                         "\U0001F30D Whole world": "world"}[focus_label]

                lat   = ds["lat"].values;  lon = ds["lon"].values
                t_idx = int(np.clip(time_index, 0, data.sizes.get("time",0)-1))
                vals  = data.isel(time=t_idx).values if "time" in dims else data.values
                vals  = np.asarray(vals, dtype=float)
                while vals.ndim > 2:
                    vals = vals[0]

                # Same unit handling as the Insight Briefing, so the globe and
                # the text never disagree about what the numbers mean.
                units = str(data.attrs.get("units", ""))
                try:
                    from ai_insight import _unit_transform
                    off, scale, units, _note = _unit_transform(
                        float(np.nanmean(vals)), classify_variable(variable), units)
                    vals = (vals + off) * scale
                except Exception:
                    pass

                fig_globe = make_globe_figure(
                    lon=lon, lat=lat, values=vals,
                    title=f"3D Globe \u2014 {variable}", focus=focus,
                    label=variable, units=units, clip_to_india=clip)
                fig_globe.update_layout(height=580)
                st.plotly_chart(fig_globe, use_container_width=True,
                                key=f"globe_{card_key}_{focus}_{int(clip)}")
            except Exception as e:
                st.info(f"Could not render 3D globe: {e}")
        else:
            st.info("3D globe requires lat/lon dimensions.")


def render_distribution(data, variable, card_key):
    with st.container(border=True):
        st.subheader("📊 Distribution Plot")
        try:
            vals = data.values.flatten(); vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                fig = px.histogram(pd.DataFrame({"value": vals}), x="value", nbins=40)
                fig.update_traces(
                    marker_color="#60b4ff",
                    marker_line_color="rgba(255,255,255,0.06)", marker_line_width=0.8,
                    hovertemplate=f"<b>{variable}:</b> %{{x:.3f}}<br><b>Count:</b> %{{y}}<extra></extra>",
                )
                mean_v = float(np.mean(vals))
                fig.add_vline(x=mean_v, line_dash="dash", line_color="#ffa040", line_width=1.8,
                              annotation_text=f"μ={mean_v:.2f}", annotation_font_color="#ffa040",
                              annotation_font_size=11)
                fig.update_layout(height=400, xaxis_title=variable, yaxis_title="Frequency", bargap=0.04)
                st.plotly_chart(fig, use_container_width=True, key=f"dist_{card_key}")
            else:
                st.info("No numerical data available.")
        except Exception:
            st.info("Distribution plot cannot be generated.")


# ----------------------------------------------------------------
# INDEX PANELS
# ----------------------------------------------------------------

def render_temperature_indices(data, variable, card_key):
    with st.container(border=True):
        st.subheader("🌡️ Temperature Indices")
        opt = st.selectbox("Choose Temperature Metric",
            ["Baseline vs Current vs Future","Monthly Seasonal Cycle","Extreme Values"],
            key=f"temp_opt_{card_key}")
        if "time" not in data.dims:
            st.info("No time dimension found.")
            st.markdown('</div>', unsafe_allow_html=True); return
        ts = _spatial_mean(data); df = ts.to_dataframe().reset_index()

        if opt == "Baseline vs Current vs Future":
            try:
                df["year"] = pd.to_datetime(df["time"],errors="coerce").dt.year
                yearly = df.groupby("year")[variable].mean().reset_index().dropna()
                base    = yearly[yearly["year"]<=2010]
                current = yearly[(yearly["year"]>2010)&(yearly["year"]<=2040)]
                future  = yearly[yearly["year"]>2040]
                fig = go.Figure()
                for subset, name, color in [(base,"Baseline","#4fffd2"),(current,"Current","#ffa040"),(future,"Future","#ff6b6b")]:
                    if not subset.empty:
                        fig.add_trace(go.Scatter(x=subset["year"],y=subset[variable],
                            mode="lines+markers",name=name,line=dict(color=color,width=2.2),
                            marker=dict(size=5,color=color),
                            hovertemplate=f"<b>{name}</b><br>Year: %{{x}}<br>{variable}: %{{y:.3f}}<extra></extra>"))
                if not (base.empty and current.empty and future.empty):
                    fig.update_layout(height=400, xaxis_title="Year", yaxis_title=variable)
                    st.plotly_chart(fig, use_container_width=True, key=f"bcf_{card_key}")
                else:
                    st.info("No data spans the baseline/current/future split.")
            except Exception:
                st.info("Cannot generate this chart.")

        elif opt == "Monthly Seasonal Cycle":
            try:
                df["month"] = pd.to_datetime(df["time"],errors="coerce").dt.month
                monthly = df.groupby("month")[variable].mean().reset_index()
                MN = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
                monthly["month_name"] = monthly["month"].apply(lambda m: MN[int(m)-1] if pd.notna(m) else "")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=monthly["month_name"],y=monthly[variable],
                    mode="lines+markers",line=dict(color=_teal,width=2.5),
                    marker=dict(size=8,color=_teal,line=dict(color="white",width=1.2)),
                    fill="tozeroy",fillcolor=f"rgba({_teal_rgb},0.07)",
                    hovertemplate=f"<b>%{{x}}</b><br>{variable}: %{{y:.3f}}<extra></extra>"))
                fig.update_layout(height=400, xaxis_title="Month", yaxis_title=variable)
                st.plotly_chart(fig, use_container_width=True, key=f"seasonal_{card_key}")
            except Exception:
                st.info("Cannot generate this chart.")

        elif opt == "Extreme Values":
            try:
                df["date"] = pd.to_datetime(df["time"],errors="coerce")
                df = df.dropna(subset=[variable])
                if not df.empty:
                    hot = df.loc[df[variable].idxmax()]; cold = df.loc[df[variable].idxmin()]
                    hot_days = int((df[variable]>30).sum()); trop_nts = int((df[variable]>20).sum())
                    c1,c2 = st.columns(2)
                    with c1:
                        st.markdown(f"""
                        <div class="metric-card metric-hot">
                            <div class="metric-label">🌡️ Hottest Day</div>
                            <div class="metric-value">{hot[variable]:.2f}°</div>
                            <div class="metric-sub">{hot['date'].date()}</div>
                        </div>
                        <div class="metric-card" style="margin-top:8px;">
                            <div class="metric-label">☀️ Hot Days &gt;30°C</div>
                            <div class="metric-value">{hot_days}</div>
                            <div class="metric-sub">days above threshold</div>
                        </div>""", unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"""
                        <div class="metric-card metric-cold">
                            <div class="metric-label">❄️ Coldest Day</div>
                            <div class="metric-value">{cold[variable]:.2f}°</div>
                            <div class="metric-sub">{cold['date'].date()}</div>
                        </div>
                        <div class="metric-card" style="margin-top:8px;">
                            <div class="metric-label">🌙 Tropical Nights &gt;20°C</div>
                            <div class="metric-value">{trop_nts}</div>
                            <div class="metric-sub">nights above threshold</div>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.info("No valid data found.")
            except Exception:
                st.info("Cannot compute extreme values.")


def render_rainfall_indices(data, variable, card_key):
    with st.container(border=True):
        st.subheader("🌧️ Rainfall & Hydrology Indices")
        opt = st.selectbox("Choose Rainfall Metric",
            ["Annual Rainfall Total","Monthly Distribution","Heavy Rainfall Days","Drought Frequency","Snowfall Days/Amounts"],
            key=f"rain_opt_{card_key}")
        if "time" not in data.dims and opt != "Snowfall Days/Amounts":
            st.info("No time dimension found.")
            st.markdown('</div>', unsafe_allow_html=True); return
        ts = _spatial_mean(data); df = ts.to_dataframe().reset_index()

        if opt == "Annual Rainfall Total":
            try:
                df["year"] = pd.to_datetime(df["time"],errors="coerce").dt.year
                yearly = df.groupby("year")[variable].sum().reset_index()
                fig = go.Figure(go.Bar(x=yearly["year"],y=yearly[variable],marker_color="#60b4ff",
                    marker_line_color="rgba(255,255,255,0.06)",marker_line_width=0.7,
                    hovertemplate="<b>Year:</b> %{x}<br><b>Total:</b> %{y:.3f}<extra></extra>"))
                fig.update_layout(height=400, xaxis_title="Year", yaxis_title=f"Total {variable}")
                st.plotly_chart(fig, use_container_width=True, key=f"ann_{card_key}")
            except Exception:
                st.info("Cannot generate Annual Rainfall Total chart.")

        elif opt == "Monthly Distribution":
            try:
                df["month"] = pd.to_datetime(df["time"],errors="coerce").dt.month
                monthly = df.groupby("month")[variable].sum().reset_index()
                fig = go.Figure(go.Bar(x=monthly["month"],y=monthly[variable],marker_color="#60b4ff",
                    hovertemplate="<b>Month:</b> %{x}<br>Total: %{y:.3f}<extra></extra>"))
                fig.update_layout(height=400, xaxis_title="Month", yaxis_title=f"Total {variable}")
                st.plotly_chart(fig, use_container_width=True, key=f"mondist_{card_key}")
            except Exception:
                st.info("Cannot generate Monthly Distribution chart.")

        elif opt == "Heavy Rainfall Days":
            threshold = st.number_input("Threshold (mm/day)",min_value=1.0,value=20.0,key=f"thresh_{card_key}")
            try:
                heavy = int((df[variable]>threshold).sum())
                st.markdown(f"""<div class="metric-card metric-rain">
                    <div class="metric-label">🌧️ Heavy Rainfall Days</div>
                    <div class="metric-value">{heavy}</div>
                    <div class="metric-sub">Days with {variable} &gt; {threshold} mm/day</div>
                </div>""", unsafe_allow_html=True)
            except Exception:
                st.info("Cannot compute heavy rainfall days.")

        elif opt == "Drought Frequency":
            try:
                df = df.dropna(subset=[variable])
                dry = (df[variable]<1).astype(int)
                groups = (dry!=dry.shift()).cumsum()
                consec = dry.groupby(groups).sum()
                st.markdown(f"""<div class="metric-card metric-rain">
                    <div class="metric-label">🏜️ Longest Drought Streak</div>
                    <div class="metric-value">{int(consec.max())} days</div>
                    <div class="metric-sub">Consecutive days with precipitation &lt; 1 unit</div>
                </div>""", unsafe_allow_html=True)
            except Exception:
                st.info("Cannot compute drought frequency.")

        elif opt == "Snowfall Days/Amounts":
            try:
                snow_days = int((df[variable]>0).sum()); snow_amount = float(df[variable].sum())
                c1,c2 = st.columns(2)
                with c1:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">❄️ Snowfall Days</div>
                        <div class="metric-value">{snow_days}</div>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">❄️ Total Snowfall</div>
                        <div class="metric-value">{snow_amount:.2f}</div>
                    </div>""", unsafe_allow_html=True)
            except Exception:
                st.info("Cannot compute snowfall data.")


def render_wind_indices(ds, u_var, v_var, fallback_data, fallback_var, card_key):
    with st.container(border=True):
        st.subheader("💨 Atmospheric & Wind Indices")
        opt = st.selectbox("Choose Atmospheric Metric",
            ["Wind Speed Distribution","Storm Frequency/Intensity","Humidity Extremes"],
            key=f"wind_opt_{card_key}")
        has_uv = (u_var is not None and v_var is not None)

        if opt == "Wind Speed Distribution":
            if has_uv:
                try:
                    u_vals=ds[u_var].values.flatten(); v_vals=ds[v_var].values.flatten()
                    spd=np.sqrt(u_vals**2+v_vals**2); spd=spd[~np.isnan(spd)]
                    mean_u=float(np.nanmean(u_vals)); mean_v=float(np.nanmean(v_vals))
                    deg=float(np.degrees(np.arctan2(mean_u,mean_v))%360)
                    dirs=["N","NE","E","SE","S","SW","W","NW"]
                    dlbl=dirs[int((deg+22.5)/45)%8]
                    st.caption(f"ℹ️ Using wind components: **`{u_var}`** + **`{v_var}`**")
                    c1,c2,c3=st.columns(3)
                    with c1: st.markdown(f"""<div class="metric-card metric-wind">
                        <div class="metric-label">💨 Avg Speed</div>
                        <div class="metric-value">{float(np.mean(spd)):.2f}</div>
                        <div class="metric-sub">m/s</div></div>""",unsafe_allow_html=True)
                    with c2: st.markdown(f"""<div class="metric-card metric-wind">
                        <div class="metric-label">💨 Max Speed</div>
                        <div class="metric-value">{float(np.max(spd)):.2f}</div>
                        <div class="metric-sub">m/s</div></div>""",unsafe_allow_html=True)
                    with c3: st.markdown(f"""<div class="metric-card metric-wind">
                        <div class="metric-label">🧭 Direction</div>
                        <div class="metric-value">{dlbl}</div>
                        <div class="metric-sub">{deg:.1f}°</div></div>""",unsafe_allow_html=True)
                    fig=px.histogram(pd.DataFrame({"Wind Speed (m/s)":spd}),x="Wind Speed (m/s)",nbins=40)
                    fig.update_traces(marker_color=_teal,
                        marker_line_color="rgba(255,255,255,0.06)",marker_line_width=0.7,
                        hovertemplate="<b>Speed:</b> %{x:.2f} m/s<br>Count: %{y}<extra></extra>")
                    fig.update_layout(height=280,yaxis_title="Frequency")
                    st.plotly_chart(fig,use_container_width=True,key=f"wdist_{card_key}")
                except Exception:
                    st.info("Cannot compute wind speed distribution.")
            elif fallback_data is not None:
                try:
                    vals=fallback_data.values.flatten(); vals=vals[~np.isnan(vals)]
                    st.write(f"💨 **Average {fallback_var}:** {float(np.mean(vals)):.2f} m/s")
                    st.write(f"💨 **Max {fallback_var}:** {float(np.max(vals)):.2f} m/s")
                    fig=px.histogram(pd.DataFrame({fallback_var:vals}),x=fallback_var,nbins=40)
                    fig.update_traces(marker_color=_teal)
                    fig.update_layout(height=320,xaxis_title=fallback_var,yaxis_title="Frequency")
                    st.plotly_chart(fig,use_container_width=True,key=f"wdist_fb_{card_key}")
                except Exception:
                    st.info("Cannot compute wind distribution.")
            else:
                st.info("Wind data (uas/vas or u10/v10) not available in this dataset.")

        elif opt == "Storm Frequency/Intensity":
            if has_uv:
                try:
                    u_vals=ds[u_var].values.flatten(); v_vals=ds[v_var].values.flatten()
                    spd=np.sqrt(u_vals**2+v_vals**2); spd=spd[~np.isnan(spd)]
                    st.caption(f"ℹ️ Using wind components: **`{u_var}`** + **`{v_var}`**")
                    c1,c2,c3=st.columns(3)
                    with c1: st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">⛈️ Storm Days &gt;20 m/s</div>
                        <div class="metric-value">{int((spd>20).sum())}</div></div>""",unsafe_allow_html=True)
                    with c2: st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">🌪️ Severe &gt;32 m/s</div>
                        <div class="metric-value">{int((spd>32).sum())}</div></div>""",unsafe_allow_html=True)
                    with c3: st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">📊 Peak Speed</div>
                        <div class="metric-value">{float(np.max(spd)):.2f}</div>
                        <div class="metric-sub">m/s</div></div>""",unsafe_allow_html=True)
                except Exception:
                    st.info("Cannot compute storm frequency/intensity.")
            else:
                st.info("Storm analysis requires both u-component and v-component wind variables.")

        elif opt == "Humidity Extremes":
            st.info("Select a humidity variable (e.g. `hurs`) for this metric, "
                    "or switch to Humidity Extremes from the Humidity section below.")


def render_humidity_indices(data, variable, card_key):
    with st.container(border=True):
        st.subheader("💧 Humidity Extremes")
        try:
            h=data.values.flatten(); h=h[~np.isnan(h)]
            c1,c2,c3=st.columns(3)
            with c1: st.markdown(f"""<div class="metric-card metric-humid">
                <div class="metric-label">💧 Average</div>
                <div class="metric-value">{float(np.nanmean(h)):.2f}%</div></div>""",unsafe_allow_html=True)
            with c2: st.markdown(f"""<div class="metric-card metric-humid">
                <div class="metric-label">💧 Maximum</div>
                <div class="metric-value">{float(np.nanmax(h)):.2f}%</div></div>""",unsafe_allow_html=True)
            with c3: st.markdown(f"""<div class="metric-card metric-humid">
                <div class="metric-label">🌵 Minimum</div>
                <div class="metric-value">{float(np.nanmin(h)):.2f}%</div></div>""",unsafe_allow_html=True)
            fig=px.histogram(pd.DataFrame({"Relative Humidity (%)":h}),x="Relative Humidity (%)",nbins=40)
            fig.update_traces(marker_color="#c89bff",
                hovertemplate="<b>Humidity:</b> %{x:.1f}%<br>Count: %{y}<extra></extra>")
            fig.update_layout(height=300,yaxis_title="Frequency")
            st.plotly_chart(fig,use_container_width=True,key=f"hum_{card_key}")
        except Exception:
            st.info("Cannot compute humidity extremes.")

# =================================================================
# SIDEBAR  — Code A design + Code B dark/light toggle pinned bottom
# =================================================================

# =================================================================
# PLAIN-LANGUAGE DATASET CARD  (replaces the raw metadata dump)
# =================================================================
_FRIENDLY_VARS = [
    ("precip", "Rainfall"), ("prate", "Rainfall rate"), ("rain", "Rainfall"),
    ("pr", "Rainfall"), ("tmax", "Maximum temperature"),
    ("tmin", "Minimum temperature"), ("tasmax", "Maximum temperature"),
    ("tasmin", "Minimum temperature"), ("air", "Air temperature"),
    ("tas", "Air temperature"), ("t2m", "Air temperature"),
    ("temp", "Temperature"), ("rhum", "Relative humidity"),
    ("hurs", "Relative humidity"), ("humid", "Humidity"),
    ("uwnd", "Wind, east\u2013west"), ("vwnd", "Wind, north\u2013south"),
    ("u10", "Wind, east\u2013west"), ("v10", "Wind, north\u2013south"),
    ("uas", "Wind, east\u2013west"), ("vas", "Wind, north\u2013south"),
    ("wind", "Wind speed"), ("snow", "Snow cover"),
]


def _friendly_var(name):
    lv = str(name).lower()
    for key, label in _FRIENDLY_VARS:
        if lv == key or lv.startswith(key):
            return label
    return str(name)


def _dataset_facts(ds):
    facts = []

    region = str(ds.attrs.get("region", "")).strip()
    if region.lower() in ("all", "india", "all india"):
        region = "All India"
    facts.append(("\U0001F4CD", "Region", region if region else "Custom area"))

    try:
        la = np.asarray(ds["lat"].values, float); lo = np.asarray(ds["lon"].values, float)
        facts.append(("\U0001F5FA\ufe0f", "Covers",
                      f"{la.min():.1f}\u2013{la.max():.1f}\u00b0N, "
                      f"{lo.min():.1f}\u2013{lo.max():.1f}\u00b0E"))
        if la.size > 1:
            res = float(np.median(np.abs(np.diff(la))))
            facts.append(("\U0001F50E", "Detail",
                          f"{res:g}\u00b0 grid \u00b7 about {res * 111:.0f} km per cell"))
    except Exception:
        pass

    try:
        if "time" in ds.coords:
            t = np.asarray(ds["time"].values)
            if np.issubdtype(t.dtype, np.datetime64):
                idx = pd.to_datetime(t)
            else:
                idx = pd.to_datetime([f"{x.year}-{x.month:02d}-{x.day:02d}" for x in t])
            freq = ""
            if idx.size > 1:
                step = float(np.median(np.diff(idx.values).astype("timedelta64[D]").astype(float)))
                freq = ("daily" if step < 2 else "monthly" if step < 40
                        else "yearly" if step < 400 else "")
            span = f"{idx[0]:%b %Y} \u2192 {idx[-1]:%b %Y}"
            years = (idx[-1] - idx[0]).days / 365.25
            facts.append(("\U0001F4C5", "Period",
                          f"{span} \u00b7 {years:.0f} yrs, {idx.size} {freq} readings".replace("  ", " ")))
    except Exception:
        pass

    names = []
    for v in ds.data_vars:
        u = str(ds[v].attrs.get("units", "")).strip()
        pretty = {"degC": "\u00b0C", "degK": "K", "mm/day": "mm/day"}.get(u, u)
        names.append(f"{_friendly_var(v)}" + (f" ({pretty})" if pretty else ""))
    if names:
        facts.append(("\U0001F9EA", "Measures", ", ".join(dict.fromkeys(names))))

    blob = " ".join(str(ds.attrs.get(k, "")) for k in
                    ("source_files", "source", "title", "institution", "history")).lower()
    if "imd" in blob:
        source = "India Meteorological Department (IMD) gridded observations"
    elif any(k in blob for k in ("ncep", "reanalysis", "mon.mean", "air_mon")):
        source = "NCEP/NCAR Reanalysis 1 \u00b7 NOAA PSL"
    else:
        source = str(ds.attrs.get("source", "")).strip() or "Not stated in the file"
    facts.append(("\U0001F3DB\ufe0f", "Source", source))
    return facts


def render_dataset_card(ds):
    rows = "".join(
        f'<div style="display:flex;gap:10px;padding:7px 0;'
        f'border-bottom:1px solid rgba(255,255,255,0.06);">'
        f'<div style="width:20px;flex-shrink:0;">{ic}</div>'
        f'<div style="flex:1;min-width:0;">'
        f'<div style="font-size:0.66rem;letter-spacing:0.07em;text-transform:uppercase;'
        f'opacity:0.55;">{label}</div>'
        f'<div style="font-size:0.85rem;line-height:1.45;margin-top:1px;'
        f'overflow-wrap:anywhere;">{val}</div></div></div>'
        for ic, label, val in _dataset_facts(ds))
    st.markdown(rows, unsafe_allow_html=True)
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    with st.expander("Technical details"):
        st.caption("Dimensions")
        st.code(", ".join(f"{k}: {v}" for k, v in ds.sizes.items()), language=None)
        st.caption("Variables")
        st.code(", ".join(ds.data_vars), language=None)



# =================================================================
# REGION FILTER — analyse any Indian state, or any box, from one file
# =================================================================
# Bounding boxes (lat_min, lat_max, lon_min, lon_max). Rectangles around each
# state, so they include some neighbouring area; the UI says so.
INDIA_STATES = {
    "Andhra Pradesh":    (12.6, 19.9, 76.8, 84.8),
    "Arunachal Pradesh": (26.6, 29.5, 91.5, 97.4),
    "Assam":             (24.1, 28.0, 89.7, 96.1),
    "Bihar":             (24.3, 27.5, 83.3, 88.3),
    "Chhattisgarh":      (17.8, 24.1, 80.2, 84.4),
    "Delhi":             (28.4, 28.9, 76.8, 77.4),
    "Goa":               (14.9, 15.8, 73.7, 74.3),
    "Gujarat":           (20.1, 24.7, 68.1, 74.5),
    "Haryana":           (27.6, 30.9, 74.5, 77.6),
    "Himachal Pradesh":  (30.4, 33.3, 75.6, 79.0),
    "Jammu & Kashmir":   (32.3, 35.0, 73.3, 76.5),
    "Jharkhand":         (21.9, 25.3, 83.3, 87.9),
    "Karnataka":         (11.6, 18.5, 74.0, 78.6),
    "Kerala":            ( 8.2, 12.8, 74.9, 77.4),
    "Ladakh":            (32.3, 36.0, 75.3, 80.3),
    "Madhya Pradesh":    (21.1, 26.9, 74.0, 82.8),
    "Maharashtra":       (15.6, 22.0, 72.6, 80.9),
    "Manipur":           (23.8, 25.7, 93.0, 94.8),
    "Meghalaya":         (25.0, 26.1, 89.8, 92.8),
    "Mizoram":           (21.9, 24.5, 92.3, 93.4),
    "Nagaland":          (25.2, 27.0, 93.3, 95.2),
    "Odisha":            (17.8, 22.6, 81.4, 87.5),
    "Punjab":            (29.5, 32.5, 73.9, 76.9),
    "Rajasthan":         (23.0, 30.2, 69.5, 78.3),
    "Sikkim":            (27.1, 28.1, 88.0, 88.9),
    "Tamil Nadu":        ( 8.1, 13.6, 76.2, 80.3),
    "Telangana":         (15.8, 19.9, 77.2, 81.3),
    "Tripura":           (22.9, 24.5, 91.2, 92.3),
    "Uttar Pradesh":     (23.9, 30.4, 77.1, 84.6),
    "Uttarakhand":       (28.7, 31.5, 77.6, 81.0),
    "West Bengal":       (21.5, 27.2, 85.8, 89.9),
}


def _subset_box(ds, la0, la1, lo0, lo1):
    """
    Cut ds to a lat/lon box. Works for either longitude convention and
    either latitude ordering. If the box is smaller than the grid (Goa on a
    2.5-degree file), it widens to the nearest cells so there is always at
    least a 2x2 block to analyse.
    Returns (subset or None, status) with status in ok / padded / outside.
    """
    lat = np.asarray(ds["lat"].values, float)
    lon = np.asarray(ds["lon"].values, float)
    if lat.size == 0 or lon.size == 0:
        return None, "outside"
    q0, q1 = lo0, lo1
    if np.nanmax(lon) > 180.0:                       # 0..360 file
        q0, q1 = lo0 % 360.0, lo1 % 360.0
    if (la1 < np.nanmin(lat) or la0 > np.nanmax(lat)
            or q1 < np.nanmin(lon) or q0 > np.nanmax(lon)):
        return None, "outside"

    rla = float(np.median(np.abs(np.diff(lat)))) if lat.size > 1 else 1.0
    rlo = float(np.median(np.abs(np.diff(lon)))) if lon.size > 1 else 1.0
    a0, a1, b0, b1 = la0, la1, q0, q1
    mla = (lat >= a0) & (lat <= a1)
    mlo = (lon >= b0) & (lon <= b1)
    widened = False
    for _ in range(6):
        if mla.sum() >= 2 and mlo.sum() >= 2:
            break
        a0 -= rla / 2; a1 += rla / 2; b0 -= rlo / 2; b1 += rlo / 2
        mla = (lat >= a0) & (lat <= a1)
        mlo = (lon >= b0) & (lon <= b1)
        widened = True
    if mla.sum() == 0 or mlo.sum() == 0:
        return None, "outside"
    out = ds.isel(lat=np.where(mla)[0], lon=np.where(mlo)[0])
    return out, ("padded" if widened else "ok")


_STATE_ALIASES = {
    "odisha": ["orissa"], "uttarakhand": ["uttaranchal"],
    "puducherry": ["pondicherry"], "delhi": ["nctofdelhi", "nationalcapitalterritoryofdelhi"],
}


@st.cache_resource(show_spinner=False)
def _state_geom(sig, name):
    """A state's real boundary from the geoBoundaries file, or None."""
    if INDIA_STATES_GDF is None:
        return None
    import re
    from shapely.ops import unary_union

    def norm(s):
        return re.sub(r"[^a-z]", "", str(s).lower().replace("&", "and"))

    targets = [norm(name)] + _STATE_ALIASES.get(norm(name), [])
    gdf = INDIA_STATES_GDF
    text_cols = [c for c in gdf.columns if c != "geometry" and gdf[c].dtype == object]
    name_cols = [c for c in ("shapeName", "NAME_1", "ST_NM", "name", "state", "STATE")
                 if c in text_cols] or text_cols
    for exact in (True, False):
        for c in (name_cols if not exact else name_cols + text_cols):
            vals = gdf[c].astype(str).map(norm)
            for t in targets:
                if exact:
                    hit = vals == t
                else:
                    hit = vals.map(lambda v: len(v) >= 4 and (t in v or v in t))
                if hit.any():
                    try:
                        return unary_union(gdf.loc[hit, "geometry"].tolist())
                    except Exception:
                        return None
    return None


def apply_region_filter(ds):
    """Sidebar control. Returns ds cut to the chosen area, or ds unchanged."""
    global ACTIVE_REGION, DS_FULL
    DS_FULL, ACTIVE_REGION = ds, None
    with st.expander("\U0001F4CD  Region Filter", expanded=False):
        try:
            lat = np.asarray(ds["lat"].values, float)
            lon = np.asarray(ds["lon"].values, float)
        except Exception:
            st.caption("This file has no latitude/longitude grid to filter.")
            return ds

        options = ["Whole file"] + sorted(INDIA_STATES) + ["Custom area\u2026"]
        choice = st.selectbox("Focus the analysis on", options, key="region_choice")

        if choice == "Whole file":
            st.caption("Every panel is analysing the full extent of the file.")
            return ds

        geom = None
        if choice == "Custom area\u2026":
            la_lo, la_hi = float(np.nanmin(lat)), float(np.nanmax(lat))
            lo_lo, lo_hi = float(np.nanmin(lon)), float(np.nanmax(lon))
            la = st.slider("Latitude (\u00b0N)", la_lo, la_hi, (la_lo, la_hi),
                           step=0.25, key="region_lat")
            lo = st.slider("Longitude (\u00b0E)", lo_lo, lo_hi, (lo_lo, lo_hi),
                           step=0.25, key="region_lon")
            box = (la[0], la[1], lo[0], lo[1])
            name = (f"Custom area {la[0]:.1f}\u2013{la[1]:.1f}\u00b0N, "
                    f"{lo[0]:.1f}\u2013{lo[1]:.1f}\u00b0E")
        else:
            name = choice
            geom = _state_geom(_BOUNDARY_SIG, choice)
            if geom is not None:
                minx, miny, maxx, maxy = geom.bounds
                box = (miny, maxy, minx, maxx)
            else:
                box = INDIA_STATES[choice]

        sub, status = _subset_box(ds, *box)
        if sub is None:
            st.warning(f"This file doesn't cover {name}. Load an All India "
                       "file to analyse any state.")
            return ds

        ny, nx = sub.sizes.get("lat", 0), sub.sizes.get("lon", 0)
        used_shape = False
        if geom is not None:
            # Statistics from cells inside the real border, not the rectangle
            try:
                LON, LAT = np.meshgrid(np.asarray(sub["lon"].values, float),
                                       np.asarray(sub["lat"].values, float))
                inside = _mask_inside(geom, LAT, LON)
                if inside is not None and int(inside.sum()) >= 4:
                    m = xr.DataArray(inside, dims=("lat", "lon"),
                                     coords={"lat": sub["lat"], "lon": sub["lon"]})
                    sub = sub.where(m)
                    used_shape = True
                    st.caption(f"Analysing {int(inside.sum())} grid cells inside "
                               f"{name}'s boundary.")
            except Exception:
                used_shape = False

        if not used_shape:
            if status == "padded":
                st.caption(f"{name} is smaller than this file's grid cells, so the "
                           f"nearest {ny}\u00d7{nx} cells are used. The 0.25\u00b0 IMD "
                           "files resolve small states far better.")
            else:
                st.caption(f"Analysing {ny}\u00d7{nx} grid cells for {name}.")
            if choice != "Custom area\u2026":
                st.caption("Uses a rectangle around the state, so a little "
                           "neighbouring area is included.")

        ACTIVE_REGION = dict(name=name, box=box, geom=geom)
        return sub.assign_attrs(region=name,
                                parent_region=str(ds.attrs.get("region", "")))


with st.sidebar:

    with st.expander("📂  Dataset", expanded=True):
        uploaded_file = st.file_uploader(
            "Upload NetCDF (.nc)", type=["nc"],
            key=f"nc_uploader_{st.session_state.get('_file_uploader_key', 0)}"
        )

    st.session_state.dataset_loaded = (uploaded_file is not None)

    if st.session_state.dataset_loaded:
        with st.expander("🌡️  Variable Selection"):
            variable_override = st.text_input("Preferred variable name (optional)", value="")
        with st.expander("⏱️  Time Controls"):
            time_index = st.number_input("Time index (for 3D data)", min_value=0, value=0, step=1)
        with st.expander("🎨  Colour Settings"):
            palette = st.selectbox("Colour Palette", ["RdBu_r","Turbo","Viridis","Plasma","RdBu"])
    else:
        variable_override = ""; time_index = 0; palette = "RdBu_r"

    # Dataset loading — Code B's robust multi-engine approach
    ds = None
    if uploaded_file is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(uploaded_file.read()); path = tmp.name
            for kwargs in [{}, {"decode_times":False,"engine":"netcdf4"}, {"decode_times":False,"engine":"h5netcdf"}]:
                try:
                    ds = xr.open_dataset(path, **kwargs); break
                except Exception:
                    continue
            if ds is None:
                st.error("Could not open file — unsupported format or corrupted.")
        except Exception:
            st.error("Error reading dataset")
        if ds is not None:
            ds = apply_region_filter(ds)
            with st.expander("🗂️  About this data", expanded=True):
                render_dataset_card(ds)

    # Dark/Light toggle — pinned at bottom of sidebar (Code A)
    st.markdown("<br>" * 2, unsafe_allow_html=True)
    st.markdown('<hr>', unsafe_allow_html=True)
    mode_label = "☀️  Switch to Light Mode" if DK else "🌙  Switch to Dark Mode"
    if st.button(mode_label, key="mode_toggle", use_container_width=True,
                 help="Toggle between dark and light theme"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()
    st.markdown(
        f'<p style="font-size:10px; color:{"rgba(255,255,255,0.20)" if DK else "rgba(12,40,70,0.30)"}; '
        f'text-align:center; margin-top:4px; letter-spacing:0.4px;">'
        f'{"🌙 Dark Mode Active" if DK else "☀️ Light Mode Active"}</p>',
        unsafe_allow_html=True
    )

# =================================================================
# LANDING PAGE — Code A clean minimal design
# =================================================================
if not st.session_state.dataset_loaded:
    st.markdown(f"""
    <div class="landing-wrapper">
        <div class="landing-logo">🌍</div>
        <div class="hero-header" style="align-items:center;display:flex;flex-direction:column;">
            <div class="hero-eyebrow">Atmospheric · Geospatial · Temporal</div></div>
        <div class="landing-title">PyClimaExplorer</div>
        <div class="landing-sub">
            Interactive climate data visualisation<br>
            powered by NetCDF — no coding required
        </div>
        <div class="landing-hint">← Upload a NetCDF (.nc) file from the sidebar to begin</div>
    </div>
    """, unsafe_allow_html=True)

# =================================================================
# FULL DASHBOARD
# =================================================================
else:
    st.markdown('<div class="topbar-title">🌍 PyClimaExplorer</div>', unsafe_allow_html=True)

    # ── NAV — segmented control with an active state
    _NAV = [("Home", "🏠"), ("Explore", "🔭"), ("Compare", "⚖️"),
            ("Story Mode", "📖"), ("Export", "📦")]

    def _go(page):
        # Runs as a callback, before the rerun — so no st.rerun() needed.
        if page == "Home":
            st.session_state.dataset_loaded = False
            st.session_state.page = "Explore"
            st.session_state["_file_uploader_key"] = st.session_state.get("_file_uploader_key", 0) + 1
        else:
            st.session_state.page = page

    try:
        _nav_box = st.container(key="pce_nav")
    except TypeError:
        _nav_box = st.container()
    with _nav_box:
        nav_cols = st.columns(len(_NAV))
        for _col, (_label, _icon) in zip(nav_cols, _NAV):
            with _col:
                _active = (_label == st.session_state.page)
                st.button(f"{_icon}  {_label}", key=f"nav_{_label}",
                          use_container_width=True,
                          type="primary" if _active else "secondary",
                          on_click=_go, args=(_label,))

    st.markdown('<hr>', unsafe_allow_html=True)
    render_breadcrumb(st.session_state.page)

    # ──────────────────────────────────────────
    # COMPARE PAGE — Code B full implementation
    # ──────────────────────────────────────────
    if st.session_state.page == "Compare":
        if ds is None:
            show_glass_placeholder("📅","Compare","Upload a dataset on the Explore page first.")
        else:
            with st.container(border=True):
                card_header("Analysis", "Compare Two Time Slices")
                time_dim_ds = "time" if "time" in ds.dims else ("TIME" if "TIME" in ds.dims else None)
                if time_dim_ds is None:
                    st.info("Dataset has no time dimension to compare.")
                else:
                    var_list = [v for v in ds.data_vars
                                if "lat" in ds[v].dims and "lon" in ds[v].dims]
                    if not var_list:
                        st.info("No variable with lat/lon found to compare.")
                    else:
                        variable_cmp = st.selectbox("Variable to compare", var_list,
                                                    index=0, key="cmp_var")
                        data_cmp = ds[variable_cmp]
                        tdv = ("time" if "time" in data_cmp.dims else
                               "TIME" if "TIME" in data_cmp.dims else None)
                        if tdv and "lat" in data_cmp.dims and "lon" in data_cmp.dims:
                            labels = time_labels(ds[tdv].values)
                            n_t = len(labels)
                            opts = list(range(n_t))
                            col_left, col_right = st.columns(2)
                            with col_left:
                                t1 = st.select_slider("Time A", options=opts, value=0,
                                                      format_func=lambda i: labels[i],
                                                      key="cmp_t1")
                            with col_right:
                                t2 = st.select_slider("Time B", options=opts, value=n_t - 1,
                                                      format_func=lambda i: labels[i],
                                                      key="cmp_t2")
                            src_c, reg_c = map_source(ds)
                            draw_c = src_c[variable_cmp]
                            lat = src_c["lat"].values; lon = src_c["lon"].values
                            u_raw = draw_c.attrs.get("units", "")
                            va, units = display_values(draw_c.isel({tdv: t1}).values,
                                                       variable_cmp, u_raw)
                            vb, _ub = display_values(draw_c.isel({tdv: t2}).values,
                                                     variable_cmp, u_raw)
                            pa = prep_for(va, lat, lon, reg_c)
                            pb = prep_for(vb, lat, lon, reg_c)
                            # ONE scale for both, or the colours can't be compared
                            vmin, vmax = map_range(pa["z"], pb["z"])
                            for col, prep, t, tag, k in (
                                    (col_left, pa, t1, "A", "cmp_chart_a"),
                                    (col_right, pb, t2, "B", "cmp_chart_b")):
                                with col:
                                    st.markdown(
                                        f'<div class="metric-label" style="margin-bottom:4px;">'
                                        f'Slice {tag} \u2014 {labels[t]}</div>',
                                        unsafe_allow_html=True)
                                    st.plotly_chart(
                                        map_figure(prep, variable_cmp, units, palette,
                                                   vmin, vmax, height=430),
                                        use_container_width=True, key=k)
                            st.caption("Both maps share one colour scale, so a colour "
                                       "means the same value on each side.")

                            # ---- what changed between the two ------------------
                            pdiff = prep_for(vb - va, lat, lon, reg_c)
                            dmin, dmax = map_range(pdiff["z"], symmetric=True)
                            diff_pal = ("BrBG" if classify_variable(variable_cmp) == "rainfall"
                                        else "RdBu_r")
                            st.markdown(
                                f'<div class="metric-label" style="margin:16px 0 4px 0;">'
                                f'Change \u2014 {labels[t2]} minus {labels[t1]}</div>',
                                unsafe_allow_html=True)
                            st.plotly_chart(
                                map_figure(pdiff, f"change in {variable_cmp}", units,
                                           diff_pal, dmin, dmax, height=470),
                                use_container_width=True, key="cmp_chart_diff")
                            dz = np.asarray(pdiff["z"], float)
                            if np.isfinite(dz).any():
                                st.caption(
                                    f"Average change across the mapped area: "
                                    f"{np.nanmean(dz):+.2f} {units}. Largest rise "
                                    f"{np.nanmax(dz):+.2f}, largest fall {np.nanmin(dz):+.2f}. "
                                    "Compare the same month in different years to see "
                                    "long-term change rather than the seasons.")
                            map_caption(pa)
                        else:
                            st.info(f"Variable '{variable_cmp}' does not have time + "
                                    "lat/lon dimensions.")
            st.markdown("<br>", unsafe_allow_html=True)
            bc = st.columns([2.5,1,2.5])
            with bc[1]:
                if st.button("← Back to Explore", key="back_compare"):
                    st.session_state.page = "Explore"; st.rerun()

        # ──────────────────────────────────────────
    # STORY MODE 2.0 — Narrative Explorer
    # ──────────────────────────────────────────
    elif st.session_state.page == "Story Mode":
        if ds is None:
            show_glass_placeholder("📖", "Story Mode",
                                   "Upload a dataset on the Explore page first.")
        else:
            # Blue card wrapper
            with st.container(border=True):

                # Header INSIDE the card (this is the blue bar title)
                card_header("◉ Story Mode", "Climate Narrative Explorer")

                # --- choose variable for story ---
                var_list = [
                    v for v in ds.data_vars
                    if (("time" in ds[v].dims) or ("TIME" in ds[v].dims))
                    and ("lat" in ds[v].dims and "lon" in ds[v].dims)
                ]
                if not var_list:
                    st.info("No variable with time + lat + lon available for story mode.")
                else:
                    variable_story = st.selectbox(
                        "Variable for story", var_list, index=0, key="story_var"
                    )
                    data_story = ds[variable_story]
                    time_dim_story = "time" if "time" in data_story.dims else "TIME"
                    times = ds[time_dim_story].values
                    nt = data_story.sizes[time_dim_story]

                    # --- pick 4–6 key steps across the record ---
                    n_steps = min(6, max(4, nt))
                    if n_steps == nt:
                        step_indices = list(range(nt))
                    else:
                        step_indices = np.linspace(0, nt - 1, n_steps, dtype=int)

                    # ensure story_step exists and is in range
                    if "story_step" not in st.session_state:
                        st.session_state.story_step = 0
                    step = int(st.session_state.story_step)
                    step = max(0, min(step, len(step_indices) - 1))
                    t_idx = int(step_indices[step])

                    # human-readable labels
                    ts_dt = pd.to_datetime(times, errors="coerce")
                    if not pd.isna(ts_dt).all():
                        labels = [str(x) for x in ts_dt]
                    else:
                        labels = [str(t) for t in times]
                    current_label = labels[t_idx]

                    # --- simple captions for each step (heatwave/flood/etc.) ---
                    captions = [
                        "Baseline conditions – a reference climate state.",
                        "First notable shift – emerging anomalies in the field.",
                        "Stronger event – larger departures from the baseline.",
                        "Persistent change – anomalies becoming the new normal.",
                        "Extreme episode – peak intensity in this record.",
                        "Post‑event climate – residual changes after extremes.",
                    ]
                    if len(step_indices) <= len(captions):
                        step_caption = captions[step]
                    else:
                        step_caption = captions[min(step, len(captions) - 1)]

                    # --- autoplay controls ---
                    col_auto, col_speed = st.columns([1, 1])
                    with col_auto:
                        autoplay = st.checkbox(
                            "Autoplay story", value=False, key="story_autoplay"
                        )
                    with col_speed:
                        speed = st.selectbox(
                            "Speed",
                            ["Slow", "Normal", "Fast"],
                            index=1,
                            key="story_speed",
                        )
                    if speed == "Slow":
                        delay = 2.5
                    elif speed == "Fast":
                        delay = 0.8
                    else:
                        delay = 1.5

                    # --- progress bar for steps ---
                    prog_html = " ".join(
                        f'<span style="width:40px;height:4px;border-radius:3px;display:inline-block;'
                        f'background:{"var(--teal)" if i <= step else "rgba(255,255,255,0.15)"}"></span>'
                        for i in range(len(step_indices))
                    )
                    st.markdown(
                        f'<div style="display:flex;align-items:center;gap:12px;'
                        f'margin-bottom:0.6rem;margin-top:0.4rem;">'
                        f'{prog_html}'
                        f'<span style="font-size:0.72rem;color:{_text_muted};">'
                        f'Step {step+1} of {len(step_indices)} · {current_label}'
                        f'</span></div>',
                        unsafe_allow_html=True,
                    )

                    # --- get data for this step ---
                    lat = ds["lat"].values
                    lon = ds["lon"].values
                    src_s, reg_s = map_source(ds)
                    draw_s = src_s[variable_story]
                    lat_d = src_s["lat"].values; lon_d = src_s["lon"].values
                    vals_s, units_s = display_values(
                        draw_s.isel({time_dim_story: t_idx}).values,
                        variable_story, draw_s.attrs.get("units", ""))
                    prep_s = prep_for(vals_s, lat_d, lon_d, reg_s)
                    smin, smax = map_range(prep_s["z"])

                    # --- 2-panel layout: map + caption ---
                    map_col, text_col = st.columns([2.2, 1])

                    with map_col:
                        fig_story = map_figure(prep_s, variable_story, units_s,
                                               palette, smin, smax, height=440)
                        # mark the highest value actually shown on the map
                        try:
                            zs = np.asarray(prep_s["z"], float)
                            iy, ix = np.unravel_index(np.nanargmax(zs), zs.shape)
                            lat_hot = float(np.asarray(prep_s["y"])[iy])
                            lon_hot = float(np.asarray(prep_s["x"])[ix])
                            peak = float(zs[iy, ix])
                            fig_story.add_trace(go.Scatter(
                                x=[lon_hot], y=[lat_hot], mode="markers",
                                marker=dict(color="#ff6b35", size=11,
                                            line=dict(color="white", width=1.6)),
                                showlegend=False, hoverinfo="skip"))
                            fig_story.add_annotation(
                                x=lon_hot, y=lat_hot, ax=44, ay=-36,
                                text=f"Highest: {peak:.1f} {units_s}",
                                showarrow=True, arrowcolor="#ff6b35", arrowwidth=1.6,
                                font=dict(color="#ff6b35", size=12),
                                bgcolor="rgba(0,0,0,0.55)")
                        except Exception:
                            pass
                        st.plotly_chart(fig_story, use_container_width=True, key="story_heatmap")
                        map_caption(prep_s)

                    with text_col:
                        st.markdown(
                            "<div style='font-size:0.78rem; font-weight:600; "
                            "letter-spacing:0.08em; text-transform:uppercase; "
                            f"color:{_text_muted}; margin-bottom:0.3rem;'>STORY STEP</div>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"<div style='font-size:1.0rem; font-weight:700; "
                            f"color:{_teal}; margin-bottom:0.4rem;'>"
                            f"{step_caption}</div>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            "<div style='font-size:0.82rem; line-height:1.6;'>"
                            "This frame shows one moment from your record. "
                            "The colours show the value across the region at that time, "
                            "and the orange marker points to where it is highest."
                            "</div>",
                            unsafe_allow_html=True,
                        )

                    # --- optional 3D globe below ---
                    fig_story_globe = make_globe_figure(
                        lon=lon_d,
                        lat=lat_d,
                        values=vals_s,
                        title=f"3D Climate Globe — {current_label}",
                        label=variable_story, units=units_s,
                    )
                    fig_story_globe.update_layout(height=560)
                    st.plotly_chart(
                        fig_story_globe,
                        use_container_width=True,
                        key="story_globe",
                    )

                    # --- previous / next controls ---
                    st.markdown("<div style='margin-top:0.4rem;'></div>", unsafe_allow_html=True)
                    c_prev, c_center, c_next = st.columns([1, 2, 1])
                    with c_prev:
                        if st.button("← Previous", disabled=(step == 0), key="story_prev"):
                            st.session_state.story_step = max(0, step - 1)
                            st.rerun()
                    with c_center:
                        st.write("")  # spacer
                    with c_next:
                        if st.button(
                            "Next →",
                            disabled=(step == len(step_indices) - 1),
                            key="story_next",
                        ):
                            st.session_state.story_step = min(
                                len(step_indices) - 1, step + 1
                            )
                            st.rerun()

                    # --- autoplay loop ---
                    if autoplay:
                        next_step = (step + 1) % len(step_indices)
                        st.session_state.story_step = next_step
                        import time as _t
                        _t.sleep(delay)
                        st.rerun()

                # Close blue card
                st.markdown("</div>", unsafe_allow_html=True)

     # EXPORT PAGE — smarter chart exports
    # ──────────────────────────────────────────
    elif st.session_state.page == "Export":
        if ds is None:
            show_glass_placeholder("📤", "Export Data",
                                   "Load a dataset first to access export options.")
        else:
            with st.container(border=True):
                card_header("📤 Export", "Download Ready‑to‑Use Data")

                var_list = list(ds.data_vars)
                exp_var = st.selectbox("Variable to export", var_list, key="export_var")
                data_var = ds[exp_var]

                dims = data_var.dims
                has_time = "time" in dims or "TIME" in dims
                time_dim = "time" if "time" in dims else ("TIME" if "TIME" in dims else None)
                has_latlon = ("lat" in dims and "lon" in dims)

                export_type = st.selectbox(
                    "What would you like to download?",
                    [
                        "Spatial slice (map) at one time",
                        "Time series (spatial mean)",
                        "Global statistics over full record",
                    ],
                    key="export_type",
                )

                if export_type == "Spatial slice (map) at one time":
                    if not has_latlon:
                        st.info("Selected variable has no lat/lon dimensions to make a map.")
                    else:
                        if not has_time:
                            st.info("Variable has no time dimension; exporting single spatial field.")
                            t_idx = None
                        else:
                            nt = data_var.sizes[time_dim]
                            t_idx = st.slider(
                                "Choose time index for the map",
                                min_value=0,
                                max_value=nt - 1,
                                value=min(nt - 1, 0),
                                key="export_slice_t",
                            )

                        if t_idx is not None:
                            slice_da = data_var.isel({time_dim: t_idx})
                        else:
                            slice_da = data_var

                        df_map = slice_da.to_dataframe(name=exp_var).reset_index()
                        st.markdown(
                            "<div style='font-size:0.8rem;color:rgba(180,220,235,0.85);"
                            "margin-bottom:0.3rem;'>Preview of map slice (first 50 rows)</div>",
                            unsafe_allow_html=True,
                        )
                        st.dataframe(df_map.head(50), use_container_width=True)

                        csv = df_map.to_csv(index=False).encode("utf-8")
                        fname = f"{exp_var}_map_slice_t{t_idx}.csv" if t_idx is not None else f"{exp_var}_map_slice.csv"
                        st.download_button(
                            label="⬇ Download spatial slice CSV",
                            data=csv,
                            file_name=fname,
                            mime="text/csv",
                            key="btn_export_map",
                        )

                elif export_type == "Time series (spatial mean)":
                    if not has_time:
                        st.info("Selected variable has no time dimension to build a time series.")
                    else:
                        # mean over lat/lon if present
                        ts_da = data_var
                        if has_latlon:
                            ts_da = ts_da.mean(dim=[d for d in ["lat", "lon"] if d in ts_da.dims])

                        df_ts = ts_da.to_dataframe(name=exp_var).reset_index()
                        st.markdown(
                            "<div style='font-size:0.8rem;color:rgba(180,220,235,0.85);"
                            "margin-bottom:0.3rem;'>Preview of time series (first 50 rows)</div>",
                            unsafe_allow_html=True,
                        )
                        st.dataframe(df_ts.head(50), use_container_width=True)

                        csv = df_ts.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="⬇ Download time‑series CSV",
                            data=csv,
                            file_name=f"{exp_var}_timeseries.csv",
                            mime="text/csv",
                            key="btn_export_ts",
                        )

                else:  # "Global statistics over full record"
                    # flatten all points and compute summary stats
                    vals = data_var.values.flatten()
                    vals = vals[np.isfinite(vals)]
                    if vals.size == 0:
                        st.info("No finite values available to summarise.")
                    else:
                        summary = {
                            "variable": [exp_var],
                            "n_points": [int(vals.size)],
                            "mean": [float(np.mean(vals))],
                            "min": [float(np.min(vals))],
                            "max": [float(np.max(vals))],
                            "std": [float(np.std(vals))],
                        }
                        df_summary = pd.DataFrame(summary)
                        st.markdown(
                            "<div style='font-size:0.8rem;color:rgba(180,220,235,0.85);"
                            "margin-bottom:0.3rem;'>Global statistics for this variable</div>",
                            unsafe_allow_html=True,
                        )
                        st.dataframe(df_summary, use_container_width=True)

                        csv = df_summary.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="⬇ Download global stats CSV",
                            data=csv,
                            file_name=f"{exp_var}_global_stats.csv",
                            mime="text/csv",
                            key="btn_export_stats",
                        )

                st.markdown("</div>", unsafe_allow_html=True)



    # ──────────────────────────────────────────
    # EXPLORE PAGE — Code A layout + Code B data logic
    # ──────────────────────────────────────────
    else:
        if ds is None:
            st.error("Dataset could not be loaded.")
        else:
            st.markdown('<span class="status-pill status-ok">Dataset Loaded</span>', unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)

            all_vars = list(ds.data_vars)

            if variable_override and variable_override in ds.data_vars:
                variable = variable_override
            else:
                variable = st.selectbox("Select Climate Variable", all_vars, index=0)

            st.session_state["_current_variable"] = variable
            data = ds[variable]; dims = data.dims
            var_type = classify_variable(variable)
            icon, label, badge_cls = CATEGORY_META[var_type]

            u_var         = auto_find(all_vars, ("uas","u10","u_wind","uwnd"))
            v_var         = auto_find(all_vars, ("vas","v10","v_wind","vwnd"))
            hum_var       = auto_find(all_vars, HUMIDITY_KEYS)
            rain_var_auto = auto_find(all_vars, RAINFALL_KEYS)

            st.markdown(
                f'<div class="var-header">{icon} Variable: <code>{variable}</code>&nbsp;&nbsp;'
                f'<span class="type-badge {badge_cls}">{label}</span></div>',
                unsafe_allow_html=True)

            section_label("🧠 AI Insight Briefing")
            if AI_INSIGHT_OK:
                render_insight_briefing(data, ds, variable, var_type,
                                        time_index, "row0_brief")
            else:
                st.warning(f"Insight Briefing unavailable — {_AI_INSIGHT_ERR}")

            section_label("📍 Spatial Pattern & Temporal Trend")
            col1, col2 = st.columns([1,1])
            with col1: render_heatmap(data, ds, variable, palette, time_index, "row1_l")
            with col2: render_timeseries(data, variable, "row1_r")

            section_label("🌐 3D Interactive Globe")
            render_globe(data, ds, variable, time_index, "row2_globe")

            section_label("📊 Climate Indices")
            col3, col4 = st.columns([1,1])

            if var_type in ("temperature","general","snow"):
                with col3: render_temperature_indices(data, variable, "row3_l")
                with col4:
                    rain_target = rain_var_auto if rain_var_auto else variable
                    rain_data   = ds[rain_target]
                    if rain_target != variable:
                        st.markdown(
                            f'<p style="color:{_text_muted}; font-size:13px; margin:0 0 6px 4px;">'
                            f'ℹ️ Rainfall section using auto-detected variable: <code>{rain_target}</code></p>',
                            unsafe_allow_html=True)
                    render_rainfall_indices(rain_data, rain_target, "row3_r")
            elif var_type == "rainfall":
                with col3: render_rainfall_indices(data, variable, "row3_l")
                with col4: render_temperature_indices(data, variable, "row3_r")
            elif var_type == "wind":
                with col3: render_wind_indices(ds, u_var, v_var, fallback_data=data, fallback_var=variable, card_key="row3_l")
                with col4: render_temperature_indices(data, variable, "row3_r")
            elif var_type == "humidity":
                with col3: render_humidity_indices(data, variable, "row3_l")
                with col4: render_wind_indices(ds, u_var, v_var, fallback_data=None, fallback_var=None, card_key="row3_r")

            section_label("📈 Distribution & Atmospheric Overview")
            col5, col6 = st.columns([1,1])
            with col5: render_distribution(data, variable, "row4_l")
            with col6:
                fb = data if var_type == "wind" else None
                fb_name = variable if var_type == "wind" else None
                render_wind_indices(ds, u_var, v_var, fallback_data=fb, fallback_var=fb_name, card_key="row4_r")

    # --- floating chat launcher (pinned bottom-right) --------------------
    try:
        from ai_chat import render_floating_chat
        _cv = st.session_state.get("_current_variable")
        if not _cv or _cv not in ds.data_vars:
            _cv = next((v for v in ds.data_vars
                        if "lat" in ds[v].dims and "lon" in ds[v].dims), None)
        if _cv:
            render_floating_chat(ds, _cv, classify_variable(_cv), time_index)
    except Exception as _ce:
        st.caption(f"Chat unavailable: {_ce}")
