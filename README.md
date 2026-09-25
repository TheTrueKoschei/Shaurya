# 🌍 PyClimaExplorer

**Climate data for the people who have to act on it.**

PyClimaExplorer turns raw NetCDF climate files into plain-language answers for district officials, farmers and planners in India. Upload a file and within seconds you get maps, trends, a 3D globe focused on India, and a written briefing on what the data means and what can be done about it — in English or Hindi, with or without an internet connection.

---

## The problem

India produces excellent climate data. IMD publishes gridded rainfall at 0.25° going back over a century, and global reanalysis covers every variable since 1948. But it ships as NetCDF — a format that needs Python, domain knowledge and several hours to read. The people making decisions about drainage, sowing calendars and heat advisories are rarely the people who can open those files.

PyClimaExplorer closes that gap.

---

## What it does

### 🧠 Insight Briefing
Every figure is computed directly from the file in Python: the long-term trend per decade, recent years against the baseline, the seasonal cycle, record years, and a clearly labelled projection if the trend continues. Those numbers are then turned into four plain-language sections: what the data shows, the shape of the year, who is affected, and what can be done.

The language layer never produces a number. It receives finished statistics and is instructed to use only those. An **audit panel** shows the exact figures it was given, so any sentence can be checked against its source.

Works fully offline with deterministic templates, in **English and Hindi**. With an API key it uses a language model for more natural phrasing.

### 💬 Ask this dataset
A chat grounded strictly in the loaded file. It has no general climate knowledge — every answer comes from statistics computed off the dataset in front of you, and it says so when it cannot answer.

It also runs **completely offline**. A local intent matcher answers from the same fact sheet with no internet, no API account and no GPU — suitable for a block office with unreliable connectivity. If the network drops mid-conversation, it falls back to offline answers automatically.

### 🇮🇳 India-focused 3D globe
A full rotating globe, opened directly over India. Data is clipped to India's coastline, fine grids stay sharp, and India's border is highlighted. Switch between India, the data region, and the whole world.

### 🗺️ Regional datasets
Ships with ready-to-load files for nine regions, each telling a different story:

| Region | Covers | What it shows |
|---|---|---|
| Northwest Arid | Rajasthan, Punjab, Haryana | heat and groundwater stress |
| Indo-Gangetic Plain | Uttar Pradesh, Bihar | food security for the most people |
| Central Plateau | Madhya Pradesh, Chhattisgarh | rain-fed agriculture |
| Western Ghats & Konkan | Maharashtra to Kerala coast | flood and landslide risk |
| Peninsular South | Karnataka, Telangana, TN, AP | two monsoons, reservoir planning |
| East Coast Delta | Odisha, West Bengal | cyclone preparedness |
| Northeast Hills | Assam and the northeast | rainfall extremes |
| Western Himalaya | J&K, Himachal, Uttarakhand | snow-fed rivers |
| All India | whole country | national overview |

Load two regions in turn and the same dashboard reaches opposite conclusions — because the data underneath does.

### Also included
Spatial heatmaps · time series with a 12-month rolling average and trend line · context-aware climate indices for temperature, rainfall, wind and humidity · Story Mode · side-by-side Compare · CSV export · light and dark themes.

---

## Data sources

| Source | Used for | Resolution | Period |
|---|---|---|---|
| **India Meteorological Department** gridded rainfall | regional rainfall files | 0.25° (~28 km) | 2015–2023 |
| **NCEP/NCAR Reanalysis 1**, NOAA PSL | temperature, humidity, wind | 2.5° (~278 km) | 1948–2026 |
| **geoBoundaries** (CC BY 4.0) | Indian state boundaries | simplified ADM1 | — |
| **Natural Earth** | neighbouring countries | 1:110m | — |

Regional files are the source data cut by extent. No values are modified or synthesised.

---

## Run it locally

Requires Python 3.10 or newer.

```bash
git clone https://github.com/TheTrueKoschei/PyClimaExplorer.git
cd PyClimaExplorer
pip install -r requirements.txt
python -m streamlit run ULTIMATE.py
```

Opens at `http://localhost:8501`. Upload any file from `india_regions/` or `india_regions_ncep/` to start.

### Optional: language-model briefings
Everything works without this. To enable model-written briefings and chat, create `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your-key"     # free tier at aistudio.google.com
# or GROQ_API_KEY / OPENAI_API_KEY
```

Never commit this file. It is already listed in `.gitignore`.

---

## Build your own regional files

```bash
# Clip any global NetCDF to India
python make_india_subset.py air.mon.mean.nc

# Build all nine regions, merging several variables into each file
python make_india_regions.py air.mon.mean.nc rhum.mon.mean.nc uwnd.mon.mean.nc vwnd.mon.mean.nc

# IMD daily rainfall: join the years and average to monthly
pip install imddata
imddata --name rain --syear 2015 --eyear 2023
python make_india_regions.py IMD_rain_*.nc --monthly
```

NCEP monthly surface files are at `psl.noaa.gov/thredds/catalog/Datasets/ncep.reanalysis.derived/surface/catalog.html`.

---

## Project structure

```
├── ULTIMATE.py              main Streamlit app
├── ai_insight.py            statistics engine + Insight Briefing
├── ai_chat.py               dataset-grounded chat
├── local_brain.py           offline answerer, no dependencies
├── theme.py                 visual polish layer
├── make_india_subset.py     clip a global file to India
├── make_india_regions.py    build the regional file set
├── india_regions/           IMD rainfall, nine regions, 0.25°
├── india_regions_ncep/      NCEP multivariable, nine regions, 2.5°
├── geoBoundaries-IND-ADM1_simplified.geojson   Indian state boundaries
├── ne_110m_admin_0_countries.*   world boundaries (all five parts)
├── photo.jpg                background image
└── requirements.txt
```

---

## Known limitations

- **NCEP is coarse.** At 2.5°, a single region is only 3–4 grid cells. Briefings and time series are unaffected since they use area averages, but regional maps look blocky. Use the IMD files for regional maps.
- **Nine years is short for a trend.** The IMD record shows the seasonal cycle clearly, but its trend-per-decade is dominated by monsoon variability. The 78-year NCEP temperature record is where long-term trends are meaningful.
- **Projections are linear extrapolations**, labelled as conditional. They are not climate model output.
- **Boundaries.** Indian states come from geoBoundaries, whose country datasets aim to represent each nation as it represents itself. Neighbouring countries use Natural Earth at 1:110m.
- **State filter uses rectangles** around each state, so a little neighbouring area is included in state-level statistics.

---

## Credits

India Meteorological Department · NOAA Physical Sciences Laboratory · geoBoundaries (Runfola et al., 2020, CC BY 4.0) · Natural Earth · Streamlit · Plotly · xarray · GeoPandas
