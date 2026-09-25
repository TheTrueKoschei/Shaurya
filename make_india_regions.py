"""
=====================================================================
 make_india_regions.py  —  build a demo set of Indian regional files
=====================================================================
 Takes the real NetCDF files you downloaded and produces one
 MULTIVARIABLE file per Indian region, so a single upload lights up
 the temperature, rainfall, humidity AND wind panels together.

 Nothing is invented. Every value is the source data, cut by region.
 If a judge asks where it came from, the answer is NCEP/NCAR
 reanalysis or IMD — the same file, sliced.

 Usage — pass every variable file you have:

     python make_india_regions.py air.mon.mean.nc rhum.mon.mean.nc \\
            uwnd.mon.mean.nc vwnd.mon.mean.nc

 --out FOLDER   write somewhere other than ./india_regions/
 --monthly      average daily data (such as IMD) down to monthly means.
 Several years of daily data otherwise runs past Streamlit's upload limit:

     python make_india_regions.py IMD_rain_*.nc --monthly

 Writes into ./india_regions/ :

     INDIA_Northwest_Arid.nc        Rajasthan, Punjab, Haryana
     INDIA_Indo_Gangetic_Plain.nc   Uttar Pradesh, Bihar
     INDIA_Central_Plateau.nc       Madhya Pradesh, Chhattisgarh
     INDIA_Western_Ghats_Konkan.nc  Maharashtra to Kerala coast
     INDIA_Peninsular_South.nc      Karnataka, Telangana, TN, AP
     INDIA_East_Coast_Delta.nc      Odisha, West Bengal
     INDIA_Northeast_Hills.nc       Assam and the seven sisters
     INDIA_Western_Himalaya.nc      J&K, Himachal, Uttarakhand
     INDIA_All.nc                   the whole country, all variables

 Grids that don't match the first file are snapped to it by nearest
 neighbour, which is how the gaussian-grid precipitation file gets
 merged with the 2.5-degree surface files.
=====================================================================
"""
import os
import sys
import numpy as np
import xarray as xr

OUTDIR = "india_regions"

# lat_min, lat_max, lon_min, lon_max  (degrees north / east)
REGIONS = {
    "Northwest_Arid":       (23.0, 32.5, 69.0, 78.0),
    "Indo_Gangetic_Plain":  (24.0, 30.5, 77.0, 88.0),
    "Central_Plateau":      (19.5, 26.5, 74.0, 84.0),
    "Western_Ghats_Konkan": ( 8.0, 21.0, 72.5, 77.5),
    "Peninsular_South":     ( 8.0, 18.5, 74.0, 84.0),
    "East_Coast_Delta":     (16.5, 24.0, 81.5, 89.0),
    "Northeast_Hills":      (22.0, 29.5, 88.0, 97.5),
    "Western_Himalaya":     (28.5, 37.0, 73.0, 81.0),
    "All":                  ( 5.0, 38.5, 66.5, 98.5),
}

# what each region is good for demonstrating — printed as a cheat sheet
STORY = {
    "Northwest_Arid":       "hottest, driest — heat action and groundwater",
    "Indo_Gangetic_Plain":  "most people, most farmland — food security",
    "Central_Plateau":      "your home region — rain-fed agriculture",
    "Western_Ghats_Konkan": "wettest coast — flood and landslide risk",
    "Peninsular_South":     "two monsoons — reservoir planning",
    "East_Coast_Delta":     "cyclone landfall — disaster preparedness",
    "Northeast_Hills":      "heaviest rainfall on earth — extremes",
    "Western_Himalaya":     "coldest, snow-fed rivers — meltwater supply",
    "All":                  "national overview — the opening slide",
}


def open_any(path):
    for kw in ({}, {"decode_times": False}):
        try:
            return xr.open_dataset(path, **kw)
        except Exception:
            continue
    return None


def standardise(ds):
    """Rename coordinates to lat/lon and put longitudes on -180..180."""
    ren = {}
    for a, b in (("latitude", "lat"), ("LAT", "lat"), ("Lat", "lat"),
                 ("longitude", "lon"), ("LON", "lon"), ("Lon", "lon")):
        if a in ds.coords:
            ren[a] = b
    if ren:
        ds = ds.rename(ren)
    if "lat" not in ds.coords or "lon" not in ds.coords:
        return None

    if float(ds["lon"].max()) > 180.0:
        ds = ds.assign_coords(lon=(((ds["lon"] + 180) % 360) - 180))
        ds = ds.sortby("lon")
    if ds["lat"].size > 1 and float(ds["lat"][0]) > float(ds["lat"][-1]):
        ds = ds.sortby("lat")
    return ds


def write_compact(ds, path):
    """
    Write with zlib compression and float32 storage.

    Source files often carry packing instructions (int16 + scale_factor,
    as NCEP does) that xarray would try to reapply on write. Those are
    cleared first so they can't clash with the new encoding.

    Typically 3-8x smaller than a plain write, which keeps every file under
    GitHub's 25 MB browser-upload limit and makes uploads to the app faster.
    """
    ds = ds.copy()
    for name in ds.variables:
        ds[name].encoding = {}
    enc = {}
    for v in ds.data_vars:
        if np.issubdtype(ds[v].dtype, np.floating):
            enc[v] = {"zlib": True, "complevel": 5, "dtype": "float32",
                      "_FillValue": np.float32(np.nan)}
    try:
        ds.to_netcdf(path, encoding=enc)
    except Exception:
        ds.to_netcdf(path)       # uncompressed, but never fail the build


def main(paths, monthly=False, outdir=OUTDIR):
    os.makedirs(outdir, exist_ok=True)

    merged, base = None, None
    for p in paths:
        ds = open_any(p)
        if ds is None:
            print(f"  skip  {p}  (could not open)")
            continue
        ds = standardise(ds)
        if ds is None:
            print(f"  skip  {os.path.basename(p)}  (no lat/lon)")
            continue

        if merged is None:
            merged, base = ds, ds
            print(f"  base  {os.path.basename(p):28s} "
                  f"{ds.sizes.get('lat')}x{ds.sizes.get('lon')} grid  "
                  f"-> {', '.join(list(ds.data_vars)[:4])}")
            continue

        try:
            if (ds.sizes.get("lat") != base.sizes.get("lat")
                    or ds.sizes.get("lon") != base.sizes.get("lon")):
                ds = ds.reindex(lat=base["lat"], lon=base["lon"], method="nearest")
                note = "  (snapped to base grid)"
            else:
                note = ""

            shared = set(ds.data_vars) & set(merged.data_vars)
            if shared and "time" in ds.dims and "time" in merged.dims:
                # Same variable, different years (e.g. IMD_rain_2022 +
                # IMD_rain_2023). These are successive slices of one record,
                # so they join end to end. Merging them instead would
                # intersect the time axes and leave nothing.
                merged = xr.concat([merged, ds], dim="time",
                                   coords="minimal", compat="override")
                merged = merged.sortby("time")
                _, keep = np.unique(merged["time"].values, return_index=True)
                if keep.size != merged.sizes["time"]:
                    merged = merged.isel(time=np.sort(keep))
                how = f"joined on time -> {merged.sizes['time']} steps"
            else:
                merged = xr.merge([merged, ds], compat="override", join="inner")
                how = f"-> {', '.join(list(ds.data_vars)[:4])}"
            print(f"  add   {os.path.basename(p):28s} {how}{note}")
        except Exception as e:
            print(f"  skip  {os.path.basename(p)}  ({type(e).__name__}: {e})")

    if merged is None:
        raise SystemExit("No usable files. Pass at least one NetCDF with lat/lon.")

    nt = merged.sizes.get("time", 0)
    if "time" in merged.dims and nt == 0:
        raise SystemExit(
            "The combined dataset has zero time steps, so the input files do "
            "not overlap and could not be joined. Pass them one at a time.")
    span = ""
    try:
        tv = merged["time"].values
        span = f"  |  {str(tv[0])[:10]} to {str(tv[-1])[:10]}"
    except Exception:
        pass
    print(f"\n  merged variables: {', '.join(list(merged.data_vars))}"
          f"  ({nt} time steps){span}")

    if monthly and "time" in merged.dims:
        # Daily IMD data over several years runs to thousands of steps and
        # hundreds of MB, which is past Streamlit's upload limit. Monthly
        # means keep every trend and season while shrinking it ~30x.
        try:
            before = merged.sizes["time"]
            merged = merged.resample(time="MS").mean(skipna=True)
            print(f"  resampled to monthly means: {before} -> "
                  f"{merged.sizes['time']} steps")
        except Exception as e:
            print(f"  monthly resample failed ({type(e).__name__}), keeping daily")
    print()
    print(f"  {'region':24s} {'grid':>9s}  {'MB':>6s}  story")
    print("  " + "-" * 86)

    written = []
    for name, (la0, la1, lo0, lo1) in REGIONS.items():
        try:
            sub = merged.sel(lat=slice(la0, la1), lon=slice(lo0, lo1))
            ny, nx = sub.sizes.get("lat", 0), sub.sizes.get("lon", 0)
            if ny == 0 or nx == 0:
                print(f"  {name:24s} {'empty':>9s}  — source does not cover it")
                continue

            sub.attrs["region"] = name.replace("_", " ")
            sub.attrs["bounds"] = f"{la0}-{la1}N, {lo0}-{lo1}E"
            sub.attrs["source_files"] = "; ".join(os.path.basename(p) for p in paths)
            sub.attrs["note"] = ("Regional subset of the source files. Values are "
                                 "unmodified; only the spatial extent differs.")

            out = os.path.join(outdir, f"INDIA_{name}.nc")
            write_compact(sub, out)
            mb = os.path.getsize(out) / 1e6
            flag = "  <- coarse, few cells" if (ny < 3 or nx < 3) else ""
            if mb > 25:
                flag += "  <- OVER 25 MB, too big for a GitHub browser upload"
            print(f"  {name:24s} {ny:3d}x{nx:<4d}  {mb:6.1f}  {STORY[name]}{flag}")
            written.append(out)
        except Exception as e:
            print(f"  {name:24s} failed ({type(e).__name__})")

    print(f"\n  {len(written)} files written to ./{outdir}/")
    print("  Upload any one of them — it carries every variable at once.\n")
    print("  If a region shows very few cells, the source grid is too coarse for it.")
    print("  IMD data at 0.25 degrees (pip install imddata) fixes that.")


if __name__ == "__main__":
    argv = sys.argv[1:]
    outdir = OUTDIR
    if "--out" in argv:
        i = argv.index("--out")
        if i + 1 >= len(argv):
            raise SystemExit("--out needs a folder name, e.g. --out india_regions_ncep")
        outdir = argv[i + 1]
        del argv[i:i + 2]
    monthly = "--monthly" in argv
    args = [a for a in argv if not a.startswith("--")]
    if not args:
        raise SystemExit(__doc__)
    main(args, monthly=monthly, outdir=outdir)
