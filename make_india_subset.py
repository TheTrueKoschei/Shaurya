"""
=====================================================================
 make_india_subset.py  —  turn any global NetCDF into an India-only one
=====================================================================
 Usage (from the project folder):

     python make_india_subset.py air.mon.mean.nc

 Writes  air.mon.mean_INDIA.nc  next to the original.

 Handles both longitude conventions automatically:
   * -180..180  (most observational datasets)
   *    0..360  (NCEP/NCAR reanalysis)
 and latitude stored either south-to-north or north-to-south.
=====================================================================
"""
import sys
import os
import numpy as np
import xarray as xr

# Generous box: includes Kashmir, the Northeast, and the island territories.
LAT_MIN, LAT_MAX = 5.0, 38.5
LON_MIN, LON_MAX = 66.5, 98.5


def subset(path_in, path_out=None):
    if path_out is None:
        stem, ext = os.path.splitext(path_in)
        path_out = f"{stem}_INDIA{ext}"

    ds = None
    for kwargs in ({}, {"decode_times": False}):
        try:
            ds = xr.open_dataset(path_in, **kwargs)
            break
        except Exception:
            continue
    if ds is None:
        raise SystemExit(f"Could not open {path_in}")

    # --- locate the coordinate names -----------------------------------
    latname = next((c for c in ("lat", "latitude", "LAT", "Lat") if c in ds.coords), None)
    lonname = next((c for c in ("lon", "longitude", "LON", "Lon") if c in ds.coords), None)
    if latname is None or lonname is None:
        raise SystemExit(f"No lat/lon coordinates found. Coords are: {list(ds.coords)}")

    lons = ds[lonname].values

    # --- 0..360 files need the box shifted, not the data rolled --------
    if float(np.nanmax(lons)) > 180.0:
        lo_min, lo_max = LON_MIN % 360, LON_MAX % 360
    else:
        lo_min, lo_max = LON_MIN, LON_MAX

    # --- latitude may run north-to-south; slice() needs matching order --
    lats = ds[latname].values
    lat_slice = (slice(LAT_MAX, LAT_MIN) if lats[0] > lats[-1]
                 else slice(LAT_MIN, LAT_MAX))

    out = ds.sel({latname: lat_slice, lonname: slice(lo_min, lo_max)})

    n_lat = out.sizes.get(latname, 0)
    n_lon = out.sizes.get(lonname, 0)
    if n_lat == 0 or n_lon == 0:
        raise SystemExit("Subset is empty — this file may not cover India.")

    # rename to what the dashboard expects
    ren = {}
    if latname != "lat":
        ren[latname] = "lat"
    if lonname != "lon":
        ren[lonname] = "lon"
    if ren:
        out = out.rename(ren)

    out.attrs["india_subset"] = (f"clipped to {LAT_MIN}-{LAT_MAX}N, "
                                 f"{LON_MIN}-{LON_MAX}E from {os.path.basename(path_in)}")
    out.to_netcdf(path_out)

    before = os.path.getsize(path_in) / 1e6
    after = os.path.getsize(path_out) / 1e6
    print(f"  grid      : {n_lat} lat x {n_lon} lon")
    print(f"  variables : {', '.join(list(out.data_vars))}")
    print(f"  size      : {before:.1f} MB  ->  {after:.1f} MB")
    print(f"  written   : {path_out}")
    return path_out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python make_india_subset.py <file.nc> [more.nc ...]")
    for f in sys.argv[1:]:
        print(f"\n{f}")
        subset(f)
