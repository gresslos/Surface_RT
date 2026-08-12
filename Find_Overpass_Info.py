import h5py
import numpy as np
import re
import sys
from datetime import datetime, timezone, timedelta

# The program reads an ACM-COM file and finds the across_index and along_index that
# matches a given lat/lon (Pyranomer ground based station)
# python Find_index_input.py /xnilu_wrk2/projects/NEVAR/data/EarthCARE_Real/Orbit_06311C_Surface/output/ECA_EXBA_ACM_COM_2B_20250708T132122Z_20250710T081425Z_06311C/ECA_EXBA_ACM_COM_2B_20250708T132122Z_20250710T081425Z_06311C.h5 69.464 25.502

FILL = 9.96920997e36  # file fill value

def haversine_distance_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    lat1 = np.deg2rad(lat1); lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2); lon2 = np.deg2rad(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def normalize_lon(lon):
    lon = (lon + 180.0) % 360.0 - 180.0
    return np.where(lon == 180.0, -180.0, lon)

def _units_to_base_dt(units_str: str) -> datetime:
    if not units_str or "since" not in units_str:
        return datetime(2000, 1, 1, tzinfo=timezone.utc)
    base = units_str.split("since", 1)[1].strip()
    m = re.match(
        r"^\s*(\d{1,4})-(\d{1,2})-(\d{1,2})\s+(\d{1,2}):(\d{1,2}):(\d{1,2}(?:\.\d+)?)"
        r"(?:\s*([+-]?\d{1,2}):(\d{2}))?\s*$", base
    )
    if not m:
        ymd = re.match(r"^\s*(\d{1,4})-(\d{1,2})-(\d{1,2})", base)
        if ymd:
            y, mo, d = map(int, ymd.groups())
            return datetime(y, mo, d, tzinfo=timezone.utc)
        raise ValueError(f"Unrecognized time units base: {base!r}")
    y, mo, d = map(int, m.group(1, 2, 3))
    H, M = map(int, m.group(4, 5))
    sec_str = m.group(6)
    if "." in sec_str:
        whole, frac = sec_str.split(".", 1)
        S = int(whole)
        micro = int(round(float("0." + frac) * 1_000_000))
    else:
        S = int(sec_str); micro = 0
    tz_h = m.group(7); tz_m = m.group(8)
    if tz_h is not None and tz_m is not None:
        sign = -1 if str(tz_h).startswith("-") else 1
        off_h = abs(int(tz_h)); off_m = int(tz_m)
        offset = timedelta(hours=off_h, minutes=off_m) * sign
    else:
        offset = timedelta(0)
    local_base = datetime(y, mo, d, H, M, S, microsecond=micro, tzinfo=timezone.utc)
    return local_base - offset  # return UTC base

def find_track_values(h5_path, MyLat, MyLon):
    MyLon = float(MyLon); MyLat = float(MyLat)
    MyLon = normalize_lon(MyLon)

    with h5py.File(h5_path, "r") as f:
        g = f["ScienceData"]
        lat = g["latitude"][:]                  # (207, 4845)
        lon = g["longitude"][:]                 # (207, 4845)
        across_track = g["across_track"][:]     # (207,)
        along_track  = g["along_track"][:]      # (4845,)
        time_vals    = g["time"][:]             # (4845,)
        u = g["time"].attrs.get("units", "")
        if isinstance(u, (bytes, bytearray)):
            u = u.decode("utf-8", errors="ignore")
        time_units = str(u)

    # Mask fill values if present
    lat = np.where(np.isfinite(lat) & (lat != FILL), lat, np.nan)
    lon = np.where(np.isfinite(lon) & (lon != FILL), lon, np.nan)
    time_vals = np.where(np.isfinite(time_vals) & (time_vals != FILL), time_vals, np.nan)

    lon = normalize_lon(lon)

    # Nearest pixel (ignore NaNs)
    dist_m = haversine_distance_m(MyLat, MyLon, lat, lon)
    if np.all(~np.isfinite(dist_m)):
        raise ValueError("No valid lat/lon pixels found.")
    i_across, j_along = np.unravel_index(np.nanargmin(dist_m), dist_m.shape)

    # Convert along-track time to UTC datetime
    base_dt = _units_to_base_dt(time_units)
    t_sec = float(time_vals[j_along])
    if np.isfinite(t_sec):
        t_dt = base_dt + timedelta(seconds=t_sec)
        t_iso = t_dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
    else:
        t_dt = None; t_iso = None

    return {
        "nearest_grid_lat": float(lat[i_across, j_along]),
        "nearest_grid_lon": float(lon[i_across, j_along]),
        "nearest_distance_m": float(dist_m[i_across, j_along]),
        "across_index": int(i_across),
        "along_index": int(j_along),
        "across_track_m": float(across_track[i_across]),
        "along_track_m": float(along_track[j_along]),
        "time_seconds_since_epoch_base": t_sec,
        "time_units": time_units,
        "time_datetime_utc": t_dt,
        "time_iso_utc": t_iso,
    }




def get_overpass_info(out):
    return out['across_index'], out['along_index'], out['time_iso_utc']



if __name__ == "__main__":

    #MyLat = 69.464 
    #MyLon = 25.502 
    #h5file = "/xnilu_wrk2/projects/NEVAR/data/EarthCARE_Real/Orbit_06311C_Surface/output/ECA_EXBA_ACM_COM_2B_20250708T132122Z_20250710T081425Z_06311C/ECA_EXBA_ACM_COM_2B_20250708T132122Z_20250710T081425Z_06311C.h5"

    if len(sys.argv) < 4:
        print("Usage: python Find_Overpass_Info.py <h5file> <latitude> <longitude>")
        sys.exit(1)

    h5file = sys.argv[1]
    MyLat = float(sys.argv[2])
    MyLon = float(sys.argv[3])

    out = find_track_values(h5file, MyLat, MyLon)
    # print("Nearest grid point:", (out["nearest_grid_lat"], out["nearest_grid_lon"]))
    # print("Distance to target (m):", f'{out["nearest_distance_m"]:.1f}')
    # print("Across index / value (m):", out["across_index"], "/", f'{out["across_track_m"]:.1f}')
    # print("Along  index / value (m):", out["along_index"],  "/", f'{out["along_track_m"]:.1f}')
    # print("Time units:", out["time_units"])
    # print("Time (UTC):", out["time_iso_utc"])


    # Create a formatted output string
    output_text = (
        f"Results for {h5file}\n"
        f"Input coordinate: ({MyLat:.4f}, {MyLon:.4f})\n"
        f"Nearest grid point: ({out['nearest_grid_lat']:.4f}, {out['nearest_grid_lon']:.4f})\n"
        f"Distance to target (m): {out['nearest_distance_m']:.1f}\n"
        f"Across index/value (m): {out['across_index']} / {out['across_track_m']:.1f}\n"
        f"Along  index/value (m): {out['along_index']} / {out['along_track_m']:.1f}\n"
        f"Time (UTC): {out['time_iso_utc']}\n"
        #f"Time units: {out['time_units']}\n"
    )

    # Print to screen
    print("\n" + output_text)

    # Write to file
    # with open("Distance.txt", "w") as f:
    #     f.write(output_text)

    get_overpass_info(out)
    
    


