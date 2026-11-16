import sys, os
import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import h5py
from netCDF4 import Dataset
from scipy.interpolate import griddata
from pathlib import Path
from matplotlib.patches import Patch

import Read_RTM as ReadEC
import Find_Overpass_Info as Find_Overpass_Info

FONTSIZE=15
INFOSIZE=13
FIGSIZE=(8,8)
cmap = plt.get_cmap('viridis') 



def read_flux(filename, REMOVE_STATIONS=None):
    with open(filename) as file:
        lines = [l.strip() for l in file if l.strip()]
    flux = []
    for line in lines:
        line = line.split(" ")
        F = float(line[0])
        if len(line) > 1:
            libRadFilename = line[1]
            if "_" in libRadFilename:
                station = libRadFilename.rsplit("_", 2)[-2]
            else:
                station = ""
            # Put in nans for fluxes at stations that should be removed
            if REMOVE_STATIONS is not None and station in REMOVE_STATIONS:
                F = np.nan
        flux.append(F)
    return np.asarray(flux)


def get_solver_info(filename):
    if "DISORT" in filename:
        # solver = "DISORT (1D RTM)"
        solver = "1D RTM"
        return solver, "red", "^"
    
    if "MYSTIC" in filename:
        solver = "MYSTIC (3D RTM)"
        solver = "3D RTM"
        return solver, "blue", "o"
    if "ACMRT1D" in filename:
        return "ACM-RT (1D RTM)", "darkorange", "s"

    return "Unknown RTM", "gray", "x"


def calculate_statistics(x, y):
    difference = y - x
    slope, intercept = np.polyfit(x, y, 1)

    return {
        "bias": np.mean(difference),
        "std": np.std(difference),
        "mae": np.mean(np.abs(difference)),
        "rel_bias": np.mean(difference / x) * 100,
        "rmse": np.sqrt(np.mean(difference**2)),

        "diff": difference,
        "absolute_diff": np.abs(difference),
        "relative_diff": difference / x * 100,

        "n": len(x),
        "slope": slope,
        "intercept": intercept,
    }

def filter_out_nans_zeros(x, y, z=False):
    print("Before filtering: len x = {}".format(len(x)))
    mask = (x != 0) & (y != 0) & (~np.isnan(x)) & (~np.isnan(y))
    if z is not False:
        mask &= (z != 0) & (~np.isnan(z))
    x_filtered = x[mask]
    y_filtered = y[mask]
    print("After filtering:  len x = {}\n".format(len(x_filtered)))   
    if z is not False:
        z_filtered = z[mask]
        return x_filtered, y_filtered, z_filtered
    return x_filtered, y_filtered

# def filter_out_nans_zeros_3_variables(x, y, z):
#     print("Before filtering: len x = {}".format(len(x)))
#     mask = (x != 0) & (y != 0) & (~np.isnan(x)) & (~np.isnan(y)) & (~np.isnan(z))
#     x_filtered = x[mask]
#     y_filtered = y[mask]
#     z_filtered = z[mask]
#     print("After filtering:  len x = {}\n".format(len(x_filtered)))   
#     return x_filtered, y_filtered, z_filtered

def style_axes(ax, fig):
    ax.set_facecolor("#f0f0f0")
    ax.grid(which="major", linestyle="--", alpha=0.4)
    ax.grid(which="minor", linestyle=":", alpha=0.2)
    ax.minorticks_on()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    fig.tight_layout()  
    
def get_acm_com_baseline(folder):
    files = sorted(Path(folder).glob("*ACM_COM*"))
    if not files:
        print(f"No ACM-COM file found in {folder}")
        return "unknown"
    filename = files[0].name
    if "ECA_EX" not in filename:
        print(f"Cannot extract baseline from {filename}")
        return "unknown"
    return filename.split("ECA_EX", 1)[1][:2]


def get_sza(ACMCOM, BMAFLX, ia):
    latitude_wanted = ACMCOM.latitude_active[ia]
    longitude_wanted = ACMCOM.longitude_active[ia]
    dist = np.sqrt((BMAFLX.latitude - latitude_wanted)**2 + (BMAFLX.longitude - longitude_wanted)**2)
    indlatBMAFLX = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
    sza = BMAFLX.solar_zenith_angle[indlatBMAFLX,1].squeeze()  # Nadir view is in element one
    return sza

def across_track_limit(x, y, limit, z=False, idx_lines=None):
    with open("DATA/Distances.txt", "r") as f:
        distances = np.array([float(line.split()[0]) for line in f])

    if idx_lines is not None:
        distances = distances[idx_lines]

    mask = distances < limit

    x[~mask] = np.nan
    y[~mask] = np.nan
    if z is not False:
        z[~mask] = np.nan
        return x, y, z
    return x, y

def get_quality_status(qs_acmcom_file, qs_bmaflx_file, qs_acmrt_file, qs_all3d_file):
    qs_ACMCOM = np.loadtxt(qs_acmcom_file, usecols=0)
    qs_BMAFLX = np.loadtxt(qs_bmaflx_file, usecols=0)
    qs_ACMRT  = np.loadtxt(qs_acmrt_file, usecols=0)
    qs_ALL3D  = np.loadtxt(qs_all3d_file, usecols=0)

    return qs_ACMCOM, qs_BMAFLX, qs_ACMRT, qs_ALL3D

def apply_qs(x, y, qs_ACMCOM, qs_BMAFLX, qs_ACMRT, qs_ALL3D, rte_solver, OBS_fluxes):
    # x: Observed fluxes
    # y: Model fluxes

    # ACMCOM and ALL3D QS to libRad fluxes
    if "1D" in rte_solver or "3D" in rte_solver:
        y = np.where(np.isin(qs_ACMCOM, (0, 1)), y, np.nan)
        y = np.where(np.isin(qs_ALL3D, (0, 1, 2)), y, np.nan)
    
    # ACM-RT and ALL3D QS to ACM-RT
    elif "ACM-RT" in rte_solver:
        y = np.where(np.isin(qs_ACMRT, (0, 1)), y, np.nan)
        y = np.where(np.isin(qs_ALL3D, (0, 1, 2)), y, np.nan)

    # ACM-RT and ALL3D QS to ACM-RT fluxes
    if "ACM-RT" in OBS_fluxes:
        rte_solver = None
        x = np.where(np.isin(qs_ACMRT, (0, 1)), x, np.nan)
        x = np.where(np.isin(qs_ALL3D, (0, 1, 2)), x, np.nan)

    # BMAFLX QS to TOA obs
    else:
        if not WANT_SUR:  # BMAFLX QS to TOA obs
            x = np.where(np.isin(qs_BMAFLX, (0, 2)), x, np.nan)

    return x, y, rte_solver








def write_info_to_file(RTMs_fluxes, property="Distance"):
    # Info for downloading ACM-COM data
    Path_base = "/xnilu_wrk2/projects/NEVAR/data/CalVal/SurfaceOverpasses/"  # OLD: "/homevip/bgre/Download/Frames_SurfaceOverpasses"


    if property == "Distance":
        filename = "DATA/Distances.txt"
        DistanceInfo = [] # List of (Distances, OrbitID, Station)
        with open(RTMs_fluxes[0]) as file:
            for lines in file:
                line = lines.strip() 
                # Ex line: ./RESULTS/libRad_v01_disort_1D_SWIA_solar_01028B_Oslo_SUR.nc 
                parts = line.split("_")
                OrbitID = parts[-3]
                Station = parts[-2]
    
                # Get ACM-COM frames to extract Distances
                Product ='ACM_COM'
                ProductPath = '*'+Product+'*'+OrbitID+'*'
                Path = Path_base + f'/{Station}'   
                ProductFile = os.path.join(Path, ProductPath, '*'+Product+'*.h5')
                try: 
                    ProductFile = sorted(glob.glob(ProductFile))[0]
                except IndexError: # If do not have for that Frame -> Skipping
                    DistanceInfo.append((np.nan, OrbitID, Station))
                    print(f"{np.nan:10.2f} km      Distance for {OrbitID} over {Station:^15}--- Across-Track Index {out['across_index']}")
                    continue
                ACMCOM = ReadEC.Scene(Name=OrbitID)
                ACMCOM.ReadEarthCAREh5(ProductFile)
    
        
                lat, lon = sites[Station]
                out = Find_Overpass_Info.find_track_values(ACMCOM.fn, lat, lon)
                D   = abs(out['across_index']-150) # [km] 
                # D_list.append(D)
                DistanceInfo.append((D, OrbitID, Station))
                print(f"{D:10.2f} km      Distance for {OrbitID} over {Station:^15}--- Across-Track Index {out['across_index']}")

        # Write distances to file
        with open(filename, "w") as f:
            for D, OrbitID, Station in DistanceInfo:
                f.write(f"{D:<10.2f} {OrbitID} {Station}\n")

    elif property == "SZA":
        filename = "DATA/SZA.txt"
        szaInfo = [] # List of (Distances, OrbitID, Station)
        with open(RTMs_fluxes[0]) as file:
            for lines in file:
                line = lines.strip() 
                # Ex line: ./RESULTS/libRad_v01_disort_1D_SWIA_solar_01028B_Oslo_SUR.nc 
                parts = line.split("_")
                OrbitID = parts[-3]
                Station = parts[-2]

                ACMCOM = None
                BMAFLX = None
    
                # Get ACM-COM frames to extract correct BMA-FLX position
                Product ='ACM_COM'
                ProductPath = '*'+Product+'*'+OrbitID+'*'
                Path = Path_base + f'/{Station}'   
                ACMCOM_ProductFile = os.path.join(Path, ProductPath, '*'+Product+'*.h5')

                # Get BMA-FLX frames to extract sza
                Product ='BMA_FLX'
                ProductPath = '*'+Product+'*'+OrbitID+'*'
                Path = Path_base + f'/{Station}'   
                BMAFLX_ProductFile = os.path.join(Path, ProductPath, '*'+Product+'*.h5')
                try: 
                    ACMCOM_ProductFile = sorted(glob.glob(ACMCOM_ProductFile))[0]
                    BMAFLX_ProductFile = sorted(glob.glob(BMAFLX_ProductFile))[0]

                    ACMCOM = ReadEC.Scene(Name=OrbitID)
                    BMAFLX = ReadEC.Scene(Name=OrbitID)
                    ACMCOM.ReadEarthCAREh5(ACMCOM_ProductFile)
                    BMAFLX.ReadEarthCAREh5(BMAFLX_ProductFile, Resolution='SmallResolution') 
                    


                    # Find ialong-track-idx for overpass for overpass
                    lat, lon = sites[Station]
                    out = Find_Overpass_Info.find_track_values(ACMCOM.fn, lat, lon)
                    ial   = out["along_index"]

                    sza = get_sza(ACMCOM, BMAFLX, ial)
                
                    szaInfo.append((sza, OrbitID, Station))
                    print(f"{sza:10.2f}        {OrbitID} over {Station:^15}")

                except IndexError: # If do not have for that Frame -> Skipping
                    szaInfo.append((np.nan, OrbitID, Station))
                    print(f"{np.nan:10.2f}        {OrbitID} over {Station:^15}")
                    continue
            

        # Write SZA to file
        with open(filename, "w") as f:
            for sza, OrbitID, Station in szaInfo:
                f.write(f"{sza:<10.2f} {OrbitID} {Station}\n")

    
    elif property == "CMF":
        filename = "DATA/CMF_MYSTIC_CompDomain" + ("_SUR" if WANT_SUR else "_TOA") + ".txt"
        CMF = []
        with open(RTMs_fluxes[0], "r") as file_all_sky, open(RTMs_fluxes[1], "r") as file_clear_sky:
            for lines_all_sky, lines_clear_sky in zip(file_all_sky, file_clear_sky):
                line_all_sky = lines_all_sky.strip() 
                line_clear_sky = lines_clear_sky.strip() 

                parts = line_all_sky.split("_") # Ex line: ./RESULTS/libRad_v01_disort_1D_SWIA_solar_01028B_Oslo_SUR.nc 
                OrbitID = parts[-3]
                Station = parts[-2]

                
                F_all_sky = float(line_all_sky.split()[0])
                F_clear_sky = float(line_clear_sky.split()[0])
    
                if F_clear_sky > 0:
                    CMF_value = F_all_sky / F_clear_sky
                else:
                    CMF_value = np.nan

                print(f"CMF = {CMF_value:<10.2f}               for Orbit {OrbitID}, Station {Station} \n\n")
                CMF.append((CMF_value, OrbitID, Station))

        with open(filename, "w") as f:
            for CMF_value, OrbitID, Station in CMF:
                f.write(f"{CMF_value:<10.2f} {OrbitID} {Station}\n")

    elif property == "overlapping_TOA_SUR_points":
        # Make file with overlap RTM fluxes and obs at SUR and TOA
        filename_TOA_DISORT_overlap = 'DATA/TOA_DISORT_overlap.txt'; filename_TOA_MYSTIC_overlap = 'DATA/TOA_MYSTIC_overlap.txt'
        filename_SUR_DISORT_overlap = 'DATA/SUR_DISORT_overlap.txt'; filename_SUR_MYSTIC_overlap = 'DATA/SUR_MYSTIC_overlap.txt'

        observations_TOA = 'DATA/TOA_Observations.txt'; observations_SUR = 'DATA/SUR_Observations_3minMean.txt'
        with open(RTMs_fluxes[0], "r") as file_1DTOA, open(RTMs_fluxes[1], "r") as file_3DTOA, open(RTMs_fluxes[2], "r") as file_1DSUR, open(RTMs_fluxes[3], "r") as file_3DSUR, open(observations_TOA, "r") as file_obs_TOA, open(observations_SUR, "r") as file_obs_SUR:
            with open(filename_TOA_DISORT_overlap, "w") as file_TOA_DISORT_overlap, open(filename_TOA_MYSTIC_overlap, "w") as file_TOA_MYSTIC_overlap, open(filename_SUR_DISORT_overlap, "w") as file_SUR_DISORT_overlap, open(filename_SUR_MYSTIC_overlap, "w") as file_SUR_MYSTIC_overlap:
                for lines_1DTOA, lines_3DTOA, lines_1DSUR, lines_3DSUR, lines_obs_TOA, lines_obs_SUR in zip(file_1DTOA, file_3DTOA, file_1DSUR, file_3DSUR, file_obs_TOA, file_obs_SUR):
                    line_1DTOA = lines_1DTOA.strip(); line_3DTOA = lines_3DTOA.strip()
                    line_1DSUR = lines_1DSUR.strip(); line_3DSUR = lines_3DSUR.strip()
                    line_obs_TOA = lines_obs_TOA.strip(); line_obs_SUR = lines_obs_SUR.strip()

                    parts = line_1DTOA.split("_") # Ex line: ./RESULTS/libRad_v01_disort_1D_SWIA_solar_01028B_Oslo_SUR.nc 
                    OrbitID = parts[-3]
                    Station = parts[-2]

                    F_1DTOA = float(line_1DTOA.split()[0])
                    F_3DTOA = float(line_3DTOA.split()[0])
                    F_1DSUR = float(line_1DSUR.split()[0])
                    F_3DSUR = float(line_3DSUR.split()[0])
                    F_obsTOA = float(line_obs_TOA.split()[0])
                    F_obsSUR = float(line_obs_SUR.split()[0])

                    print(f"Orbit {OrbitID}, Station {Station} -> 1D TOA: {F_1DTOA}, 3D TOA: {F_3DTOA}, 1D SUR: {F_1DSUR}, 3D SUR: {F_3DSUR}")

                    if not (F_1DTOA > 0 and F_1DSUR > 0 and F_obsTOA > 0 and F_obsSUR > 0):
                        F_1DTOA = np.nan
                        F_1DSUR = np.nan

                    if not (F_3DTOA > 0 and F_3DSUR > 0 and F_obsTOA > 0 and F_obsSUR > 0):
                        F_3DTOA = np.nan
                        F_3DSUR = np.nan
  

                    file_TOA_DISORT_overlap.write(f"{F_1DTOA:10} {OrbitID} {Station}\n")
                    file_SUR_DISORT_overlap.write(f"{F_1DSUR:10} {OrbitID} {Station}\n")
                    file_TOA_MYSTIC_overlap.write(f"{F_3DTOA:10} {OrbitID} {Station}\n")
                    file_SUR_MYSTIC_overlap.write(f"{F_3DSUR:10} {OrbitID} {Station}\n")

    elif property == "QS":
        filename_ACM_COM = "DATA/QS_ACM_COM.txt"
        filename_BMA_FLX = "DATA/QS_BMA_FLX.txt"
        filename_ACM_RT  = "DATA/QS_ACM_RT.txt"
        filename_ALL_3D  = "DATA/QS_ALL_3D.txt"

        ACM_COM_LIST = [] # List of (QS, OrbitID, Station)
        BMA_FLX_LIST = [] # List of (QS, OrbitID, Station)
        ACM_RT_LIST  = [] # List of (QS, OrbitID, Station)
        ALL_3D_LIST  = [] # List of (QS, OrbitID, Station)
        with open(RTMs_fluxes[0]) as file:
            for lines in file:
                line = lines.strip() 
                # Ex line: ./RESULTS/libRad_v01_disort_1D_SWIA_solar_01028B_Oslo_SUR.nc 
                parts = line.split("_")
                OrbitID = parts[-3]
                Station = parts[-2]

                ACMCOM,     BMAFLX,     ACMRT,     ALL3D     = None, None, None, None
                QS_ACM_COM, QS_BMA_FLX, QS_ACM_RT, QS_ALL_3D = np.nan, np.nan, np.nan, np.nan
                ial = None
                
    
                Product ='ACM_COM'
                ProductPath = '*'+Product+'*'+OrbitID+'*'
                Path = Path_base + f'/{Station}'   
                ACMCOM_ProductFile = os.path.join(Path, ProductPath, '*'+Product+'*.h5')
                Product ='BMA_FLX'
                ProductPath = '*'+Product+'*'+OrbitID+'*'
                Path = Path_base + f'/{Station}'   
                BMAFLX_ProductFile = os.path.join(Path, ProductPath, '*'+Product+'*.h5')
                Product ='ACM_RT'
                ProductPath = '*'+Product+'*'+OrbitID+'*'
                Path = Path_base + f'/{Station}'   
                ACMRT_ProductFile = os.path.join(Path, ProductPath, '*'+Product+'*.h5')
                Product ='ALL_3D'
                ProductPath = '*'+Product+'*'+OrbitID+'*'
                Path = Path_base + f'/{Station}'   
                ALL3D_ProductFile = os.path.join(Path, ProductPath, '*'+Product+'*.h5')

                try:
                    ALL3D_ProductFile = sorted(glob.glob(ALL3D_ProductFile))[0]
                    ALL3D = ReadEC.Scene(Name=OrbitID)
                    ALL3D.ReadEarthCAREh5(ALL3D_ProductFile)
                except IndexError:
                    print(f"Missing ALL3D file for OrbitID {OrbitID} over {Station:^15}")

                try:
                    ACMCOM_ProductFile = sorted(glob.glob(ACMCOM_ProductFile))[0]
                    ACMCOM = ReadEC.Scene(Name=OrbitID)
                    ACMCOM.ReadEarthCAREh5(ACMCOM_ProductFile)
                except IndexError:
                    print(f"Missing ACMCOM file for OrbitID {OrbitID} over {Station:^15}")

                try: 
                    BMAFLX_ProductFile = sorted(glob.glob(BMAFLX_ProductFile))[0]
                    BMAFLX = ReadEC.Scene(Name=OrbitID)
                    BMAFLX.ReadEarthCAREh5(BMAFLX_ProductFile, Resolution='SmallResolution') 
                except IndexError:
                    print(f"Missing BMAFLX file for OrbitID {OrbitID} over {Station:^15}")

                try: 
                    ACMRT_ProductFile = sorted(glob.glob(ACMRT_ProductFile))[0]
                    ACMRT = ReadEC.Scene(Name=OrbitID)
                    ACMRT.ReadEarthCAREh5(ACMRT_ProductFile)
                except IndexError:
                    print(f"Missing ACMRT file for OrbitID {OrbitID} over {Station:^15}")



                try: 
                    lat, lon = sites[Station]
                    out  = Find_Overpass_Info.find_track_values(ACMCOM.fn, lat, lon)
                    ial  = out["along_index"]
                    iacr = out["across_index"]
                    irec = ALL3D.index_construction[iacr, ial]

                       
                    latitude_wanted = ACMCOM.latitude_active[ial]
                    longitude_wanted = ACMCOM.longitude_active[ial]
                    dist = np.sqrt((BMAFLX.latitude - latitude_wanted)**2 + (BMAFLX.longitude - longitude_wanted)**2)
                    indlatBMAFLX = np.unravel_index(np.argmin(dist, axis=None), dist.shape)

                
                    QS_ACM_COM = ACMCOM.quality_status[irec]            if ACMCOM is not None else np.nan
                    QS_BMA_FLX = BMAFLX.quality_status[indlatBMAFLX]    if BMAFLX is not None else np.nan
                    QS_ACM_RT  = ACMRT.quality_status[irec]             if ACMRT  is not None else np.nan
                    QS_ALL_3D  = ALL3D.quality_status[iacr, ial]        if ALL3D  is not None else np.nan
                
                    print(f"OrbitID {OrbitID} over {Station:^20}           QS_ACM_COM {QS_ACM_COM:4}   |  QS_BMA_FLX {QS_BMA_FLX:4}   |  QS_ACM_RT {QS_ACM_RT:4}   |  QS_ALL_3D {QS_ALL_3D:4}")
                    ACM_COM_LIST.append((QS_ACM_COM, OrbitID, Station))
                    BMA_FLX_LIST.append((QS_BMA_FLX, OrbitID, Station))
                    ACM_RT_LIST.append((QS_ACM_RT, OrbitID, Station))
                    ALL_3D_LIST.append((QS_ALL_3D, OrbitID, Station))
                except KeyError:
                    print(f"Failed to extract QS values for OrbitID {OrbitID} over {Station:^15}")
                

        # Write QS to files
        for filename, list in [(filename_ACM_COM, ACM_COM_LIST), (filename_BMA_FLX , BMA_FLX_LIST), (filename_ACM_RT, ACM_RT_LIST), (filename_ALL_3D, ALL_3D_LIST)]:
            with open(filename, "w") as f:
                for QS, OrbitID, Station in list:
                    f.write(f"{QS:<10.0f} {OrbitID} {Station}\n")


       






def plot_scatter(RTMs_fluxes_list, OBS_fluxes): 
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(1,1,1)
    pngfile = 'FIGS/correlation_' + zout[:-1] + '.png'                                                                                                                                                                 
    xlabel_specs = r' [W/m$^2$]'
    xlabel = 'Pyranometer Observations' if WANT_SUR else 'BMA-FLX Observations'
    ylabel = 'RTM' 
    ylabel_specs = xlabel_specs
    title =  f'Surface SW Flux Correlation' if WANT_SUR else f'TOA SW Flux Correlation'
    pl_list = []

    if WANT_ACMRT: 
        pngfile = f'FIGS/correlation_{zout[:-1]}_ACMRT1D.png' 
        xlabel = 'ACM-RT 1D' 
        ylabel = 'DISORT'

    # Extract data to plot
    for RTM_fluxes in RTMs_fluxes_list:
        rte_solver, color, marker = get_solver_info(RTM_fluxes)
        x = read_flux(OBS_fluxes, REMOVE_STATIONS) # model truth
        y = read_flux(RTM_fluxes, REMOVE_STATIONS)

        # Apply QS
        x, y, rte_solver = apply_qs(x, y, qs_ACMCOM, qs_BMAFLX, qs_ACMRT, qs_ALL3D, rte_solver, OBS_fluxes)
        

        


        if WANT_CMF:
            # Plot all x,y points that has a CMF value in the file Cloudiness-file 
            CMF_clearsky_limit      = 0.98
            CFM_heavy_clouds_limit  = 0.4
            color = "gray"
            marker = "o"
            filename = "DATA/CMF_MYSTIC_CompDomain_SUR.txt" 
            cloudiness_data = []
            with open(filename, 'r') as f:
                for line in f:
                    parts = line.split()
                    cloudiness_data.append([float(parts[0])])
            cloudiness_data = np.array(cloudiness_data)
        
            cmf_values = np.asarray(cloudiness_data[:, 0]) # First column contains the CMF values

            # Apply across-track limit
            x,y,cmf_values = across_track_limit(x, y, ACROSS_TRACK_LIMIT, z=cmf_values)
            x, y, cmf_values = filter_out_nans_zeros(x, y, z=cmf_values)
            stats = calculate_statistics(x, y)

            for cmf, x_val, y_val in zip(cmf_values, x, y):
                if cmf > CMF_clearsky_limit:
                    scatter1 = ax.scatter(x_val, y_val, s=14, alpha=.6, c='blue', marker=marker, label='Cloud-free')
                elif cmf < CFM_heavy_clouds_limit:
                    scatter2 = ax.scatter(x_val, y_val, s=14, alpha=.6, c='red', marker=marker, label='Heavily cloudy')
                else:
                    scatter3 = ax.scatter(x_val, y_val, s=14, alpha=.6, c='green', marker=marker, label='Cloudy')
            pl_list.extend([scatter1, scatter3, scatter2])

            pngfile = pngfile[:-4] + '_CMF.png'


           

        else:
            # Filter out SZA above 90 for ACM-RT to match libRad
            if WANT_SUR or ('overlap' in RTM_fluxes):
                # Apply across-track limit
                x, y = across_track_limit(x, y, ACROSS_TRACK_LIMIT)

            if "ACM-RT" in rte_solver: 
                with open("DATA/SZA.txt", "r") as f:
                    sza = np.array([float(line.split()[0]) for line in f])
                mask = sza < 80
                print(f"len(mask) = {len(mask)} len(x) = {len(x)} len(y) = {len(y)}")
                x = x[mask]
                y = y[mask]

            

            x, y  = filter_out_nans_zeros(x, y)
            stats = calculate_statistics(x, y)

                
            want_color_mapping = False # If want colorbars to points
            if want_color_mapping:
                res = y - x
                sc = ax.scatter(x, y, s=8, alpha=.8, c=res, cmap=cmap)

                # Colorbar
                cb = fig.colorbar(
                    sc, ax=ax, 
                    shrink=.8, pad=0.0, 
                    extend='both',
                    extendrect=True)
                cb.set_label(rf"{ylabel} - {xlabel} [W/m$^2$]", size=INFOSIZE)
                cb.ax.tick_params(labelsize=INFOSIZE)

                # soften colorbar box + background
                cb.outline.set_visible(False)             # remove black frame
                cb.ax.set_facecolor('#f7f7f7')            # subtle bg behind ramp
            else: 
                ax.scatter(x, y, s=14, alpha=.6, c=color,marker=marker,) 

            


        # -------- Ranges for plotting --------
        xy_min = 0 #np.nanmin([x.min(), y.min()])
        xy_max = np.nanmax([x.max(), y.max()])
        pad = 0.10 * (xy_max - xy_min if xy_max > xy_min else 1.0)
        lo, hi = xy_min, xy_max + pad #xy_min - pad, xy_max + pad

        # For setting ax.set_ylim/xlim
        ymin, xmin, ymax, xmax, pad = xy_min, xy_min, xy_max, xy_max, 0
        ax.set_xlim(xmin,hi); ax.set_ylim(ymin,hi)


        # --------- Plot Regression -----------
        xx = np.linspace(lo, hi, 100)
        if WANT_ACMRT:  n_decimals = 2 
        else:           n_decimals = 1    
        p, = ax.plot(xx, stats['intercept'] + stats['slope'] * xx, lw=1.5, c=color, label=f"{rte_solver}     y = {stats['intercept']:.{n_decimals}f} + {stats['slope']:.{n_decimals}f}x") # WRITE IN FIG-TEXT: Linear Least Squares (fit)") 
        pl_list.append(p)

        # ---------- Textbox with stats ---------
        data_str = (
            r"$\mathbf{" + rte_solver[:2] + r"\,\,\, " + rte_solver[2:] + "}$" + "\n" +
            f"MBE = {stats['bias']:5.1f}" + r" W/m$^2$" + f"\nSTD = {stats['std']:6.1f}" + r" W/m$^2$" + f"\nRMSE = {stats['rmse']:6.1f}" + r" W/m$^2$"
            f"\nN = {stats['n']}"
        )

        y_position = 0.01 if '1D' in rte_solver else 0.15
        if len(RTMs_fluxes_list) == 1: y_position = 0.01
        fig.text(.99, y_position,          
                data_str,transform=ax.transAxes, ha='right', va='bottom',fontsize=FONTSIZE*.6, color='k',  #color=color,    #
                bbox=dict(boxstyle='round,pad=0.3',facecolor='none',edgecolor=color,linewidth=1.5))



    # y=x line
    p, = ax.plot([lo, hi], [lo, hi], lw=1.2, c='black', linestyle="--", label="y=x")
    pl_list.append(p)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    ax.legend(  handles=pl_list,
                loc='upper left', framealpha=0.7, 
                borderaxespad=0.0,                   # space to axes
                borderpad=0.25, labelspacing=0.25,   # compact box)
                fontsize=INFOSIZE*.8)

    ax.set_xlabel(xlabel + xlabel_specs, fontsize=INFOSIZE)
    ax.set_ylabel(ylabel + ylabel_specs, fontsize=INFOSIZE)


    fig.suptitle(title, fontsize=FONTSIZE, y=0.98)

    # ------------ Spesify ACM-COM baseline ----
    # baseline = get_acm_com_baseline("/homevip/bgre/Download/Frames_SurfaceOverpasses/Oslo")
    # plt.figtext(0.001, 0.003, f"Baseline ACM-COM: {baseline}", fontsize=FONTSIZE*0.45)

    
    style_axes(ax, fig)


    print("pngfile", pngfile)
    plt.savefig(pngfile)
    plt.close()








def plot_scatter_single_station(RTMs_fluxes, OBS_fluxes, station): 
   
    """ NOTE:  Plots without QS """

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(1,1,1)
    pngfile = f'FIGS/STATION/{zout[:-1]}/{station}_correlation_{zout[:-1]}.png'                                                                                                                                               
    xlabel_specs = r' [W/m$^2$]'
    xlabel = 'Pyranometer Observations' if WANT_SUR else 'BMA-FLX Observations'
    ylabel = 'RTM' # rte_solver
    ylabel_specs = xlabel_specs
    title =  f'Surface' if WANT_SUR else f'TOA' + f' SW Correlation         {station}'
    pl_list = []

    # Extract data to plot
    for RTM_fluxes in RTMs_fluxes:
        rte_solver, color, marker = get_solver_info(RTM_fluxes)
        x = []
        y = []
        if "ACMRT" in RTM_fluxes: 
            # Assumes that has already made idx_lines
            with open(RTM_fluxes) as rtm, open(OBS_fluxes) as obs:
                for i, (rtm_line, obs_line) in enumerate(zip(rtm, obs)):
                    if i in idx_lines:
                        rtm_parts = rtm_line.split()
                        y.append(float(rtm_parts[0]))
                        x.append(float(obs_line.split()[0])) 
        else:
            idx_lines = []
            with open(RTM_fluxes) as rtm, open(OBS_fluxes) as obs:
                for i, (rtm_line, obs_line) in enumerate(zip(rtm, obs)):

                    rtm_parts = rtm_line.split()
                    libRadFn = rtm_parts[1]

                    if station in libRadFn:
                        idx_lines.append(i)
                        y.append(float(rtm_parts[0]))
                        x.append(float(obs_line.split()[0])) 
        x, y = np.asarray(x), np.asarray(y)

        if WANT_SUR:
            x, y = across_track_limit(x, y, ACROSS_TRACK_LIMIT, idx_lines=idx_lines)

        x, y = filter_out_nans_zeros(x, y)
        stats = calculate_statistics(x, y)
                

        # -------- Ranges for plotting --------
        xy_min = 0 #np.nanmin([x.min(), y.min()])
        xy_max = np.nanmax([x.max(), y.max()])
        pad = 0.03 * (xy_max - xy_min if xy_max > xy_min else 1.0)
        lo, hi = xy_min, xy_max + pad #xy_min - pad, xy_max + pad

        # For setting ax.set_ylim/xlim
        ymin, xmin, ymax, xmax, pad = xy_min, xy_min, xy_max, xy_max, 0
        ax.set_xlim(xmin,hi); ax.set_ylim(ymin,hi)

        # --------- Plot Regression -----------
        xx = np.linspace(lo, hi, 100)
        p, = ax.plot(xx, stats['intercept'] + stats['slope'] * xx, lw=1.5, c=color, label=f"{rte_solver}     y = {stats['intercept']:.1f} + {stats['slope']:.1f}x") # WRITE IN FIG-TEXT: Linear Least Squares (fit)") 
        pl_list.append(p)


        ax.scatter(x, y, s=14, alpha=.6, c=color, marker=marker) 



        # ---------- Textbox with stats ---------
        solver_str = r"$\mathbf{{DISORT\,\,\, (1D\,\, RTM)}}$" if "1D" in rte_solver else r"$\mathbf{{MYSTIC\,\,\, (3D\,\, RTM)}}$"
        solver_str = rte_solver
        bias_str = r"$\Delta F_{\mathrm{SUR}}^{\downarrow}$ (Bias) = " if WANT_SUR else r"$\Delta F_{\mathrm{TOA}}^{\uparrow}$ (Bias) = "
        bias_str = "Bias = "
        data_str = (
            solver_str + "\n"
            # f"R² = {r2:.2f}                   \n"
            + f"{bias_str} {stats['bias']:5.1f}" + r" $\pm$ " + f"{stats['std']:6.1f}" + r" W/m$^2$"
            # f"\nMAE = {stats['mae']:.1f}\n"
            + f"\nN = {stats['n']}"
        )
        fig.text(.99, 0.01 if 'DISORT' in rte_solver else 0.11,          
                data_str,
                transform=ax.transAxes, 
                ha='right', va='bottom',
                fontsize=FONTSIZE*.7, color='k',  #color=color,    #
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, #facecolor='w',
                alpha=0.5))



    # y=x line
    p, = ax.plot([lo, hi], [lo, hi], lw=1.2, c='red', linestyle="--", label="y=x")
    pl_list.append(p)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    ax.legend(  handles=pl_list,
                loc='upper left', framealpha=0.7, 
                borderaxespad=0.0,                   # space to axes
                borderpad=0.25, labelspacing=0.25,   # compact box)
                fontsize=INFOSIZE*.8)

    ax.set_xlabel(xlabel + xlabel_specs, fontsize=INFOSIZE)
    ax.set_ylabel(ylabel + ylabel_specs, fontsize=INFOSIZE)
    fig.suptitle(title, fontsize=FONTSIZE, y=0.98)

    # ------------ Spesify ACM-COM baseline ----
    # baseline = get_acm_com_baseline("/homevip/bgre/Download/Frames_SurfaceOverpasses/Oslo")
    # plt.figtext(0.001, 0.003, f"Baseline ACM-COM: {baseline}", fontsize=FONTSIZE*0.45)
    
    style_axes(ax, fig)

    print("pngfile", pngfile)
    plt.savefig(pngfile)
    plt.close()







def plot_across_track_BOX(RTMs_fluxes, OBS_fluxes, data_idx=0, step_bins=5, WHICH_PLOT=0): 
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(1,1,1)
    pngfile = 'FIGS/Error_vs_Distance' + zout[:-1] + '.png'                                                                                                                                                   
    title =  f'Error vs Across-Track Distance\nSurface SW Flux'
    xlabel = 'Distance to Satellite Overpass'
    xlabel_specs = r' [km]'
    ylabel_specs =  r' [W/m$^2$]'
    pl_list = []

    

    # Extract distances once
    # D_list = []
    # with open("DATA/Distances.txt", "r") as f:
    #     for line in f:
    #         D_list.append(float(line.split()[0]))
    # D_list = np.asarray(D_list)
    D_list = np.loadtxt("DATA/Distances.txt", usecols=0)
    



    
    for RTM_fluxes in RTMs_fluxes:
        rte_solver, color, marker = get_solver_info(RTM_fluxes)
        x = read_flux(OBS_fluxes, REMOVE_STATIONS) # model truth
        y = read_flux(RTM_fluxes, REMOVE_STATIONS)

        x, y, rte_solver = apply_qs(x, y, qs_ACMCOM, qs_BMAFLX, qs_ACMRT, qs_ALL3D, rte_solver, OBS_fluxes)
        x, y = across_track_limit(x, y, ACROSS_TRACK_LIMIT)
        x, y, D_filtered = filter_out_nans_zeros(x, y, D_list)
        stats = calculate_statistics(x, y)
                
        
      
        data            = [stats['diff'],                                       stats['absolute_diff']  , stats['relative_diff']][data_idx]
        ylabel          = [r'$\Delta F_{\mathrm{SUR}}^{\downarrow}$ (Bias)',    'Absolute Error'        , "Relative Error"      ][data_idx]
        textboxlabel    = ["Bias",                                              "MAE"                   , "Rel. Bias"           ][data_idx]
        ylabel_specs    = [ylabel_specs,                                        ylabel_specs            , " [%]"                ][data_idx]
    
        bins = np.arange(0, ACROSS_TRACK_LIMIT + step_bins, step_bins) #make bins go with steps of 10 from 0 to ACROSS_TRACK_LIMIT
    
        
        box_data = []
    
        for i in range(len(bins) - 1):
            mask = (
                (D_filtered >= bins[i]) &
                (D_filtered < bins[i + 1]) &
                np.isfinite(data)
            )
            box_data.append(data[mask])

        positions = np.arange(len(box_data))
        # positions = np.arange(len(box_data)) + (-0.18 if 'MYSTIC' in rte_solver else 0.18)
        
        # Boxplot
        if WHICH_PLOT == 0:     
            bp = ax.boxplot(
                box_data,
                positions=positions,
                widths=0.3,
                patch_artist=True,
                showfliers=False,
                boxprops=dict(facecolor=color, alpha=0.5),
                medianprops=dict(color='none'),
                whiskerprops=dict(color=color),
                capprops=dict(color=color)
            )
            # Add data-point per box
            for pos, values, whiskers in zip(positions, box_data, bp['caps'][1::2]): 
                # bp['caps'] = [lower_1, upper_1, lower_2, upper_2, lower_3, upper_3, ...]
                if len(values) > 0:
                    y = whiskers.get_ydata()[1]
                    ax.text(
                        pos, y, f'N = {len(values)}',
                        ha='center', va='bottom',
                        fontsize=INFOSIZE * 0.65
                    )
        # Violin plot
        elif WHICH_PLOT == 1:
            for pos, values in zip(positions, box_data):
                vp = ax.violinplot(
                    [values], positions=[pos],
                    widths=0.3,showmeans=False, showmedians=True, showextrema=False,
                )
                for body in vp['bodies']:
                    body.set_facecolor(color)
                    body.set_edgecolor(color)
                    body.set_alpha(0.5)
                vp['cmedians'].set_color('black')

                ax.scatter(
                    np.full(len(values), pos),
                    values,
                    s=8,
                    color=color,
                    alpha=0.5,
                    zorder=3,
                )

                ax.text(
                    pos, np.max(values), f'N = {len(values)}',
                    ha='center', va='bottom',
                    fontsize=INFOSIZE * 0.65,
                )
        # Distribution plot  (e.g., histogram)
        elif WHICH_PLOT == 2:
            for pos, values in zip(positions, box_data):
                values = np.sqrt(np.mean(values**2)) # RMSE
                ylabel = 'RMSE'
                ax.bar(pos, values, width=1.0,color=color,alpha=0.5,linewidth=0.3)

        pl_list.append(Patch(facecolor=color, alpha=0.5, label=rte_solver))


    
       
    # Reference line at zero bias
    ax.axhline(0, color='gray', linewidth=1)    
    ax.margins(y=0.1) # leave space on top and bottom
    ymin, ymax = ax.get_ylim()
    ymax = max(abs(ymin), abs(ymax))
    # ymax = 2 if data_idx == 2 else 150 # set fixed
    ax.set_ylim(-ymax, ymax)
    if data_idx == 1 or WHICH_PLOT == 2: # MAB or Distribution plot
        ax.set_ylim(0, ymax)

    
    ax.legend(  handles=pl_list,
                # loc='upper left', 
                framealpha=0.7, 
                borderaxespad=0.0,                   # space to axes
                borderpad=0.25, labelspacing=0.25,   # compact box)
                fontsize=INFOSIZE*.8)



    ax.set_xticks(np.arange(len(bins)-1))
    # if WANT_MANY_BINS:  
    #     # ax.set_xticklabels(['0-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30-35', '35-40'])
    #     ax.set_xticklabels(['0-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30-35', '35-40'])
    # else:               
    #     # ax.set_xticklabels(['0-10', '10-20', '20-30', '30-40'])
    #     # ax.set_xticklabels(['0-10', '10-20', '20-30', '30-40', '40-50', '50-60'])
    #     ax.set_xticklabels(['0-10', '10-20', '20-30', '30-40', '40-50'])
    ax.set_xticklabels([f'{bins[i]}-{bins[i+1]}' for i in range(len(bins)-1)])

    ax.set_xlabel(xlabel + xlabel_specs, fontsize=INFOSIZE)
    ax.set_ylabel(ylabel + ylabel_specs, fontsize=INFOSIZE)
    fig.suptitle(title, fontsize=FONTSIZE, y=0.98)

    # ------------ Spesify ACM-COM baseline ----
    # baseline = get_acm_com_baseline("/homevip/bgre/Download/Frames_SurfaceOverpasses/Oslo")
    # plt.figtext(0.001, 0.003, f"Baseline ACM-COM: {baseline}", fontsize=FONTSIZE*0.45)
    
    style_axes(ax, fig)

    print("pngfile", pngfile)
    plt.savefig(pngfile)
    plt.close()






def plot_across_track_scatter(RTMs_fluxes, OBS_fluxes, data_idx=0, step_bins=5): 
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(1,1,1)
    pngfile = 'FIGS/Error_vs_Distance_SCATTER_' + zout[:-1] + '.png'                                                                                                                                                  
    title =  f'Error vs Across-Track Distance\nSurface SW Flux'
    xlabel = 'Distance to Satellite Overpass'
    xlabel_specs = r' [km]'
    ylabel_specs =  r' [W/m$^2$]'
    pl_list = []

    # Extract distances once
    D_list = []
    with open("DATA/Distances.txt", "r") as f:
        for line in f:
            D_list.append(float(line.split()[0]))
    D_list = np.asarray(D_list)
    # Extract data to plot
    for RTM_fluxes in RTMs_fluxes:
        rte_solver, color, marker = get_solver_info(RTM_fluxes)
        x = read_flux(OBS_fluxes, REMOVE_STATIONS) # model truth
        y = read_flux(RTM_fluxes, REMOVE_STATIONS)
        x, y, rte_solver = apply_qs(x, y, qs_ACMCOM, qs_BMAFLX, qs_ACMRT, qs_ALL3D, rte_solver, OBS_fluxes)
        x, y = across_track_limit(x, y, ACROSS_TRACK_LIMIT)
        x, y, D_filtered = filter_out_nans_zeros(x, y, D_list)
        stats = calculate_statistics(x, y)

        data            = [stats['diff'],                                       stats['absolute_diff']  , stats['relative_diff']][data_idx]
        ylabel          = [r'$\Delta F_{\mathrm{SUR}}^{\downarrow}$ (Bias)',    'Absolute Error'        , "Relative Error"      ][data_idx]
        textboxlabel    = ["Bias",                                              "MAE"                   , "Rel. Bias"           ][data_idx]
        ylabel_specs    = [ylabel_specs,                                        ylabel_specs            , " [%]"                ][data_idx]


        WANT_TREND = True
        if WANT_TREND:
            WANT_RMSE = True # else: MAB per distance bin
            if WANT_RMSE:
                data = np.asarray(stats['diff'])
                ylabel = 'RMSE'
            else: 
                data = np.asarray(stats['absolute_diff'])
                ylabel = 'MAE'

            textboxlabel = ylabel
            ylabel_specs = r' [W/m$^2$]'
            

            pngfile = pngfile.replace('.png', '_TREND.png')
            distances = np.asarray(D_filtered)

            data_binned  = []
            D_binned     = []
            error_spread = []

            for lower in np.arange(0, np.nanmax(distances) + step_bins, step_bins):
                upper = lower + step_bins
                mask = (distances >= lower) & (distances < upper)
                values = data[mask]

                if not np.any(np.isfinite(values)):
                    continue

                if WANT_RMSE:
                    bin_value = np.sqrt(np.nanmean(values**2))
                else:
                    bin_value = np.nanmean(np.abs(values))

                data_binned.append(bin_value)
                D_binned.append((lower + upper) / 2)
                error_spread.append(np.nanstd(values))

            p, = ax.plot(
                D_binned, data_binned,
                marker='o', lw=1, color=color, alpha=0.5,
                label=f"{rte_solver}"
            )
            # Plot error spread as shaded area around the binned data
            WANT_ERROR_SPREAD = True
            if WANT_ERROR_SPREAD:
                data_binned = np.array(data_binned)
                error_spread = np.array(error_spread)
                ax.fill_between(
                    D_binned,
                    np.maximum(0, data_binned - error_spread),
                    data_binned + error_spread,
                    color=color,   # Uses the solver's color; use "blue" to force blue
                    alpha=0.2,
                    zorder=1,
                )
            pl_list.append(p)

            # Fix data values for calculating textbox value on figure
            if WANT_RMSE: textbox_value = np.sqrt(np.nanmean(data**2))
            else: textbox_value = np.nanmean(np.abs(data))
        else:
            p = ax.scatter(D_filtered, data,  lw=1, c=color, alpha=.5, label=f"{rte_solver}") 
            pl_list.append(p)
            textbox_value = np.nanmean(data)


            WANT_REGRESSION = True
            if WANT_REGRESSION:
                b, a = np.polyfit(D_filtered, data, 1)
                # --------- Plot Regression -----------
                xx = np.linspace(0, np.nanmax(D_filtered), 100)
                p, = ax.plot(xx, a + b * xx, lw=1.5, c=color, label=f"{rte_solver}     y = {a:.1f} + {b:.1f}x") # WRITE IN FIG-TEXT: Linear Least Squares (fit)") 
                pl_list.append(p)

            # ---------- Textbox with stats ---------
            data_str = (
                r"$\mathbf{" + rte_solver[:2] + r"\,\,\, " + rte_solver[2:] + "}$" + "\n" +
                f"MBE = {stats['bias']:5.1f}" + r" W/m$^2$" + f"\nSTD = {stats['std']:6.1f}" + r" W/m$^2$" + f"\nRMSE = {stats['rmse']:6.1f}" + r" W/m$^2$"
                f"\nN = {stats['n']}"
            )
            
            y_position = 0.87 if "3D" in rte_solver else 0.73   # 0.01 if '1D' in rte_solver else 0.15
            if len(RTMs_fluxes) == 1: y_position = 0.87
            fig.text(.99, y_position,          
                    data_str,transform=ax.transAxes, ha='right', va='bottom',fontsize=FONTSIZE*.6, color='k',  #color=color,    #
                    bbox=dict(boxstyle='round,pad=0.3',facecolor='none',edgecolor=color,linewidth=1.5))
        



       
    ax.legend(  handles=pl_list,
                loc='upper left', 
                framealpha=0.7, 
                borderaxespad=0.0,                   # space to axes
                borderpad=0.25, labelspacing=0.25,   # compact box)
                fontsize=INFOSIZE*.8)
    
    ax.set_xlabel(xlabel + xlabel_specs, fontsize=INFOSIZE)
    ax.set_ylabel(ylabel + ylabel_specs, fontsize=INFOSIZE)
    fig.suptitle(title, fontsize=FONTSIZE, y=0.98)

    # ------------ Spesify ACM-COM baseline ----
    # baseline = get_acm_com_baseline("/homevip/bgre/Download/Frames_SurfaceOverpasses/Oslo")
    # plt.figtext(0.001, 0.003, f"Baseline ACM-COM: {baseline}", fontsize=FONTSIZE*0.45)
    
    style_axes(ax, fig)

    print("pngfile", pngfile)
    plt.savefig(pngfile)
    plt.close()






def plot_error_vs_latitude_scatter(RTMs_fluxes, OBS_fluxes):
    FIGSIZE = (12,8)
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(1,1,1)
    pngfile = 'FIGS/Error_vs_Latitude' + zout[:-1] + '.png'                                                    

    if WANT_SUR:
        title =  f'Error vs Latitude\nSurface SW Flux'      
    else: 
        title =  f'Error vs Latitude\nTOA SW Flux'
    xlabel = 'Latitude (Station)'
    xlabel_specs = r' [°N]'
    ylabel_specs =  r' [W/m$^2$]'
    pl_list = []


    # Extract latitudes for each simulated station-pixel
    lat_list = [] # List of (Distances, OrbitID, Station)
    with open(RTMs_fluxes[0]) as file:
        for lines in file:
            line = lines.strip() 
            # Ex line: ./RESULTS/libRad_v01_disort_1D_SWIA_solar_01028B_Oslo_SUR.nc 
            parts = line.split("_")
            OrbitID = parts[-3]
            Station = parts[-2]
            lat, _ = sites[Station]
            lat_list.append(lat)
        lat_list = np.asarray(lat_list)

    
    # Extract data to plot
    for RTM_fluxes in RTMs_fluxes:
        rte_solver, color, marker = get_solver_info(RTM_fluxes)
        x = read_flux(OBS_fluxes, REMOVE_STATIONS) # model truth
        y = read_flux(RTM_fluxes, REMOVE_STATIONS)
        x, y, rte_solver = apply_qs(x, y, qs_ACMCOM, qs_BMAFLX, qs_ACMRT, qs_ALL3D, rte_solver, OBS_fluxes)
        x, y = across_track_limit(x, y, ACROSS_TRACK_LIMIT)
        x, y, lat_filtered = filter_out_nans_zeros(x, y, lat_list)
        stats = calculate_statistics(x, y)


        ######## Choose what data to plot ########
        data_idx = 0
            # 0: Bias
            # 1: MAB
            # 2: Rel. Bias
        ##########################################
        data            = [stats['diff'],                                       stats['absolute_diff']  , stats['relative_diff']][data_idx]
        ylabel          = [r'$\Delta F_{\mathrm{SUR}}^{\downarrow}$ (Bias)',    'Absolute Error'        , "Relative Error"      ][data_idx]
        textboxlabel    = ["Bias",                                              "MAE"                   , "Rel. Bias"           ][data_idx]
        ylabel_specs    = [ylabel_specs,                                        ylabel_specs            , " [%]"                ][data_idx]

        p = ax.scatter(lat_filtered, data,  lw=1, c=color, alpha=.5, label=f"{rte_solver}") 
        pl_list.append(p)

        WANT_REGRESSION = True
        if WANT_REGRESSION:
            b, a = np.polyfit(lat_filtered, data, 1)
            # --------- Plot Regression -----------
            xx = np.linspace(0, np.nanmax(lat_filtered), 100)
            p, = ax.plot(xx, a + b * xx, lw=1.5, c=color, label=f"{rte_solver}     y = {a:.1f} + {b:.1f}x") # WRITE IN FIG-TEXT: Linear Least Squares (fit)") 
            pl_list.append(p)
        


    

        # ---------- Textbox with stats ---------
        solver_str = r"$\mathbf{{DISORT\,\,\, (1D\,\, RTM)}}$" if "1D" in rte_solver else r"$\mathbf{{MYSTIC\,\,\, (3D\,\, RTM)}}$"
        solver_str = rte_solver
        data_str = solver_str + "\n"+ ylabel + f" = {np.nanmean(data):.1f}"
        if data_idx==0: 
            data_str += r" $\pm$ " + f"{np.nanstd(data):.1f}" 
        data_str += r" W/m$^2$" if data_idx!=2 else " %"
        data_str += f"\nN = {len(data)}"
        
        fig.text(0.99, 0.90, #.99, 0.85 if '1D' in rte_solver else 0.93,          
                data_str,
                transform=ax.transAxes, 
                ha='right', va='bottom',
                fontsize=FONTSIZE*.7, color='k',  #color=color,    #
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, #facecolor='w',
                alpha=0.5))
    

    
    ax.legend(  
        handles=pl_list,
        loc='upper left', 
        framealpha=0.7, 
        borderaxespad=0.0,                   # space to axes
        borderpad=0.25, labelspacing=0.25,   # compact box)
        fontsize=INFOSIZE*.8)


    # Reference line at zero bias
    ax.axhline(0, color='gray', linewidth=1)
    ax.margins(y=0.1)
    ymin, ymax = ax.get_ylim()
    ymax = max(abs(ymin), abs(ymax))
    ax.set_ylim(-ymax, ymax)
    

    # Labels
    ax.set_xlim(59, 77)
    ax.set_xticks(np.arange(60, 78, 2.5))

    # Station names below the axis
    y = ax.get_ylim()[0] + 5

    # y = -200 #np.nanmin(data) #- 200
    # Small vertical offsets for overlapping labels
    offsets = {
            'Oslo': 0,
            'Østerås': 70,
    
            'Flesland': 0,
            'Bergen': 120,
    
            'Trondheim': 0,
            'Værnes': 150,
    
            'Filefjell': 0,
            'Beitostølen': 90,
            'Rena': 0,
    
            'Gratangen': 0,
            'Iskoras': 0,
            'Karasjok': 90,
            'Tromsø': 0,
            'Korgåsen': 0,
        }

    for station, (lat, _) in sites.items():
        ax.annotate(
            station,
            xy=(lat, y + offsets.get(station, 0)),
            ha='center',
            va='bottom',
            fontsize=INFOSIZE * 0.6,
            rotation=90
        )

    ax.set_xlabel(xlabel + xlabel_specs, fontsize=INFOSIZE)
    ax.set_ylabel(ylabel + ylabel_specs, fontsize=INFOSIZE)
    fig.suptitle(title, fontsize=FONTSIZE, y=0.98)

    # ------------ Spesify ACM-COM baseline ----
    # baseline = get_acm_com_baseline("/homevip/bgre/Download/Frames_SurfaceOverpasses/Oslo")
    # plt.figtext(0.001, 0.003, f"Baseline ACM-COM: {baseline}", fontsize=FONTSIZE*0.45)
    
    style_axes(ax, fig)

    print("pngfile", pngfile)
    plt.savefig(pngfile)
    plt.close()







def plot_error_vs_latitude_BOX(RTMs_fluxes, OBS_fluxes):
    FIGSIZE = (15,8)
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(1,1,1)
    pngfile = 'FIGS/Error_vs_Latitude_BOX_' + zout[:-1] + '.png'                                                    

    if WANT_SUR:
        title =  f'Error vs Latitude\nSurface SW Flux'      
    else: 
        title =  f'Error vs Latitude\nTOA SW Flux'
    xlabel = 'Latitude (Station)'
    xlabel_specs = r' [°N]'
    ylabel_specs =  r' [W/m$^2$]'
    pl_list = []


    # Extract latitudes for each simulated station-pixel
    lat_list = [] # List of (Distances, OrbitID, Station)
    with open(RTMs_fluxes[0]) as file:
        for lines in file:
            line = lines.strip() 
            # Ex line: ./RESULTS/libRad_v01_disort_1D_SWIA_solar_01028B_Oslo_SUR.nc 
            parts = line.split("_")
            OrbitID = parts[-3]
            Station = parts[-2]
            lat, _ = sites[Station]
            lat_list.append(lat)
        lat_list = np.asarray(lat_list)

    
    # Extract data to plot
    for RTM_fluxes in RTMs_fluxes:
        rte_solver, color, marker = get_solver_info(RTM_fluxes)
        x = read_flux(OBS_fluxes, REMOVE_STATIONS) # model truth
        y = read_flux(RTM_fluxes, REMOVE_STATIONS)
        x, y, rte_solver = apply_qs(x, y, qs_ACMCOM, qs_BMAFLX, qs_ACMRT, qs_ALL3D, rte_solver, OBS_fluxes)
        x, y = across_track_limit(x, y, ACROSS_TRACK_LIMIT)
        x, y, lat_filtered = filter_out_nans_zeros(x, y, lat_list)
        stats = calculate_statistics(x, y)

        ######## Choose what data to plot ########
        data_idx = 0
            # 0: Bias
            # 1: MAB
            # 2: Rel. Bias
        ##########################################
        data            = [stats['diff'],                                       stats['absolute_diff']  , stats['relative_diff']][data_idx]
        ylabel          = [r'$\Delta F_{\mathrm{SUR}}^{\downarrow}$ (Bias)',    'Absolute Error'        , "Relative Error"      ][data_idx]
        textboxlabel    = ["Bias",                                              "MAE"                   , "Rel. Bias"           ][data_idx]
        ylabel_specs    = [ylabel_specs,                                        ylabel_specs            , " [%]"                ][data_idx]

        station_lats = [lat for lat, lon in sites.values()]
        station_names = list(sites.keys())

        box_data = []

        for station_lat in station_lats:
            mask = ((lat_filtered == station_lat) & np.isfinite(data))
            box_data.append(data[mask])
    
        # Boxplot
        positions = np.array(station_lats) #+ (-0.18 if 'MYSTIC' in rte_solver else 0.18)
        bp = ax.boxplot(
            box_data,
            positions=positions,
            widths=0.15,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor=color, alpha=0.5),
            medianprops=dict(color='none'),
            whiskerprops=dict(color=color),
            capprops=dict(color=color)
        )
        # Add data-point per box
        for pos, values, whiskers in zip(positions, box_data, bp['caps'][1::2]): 
            # bp['caps'] = [lower_1, upper_1, lower_2, upper_2, lower_3, upper_3, ...]
            if len(values) > 0:
                y = whiskers.get_ydata()[1]
                ax.text(
                    pos, y, f'N = {len(values)}',
                    ha='center', 
                    va='bottom',
                    fontsize=INFOSIZE * 0.5
                )

    
        for station, values in zip(station_names, box_data):
            print(f"{station:15} {len(values)}")
        print()
        
        # ---------- Textbox with stats ---------
        solver_str = r"$\mathbf{{DISORT\,\,\, (1D\,\, RTM)}}$" if "1D" in rte_solver else r"$\mathbf{{MYSTIC\,\,\, (3D\,\, RTM)}}$"
        solver_str = rte_solver
        data_str = solver_str + "\n"+ ylabel + f" = {np.nanmean(data):.1f}"
        if data_idx==0: 
            data_str += r" $\pm$ " + f"{np.nanstd(data):.1f}" 
        data_str += r" W/m$^2$" if data_idx!=2 else " %"
        data_str += f"\nN = {len(data)}"
        
        fig.text(0.995, 0.91, #.99, 0.85 if '1D' in rte_solver else 0.93,          
                data_str,
                transform=ax.transAxes, 
                ha='right', va='bottom',
                fontsize=FONTSIZE*.7, color='k',  #color=color,    #
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, #facecolor='w',
                alpha=0.5))

        pl_list.append(Patch(facecolor=color, alpha=0.5, label=rte_solver))
    
    ax.legend(  
        handles=pl_list,
        loc='upper left', 
        framealpha=0.7, 
        borderaxespad=0.0,                   # space to axes
        borderpad=0.25, labelspacing=0.25,   # compact box)
        fontsize=INFOSIZE*.8)

    # Reference line at zero bias
    ax.axhline(0, color='gray', linewidth=1)
    ax.margins(y=0.1) # leave space on top and bottom
    ymin, ymax = ax.get_ylim()
    ymax = max(abs(ymin), abs(ymax))
    # ymax = 380 # 3.7 if data_idx == 2 else 370 # set fixed
    ax.set_ylim(-ymax, ymax)
    

    # Labels
    ax.set_xlim(59, 77)
    ax.set_xticks(np.arange(60, 78, 2.5))

    # Station names below the axis
    y = ax.get_ylim()[0] 
    y -= y*0.01

    # Small vertical offsets for overlapping labels
    offsets = {
        'Oslo': 0,
        'Østerås': 40,

        'Flesland': 0,
        'Bergen': 60,

        'Trondheim': 0,
        'Værnes': 70,

        'Filefjell': 0,
        'Beitostølen': 50,
        'Rena': 0,

        'Gratangen': 0,
        'Iskoras': 0,
        'Karasjok': 40,
        'Tromsø': 0,
        'Korgåsen': 0,
    }
    # offset_factor = 0.01 if data_idx == 2 else 1

    for station, (lat, _) in sites.items():
        ax.annotate(
            station,
            xy=(lat, y + (offsets.get(station, 0))),
            ha='center',
            va='bottom',
            fontsize=INFOSIZE * 0.6,
            rotation=90
        )

    ax.set_xlabel(xlabel + xlabel_specs, fontsize=INFOSIZE)
    ax.set_ylabel(ylabel + ylabel_specs, fontsize=INFOSIZE)
    fig.suptitle(title, fontsize=FONTSIZE, y=0.98)

    # ------------ Spesify ACM-COM baseline ----
    # baseline = get_acm_com_baseline("/homevip/bgre/Download/Frames_SurfaceOverpasses/Oslo")
    # plt.figtext(0.001, 0.003, f"Baseline ACM-COM: {baseline}", fontsize=FONTSIZE*0.45)

    style_axes(ax, fig)

    print("pngfile", pngfile)
    plt.savefig(pngfile)
    plt.close()





def plot_error_per_station(RTMs_fluxes, OBS_fluxes, WHICH_PLOT=0):
    title = "Error per Station\n" + ("Surface SW Flux" if WANT_SUR else "TOA SW Flux")
    station_names = [
        name for name in sites if name not in REMOVE_STATIONS
    ]
    station_ids = {name: i + 1 for i, name in enumerate(sites)}
    positions = np.arange(len(station_names))

    fig, ax = plt.subplots(figsize=(8, 15))
    handles = []
    nsolvers = len(RTMs_fluxes)
    width = 0.7 / max(nsolvers, 1)

    for solver_index, RTM_fluxes in enumerate(RTMs_fluxes):
        rte_solver, color, marker = get_solver_info(RTM_fluxes)

        # Match station IDs to the entries read by read_flux.
        with open(RTM_fluxes) as file:
            entries = [line.strip() for line in file if line.strip()]

        entry_stations = [
            line.rsplit("_", 2)[-2] for line in entries
        ]
        ids = np.array([station_ids[name] for name in entry_stations])

        x = read_flux(OBS_fluxes, REMOVE_STATIONS)
        y = read_flux(RTM_fluxes, REMOVE_STATIONS)
        x, y, rte_solver = apply_qs(x, y, qs_ACMCOM, qs_BMAFLX, qs_ACMRT, qs_ALL3D, rte_solver, OBS_fluxes)
        x, y = across_track_limit(x, y, ACROSS_TRACK_LIMIT)
        x, y, ids = filter_out_nans_zeros(x, y, ids)

        errors = y - x
        offset = (solver_index - (nsolvers - 1) / 2) * 0.2

        for row, station in enumerate(station_names):
            values = errors[(ids == station_ids[station]) & np.isfinite(errors)]

            # Boxplot 
            if WHICH_PLOT == 0:
                ax.boxplot(
                    [values], positions=[row + offset],
                    orientation="horizontal", widths=width * 0.4, manage_ticks=False, patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor=color, alpha=0.5), medianprops=dict(color="black"), whiskerprops=dict(color=color), capprops=dict(color=color))

            # Violin plot
            if WHICH_PLOT == 1:
                offset = offset 
                vp = ax.violinplot(
                    [values], positions=[row + offset],
                    orientation="horizontal",
                    widths=width * 0.4,
                    showmeans=False,
                    showmedians=True,
                    showextrema=False,
                )
                for body in vp["bodies"]:
                    body.set_facecolor(color)
                    body.set_edgecolor(color)
                    body.set_alpha(0.5)

                vp["cmedians"].set_color("black")


            ax.text(
                0.95, row + offset - 0.04,
                f"N = {len(values)}",
                transform=ax.get_yaxis_transform(),
                ha="right", va="center",
                fontsize=INFOSIZE * 0.5,
                color=color,
            )

        handles.append(Patch(facecolor=color, alpha=0.5, label=rte_solver))

    ax.axvline(0, color="gray", linewidth=1)
    ax.set_yticks(positions)
    ax.set_yticklabels(station_names)
    ax.set_ylim(len(station_names) - 0.5, -0.5)

    lower, upper = ax.get_xlim()
    limit = max(abs(lower), abs(upper), 1)
    ax.set_xlim(-limit, limit)

    level = "SUR" if WANT_SUR else "TOA"
    ax.set_xlabel(r"Bias [W/m$^2$]",fontsize=INFOSIZE,)
    ax.set_ylabel("Station", fontsize=INFOSIZE)
    ax.set_title(title, fontsize=FONTSIZE,)
    ax.legend(handles=handles, fontsize=INFOSIZE * 0.8)

    style_axes(ax, fig)
    ax.yaxis.set_minor_locator(plt.NullLocator())

    pngfile = "FIGS/Error_per_station_" + zout[:-1] + ".png"
    plt.savefig(pngfile)
    plt.close(fig)























########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################


if __name__ == "__main__":
    # Station coordinates
    sites = {
        "Oslo":        (59.942, 10.720),
        "Karasjok":    (69.464, 25.502),
        "Gratangen":   (68.732, 17.241),
        "Beitostølen": (61.251, 8.923),
        "Værnes":      (63.460, 10.931),
        "Østerås":     (59.948, 10.603),
        "Trondheim":   (63.415, 10.407),
        "Tromsø":      (69.654, 18.937),
        "Bergen":      (60.383, 5.333),
        "Hopen":       (76.510, 25.013),
        "Brusdalen":   (62.485, 6.480),
        "Jan-Mayen":   (70.939, -8.669),
        "Flesland":    (60.289, 5.227),
        "Iskoras":     (69.300, 25.346),
        "Rena":        (61.376, 11.499),
        "Korgåsen":    (69.936, 28.377),
        "Bjørnøya":    (74.504, 18.998),
        "Filefjell":   (61.178, 8.113),
        "Juvvasshøe":  (61.678, 8.369),
    }

    REMOVE_STATIONS = ["Rena", "Korgåsen", "Iskoras", "Hopen", "Bjørnøya", "Gratangen", "Tromsø"]
    # Stations to be removed from the plots (validation relied on clear-sky fluxes over an full day)
    ACROSS_TRACK_LIMIT = 40
    # Longest across-track distance to consider in plots



    WANT_SUR    = True

    WANT_ACMRT  = False 
        # ref: ACM-RT or Observations

    WANT_CMF = True
        # CMF color mapping scatter plot

    

   
    WANT_RTM = 1
    """
            0: DISORT
            1: MYSTIC
            2: DISORT & MYSTIC
            3: DISORT & ACMRT1D
            4: MYSTIC1e6 & MYSTIC1e7
            5: ACMRT1D_ClearSky
            6: DISORT & ACMRT1D_ClearSky 
            7: MYSTIC_CompDomain & MYSTIC_ClearSky_CompDomain
            8: DISORT_overlap & MYSTIC_overlap
    """

    SELECT_PLOT = 0
    """
            0: scatter plot
            1: scatter plot per station
            2: Error vs. across-track 
            3: Error vs. latitude
            4: Error per station
            5: Write info-file
    """
    

    




    # ------------------------ FILES -------------------------------------------------------------------------------------
    zout = 'SUR_' if WANT_SUR else 'TOA_' 

    RTMs_fluxes = [
            [f'DATA/{zout}DISORT.txt'],
            [f'DATA/{zout}MYSTIC.txt'],
            [f'DATA/{zout}MYSTIC.txt', f'DATA/{zout}DISORT.txt'],
            [f'DATA/{zout}DISORT.txt', f'DATA/{zout}ACMRT1D.txt'],
            [f'DATA/{zout}MYSTIC_1e6.txt', f'DATA/{zout}MYSTIC.txt'],
            [f'DATA/{zout}ACMRT1D_ClearSky.txt'],
            [f'DATA/{zout}DISORT.txt', f'DATA/{zout}ACMRT1D_ClearSky.txt'],
            [f'DATA/{zout}MYSTIC_CompDomain.txt', f'DATA/{zout}MYSTIC_ClearSky_CompDomain.txt'],
            [f'DATA/{zout}DISORT_overlap.txt', f'DATA/{zout}MYSTIC_overlap.txt'],

        ][WANT_RTM]
            #####################################################
            #                       Specs                       #
            # DISORT.txt and MYSTIC.txt: ACMCOM QS 0: Good      #
            # TOA_Observation.txt: SmallResolution, qs = 0,2    #
            #####################################################
  

    OBS_fluxes = 'DATA/SUR_Observations_3minMean_QualityFiltered.txt' if WANT_SUR else 'DATA/TOA_Observations.txt'           
    # Note: QualityFiltered -> remove Beintostølen obs from 18.june 2026                                      
    if WANT_ACMRT:
        OBS_fluxes = 'DATA/SUR_ACMRT1D.txt' if WANT_SUR else 'DATA/TOA_ACMRT1D.txt' 
        
    



    # Extract QS-products
    qs_ACMCOM, qs_BMAFLX, qs_ACMRT, qs_ALL3D = get_quality_status("DATA/QS_ACM_COM.txt", "DATA/QS_BMA_FLX.txt", "DATA/QS_ACM_RT.txt", "DATA/QS_ALL_3D.txt")
    # QS_ACMCOM: 0, 1       -> Good + Good with warnings
    # QS_BMAFLX: 0, 2       -> Both thermal and solar fluxes fully trustful + Only solar flux fully trustful
    # QS_ACMRT:  0, 1       -> Very good + Good
    # QS_ALL3D:  0, 1, 2    -> Good construction conditions + limited search distance due to frame edge + limited search distance due to large variability of solar zenith angle    (not use: 3. presence of unreliable data (filled data) during the cost calculation, 4. presence of margin)
                                                  
    



    ##################### TESTING #######################
    # RTMs_fluxes = ['DATA/SUR_ACMRT1D_Nadir.txt']
    # RTMs_fluxes = ['DATA/SUR_ACMRT1D_Nadir.txt', 'DATA/SUR_ACMRT1D.txt']
    # RTMs_fluxes = ['DATA/SUR_ACMRT1D.txt']
    #####################################################
    # ------------------------- PLOTS  --------------------------------------------------------------------------------------




    if SELECT_PLOT == 0: 
        plot_scatter(RTMs_fluxes, OBS_fluxes)

    elif SELECT_PLOT == 1: 
        stations = [
            "Oslo",     
            "Karasjok", 
            "Gratangen",
            "Beitostølen",           
            "Værnes",   
            "Østerås",  
            "Trondheim",
            "Tromsø",   
            "Bergen",   
            "Hopen",    
            "Brusdalen",
            "Jan-Mayen",
            "Flesland", 
            "Iskoras",  
            "Rena",     
            "Korgåsen", 
            "Bjørnøya", 
            "Filefjell",
            "Juvvasshøe",                  
        ]
        for station in stations:
            plot_scatter_single_station(RTMs_fluxes, OBS_fluxes, station)
            


    elif SELECT_PLOT == 2: 
        if WANT_SUR: 
            # Chose statistic
            data_idx = 0
                # 0: Bias
                # 1: MAB
                # 2: Rel. Bias
         
            # Chose distance bins
            step_bins = 5
            # step_bins = 10

            # Chose plot type
            WHICH_PLOT = 1
                # 0: Box
                # 1: Violin
                # 2: Distribution

            plot_across_track_BOX(RTMs_fluxes, OBS_fluxes, data_idx=data_idx, step_bins=step_bins, WHICH_PLOT=WHICH_PLOT)
            plot_across_track_scatter(RTMs_fluxes, OBS_fluxes, data_idx=data_idx, step_bins=step_bins)
        else:
            print("Need to select WANT_SUR")

    elif SELECT_PLOT == 3:
        plot_error_vs_latitude_scatter(RTMs_fluxes, OBS_fluxes)
        plot_error_vs_latitude_BOX(RTMs_fluxes, OBS_fluxes)

    elif SELECT_PLOT == 4:
        WHICH_PLOT = 0
            # 0: Box
            # 1: Violin 
        plot_error_per_station(RTMs_fluxes, OBS_fluxes, WHICH_PLOT=WHICH_PLOT)

    elif SELECT_PLOT == 5:
        # property = "SZA"
        # property = "Distance"
        property = "CMF"
        # property = "overlapping_TOA_SUR_points"; RTMs_fluxes  =  [f'DATA/TOA_DISORT.txt', f'DATA/TOA_MYSTIC.txt', f'DATA/SUR_DISORT.txt', f'DATA/SUR_MYSTIC.txt']
        # property = "QS"
        write_info_to_file(RTMs_fluxes, property=property)