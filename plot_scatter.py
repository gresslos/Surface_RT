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

import Read_RTM as ReadEC
import Find_Overpass_Info as Find_Overpass_Info

FONTSIZE=15
INFOSIZE=13
FIGSIZE=(8,8)
cmap = plt.get_cmap('viridis') 



def read_flux(filename):
    with open(filename) as file:
        lines = [l.strip() for l in file if l.strip()]
    flux = []
    for line in lines:
        line = line.split(" ")[0]
        flux.append(float(line))
    return np.asarray(flux)








def plot_scatter(RTMs_fluxes, OBS_fluxes): 
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(1,1,1)
                # gs = fig.add_gridspec(2, 1, height_ratios=[10,1]) #, hspace=0.0)
                # ax = fig.add_subplot(gs[0, 0])

    pngfile = 'FIGS/correlation_' + zout[:-1] + '.png'                                                    
                                                                                                        
    xlabel_specs = r' [W/m$^2$]'
    xlabel = 'Pyranometer Observations' if WANT_SUR else 'BMA-FLX Observations'
    ylabel = 'RTM' # rte_solver
    ylabel_specs = xlabel_specs

    title =  f'Surface SW Correlation' if WANT_SUR else f'TOA SW Correlation'
    pl_list = []

    # Extract data to plot
    for RTM_fluxes in RTMs_fluxes:
        color = 'mediumseagreen' if 'DISORT' in RTM_fluxes else 'blue'
        rte_solver = 'DISORT (1D RTM)' if 'DISORT' in RTM_fluxes else 'MYSTIC (3D RTM)'

        x = read_flux(filename=OBS_fluxes) # model truth
        y = read_flux(filename=RTM_fluxes)


        print("Before filtering ", len(x), f"   ({"1D" if "DISORT" in RTM_fluxes else "3D"})")  #, filename_libRad)
        # Filter out Nans
        mask = ~np.isnan(x) & ~np.isnan(y)
        x = x[mask]
        y = y[mask]
        print("After filtering  ", len(x))

        


        # ------------ Stats --------------
        r = np.corrcoef(x, y)[0, 1] # Correlation coefficient
        r2 = r**2 # The strength of linear association

        # Least squares fit y = a + b x
        b, a = np.polyfit(x, y, 1)
            # yhat = a + b * x
        diff = y - x
            # rmse = np.sqrt(np.mean(diff**2))
        mae = np.mean(np.abs(diff))
        bias = np.mean(diff)
        std = np.std(diff) 
        


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
        p, = ax.plot(xx, a + b * xx, lw=1.5, c=color, label=f"{rte_solver}     y = {a:.1f} + {b:.1f}x") # WRITE IN FIG-TEXT: Linear Least Squares (fit)") 
        pl_list.append(p)


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

        else: ax.scatter(x, y, 
                        s=14, alpha=.6, c=color,
                        marker='^' if "DISORT" in RTM_fluxes else 'o',
                        ) 



        # ---------- Textbox with stats ---------
        solver_str = r"$\mathbf{{DISORT\,\,\, (1D\,\, RTM)}}$" if "1D" in rte_solver else r"$\mathbf{{MYSTIC\,\,\, (3D\,\, RTM)}}$"
        bias_str = r"$\Delta F_{\mathrm{SUR}}^{\downarrow}$ (Bias) = " if WANT_SUR else r"$\Delta F_{\mathrm{TOA}}^{\uparrow}$ (Bias) = "
        bias_str = "Bias = "
        data_str = (
            solver_str + "\n"
            # f"n = {x.size}\n"
            # f"R² = {r2:.2f}                   \n"
            # f"RMSE = {rmse:.1f}\n"
            + f"{bias_str} {bias:5.1f}" + r" $\pm$ " + f"{std:6.1f}" + r" W/m$^2$"
            # f"\nMAE = {mae:.1f}\n"
        )
        fig.text(.99, 0.01 if '1D' in rte_solver else 0.09,          
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

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis='both', which='major', labelsize=INFOSIZE*.8)
    ax.tick_params(axis='both', which='minor', labelsize=INFOSIZE*.8)
    fig.suptitle(title, fontsize=FONTSIZE, y=0.98)

    # ------------ Spesify ACM-COM baseline ----
    folder = "/homevip/bgre/Download/Frames_SurfaceOverpasses/Oslo"
    files = glob.glob(os.path.join(folder, "*ACM_COM*"))
    if files:
        first_file = os.path.basename(files[0])
        baseline = first_file.split("ECA_EX", 1)[1][:2]
        print(baseline)
    else:
        print("No ACM_COM file found. Cannot extract BASELINE")
    plt.figtext(0.001, 0.003, f"Baseline ACM-COM: {baseline}", fontsize=FONTSIZE*0.45)
    
    # BG: ----- plot-adjustments for nicer looking plots -----------
    fig.tight_layout()
    # Axes background (warm light grey)
    ax.set_facecolor('#f0f0f0')
    # Grid: major dashed, minor dotted
    ax.grid(which='major', linestyle='--', alpha=0.4)
    ax.grid(which='minor', linestyle=':',  alpha=0.2)
    ax.minorticks_on()
    for spine in ['top','right']: # remove top/right border
        ax.spines[spine].set_visible(False)
    # -------------------------------------------------------------

    print("pngfile", pngfile)
    plt.savefig(pngfile)
    plt.close()



















def plot_scatter_single_station(RTMs_fluxes, OBS_fluxes, station): 
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(1,1,1)
                # gs = fig.add_gridspec(2, 1, height_ratios=[10,1]) #, hspace=0.0)
                # ax = fig.add_subplot(gs[0, 0])

    pngfile = f'FIGS/STATION/{zout[:-1]}/{station}_correlation_{zout[:-1]}.png'                                                    
                                                                                                        
    xlabel_specs = r' [W/m$^2$]'
    xlabel = 'Pyranometer Observations' if WANT_SUR else 'BMA-FLX Observations'
    ylabel = 'RTM' # rte_solver
    ylabel_specs = xlabel_specs

    title =  f'Surface' if WANT_SUR else f'TOA' + f' SW Correlation         {station}'
    pl_list = []

    # Extract data to plot
    for RTM_fluxes in RTMs_fluxes:
        color = 'mediumseagreen' if 'DISORT' in RTM_fluxes else 'blue'
        rte_solver = 'DISORT (1D RTM)' if 'DISORT' in RTM_fluxes else 'MYSTIC (3D RTM)'


        x = []
        y = []
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
       

        print("Before filtering ", len(x), f"   ({"1D" if "DISORT" in RTM_fluxes else "3D"})")  #, filename_libRad)
        # Filter out Nans
        mask = ~np.isnan(x) & ~np.isnan(y)
        x = x[mask]
        y = y[mask]
        print("After filtering  ", len(x))


        # ------------ Stats --------------
        r = np.corrcoef(x, y)[0, 1] # Correlation coefficient
        r2 = r**2 # The strength of linear association

        # Least squares fit y = a + b x
        b, a = np.polyfit(x, y, 1)
            # yhat = a + b * x
        diff = y - x
            # rmse = np.sqrt(np.mean(diff**2))
        mae = np.mean(np.abs(diff))
        bias = np.mean(diff)
        std = np.std(diff) 
        


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
        p, = ax.plot(xx, a + b * xx, lw=1.5, c=color, label=f"{rte_solver}     y = {a:.1f} + {b:.1f}x") # WRITE IN FIG-TEXT: Linear Least Squares (fit)") 
        pl_list.append(p)


       
        ax.scatter(x, y, 
                s=14, alpha=.6, c=color,
                marker='^' if "DISORT" in RTM_fluxes else 'o',
        ) 



        # ---------- Textbox with stats ---------
        solver_str = r"$\mathbf{{DISORT\,\,\, (1D\,\, RTM)}}$" if "1D" in rte_solver else r"$\mathbf{{MYSTIC\,\,\, (3D\,\, RTM)}}$"
        bias_str = r"$\Delta F_{\mathrm{SUR}}^{\downarrow}$ (Bias) = " if WANT_SUR else r"$\Delta F_{\mathrm{TOA}}^{\uparrow}$ (Bias) = "
        bias_str = "Bias = "
        data_str = (
            solver_str + "\n"
            # f"n = {x.size}\n"
            # f"R² = {r2:.2f}                   \n"
            # f"RMSE = {rmse:.1f}\n"
            + f"{bias_str} {bias:5.1f}" + r" $\pm$ " + f"{std:6.1f}" + r" W/m$^2$"
            # f"\nMAE = {mae:.1f}\n"
        )
        fig.text(.99, 0.01 if '1D' in rte_solver else 0.09,          
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

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis='both', which='major', labelsize=INFOSIZE*.8)
    ax.tick_params(axis='both', which='minor', labelsize=INFOSIZE*.8)
    fig.suptitle(title, fontsize=FONTSIZE, y=0.98)

    # ------------ Spesify ACM-COM baseline ----
    folder = "/homevip/bgre/Download/Frames_SurfaceOverpasses/Oslo"
    files = glob.glob(os.path.join(folder, "*ACM_COM*"))
    if files:
        first_file = os.path.basename(files[0])
        baseline = first_file.split("ECA_EX", 1)[1][:2]
        print(baseline)
    else:
        print("No ACM_COM file found. Cannot extract BASELINE")
    plt.figtext(0.001, 0.003, f"Baseline ACM-COM: {baseline}", fontsize=FONTSIZE*0.45)
    
    # BG: ----- plot-adjustments for nicer looking plots -----------
    fig.tight_layout()
    # Axes background (warm light grey)
    ax.set_facecolor('#f0f0f0')
    # Grid: major dashed, minor dotted
    ax.grid(which='major', linestyle='--', alpha=0.4)
    ax.grid(which='minor', linestyle=':',  alpha=0.2)
    ax.minorticks_on()
    for spine in ['top','right']: # remove top/right border
        ax.spines[spine].set_visible(False)
    # -------------------------------------------------------------

    print("pngfile", pngfile)
    plt.savefig(pngfile)
    plt.close()













def plot_scatter_across_track_uncertainty(RTMs_fluxes, OBS_fluxes): 
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(1,1,1)
        
    pngfile = 'FIGS/Error_vs_Distance' + zout[:-1] + '.png'                                                    
                                                                                                        
    title =  f'Surface SW Flux Error vs Across-Track Distance'

    xlabel = 'Across-Track Distance (GB Station to Satellite Track)'
    xlabel_specs = r' [km]'
    ylabel_specs =  r' [W/m$^2$]'

    pl_list = []

    # Info for downloading ACM-COM data
    Path_base = "/homevip/bgre/Download/Frames_SurfaceOverpasses"
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

    Download_Distances_Data = False
    if Download_Distances_Data:
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
        with open("DATA/Distances.txt", "w") as f:
            for D, OrbitID, Station in DistanceInfo:
                f.write(f"{D:<10.2f} {OrbitID} {Station}\n")


    # Extract data to plot
    for RTM_fluxes in RTMs_fluxes:
        color = 'mediumseagreen' if 'DISORT' in RTM_fluxes else 'blue'
        rte_solver = 'DISORT (1D RTM)' if 'DISORT' in RTM_fluxes else 'MYSTIC (3D RTM)'

        D_list = []
        with open("DATA/Distances.txt", "r") as f:
            for line in f:
                D = float(line.split(" ")[0])
                D_list.append(D)

        
        x = read_flux(filename=OBS_fluxes) # model truth
        y = read_flux(filename=RTM_fluxes)

        # print(f"len x {len(x)}, len y {len(y)}, len D {len(D_list)}")

        # Stats
        F_diff = y - x  # Simulated - Observed fluxes
        F_diff_abs = np.abs(F_diff)
        rmse = np.sqrt(np.nanmean(F_diff**2))


        ######## Choose what data to plot ########
        data_idx = 1
            # 0: Bias
            # 1: MAB
        ##########################################
        data            = [F_diff,                                              F_diff_abs      ][data_idx]
        ylabel          = [r'$\Delta F_{\mathrm{SUR}}^{\downarrow}$ (Bias)',    'Absolute Error'][data_idx]
        textboxlabel    = ["Bias",                                              "MAE"           ][data_idx]

        p = ax.scatter(D_list, data,  lw=1, c=color, label=f"{rte_solver}") 
        pl_list.append(p)


        WANT_REGRESSION = True
        if WANT_REGRESSION:
            D_list = np.asarray(D_list)
            data = np.asarray(data)
            mask = np.isfinite(D_list) & np.isfinite(data)

            b, a = np.polyfit(D_list[mask], data[mask], 1)
            # --------- Plot Regression -----------
            xx = np.linspace(0, np.nanmax(D_list), 100)
            p, = ax.plot(xx, a + b * xx, lw=2, c=color, label=f"Regression {rte_solver}\ny = {a:.1f} + {b:.1f}x") 
            pl_list.append(p)
        

   
    

        # ---------- Textbox with stats ---------
        solver_str = r"$\mathbf{{DISORT\,\,\, (1D\,\, RTM)}}$" if "1D" in rte_solver else r"$\mathbf{{MYSTIC\,\,\, (3D\,\, RTM)}}$"
        data_str = solver_str + "\n"+ ylabel + f" = {np.nanmean(data):.1f}"
        if data_idx==0: 
            data_str += r" $\pm$ " + f"{np.nanstd(data):.1f}" 
        data_str += r" W/m$^2$"
        
        fig.text(.82, 0.12 if '1D' in rte_solver else 0.01, 
                            # 0.5 if '1D' in rte_solver else .9, 0.01, 
                data_str,
                transform=ax.transAxes, 
                ha='center', va='center',
                fontsize=FONTSIZE*.7, color='k',  #color=color,    #
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, #facecolor='w',
                alpha=0.6))


    
    ax.legend(  handles=pl_list,
                loc='upper left', framealpha=0.7, 
                borderaxespad=0.0,                   # space to axes
                borderpad=0.25, labelspacing=0.25,   # compact box)
                fontsize=INFOSIZE*.8)
    
    ax.set_xlabel(xlabel + xlabel_specs, fontsize=INFOSIZE)
    ax.set_ylabel(ylabel + ylabel_specs, fontsize=INFOSIZE)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis='both', which='major', labelsize=INFOSIZE*.8)
    ax.tick_params(axis='both', which='minor', labelsize=INFOSIZE*.8)
    fig.suptitle(title, fontsize=FONTSIZE, y=0.98)

    # ------------ Spesify ACM-COM baseline ----
    folder = "/homevip/bgre/Download/Frames_SurfaceOverpasses/Oslo"
    files = glob.glob(os.path.join(folder, "*ACM_COM*"))
    if files:
        first_file = os.path.basename(files[0])
        baseline = first_file.split("ECA_EX", 1)[1][:2]
        # print(baseline)
    else:
        print("No ACM_COM file found. Cannot extract BASELINE")
    plt.figtext(0.001, 0.003, f"Baseline ACM-COM: {baseline}", fontsize=FONTSIZE*0.45)
    
    # BG: ----- plot-adjustments for nicer looking plots -----------
    fig.tight_layout()
    # Axes background (warm light grey)
    ax.set_facecolor('#f0f0f0')
    # Grid: major dashed, minor dotted
    ax.grid(which='major', linestyle='--', alpha=0.4)
    ax.grid(which='minor', linestyle=':',  alpha=0.2)
    ax.minorticks_on()
    for spine in ['top','right']: # remove top/right border
        ax.spines[spine].set_visible(False)
    # -------------------------------------------------------------

    print("pngfile", pngfile)
    plt.savefig(pngfile)
    plt.close()
















if __name__ == "__main__":
    # Plot correlation at SUR or TOA







    # WANT_SUR    = True
    WANT_SUR    = False







    # WANT_RTM = 0  
    # WANT_RTM = 1
    WANT_RTM = 2
    """
    0: DISORT
    1: MYSTIC
    2: DISORT & MYSTIC
    """



    SELECT_PLOT = 0
    """
    0: scatter plot
    1: scatter plot per station
    2: Error vs. across-track 
    """
    

    




    # ------------------------ FILES -------------------------------------------------------------------------------------
    zout = 'SUR_' if WANT_SUR else 'TOA_' 

    RTMs_fluxes = [
        [f'DATA/{zout}DISORT.txt'],
        [f'DATA/{zout}MYSTIC.txt'],
        [f'DATA/{zout}MYSTIC.txt', f'DATA/{zout}DISORT.txt']
        ][WANT_RTM]

    OBS_fluxes = 'DATA/SUR_Observations.txt' if WANT_SUR else 'DATA/TOA_Observations_SR.txt' 

        # NOTE:
        # TOA_Observation.txt: SmallResolution, qs = 0,2
    
    
    
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
        if WANT_SUR: plot_scatter_across_track_uncertainty(RTMs_fluxes, OBS_fluxes)
        

    

    
