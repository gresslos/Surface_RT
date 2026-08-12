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
from matplotlib.colors import LogNorm
import matplotlib.dates as mdates
from datetime import datetime, timedelta, timezone
from pathlib import Path

import Find_Overpass_Info as Find_Overpass_Info



FONTSIZE=14
INFOSIZE=13

WP_min, WP_max = 1e-3, 1e5 
WC_min, WC_max = 1e-7, 10**4.5      # Modify from 1e-6 -> 1e-7 form cmaps-sake! #1e-6, 10**4.5     # 1e-5, 10**4.5
cmap = plt.get_cmap('viridis')  # 'viridis' 'inferno'

def find_nearest_id(array,value):
    idx=(np.abs(array-value)).argmin()
    return idx

def moving_average(x, w=3):
            # BG: Old version
            # # https://stackoverflow.com/questions/13728392/moving-average-or-running-mean?noredirect=1&lq=1
            # return np.convolve(x, np.ones(w), 'same') / w #'full') / w #'valid') / w
   
    x = np.asarray(x, dtype=float)

    kernel = np.ones(w)

    valid = np.isfinite(x)
    x0 = np.where(valid, x, 0.0)

    num = np.convolve(x0, kernel, mode="same")
    den = np.convolve(valid.astype(float), kernel, mode="same")

    out = num / den
    out[den == 0] = np.nan

    return out



# BG: My interpolation function 
def interpolate(x1, data1_raw, fill1, x2, data2_raw, fill2):
    """
    1) Masks fill values in data1 and data2  
    2) Extracts valid points from data1 and sorts them  
    3) Interpolates data1 onto x2  
    4) Sets out‑of‑bounds interpolation to NaN  
    5) Masks data2 zeros/nans  
    6) Returns only the indices where both interpolated data1 and data2 are valid

    Parameters
    - 1D arrays
    - x1, x2 (target)   : latitudes
    - d1, d2 (target)   : data
    - fill1, fill2      : fill values 


    Returns
    - x_valid               : subset of x2, valid for both data
    - data1_interp_valid    : data1 interpolated onto x2 + masked to x_valid.
    - data2_valid           : data2_raw (with fill2→NaN) + masked to ly x_valid.
    """

    # mask fill values → NaN
    d1 = np.where(data1_raw == fill1, np.nan, data1_raw)
    d2 = np.where(data2_raw == fill2, np.nan, data2_raw)

    # select non‑NaN points in data1
    valid1 = ~np.isnan(d1)
    x1v, d1v = x1[valid1], d1[valid1]

    # sort for np.interp requirements
    order = np.argsort(x1v)
    x1v, d1v = x1v[order], d1v[order]

    # interpolate onto x2
    interp_raw = np.interp(x2, x1v, d1v)

    # mask out‑of‑bounds
    oob = (x2 < x1v[0]) | (x2 > x1v[-1])
    interp = np.where(oob, np.nan, interp_raw)

    # mask invalid or zero in data2
    valid2 = ~np.isnan(d2) & (d2 != 0)

    # final valid mask
    valid = (~np.isnan(interp)) & valid2

    # return only the good points
    x_valid            = x2[valid]
    data1_interp_valid = interp[valid]
    data2_valid        = d2[valid]

    return x_valid, data1_interp_valid, data2_valid

# BG: My add-profile-to-plot function
def add_profile_to_plot(fig, ax, ACMCOM, fsize, legend_list, quantity=None, quantity_list=None, iacr=151, stacked=True, normalize=False, name=None, data=None, slice=None, h_max=20):
    """
    Add a single profile (ex. elevation, lwc, iwc, ...) to the given axis.
    
    Parameters:
        ax       : main matplotlib axis
        pl_list  : list to append plotted Line2D objects
        quantity : one of ['elevation', 'lwc', ....]
        iacr     : nadir column index
    
    Returns:
        ax2      : twin axis used for plotting
    """
    
    # Define thresholds (might need to be adjusted)
    THRESHOLD_lwc = 1e30 
    THRESHOLD_iwc = 1e30
    THRESHOLD_lwp = 400  + 1e30
    THRESHOLD_iwp = 1000 + 1e30

    if quantity is None:
        # No profile to plot
        return None

    x = ACMCOM.latitude_active
    n_lat = x.shape[0]
    profile = []

    if quantity == 'elevation':
        for ia in range(n_lat):
            heights = ACMCOM.height_level[1:, ia]
            pressures = ACMCOM.pressure_level[1:, ia]
            valid_heights = [h for h, p in zip(heights, pressures) if h > 0 and p < 1e10]
            profile.append(min(valid_heights) if valid_heights else np.nan)
        ylabel = 'Surface Elevation [m]'
        color = 'black'
        label = 'Surface Elevation'

    elif quantity == 'lwp':
        for ia in range(n_lat):
            heights = ACMCOM.height_layer[:, ia]
            lwc = ACMCOM.liquid_water_content[:, ia]
            valid_heights = []
            valid_lwc = []
            for h, l in zip(heights, lwc):
                if h >= 0 and h < 1e35 and l < THRESHOLD_lwc: 
                    valid_heights.append(h)
                    valid_lwc.append(l)
            if len(valid_heights) > 1:
                totcol = -np.trapezoid(valid_lwc, x=valid_heights)
                profile.append(totcol if (totcol < THRESHOLD_lwp) else 0.0)
            else:
                profile.append(0.0)
        ylabel = 'LWP [kg/m$^2$]'
        color = (0.2, 0.4, 0.8)   # stronger blue
        label = 'LWP'

    elif quantity == 'iwp':
        for ia in range(n_lat):
            heights = ACMCOM.height_layer[:, ia]
            iwc = ACMCOM.ice_water_content[:, ia]
            valid_heights = []
            valid_iwc = []
            for h, l in zip(heights, iwc):
                if h >= 0 and h < 1e35 and l < THRESHOLD_iwc:
                    valid_heights.append(h)
                    valid_iwc.append(l)
            if len(valid_heights) > 1:
                totcol = -np.trapezoid(valid_iwc, x=valid_heights)
                profile.append(totcol if (totcol < THRESHOLD_iwp) else 0.0)
            else:
                profile.append(0.0)
        ylabel = 'IWP [kg/m$^2$]'
        color = (0.0, 0.6, 0.5)   # greenish teal
        label = 'IWP'

    elif quantity == 'tot_wp':
        profile = [[],[]]
        for ia in range(n_lat):
            heights = ACMCOM.height_layer[:, ia]
            lwc     = ACMCOM.liquid_water_content[:, ia]
            iwc     = ACMCOM.ice_water_content[:, ia]

            
            valid_h = []
            valid_iwc = []
            valid_lwc = []
            for h, l, i in zip(heights, lwc, iwc):
                if 0 <= h < 1e35 and l < THRESHOLD_lwc and i < THRESHOLD_iwc:
                    # replace any non‑finite with zero
                    l_val = l if np.isfinite(l) else 0.0
                    i_val = i if np.isfinite(i) else 0.0

                    valid_h.append(h)
                    valid_iwc.append(i_val)
                    valid_lwc.append(l_val)

            if len(valid_h) > 1:
                totcol_i = -np.trapezoid(valid_iwc, x=valid_h)
                totcol_l = -np.trapezoid(valid_lwc, x=valid_h)
                profile[0].append(totcol_i if (totcol_i < THRESHOLD_iwp) else 0.0)
                profile[1].append(totcol_l if (totcol_l < THRESHOLD_lwp) else 0.0)
            else:
                profile[0].append(0.0)
                profile[1].append(0.0)
        ylabel = 'IWP & LWP \n [kg/m$^2$]' #'Water Path [kg/m$^2$]'
        color  =  'black' # (0.1, 0.5, 0.65) # combo LWP and IWP
        label  = 'LWP & IWP'

    elif quantity == 'tot_wc': # lat vs altitude
        profile = [[],[]]
        ylabel = ['IWC [kg/m$^3$]','LWC [kg/m$^3$]'] #'Water content 
        color  =  'black' 
        label  = 'IWC & LWC'
        iwc = ACMCOM.ice_water_content; lwc = ACMCOM.liquid_water_content
        iwc = np.where((iwc >= 1e30) | (iwc == 0), np.nan, iwc)
        lwc = np.where((lwc >= 1e30) | (lwc == 0), np.nan, lwc)
        vmax = WC_max
        vmin = 0

        profile[0].append(iwc)
        profile[1].append(lwc)
        
        # Find highest cloudy altitude
                    # heights = ACMCOM.height_layer
                    # valid = ((~np.isnan(iwc)) & (iwc != 0)) | ((~np.isnan(lwc)) & (lwc != 0))
                    # cloud_top_height = np.nanmax(heights[valid])/1000 # units= m -> km
        cloud_top_height = h_max


    elif quantity == 'albedo' or quantity == 'SWalbedo' or quantity == 'LWalbedo':
        if 'thermal' in plot_type or quantity == 'LWalbedo':
            # long‑wave albedo averaged over all thermal wavelengths
            nwvl = ACMCOM.wavelengths_thermal_surface_emissivity.shape[0]
            for ia in range(n_lat):
                type_idx = ACMCOM.surface_emissivity_type_index[iacr, ia]
                if type_idx < 0 or type_idx > ACMCOM.surface_emissivity_table.shape[0]:
                    # missing or invalid type
                    profile.append(0.0)
                else:
                    # compute albedo = 1 - emissivity for each wavelength
                    albs = []
                    for iwvl in np.arange(nwvl-1,-1,-1):
                        eps = ACMCOM.surface_emissivity_table[type_idx-1, iwvl]
                        if np.isfinite(eps):
                            albs.append(1.0 - eps)
                    profile.append(np.nanmean(albs) if albs else 0.0)
            ylabel = 'Albedo (LW) []'
            label  = 'LW Albedo'
            color  = (1.00, 0.40, 0.00) # orange
        elif 'solar' in plot_type or quantity == 'SWalbedo':
            # short‑wave albedo = mean(vis, NIR)
            for ia in range(n_lat):
                a1 = ACMCOM.albedo_diffuse_radiation_surface_visible[iacr, ia]
                a2 = ACMCOM.albedo_diffuse_radiation_surface_near_infrared[iacr, ia]
                # filter out sentinel large values and nonfinite
                if a1 >= 1e35 or a2 >= 1e35 or not np.isfinite(a1) or not np.isfinite(a2):
                    profile.append(0.0)
                else:
                    profile.append(np.nanmean([a1, a2]))
            ylabel = 'Albedo (SW) []'
            label  = 'SW Albedo'
            color  = (1.00, 0.84, 0.00) # gold


    elif quantity == 'aerosols':
        for ia in range(n_lat):
            nheights = ACMCOM.aerosol_extinction.shape[0] - 1
            aero_tau_tot = 0.0
            for ih in range(nheights):
                h = ACMCOM.height_level[ih+1, ia]
                dz = (ACMCOM.height_level[ih, ia] - h) / 1000.0
                ext = ACMCOM.aerosol_extinction[ih, ia]
                if ACMCOM.aerosol_classification[ih, ia] >= 0 and ext < 1e30: #np.isfinite(ext) -> do not work
                    tau = ext * dz
                    aero_tau_tot += tau
            if not np.isfinite(aero_tau_tot):
                aero_tau_tot = 0.0
            if 'thermal' in plot_type:
                # scale for thermal: 
                import MakeRTM as MakeRTM
                aero_tau_tot = aero_tau_tot * 0.6 if MakeRTM.aerosol_thermal_impact_bool(ia, ACMCOM) else 0.0 # Conversion Factor
            profile.append(aero_tau_tot)
        ylabel = 'AOD []' #'Aerosol optical depth []'
        color  = 'gray'
        label  = 'AOD'

    elif quantity == 'surface_temperature':
        # surface_T = min(T for T,v,p in zip(ACMCOM.temperature_level[1:,ia], ACMCOM.height_level[1:,ia], ACMCOM.pressure_level[1:,ia]) if v > 0 and p < 1e10)
        # surface_T: ", ACMCOM.surface_temperature[iacr,ia]) 
        for ia in range(n_lat):
            # T = ACMCOM.surface_temperature[iacr,ia]

            T = min((T for T, v, p in zip(ACMCOM.temperature_level[1:, ia],
                        ACMCOM.height_level[1:, ia],
                        ACMCOM.pressure_level[1:, ia])
                        if v > 0 and p < 1e10 and np.isfinite(T)), default=np.nan)

            if np.isfinite(T): profile.append(T)
            else: profile.append(np.nan)
        ylabel = 'Surf.Temp. [K]' #'Surface Temperature [K]'
        color  = 'red'
        label  = 'Surface Temperature'

    elif quantity == 'DEM_elevation':
        # Import GetElevation() from MakeRTMInpitFile.py to compare DEM-elevation to EarthCARE-elevation
        from MakeRTMInputFile_bg import GetElevation

        for ia in range(n_lat):
            h = GetElevation(ACMCOM.latitude_active[ia], ACMCOM.longitude_active[ia], ia)
            if np.isfinite(h): profile.append(h)
            else: profile.append(np.nan)
        ylabel = 'Surface Elevation (DEM) [m]'
        color  = 'pink'
        label  = 'Surface Elevation (DEM)'

    elif quantity == 'CF':
        THRESHOLD = 0 # kg/m3
        MAX = 1e30    # kg/m3
    
        want_large_buffer =  True     # True False
        Buffer_Along  = 12 if want_large_buffer else 6
        Buffer_Across = 10 if want_large_buffer else 6
        # handy refs
        LWC = ACMCOM.liquid_water_content    # (levels, columns)
        IWC = ACMCOM.ice_water_content       # (levels, columns)
        n_levels, n_cols = LWC.shape
        # output
        profile = np.full(n_cols, np.nan, dtype=float)
        # 1) per-column flags (compute once)
        valid_col = np.any(np.isfinite(LWC) & (LWC < MAX)    &  np.isfinite(IWC) & (IWC < MAX), axis=0)
        cloud_col = np.any(((LWC > THRESHOLD) & (LWC < MAX)) | ((IWC > THRESHOLD) & (IWC < MAX)), axis=0)
        # 2) adjusted index map (once)
        offset = 0
        if   "Orbit_05926C" in SceneName: offset = 2700
        elif "Orbit_06888C" in SceneName: offset = 2527
        elif "Orbit_07277C" in SceneName: offset = 2527
        elif "Orbit_06331C" in SceneName: offset = 2636
        idx = (ACM3D.index_construction - offset)          # (cross, along)
        # Valid where >= 0 and finite
        valid_idx_mask = (idx >= 0) & ~np.isnan(idx)

        acr_lo = ACM3D.nadir_pixel_index - Buffer_Across
        acr_hi = ACM3D.nadir_pixel_index + Buffer_Across + 1
        # 3) slide along
        for ia in range(Buffer_Along, n_cols - Buffer_Along):
            al_lo = ia - Buffer_Along
            al_hi = ia + Buffer_Along + 1
            # Column indices in Buffer
            # Rectangle of column IDs -> 1D 
            cols = idx[acr_lo:acr_hi, al_lo:al_hi].ravel()
            # Remove out of bounce idx from cutting swath
            valid_cols_mask = valid_idx_mask[acr_lo:acr_hi, al_lo:al_hi].ravel()
            cols = cols[valid_cols_mask]

            v = valid_col[cols]
            c = cloud_col[cols]
            v_sum = int(v.sum())
            CF = (int(c.sum()) / v_sum) if v_sum > 0 else np.nan
            profile[ia] = CF

        want_compare_to_small_buffer = False
        if want_compare_to_small_buffer: # Same procedure as above
            want_large_buffer =  False   
            Buffer_Along  = 12 if want_large_buffer else 6
            Buffer_Across = 10 if want_large_buffer else 6
            acr_lo = ACM3D.nadir_pixel_index - Buffer_Across
            acr_hi = ACM3D.nadir_pixel_index + Buffer_Across + 1
            for ia in range(Buffer_Along, n_cols - Buffer_Along):
                al_lo = ia - Buffer_Along
                al_hi = ia + Buffer_Along + 1
                cols = idx[acr_lo:acr_hi, al_lo:al_hi].ravel()
                # Remove out of bounce idx from cutting swath
                valid_cols_mask = valid_idx_mask[acr_lo:acr_hi, al_lo:al_hi].ravel()
                cols = cols[valid_cols_mask]
                v = valid_col[cols]
                c = cloud_col[cols]
                v_sum = int(v.sum())
                CF = (int(c.sum()) / v_sum) if v_sum > 0 else np.nan
                profile[ia] -= CF
            profile[:int(np.flatnonzero(valid_col)[0])], profile[int(np.flatnonzero(valid_col)[-1]):] = np.nan, np.nan # Eliminate boundary effects

        ylabel = 'CF []' if not want_compare_to_small_buffer else 'CF Diff.\n(Large - Small Buffer) []'
        color  = 'black'
        label  = 'CF' if not want_compare_to_small_buffer else 'CF Diff.'


    else:
        raise ValueError("Invalid quantity. Choose from: 'elevation', 'lwp', 'iwp', 'tot_wp', 'albedo', 'aerosols")








    alpha = 0.25
    # -------------------------------------- Handling Stacked Figures --------------------------------------------------------
    if stacked: # Stack the Add_Profile to a subplot under OG-figure (stacked on top)
        if not hasattr(fig, "_stacked_inited"): # check if first time call this function
            fig.clf()  # clear figure to rebuild layout
            if ('CF' in quantity_list and len(quantity_list) != 1) or ('tot_wc' in quantity_list and 'tot_wp' in quantity_list): # Have 2 stacked figures
                fig.set_figheight( fig.get_figheight() * 1.6) #increase fig-height for stacked figures
                gs = fig.add_gridspec(3, 1, height_ratios=[3, 1.5, 2], hspace=0.0)
                ax = fig.add_subplot(gs[0, 0])
                ax2 = fig.add_subplot(gs[1, 0], sharex=ax)
                ax3 = fig.add_subplot(gs[2, 0], sharex=ax)
               
                fig._ax_top = ax
                fig._ax_mid = ax2
                fig._ax_bot = ax3
                plt.setp(fig._ax_mid.get_xticklabels(), visible=False) # Remove x-ticks for midle-fig
            else: 
                fig.set_figheight( fig.get_figheight() * 1.5) #increase fig-height for stacked figures
                gs = fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.0)
                ax = fig.add_subplot(gs[0, 0])
                ax2 = fig.add_subplot(gs[1, 0], sharex=ax)
        
                fig._ax_top = ax
                fig._ax_bot = ax2

        if ('CF' in quantity_list) or ('tot_wc' in quantity_list and 'tot_wp' in quantity_list): # 3-Figures Stacked
            if ('CF' in quantity) or ('tot_wp' in quantity):  # Plotting CF into _ax_bot
                ax  = fig._ax_top
                ax2 = fig._ax_mid
            else: # Plot other quantities into _ax_mid
                ax  = fig._ax_top
                ax2 = fig._ax_bot
        else:
            ax  = fig._ax_top
            ax2 = fig._ax_bot
        alpha = 0.6

        # Axes background (warm light grey)
        ax2.set_facecolor('#f0f0f0')
        # Grid: major dashed, minor dotted
        ax2.grid(which='major', linestyle='--', alpha=0.4)
        ax2.grid(which='minor', linestyle=':',  alpha=0.2)
        ax2.spines['right'].set_visible(False)
        
        plt.setp(ax.get_xticklabels(), visible=False) # Same x-ticks (but remove)
        # -------------------------------------------

        # Make twin axis
        if ('wc' not in quantity) and ('CF' not in quantity or len(quantity_list) == 1): 
            ax2.tick_params(left=False, labelleft=False) # remove left y-ticks
            ax2.spines['left'].set_visible(False)
            ax2.tick_params(axis='y', which='both', left=True, labelleft=False, length=0)
            ax2 = ax2.twinx()
            ax2.spines['left'].set_visible(False)
           
        fig._stacked_inited = True # Set True -> function run at least once
        # -----------------------------------------------------------------------------------------------------------------------
    else:
        # Make twin axis
        ax2 = ax.twinx()




    #------------ Modify for multiple add-profiles-verticle spine position ---------
    # per-parent-axis counter
    n_right = getattr(ax, "right_twin_count", 0)
    # only push outward for the 2nd, 3rd, ... twins
    if n_right > 0 and 'CF' not in quantity :
        offset_step = 80 #60  # px
        ax2.spines["right"].set_position(("outward", offset_step * n_right))

    # update counter for next call
    if ('CF' not in quantity) or ('tot_wc' not in quantity): setattr(ax, "right_twin_count", n_right + 1)
    # ---------------------------------------------------------------------------

    ax2.set_ylabel(ylabel, fontsize=fsize*0.8, color=color, labelpad=5)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.spines["right"].set_edgecolor(color)

    # PLOT
    if 'C' in name: # lat [small->large->small] -> not monotonously increasing -> problem plotting
        x = x[slice]


    if 'wp' in quantity or 'wc' in quantity:
        profile = np.asarray(profile, float)
        # break the line at non-positive values (important for log)
        profile[profile <= 0] = np.nan  # no zeros/negatives on log y
        if quantity == 'tot_wc': 
            profile = profile.squeeze() 
            y = ACMCOM.height_layer[:,:]/1000  # Convert from m to km
            label_list = ylabel
            color_list = [0, 0]
            ax2.set_ylabel('Altitude [km]', fontsize=INFOSIZE)
            fig.subplots_adjust(right=1.05, bottom=.1)   # Adjuct without tight_layout
            cs_list = []


        elif quantity == 'tot_wp': label_list = ['IWP','LWP']; color_list = [(0.0, 0.6, 0.5), (0.2, 0.4, 0.8)]
        else: profile = [profile]; label_list = [label]; color_list = [color]

        for i, (p, l, c) in enumerate(zip(profile, label_list, color_list)):
            if 'wc' in quantity: # Plot Colormesh + Colorbar
                if   'I' in l: cmap = plt.get_cmap('Blues') # plt.get_cmap('viridis')
                elif 'L' in l: cmap = plt.get_cmap('Reds') # plt.get_cmap('magma')  
                else: print("ERROR in cmap")


    
                if 'C' in name: # lat [small->large->small] -> not monotonously increasing -> problem plotting
                    p = p[:, slice]
                    y = y[:, slice] if i == 0 else y
        

                cs = ax2.pcolormesh(x,y,p, shading='auto', cmap=cmap, norm=LogNorm(WC_min, WC_max))
                cs_list.append(cs)
                ax2.tick_params(axis='y', which='both', right=True, labelright=False, length=0) # clears ticks

                fig.tight_layout = lambda *args, **kwargs: None # turnes off tight_layout -> wont work later in code        
              
                
            else: # Plot Line(s)
                if 'C' in name: # lat [small->large->small] -> not monotonously increasing -> problem plotting
                    p = p[slice]
                

                line, = ax2.plot(x, p,  
                            label=l, 
                            color=c,
                            linewidth=0.8,
                            markersize=1, marker='o',
                            alpha=alpha)
                legend_list.append(line) 
            
    else: # Add property
        if 'C' in name: # lat [small->large->small] -> not monotonously increasing -> problem plotting
            profile = profile[slice]


        line, = ax2.plot(x, profile,  
                    label=label, 
                    color=color,
                    # linestyle='--',
                    linewidth=0.8,
                    alpha=alpha)

    if 'cs' in locals():  # If cs is defined
        if 'tot_wc' in quantity:
            three_fig_bool = ('CF' in quantity_list and len(quantity_list) != 1) or ('tot_wp' in quantity_list)
            ax_list = [ax, fig._ax_mid, ax2] if three_fig_bool else [ax, ax2]
            anchor = (0, 0)
            shrink = .3 if three_fig_bool else .4
            for cs, l in zip(cs_list, label_list):
                cb = fig.colorbar(
                    cs, ax=ax_list, shrink=shrink,
                    pad=0.0,
                    anchor=anchor,
                )
                cb.set_label(l, size=INFOSIZE)
                cb.ax.tick_params(labelsize=INFOSIZE * .8)
                cb.outline.set_visible(False)
                cb.ax.set_facecolor('#f7f7f7')

    if 'wc' in quantity or 'wp' in quantity:
        pass  # no legend for tot_wc
    else:
        if 'CF' not in quantity or len(quantity_list) == 1:
            legend_list.append(line)
        else: 
            # CF gets its own legend on ax2 (only if there is something to show)
            ax2.legend( loc='upper right', framealpha=0.7, 
                        borderaxespad=0.0,                  # space to axes
                        borderpad=0.25, labelspacing=0.25,   # compact box)
                        fontsize=INFOSIZE*.8)
    
    # Set y-limits:
    if 'wp' in quantity:
        ax2.set_yscale('log')
        ax2.set_ylim(WP_min, WP_max)
    elif 'wc' in quantity: ax2.set_ylim(ymin=0,ymax=cloud_top_height) #ymax=19) # altitude
    else:
        ymax, ymin = np.nanmax(profile), np.nanmin(profile)
        pad = 0.15 * (ymax-ymin)
        ax2.set_ylim(ymin, ymax + pad)
    
    if 'tot_wc' in quantity_list and 'tot_wp' in quantity_list: ax2 = fig._ax_mid  # Make sure legends is made to mid_fig
    return ax, ax2 

    
def calculate_cloud_fraction(ACMCOM, ACM3D, want_2D=True, want_ice=False):
    # CF of whole swat
    THRESHOLD = 0 # kg/m3
    MAX = 1e30    # kg/m3
    THRESHOLD_lwp = 400  + 1e30 
    THRESHOLD_iwp = 1000 + 1e30


    
    # ----------- Quality-Status -----------------
    # Create a combined mask: quality 0 or 1
    quality = np.asarray(ACMCOM.quality_status[:])
    quality_status_mask = np.isin(quality, [0, 1])
    # ------------------------------------------------¨

    if want_2D:
        # flatten horizontal mapping
        irec_flat = ACM3D.index_construction.ravel().copy()   # shape (nx*ny)
        # Modify for cutted swats:
        if "Orbit_06888C" in SceneNames[0]: # Svaldbard swat
            irec_flat -= 2527 
            # keep only non-negative values
            irec_flat = irec_flat[irec_flat >= 0]
        elif SceneNames[0] in ["Orbit_06662C", "Orbit_06600C"]: # Greenland swats
            # keep only in-bounce values from cutting swat 
            irec_flat = irec_flat[irec_flat < ACM3D.index_construction.shape[1]] # length of along-track

        # gather arrays for those columns: shapes -> (layer_number, along_track)
        lwc = ACMCOM.liquid_water_content[:, irec_flat]     
        iwc = ACMCOM.ice_water_content[:, irec_flat]        

        # per-level valid / cloud masks (2D: levels x npoints)
        if want_ice:
            valid_level = np.isfinite(iwc) & (iwc < MAX)  
            cloud_level = (iwc > THRESHOLD) & (iwc < MAX)  
        else:
            valid_level = np.isfinite(lwc) & (lwc < MAX) & np.isfinite(iwc) & (iwc < MAX)
            cloud_level = (lwc > THRESHOLD) & (lwc < MAX) | (iwc > THRESHOLD) & (iwc < MAX)
                    
        # collapse to 1D per horizontal pixel: True if ANY level satisfies condition
        valid_mask_1d = valid_level.any(axis=0)   # shape (npoints,), boolean
        cloud_mask_1d = cloud_level.any(axis=0)   # shape (npoints,), boolean

        # apply quality filter on the selected columns
        quality_status_mask = quality_status_mask[irec_flat]             # align with selected columns
        valid_mask_1d &= quality_status_mask
        cloud_mask_1d &= quality_status_mask

        # global counts over the whole 2D swath (number of horizontal pixels)
        valid_count = int(valid_mask_1d.sum())   # how many horizontal pixels have any valid data
        cloud_count = int(cloud_mask_1d.sum())   # how many horizontal pixels have any cloud

        CF = cloud_count / valid_count if valid_count > 0 else np.nan 
            
    # CF of nadir swat (1D)
    else:
        lwc = ACMCOM.liquid_water_content    # shape (layer_number, along_track)
        iwc = ACMCOM.ice_water_content       # shape (layer_number, along_track)

        # valid pixel mask: both lwc and iwc finite                                                              
        # cloud mask: either lwc>0 or iwc>0, but only where values are valid  
        if want_ice:
            valid_level = (np.isfinite(iwc) & (iwc < MAX))  # Dimention = 2D
            cloud_level = (iwc > THRESHOLD) & (iwc < MAX)   
        else:
            valid_level = (np.isfinite(lwc) & (lwc < MAX)) & (np.isfinite(iwc) & (iwc < MAX))   # Dimention = 2D
            cloud_level = ((lwc > THRESHOLD) & (lwc < MAX)) | ((iwc > THRESHOLD) & (iwc < MAX))

        valid_mask = valid_level.any(axis=0)    # Dimention = 1D
        cloud_mask = cloud_level.any(axis=0)                                            
        # Explenation: .any(axis=0) returns True if any element is True -> at least one element True for all layer_number
        
        # apply quality filter
        valid_mask &= quality_status_mask
        cloud_mask &= quality_status_mask

        # Sum up valid- or CP-pixel mask:
        # collapse to 1D: counts per along-track column
        valid_count = valid_mask.sum(axis=0).astype(int)   # shape (along_track,)
        cloud_count = cloud_mask.sum(axis=0).astype(int)   # shape (along_track,)

        CF = cloud_count / valid_count if valid_count > 0 else np.nan

        # Caclulate All-Sky mean LWP and IWP: 
        # -------------------------------------------------------------------------------
        if not want_ice:
            z = np.asarray(ACMCOM.height_layer)
            L,N = lwc.shape
            LWP = np.full(N, 0.0)
            IWP = np.full(N, 0.0)
            for i in range(N):
                if not valid_mask[i]:
                    continue # -> invalid column, exclude from mean
                
                zi = z[:,i]
                # Create mask for valid heights
                mL = (lwc[:,i] < MAX) & np.isfinite(lwc[:, i]) & np.isfinite(zi)
                mI = (iwc[:,i] < MAX) & np.isfinite(iwc[:, i]) & np.isfinite(zi)

                # Integrate; abs() handles increasing or decreasing height
                LWP[i] = np.abs(np.trapezoid(lwc[mL, i], x=zi[mL])) if mL.any() else 0.0
                IWP[i] = np.abs(np.trapezoid(iwc[mI, i], x=zi[mI])) if mI.any() else 0.0


            # All-sky means: clear valid columns contribute 0; invalid columns are NaN
            # Update valid mask to lwp- and iwp-threshold
            valid_mask_l = valid_mask & (LWP < THRESHOLD_lwp)
            valid_mask_i = valid_mask & (IWP < THRESHOLD_iwp)
            LWP_all = np.where(valid_mask_l, LWP, np.nan)
            IWP_all = np.where(valid_mask_i, IWP, np.nan)

            LWP_mean = float(np.nanmean(LWP_all)) if np.any(valid_mask_l) else np.nan
            IWP_mean = float(np.nanmean(IWP_all)) if np.any(valid_mask_i) else np.nan

            print(f"Mean All-Sky (1D):   <LWP> = {LWP_mean:.2f} kg m^-2   <IWP> = {IWP_mean:.2f} kg m^-2")
            print(f'     All-Sky (1D): Max(LWP) = {np.nanmax(LWP_all):.2f} kg m^-2   Max(IWP) = {np.nanmax(IWP_all):.2f} kg m^-2')
            # -----------------------------------------------------------------------------

    return CF

def get_property(BMAFLX, ACMCOM, idx=None):
    SZA_list = BMAFLX.solar_zenith_angle[:,1]  # Nadir view is in element one
    PHI_list = BMAFLX.solar_azimuth_angle[:,1]

    
    if idx is not None: SZA_list = SZA_list[idx]; PHI_list = PHI_list[idx]
    SZA_min = np.nanmin(SZA_list)
    SZA_max = np.nanmax(SZA_list)
    PHI = np.nanmean(PHI_list)
   
    h = ACMCOM.height_level[1, :]   # select level index 1 (as in MakeRTM.py)
    mask = (h < 1e30) & np.isfinite(h)
    zout = np.nanmean(h[mask]) / 1000.0  # km

    return SZA_min, SZA_max, PHI, zout

def AssDomain(data, shape=(11,11)):
    Nx, Ny = shape
    # print('\n\n---------------------- AssDomain -------------------')
    # print('     AssDomainSize', Nx, Ny)
    # print('     Initial data.shape:          ', data.shape)
    start_x = int(data.shape[-1]/2 - Nx/2); start_y = int(data.shape[-2]/2 - Ny/2)
    end_x   = int(data.shape[-1]/2 + Nx/2); end_y   = int(data.shape[-2]/2 + Ny/2)
    data = data[start_x:end_x, start_y:end_y]
    # print('     start, end:', start_x, end_x)
    # print('     Removed Buffer -> data.shape:', data.shape)
    # print('----------------------------------------------------\n\n')

    data = np.nanmean(data, axis=(0,1))
    return data




class Scene:
    def __init__(self,  Name='',  verbose=False):
        self.Name=Name
        self.verbose=verbose
        return

    def ExtractData(self, ia, shape=(1,1), idx_scene=None, BMAFLX=None, pyranometer_data=None, want_info=False):
        data = self.solar_eup[ia]
        
        if 'montecarlo' in self.fn:
            data = AssDomain(data, size=shape)
        elif 'disort' in self.fn and want_average_line:
            w_size = 10 # Match BBR footprint
            data = np.convolve(data, np.ones(w_size), 'same') / w_size

        if want_info:
            print(f"Extracted data from {self.fn}:\n"
                f"shape {shape}\n\n |------------------------|\n"
                f" |     data  = {data:6.2f}     |\n" 
                f" |------------------------|\n"
            )
            if 'TOA' in self.fn: 
                # BMAFLX = BMAFLX.solar_combined_top_of_atmosphere_flux
                # print(f'    BMAFLX = {BMAFLX}')
                1
            else:
                # pyranometer_data = np.loadtxt('DATA/SUR_Observations.txt') # Called outside OrbitIDs loop
                pyranometer_data = pyranometer_data[idx_scene]
                print(f'    pyranometer = {pyranometer_data}\n\n\n')


        
        flux_file.write(f'{data:8.2f} {libRad.fn} \n')
        # info_file.write(f'{SceneName} {ia:4},{iacr:3} {source_str:8} {data:.2f} W/m2   ({ProductFile})\n')

        return
                                                                                                       
    
    def plot_temporal_pyranometer(self, filename="SUR/pyranometer_temporal.csv"):
        data = np.genfromtxt(
            filename,
            delimiter=";",
            skip_header=1, # skip header
            dtype=str
        )
       # Keep only rows with valid time + flux
        time = []
        flux = []

        for row in data:
            try:
                t = datetime.strptime(row[2], "%d.%m.%Y %H:%M") + timedelta(hours=1)
                f = float(row[3].replace(",", "."))
                time.append(t)
                flux.append(f)
            except:
                continue  # skip footer / bad rows

        time = np.array(time)
        flux = np.array(flux)

        # Time of interest
        mark_time = datetime(2024, 9, 14, 16, 5)
        # Find closest index
        idx = np.argmin(np.abs(time - mark_time))

        # ---- Plot ----
        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(time, flux)
        ax.plot(time[idx], flux[idx], "ro", markersize=5, label="Reference observation")  # red dot


        ax.set_ylabel(r"Pyranometer $F_{\mathrm{SUR}}^{\downarrow}$ [W/m$^2$]")
        ax.set_xlabel("Local Time [UTC+2]")


        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    
        title = "Observed Temporal Surface Downward SW Flux (Pyranometer)"
        fig.suptitle(title, fontsize=FONTSIZE, y=.93)

        ax.legend(  loc='upper left', 
                    framealpha=0.9, 
                    borderaxespad=0.0,                   # space to axes
                    borderpad=0.25, labelspacing=0.25,   # compact box
                    markerscale=1.5, # -> increase dot sizes to see on plot
                    fontsize=INFOSIZE*.8)
       
        # BG: ----- plot-adjustments for nicer looking plots -----------
        fig.tight_layout()
        ax.set_facecolor('#f0f0f0') # Axes background (warm light grey)
        # Grid: major dashed, minor dotted
        ax.grid(which='major', linestyle='--', alpha=0.4)
        ax.grid(which='minor', linestyle=':',  alpha=0.2)
        ax.minorticks_on()

        # remove top/right border
        for spine in ['top','right']:
            ax.spines[spine].set_visible(False)
        # -------------------------------------------------------------
        
       
        pngfile = (plotdir_base + 'pyranometer_temporal_Orbit01690C.png')

        print("pngfile", pngfile)
        plt.savefig(pngfile)
        plt.close()

        return
            
    def plot_all_levels(self, Scene2=None, idx_scene=None, ia=None):
        fig = plt.figure(figsize=(10,4))
        ax = fig.add_subplot(1,1,1)

        x = self.latitude

        xlabel = "Latitude"; xlabel_specs = r" [N$^\circ$]"
        ylabel = "Altitude"; ylabel_specs = r" [km]"
        if        'eup' in self.fn: cblabel = r"$F_{\mathrm{TOA}}^{\uparrow}$ [W/m$^2$]"
        else:                       cblabel = r"$F_{\mathrm{TOA}}^{\downarrow}$ [W/m$^2$]"

        # 'viridis' 'inferno' 'coolwarm'    cmap = plt.get_cmap('jet')
        cmap = plt.get_cmap('inferno')
        xmin = self.latitude.min() #-90 #
        xmax = self.latitude.max() #90 #

       
        title = ''
        pl_list = []
        if plot_type == 'all_levels_solar': #############################################################
            z = ACMCOM.height_level / 1000.0  # [km]¨

            data = self.solar_eup   # (along_track, height_level)
            data = data.T           # (height_level, along_track)

            
            if       'disort' in self.fn: label = '  [DISORT]'
            elif 'montecarlo' in self.fn: label = '  [MYSTIC]'
            else: label= 'something went wrong';clabel=''

            if        'eup' in self.fn: title += "SW Upward Flux"
            else:                       title += "SW Downward Flux"
        
            # Print out calculated STDs
            if 'montecarlo' in self.fn:
                data_std = np.nanmean(self.solar_eup_std[data.T > 0])
                print(f'Mean STD = {data_std:.2f}  W/m2')

            ax_list         = [ax]
            data_list       = [data]
            cblabel_list    = [cblabel]
            label_list      = [label]
            data_diff       = None

        elif plot_type == 'all_levels_solar_diff': #############################################################
            z = ACMCOM.height_level / 1000.0  # [km]
            data = self.solar_eup.T - Scene2.solar_eup.T
            

            if   'disort' in self.fn and 'montecarlo' in Scene2.fn: label = '  [1D - 3D]'
            elif 'disort' in Scene2.fn and 'montecarlo' in self.fn: label = '  [3D - 1D]'
            else: label= 'something went wrong';clabel=''

            
            if        'eup' in self.fn: title += "SW Upward Flux Difference";   cblabel = r"$\Delta F_{\mathrm{TOA}}^{\uparrow}$ [W/m$^2$]"
            else:                       title += "SW Downward Flux Difference"; cblabel = r"$\Delta F_{\mathrm{TOA}}^{\downarrow}$ [W/m$^2$]"
            

            ax_list         = [ax]
            data_list       = [data]
            cblabel_list    = [cblabel]
            label_list      = [label]
            data_diff       = None
        
        elif plot_type == 'all_levels_solar_subplots': #############################################################
            fig, axes = plt.subplots(1, 3, figsize=(10,4), sharey=True)
            fig, axes = plt.subplots(3, 1, figsize=(10,10), sharey=True)
            ax1, ax2, ax3 = axes

            

            z = ACMCOM.height_level / 1000.0  # [km]


            data1 = self.solar_eup   # (along_track, height_level)
            data1 = data1.T          # (height_level, along_track)

            data2 = Scene2.solar_eup  
            data2 = data2.T        

            data_diff = self.solar_eup.T - Scene2.solar_eup.T


            if        'eup' in self.fn: title += "SW Upward Flux";   cblabel_diff = r"$\Delta F_{\mathrm{TOA}}^{\uparrow}$ [W/m$^2$]"
            else:                       title += "SW Downward Flux"; cblabel_diff = r"$\Delta F_{\mathrm{TOA}}^{\downarrow}$ [W/m$^2$]"

            
            if   'disort' in self.fn and 'montecarlo' in Scene2.fn:  label1 = '  [DISORT]'; label2 = '  [MYSTIC]'; label_diff = '  [1D - 3D]'
            elif 'disort' in Scene2.fn and 'montecarlo' in self.fn:  label1 = '  [MYSTIC]'; label2 = '  [DISORT]'; label_diff = '  [3D - 1D]'



            ax_list         = [ax1, ax2, ax3]
            data_list       = [data1, data2, data_diff]
            cblabel_list    = [cblabel, cblabel, cblabel_diff]
            label_list      = [label1, label2, label_diff]

            



        #----------------- Add clouds ----------------------
        if want_cloud_on_plot:  
            # For cloudmask
            # 0: clear sky, 1: cloudy sky
            cloud_mask = (ACMCOM.cloud_flag == 1)   

            if 'C' in self.Name: # lat [small->large->small] -> not monotonously increasing -> problem plotting
                N = len(x)
                half_N = int(N/2)
                left = slice(0, half_N)
                right = slice(half_N, N)

                if np.nanmax(data_list[0][:, left]) > np.nanmax(data_list[0][:, right]):
                    sl = left
                else:
                    sl = right
                    ia_station = ia - half_N

                x = x[sl]
                z = z[:, sl]
                data_list = [d[:, sl] for d in data_list]
                cloud_mask = cloud_mask[:, sl]

                
            
        # Set limits -----------------------------
        if modify_xlim:
            lat_min, lat_max = lat_ranges
            mask = (x > lat_min) & (x < lat_max) 
        else:
            lat_min, lat_max = np.nanmin(x), np.nanmax(x)

        SZA_min, SZA_max, PHI_mean, zout = get_property(BMAFLX, ACMCOM, idx=ia)

        mask = (x > lat_min) & (x < lat_max)
    




        for i, (ax, data, cblabel, label) in enumerate(zip(ax_list, data_list, cblabel_list, label_list)):
                        
                        #------------------------------------------------------------------------#
                        # Fix .nc file given "old" bug in code. Is fixed in newer versions
                        # arr = np.empty_like(data)
                        # for j in range(data.shape[1]):   # loop over along-track
                        #     col = data[:, j]
                            
                        #     is_zero = (col == 0) | (np.isnan(col))
                        #     arr[:, j] = np.concatenate([col[is_zero], col[~is_zero]])
                        # data = arr
                        #------------------------------------------------------------------------#


            
            data = data[::-1, :]         # reverse altitude-order   
            
            if i != 1:
                vmin = np.nanmin(data[:,mask])
                vmax = np.nanmax(data[:,mask])
                if '_eup' in additional_spesifications: vmax = 380 
                else: vmax = 500
            if plot_type == 'all_levels_solar_diff' or (plot_type == 'all_levels_solar_subplots' and i == 2): # normalization of diff-plot
                vmax = np.max(np.abs([vmin, vmax]))
                vmax = 400
                vmin = -vmax

                cmap = plt.get_cmap('coolwarm')
            # ---------------------------------------

            # (height_level, along-track)
            h_max = np.nanmax(z[1:,:][cloud_mask & mask]) + 1      # IMPORTANT NOTE: do not calculate Fluxes above around 40 km!
            # h_max = 12
            # h_max = 100





            ########################## PLOTTING #########################################
            # ---------- BG: Get additional data to plot (twin or stacked-axis) ----------------
            if quantity_list:
                add_profile_list = [] # List for ax2-legends
                for quantity in quantity_list:
                    ax, ax2 = add_profile_to_plot(fig, ax, ACMCOM, fsize=FONTSIZE, legend_list=add_profile_list, quantity=quantity, quantity_list=quantity_list, stacked=True, name=self.Name, data=data, slice=sl, h_max=h_max)
                if add_profile_list: ax2.legend(handles=add_profile_list, 
                                                loc='upper right', framealpha=0.7, 
                                                borderaxespad=0.0,                  # space to axes
                                                borderpad=0.25, labelspacing=0.25,   # compact box)
                                                fontsize=INFOSIZE*.8)
            # ------------------------------------------------------------------------


            if want_cloud_on_plot:
                cloud_plot = np.ma.masked_where(~cloud_mask, cloud_mask)
                ax.pcolormesh(
                    x, z[1:,:],cloud_plot,
                    shading="auto",cmap="gray_r",alpha=0.5,zorder=10,
                )
                # Fake scatter for legend: white square
                ax.scatter([], [], s=10, marker="s", color="white", alpha=0.4, label="Cloud-Mask")
            
            ##### Add Station Point on Figure #####
            ax.plot(x[ia_station], 0.03, 'ro', markersize=7, zorder=20, label='Observation site')
        
            im = ax.pcolormesh(x, z, data,
                    shading='auto', cmap=cmap, vmin=vmin, vmax=vmax
                )
            ######################################################################################
            




            # add colorbar with flux label
            if quantity_list: 
                cax = fig.add_axes([0.8, 0.55, 0.02, 0.3]) # create colorbar axis: [left, bottom, width, height] in figure coordinates
                cb = fig.colorbar(im, cax=cax, pad=0.005, extend='both', extendrect=True)
            else: 
                cb = fig.colorbar(im, ax=ax, pad=0.005, extend='both', extendrect=True)
            cb.set_label(cblabel, size=INFOSIZE*.8)
            cb.ax.tick_params(labelsize=INFOSIZE * .8)
            cb.outline.set_visible(False)
            cb.ax.set_facecolor('#f7f7f7')

            if i == 0:
                leg = ax.legend(  loc='upper left', 
                            framealpha=0.9, 
                            borderaxespad=0.0,                   # space to axes
                            borderpad=0.25, labelspacing=0.25,   # compact box
                            markerscale=1.5, # -> increase dot sizes to see on plot
                            fontsize=INFOSIZE*.8)
                leg.get_frame().set_facecolor("lightgray")   # light grey background          

            ax.set_xlim(lat_min, lat_max)
            ax.set_ylim(0, h_max)
            # ax.set_xlabel(xlabel + xlabel_specs, fontsize=INFOSIZE*.9)
            # if i == 0: ax.set_ylabel(ylabel + ylabel_specs, fontsize=INFOSIZE)
            ax.set_ylabel(ylabel + ylabel_specs, fontsize=INFOSIZE*.9)
            if i == 2: ax.set_xlabel(xlabel + xlabel_specs, fontsize=INFOSIZE) 
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.tick_params(axis='both', which='major', labelsize=INFOSIZE*.8)
            ax.tick_params(axis='both', which='minor', labelsize=INFOSIZE*.8)
            ax.set_title(label, fontsize=INFOSIZE)
        
        fig.suptitle(title, fontsize=FONTSIZE, y=1, x=.47)
        line1 = f"{self.Name} - {date} {time} (UTC)"
        if modify_xlim:
            line2 = fr"SZA(obs.site)={SZA_min:.0f}$^\circ$ $\phi$(obs.site)={PHI_mean:.0f}$^\circ$"
        else: 
            # line2 = fr"SZA=⟨{SZA_min:.0f}, {SZA_max:.0f}⟩$^\circ$ $⟨\phi⟩$={PHI_mean:.0f}$^\circ$"
            line2 = ''
        plt.figtext(0.49, 0.002, f"{line1}      {line2}", fontsize=INFOSIZE*.7)         
        plt.figtext(0.001, 0.004, baseline_str, fontsize=FONTSIZE*0.45)
        # Orbit nr: self.Name

        # BG: ----- plot-adjustments for nicer looking plots -----------
        fig.tight_layout()
        # Axes background (warm light grey)
        ax.set_facecolor('#f0f0f0')
        # Grid: major dashed, minor dotted
        ax.grid(which='major', linestyle='--', alpha=0.4)
        ax.grid(which='minor', linestyle=':',  alpha=0.2)
        ax.minorticks_on()

        # remove top/right border
        for spine in ['top','right']:
            ax.spines[spine].set_visible(False)
        # -------------------------------------------------------------
        
        add_spec = '_eup' if 'eup' in additional_spesifications else '_edn_edir'
        pngfile = (
            plotdir_base + SceneName + '_' + librad_version + '_' + plot_type + add_spec + '.png'
            if not want_closeup else
            plotdir_base + SceneName + '_' + librad_version + '_' + plot_type + add_spec + '_Closeup.png'
        )

        
        print("pngfile", pngfile)
        plt.savefig(pngfile)
        plt.close()

        return


        
    def InterpolateAngstrom(self, ACMCOM, angstrom_exponent=1):
        AMACD_lat = self.latitude
        AMACD_lon = self.longitude
        #
        # MSI provides angstrom_exponent for different wavelength
        # intervals. For the moment choose the longest wavelength one as
        # for the synthetic data the other one appears unrealistic.
        #
        AMACD_ang = self.aerosol_angstrom_exponent[:,:,angstrom_exponent] #lambda [670,865]
        ACMCOM_lat = np.transpose(ACMCOM.latitude)
        ACMCOM_lon = np.transpose(ACMCOM.longitude)

        # Geometry information is on tie grid which is similar to data grid
        # in along-track direction (and slightly not-aligned),
        # but with fewer points in the across-track direction.
        # Interpolate to full grid
        points = (AMACD_lon.flatten(), AMACD_lat.flatten())
        values = AMACD_ang.flatten()
        xi = (ACMCOM_lon.flatten(), ACMCOM_lat.flatten())

        AMACD_ang = griddata(points, values, xi, method='nearest')
        AMACD_ang = np.reshape(AMACD_ang, ACMCOM_lat.shape)
        self.aerosol_angstrom_exponent = AMACD_ang
        return

    def ReadEarthCAREh5(self, fn, ACM3D=None, verbose=False, Resolution='', RemoveMissingData=False):

        atmosphere=1 ################ based on ACM-CAP -> GOOD
        # atmosphere=0 ################ Composit profile -> BAD
     
        if verbose:
            print("Reading EarthCARE L2 file:", fn)

        file    = h5py.File(fn, 'r')
        self.fn= fn
        
        if 'libRad' in fn:
            SD=file
        elif Resolution != '':
            SD=file['ScienceData'][Resolution]
        else:
            SD=file['ScienceData']

        if verbose:
            for key in SD.keys():
                print(key)
            #            for item in SD.items():
            #                print(item)


        #if 'ACM_3D_' in fn:
        if 'ALL_3D_' in fn:
            #            print(SD['index_construction'].attrs.keys())
            #            print(SD['index_construction'].attrs['missing_value'])
            SDSpecificProductHeader = file['HeaderData']['VariableProductHeader']['SpecificProductHeader']
            #print('Tove2')
            self.nadir_pixel_index = SDSpecificProductHeader['nadir_pixel_index'][()]
            #print('Tove3')
            self.number_pixels_along_track_assessment_domain = SDSpecificProductHeader['number_pixels_along_track_assessment_domain'][()]
            self.number_pixels_across_track_assessment_domain = SDSpecificProductHeader['number_pixels_across_track_assessment_domain'][()]
            self.index_construction=SD['index_construction'][()]
            self.along_track_shape=self.index_construction.shape[1]
            self.across_track_shape=self.index_construction.shape[0]
            #            self.index_construction_quality_status=SD['index_construction_quality_status'][()]
            self.latitude=SD['latitude'][()]
            self.longitude=SD['longitude'][()]
            self.missing_value = SD['index_construction'].attrs['missing_value']
            self.number_pixels_along_track_buffer_zone_back_view = SD['number_pixels_along_track_buffer_zone_back_view'][()]
            self.number_pixels_along_track_buffer_zone_fore_view = SD['number_pixels_along_track_buffer_zone_fore_view'][()]
            self.number_pixels_across_track_buffer_zone = SD['number_pixels_across_track_buffer_zone'][()]
            self.start_along_track_assessment_domain_day_3d=SD['start_along_track_assessment_domain_day_3d'][()]
            self.number_pixels_along_track_buffer_zone_back_view=SD['number_pixels_along_track_buffer_zone_back_view'][()]
            self.number_pixels_along_track_buffer_zone_fore_view=SD['number_pixels_along_track_buffer_zone_fore_view'][()]
            self.number_pixels_across_track_buffer_zone=SD['number_pixels_across_track_buffer_zone'][()]
            
            # Remove all indices equal 0. Does not appear to have any valid data
            if RemoveMissingData:
                self.indx = np.where(self.index_construction >0 )#> self.missing_value)
                self.along_track_shape =  int(len(self.indx[0])/self.across_track_shape)
                self.index_construction= self.index_construction[self.indx]
                self.index_construction = np.reshape(self.index_construction,(self.across_track_shape,self.along_track_shape))
                self.latitude = np.reshape(self.latitude[self.indx],(self.across_track_shape,self.along_track_shape))
                self.longitude = np.reshape(self.longitude[self.indx],(self.across_track_shape,self.along_track_shape))

        elif 'AM__ACD' in fn:
            SDSpecificProductHeader = file['HeaderData']['VariableProductHeader']['SpecificProductHeader']
            self.latitude=SD['latitude'][()]
            self.longitude=SD['longitude'][()]
            self.aerosol_angstrom_exponent=SD['aerosol_angstrom_exponent'][()]
            self.aerosol_angstrom_exponent = self.aerosol_angstrom_exponent[:,:,:]

        elif 'ACM_RT' in fn:
            SDSpecificProductHeader = file['HeaderData']['VariableProductHeader']['SpecificProductHeader']
            #           self. = SDSpecificProductHeader[''][()]
            self.latitude=SD['latitude'][()]
            self.longitude=SD['longitude'][()]
            self.latitude_active=SD['latitude_active'][()]
            self.longitude_active=SD['longitude_active'][()]
            self.height_layers=SD['height_layers'][()]
            self.height_levels=SD['height_levels'][()]
            self.flux_up_solar_1d_all_sky=SD['flux_up_solar_1d_all_sky'][()]
            self.flux_up_solar_3d_all_sky=SD['flux_up_solar_3d_all_sky'][()]
            
            self.flux_up_thermal_1d_all_sky=SD['flux_up_thermal_1d_all_sky'][()]
            self.flux_up_thermal_3d_reference_height_all_sky=SD['flux_up_thermal_3d_reference_height_all_sky'][()]

            # BG: only single atmosphere here!
            self.flux_up_solar_1d_all_sky=self.flux_up_solar_1d_all_sky[0,:,:]
            self.flux_up_solar_3d_all_sky=self.flux_up_solar_3d_all_sky[0,:,:]
            self.flux_up_thermal_1d_all_sky=self.flux_up_thermal_1d_all_sky[0,:,:]
            self.flux_up_thermal_3d_reference_height_all_sky=self.flux_up_thermal_3d_reference_height_all_sky[0,:]   

            # BG: add quality-status
            self.quality_status = SD['quality_status'][()]         
            
        elif 'ACM_COM' in fn:
            # self.ice_water_content=SD['ice_water_content'][()]*1000 # Convert from kg/m**3 to g/m**3
            # self.ice_effective_radius=SD['ice_effective_radius'][()]*1e+6 # Convert from m to um
            # self.liquid_water_content=SD['liquid_water_content'][()]*1000 # Convert from kg/m**3 to g/m**3
            # self.liquid_effective_radius=SD['liquid_effective_radius'][()]*1e+6 # Convert from m to um
            # self.aerosol_extinction=SD['aerosol_extinction'][()]*1000 # Convert from /m to /km
            self.ice_water_content = np.asarray(SD['ice_water_content'][()], dtype=np.float64) * 1000.0
            self.ice_effective_radius = np.asarray(SD['ice_effective_radius'][()], dtype=np.float64) * 1e6
            self.liquid_water_content = np.asarray(SD['liquid_water_content'][()], dtype=np.float64) * 1000.0
            self.liquid_effective_radius = np.asarray(SD['liquid_effective_radius'][()], dtype=np.float64) * 1e6
            self.aerosol_extinction = np.asarray(SD['aerosol_extinction'][()], dtype=np.float64) * 1000.0
            self.cloud_flag = np.asarray(SD['cloud_flag'][()], dtype=np.float64)

            self.ice_water_content_units = SD['ice_water_content'].attrs['units']
            self.ice_water_content_units = 'kg/m**3'
            self.ice_effective_radius_units = SD['ice_effective_radius'].attrs['units']
            self.ice_effective_radius_units = 'um'    
            self.liquid_water_content_units = SD['liquid_water_content'].attrs['units']
            self.liquid_water_content_units = 'kg/m**3'
            self.aerosol_extinction_units = SD['aerosol_extinction'].attrs['units']
            self.aerosol_extinction_units = '1/km'
            self.aerosol_classification=SD['aerosol_classification'][()]
                    # 0:Clear/not aerosol
                    # 10:Dust
                    # 11:Sea Salt
                    # 12:Continental Pollution
                    # 13:Smoke
                    # 14:Dusty smoke
                    # 15:Dusty mix
                    # 25:Stratospheric Ash
                    # 26:Stratospheric Sulfate
                    # 27:Stratospheric Smoke
            self.time = SD['time'][()]
            self.time_units = 'seconds since 2000-1-1 00:00:00.0 0:00'
            self.latitude=SD['latitude'][()]
            self.longitude=SD['longitude'][()]
            self.latitude_active=SD['latitude_active'][()]
            self.longitude_active=SD['longitude_active'][()]
            self.height_layer=SD['height_layer'][()]
            self.height_level=SD['height_level'][()]
            self.pressure_level=SD['pressure_level'][()]
            self.pressure_layer_mean=SD['pressure_layer_mean'][()]
            self.temperature_level=SD['temperature_level'][()]
            self.temperature_layer_mean=SD['temperature_layer_mean'][()]
            self.volume_mixing_ratio_layer_mean_O3=SD['volume_mixing_ratio_layer_mean_O3'][()]
            self.volume_mixing_ratio_layer_mean_O2=SD['volume_mixing_ratio_layer_mean_O2'][()]
            self.specific_humidity_layer_mean=SD['specific_humidity_layer_mean'][()]
            self.specific_humidity_layer_mean_units = SD['specific_humidity_layer_mean'].attrs['units']
            self.volume_mixing_ratio_layer_mean_CO2=SD['volume_mixing_ratio_layer_mean_CO2'][()]
            self.volume_mixing_ratio_layer_mean_CH4=SD['volume_mixing_ratio_layer_mean_CH4'][()]
            self.volume_mixing_ratio_layer_mean_N2O=SD['volume_mixing_ratio_layer_mean_N2O'][()]

            self.liquid_water_content = self.liquid_water_content[atmosphere,:,:]
            self.liquid_effective_radius = self.liquid_effective_radius[atmosphere,:,:]
            self.ice_water_content = self.ice_water_content[atmosphere,:,:]
            self.ice_effective_radius = self.ice_effective_radius[atmosphere,:,:]
            self.aerosol_extinction = self.aerosol_extinction[atmosphere,:,:]
            self.aerosol_classification = self.aerosol_classification[atmosphere,:,:]
            self.cloud_flag = self.cloud_flag[atmosphere,:,:]

            self.surface_temperature = SD['surface_temperature'][()]
            self.albedo_direct_radiation_surface_visible = SD['albedo_direct_radiation_surface_visible'][()]
            self.albedo_direct_radiation_surface_near_infrared = SD['albedo_direct_radiation_surface_near_infrared'][()]
            self.albedo_diffuse_radiation_surface_visible = SD['albedo_diffuse_radiation_surface_visible'][()]
            self.albedo_diffuse_radiation_surface_near_infrared = SD['albedo_diffuse_radiation_surface_near_infrared'][()]

            # self.wavelengths_thermal_surface_emissivity = SD['wavelengths_thermal_surface_emissivity'][()]
            self.wavelengths_thermal_surface_emissivity = SD['wavenumbers_thermal_surface_emissivity'][()]
            self.types_surface_emissivity  = SD['types_surface_emissivity'][()]
            self.surface_emissivity_table  = SD['surface_emissivity_table'][()]
            self.surface_emissivity_type_index  = SD['surface_emissivity_type_index'][()]

            # BG: Variables needed for Ovean-Wave model 
            self.surface_albedo_classification = SD['surface_albedo_classification'][()]
            self.wind_speed_at_10_meters = SD['wind_speed_at_10_meters'][()]

            # BG: add quality-status
            self.quality_status = SD['quality_status'][()]
            self.quality_status = self.quality_status[atmosphere,:]

            
            #            print('self.liquid_water_content.shape', self.liquid_water_content.shape, self.liquid_water_content.max(),
            #                  self.aerosol_extinction.shape, self.aerosol_classification.shape)
            # Remove missing data. Not fully implemented
            if RemoveMissingData:
                self.surface_temperature = np.reshape(self.surface_temperature[ACM3D.indx],(ACM3D.across_track_shape,ACM3D.along_track_shape))
                self.latitude = np.reshape(self.latitude[ACM3D.indx],(ACM3D.across_track_shape,ACM3D.along_track_shape))
                self.longitude = np.reshape(self.longitude[ACM3D.indx],(ACM3D.across_track_shape,ACM3D.along_track_shape))
                
        elif 'BMA_FLX' in fn:
            SDGroup=file['ScienceData']
            self.bbr_direction = SDGroup['bbr_direction']
            if verbose:
                for key in SDGroup.keys():
                    print('key', key)
                    print('self.bbr_direction ', self.bbr_direction, self.bbr_direction.shape )
            self.bbr_directions = self.bbr_direction.shape[0]
            self.latitude=SD['latitude'][()]
            self.longitude=SD['longitude'][()]
            # Is time in this product WRONG??? It differs from time in acm_com product
            #            self.time = SD['time'][()]  
            #            self.time_units = 'seconds since 2000-1-1 00:00:00.0 0:00'
            self.solar_zenith_angle = SD['solar_zenith_angle'][()]
            self.solar_zenith_angle_units = 'degrees'
            # Azimuth angle between the sun and the north. Measured clockwise
            self.solar_azimuth_angle = SD['solar_azimuth_angle'][()]
            self.solar_azimuth_angle_units = 'degrees'
            self.viewing_zenith_angle = SD['viewing_zenith_angle'][()]
            self.viewing_zenith_angle_units = 'degrees'
            self.viewing_azimuth_angle = SD['viewing_azimuth_angle'][()]
            self.viewing_azimuth_angle_units = 'degrees'

            self.solar_top_of_atmosphere_flux = SD['solar_top_of_atmosphere_flux'][()]
            self.solar_top_of_atmosphere_flux_units = 'W/m**2'
            self.solar_top_of_atmosphere_flux_error = SD['solar_top_of_atmosphere_flux_error'][()]
            self.solar_top_of_atmosphere_flux_error_units = 'W/m**2'
            self.solar_top_of_atmosphere_flux_quality_status = SD['solar_top_of_atmosphere_flux_quality_status'][()]

            self.solar_combined_top_of_atmosphere_flux = SD['solar_combined_top_of_atmosphere_flux'][()]
            self.solar_combined_top_of_atmosphere_flux_units = 'W/m**2'
            self.solar_combined_top_of_atmosphere_flux_error = SD['solar_combined_top_of_atmosphere_flux_error'][()]
            self.solar_combined_top_of_atmosphere_flux_error_units = 'W/m**2'
            self.solar_combined_top_of_atmosphere_flux_quality_status = SD['solar_combined_top_of_atmosphere_flux_quality_status'][()]

            self.thermal_combined_top_of_atmosphere_flux = SD['thermal_combined_top_of_atmosphere_flux'][()]
            self.thermal_combined_top_of_atmosphere_flux_units = 'W/m**2'
            self.thermal_combined_top_of_atmosphere_flux_error = SD['thermal_combined_top_of_atmosphere_flux_error'][()]
            self.thermal_combined_top_of_atmosphere_flux_error_units = 'W/m**2'
            self.thermal_combined_top_of_atmosphere_flux_quality_status = SD['thermal_combined_top_of_atmosphere_flux_quality_status'][()]

            # BG: add quality-status
            self.quality_status = SD['quality_status'][()]

            # Remove missing data. Not fully implemented
            RemoveMissingData=True
            if RemoveMissingData:
                #               print('latitude', self.latitude.shape, self.solar_zenith_angle.shape, self.solar_top_of_atmosphere_flux.shape); 
                self.indx = np.where(self.latitude <10000 )#> self.missing_value)
                self.along_track_shape=self.latitude.shape
                self.latitude = self.latitude[self.indx]
                self.longitude  = self.longitude[self.indx]
                self.solar_combined_top_of_atmosphere_flux=self.solar_combined_top_of_atmosphere_flux[self.indx]
                self.solar_combined_top_of_atmosphere_flux_quality_status = self.solar_combined_top_of_atmosphere_flux_quality_status[self.indx]
                self.solar_top_of_atmosphere_flux=self.solar_top_of_atmosphere_flux[self.indx[0],:]
                self.solar_top_of_atmosphere_flux_quality_status = self.solar_top_of_atmosphere_flux_quality_status[self.indx[0],:]
                self.thermal_combined_top_of_atmosphere_flux=self.thermal_combined_top_of_atmosphere_flux[self.indx]
                self.thermal_combined_top_of_atmosphere_flux_quality_status = self.thermal_combined_top_of_atmosphere_flux_quality_status[self.indx]
                #                print('latitude', self.latitude.shape, self.solar_zenith_angle.shape, self.solar_top_of_atmosphere_flux.shape); exit()
                tmp = np.zeros((len(self.indx[0]), self.bbr_directions))
                for ib in np.arange(self.bbr_directions):
                    tmp[:, ib]  = self.solar_zenith_angle[self.indx, ib]
                self.solar_zenith_angle  = tmp
                tmp = np.zeros((len(self.indx[0]), self.bbr_directions))
                for ib in np.arange(self.bbr_directions):
                    tmp[:, ib]  = self.solar_azimuth_angle[self.indx, ib]
                self.solar_azimuth_angle  = tmp
                #                print('latitude', self.latitude.shape, self.solar_zenith_angle.shape); #exit()

        elif 'libRad' in fn:
            SD=file
            if verbose:
                for key in SD.keys():
                    print('key', key)
 
            self.latitude=SD['latitude'][()]
            # self.longitude=SD['longitude'][()] 
            # self.solar_zenith_angle = SD['solar_zenith_angle'][()]
            # self.solar_zenith_angle_units = 'degrees'
            # # Azimuth angle between the sun and the north. Measured clockwise
            # self.solar_azimuth_angle = SD['solar_azimuth_angle'][()]
            # self.solar_azimuth_angle_units = 'degrees'
            # self.viewing_zenith_angle = SD['viewing_zenith_angle'][()]
            # self.viewing_zenith_angle_units = 'degrees'
            # self.viewing_azimuth_angle = SD['viewing_azimuth_angle'][()]
            # self.viewing_azimuth_angle_units = 'degrees'

            self.solar_eup = SD['solar_eup'][()]
            try:
                self.thermal_eup = SD['thermal_eup'][()]
            except:
                1;
            # if 'mystic' in fn:
            ######### BG: testing something new 
            if 'montecarlo' in fn:
                self.solar_eup_std = SD['solar_eup_std'][()]
                self.thermal_eup_std = SD['thermal_eup_std'][()]
            self.solar_eup_units = 'W/m**2'
            self.thermal_eup_units = 'W/m**2'

        return

    def SetExtent(self):
        try:
            self.extent_left = self.longitude.min()
            self.extent_right = self.longitude.max()
            self.extent_bottom = self.latitude.min()
            self.extent_top = self.latitude.max()
        except:
            self.extent_bottom = self.latitude.min()
            self.extent_top = self.latitude.max()

    def WriteNetcdf(self, fn, verbose=True):
        if verbose:
            print("Writing libRadtran output to netcdf file: ", fn)

        ncf = Dataset(fn, 'w')

        along_tracks = self.latitude_active.shape[0]
        ncf.createDimension('along_track', along_tracks)

        latitude = ncf.createVariable('latitude',np.dtype('float').char,('along_track',))
        latitude.units = "degree_north" 
        latitude.long_name = "Latitude"
        latitude[:] = self.latitude_active

        solar_eup = ncf.createVariable('solar_eup',np.dtype('float').char,('along_track',))
        solar_eup.units = "W m-2" 
        solar_eup.long_name = "Solar upward flux at TOA"
        solar_eup[:] = self.solar_e
        
        solar_eup_std = ncf.createVariable('solar_eup_std',np.dtype('float').char,('along_track',))
        solar_eup_std.units = "W m-2" 
        solar_eup_std.long_name = "Standard deviation of solar upward flux at TOA"
        solar_eup_std[:] = self.solar_e_std

        thermal_eup = ncf.createVariable('thermal_eup',np.dtype('float').char,('along_track',))
        thermal_eup.units = "W m-2" 
        thermal_eup.long_name = "Thermal upward flux at TOA"
        thermal_eup[:] = self.thermal_e
        
        thermal_eup_std = ncf.createVariable('thermal_eup_std',np.dtype('float').char,('along_track',))
        thermal_eup_std.units = "W m-2" 
        thermal_eup_std.long_name = "Standard deviation of thermal upward flux at TOA"
        thermal_eup_std[:] = self.thermal_e_std
                
        ncf.close()
        
    def WriteNetcdf_all_levels(self, fn, shape=(1,1), verbose=True):
        if verbose:
            print("Writing libRadtran output to netcdf file: ", fn)

        ncf = Dataset(fn, 'w')

        along_tracks = self.latitude_active.shape[0]
        nHeightLevels = self.solar_e.shape[1]
        ncf.createDimension('along_track', along_tracks)
        ncf.createDimension('height_level', nHeightLevels)
                        # ncf.createDimension('Nx', shape[0])
                        # ncf.createDimension('Ny', shape[1])

        latitude = ncf.createVariable('latitude',np.dtype('float').char,('along_track',))
        latitude.units = "degree_north" 
        latitude.long_name = "Latitude"
        latitude[:] = self.latitude_active

        # --- Flux variables: 2D (along_track, height_level) ---
        if 'montecarlo' in fn:
            ### BG: Explenation ###
            #   Only take care of center pixel 
            #   Load as 2D matricies
            ix_center = self.solar_e.shape[2] // 2
            iy_center = self.solar_e.shape[3] // 2

            # print(f'center pixels for {self.solar_eup.shape} -> ({ix_center}, {iy_center})')
            # print(solar_eup.shape, self.solar_eup.shape)
            # print(self.solar_eup[1500, :, ix_center, iy_center])

            solar_eup = ncf.createVariable('solar_eup', np.dtype('float').char, ('along_track', 'height_level'))
            solar_eup.units = "W m-2"
            solar_eup.long_name = "Solar upward flux"
            solar_eup[:, :] = self.solar_e[:, :, ix_center, iy_center] 

            solar_eup_std = ncf.createVariable('solar_eup_std', np.dtype('float').char, ('along_track', 'height_level'))
            solar_eup_std.units = "W m-2"
            solar_eup_std.long_name = "Standard deviation of solar upward flux"
            solar_eup_std[:, :] = self.solar_e_std[:, :, ix_center, iy_center] 

            thermal_eup = ncf.createVariable('thermal_eup', np.dtype('float').char, ('along_track', 'height_level'))
            thermal_eup.units = "W m-2"
            thermal_eup.long_name = "Thermal upward flux"
            thermal_eup[:, :] = self.thermal_e[:, :, ix_center, iy_center] 

            thermal_eup_std = ncf.createVariable('thermal_eup_std', np.dtype('float').char, ('along_track', 'height_level'))
            thermal_eup_std.units = "W m-2"
            thermal_eup_std.long_name = "Standard deviation of thermal upward flux"
            thermal_eup_std[:, :] = self.thermal_e_std[:, :, ix_center, iy_center] 
        else: 
            solar_eup = ncf.createVariable('solar_eup', np.dtype('float').char, ('along_track', 'height_level'))
            solar_eup.units = "W m-2"
            solar_eup.long_name = "Solar upward flux"
            solar_eup[:, :] = self.solar_e

            solar_eup_std = ncf.createVariable('solar_eup_std', np.dtype('float').char, ('along_track', 'height_level'))
            solar_eup_std.units = "W m-2"
            solar_eup_std.long_name = "Standard deviation of solar upward flux"
            solar_eup_std[:, :] = self.solar_e_std

            thermal_eup = ncf.createVariable('thermal_eup', np.dtype('float').char, ('along_track', 'height_level'))
            thermal_eup.units = "W m-2"
            thermal_eup.long_name = "Thermal upward flux"
            thermal_eup[:, :] = self.thermal_e

            thermal_eup_std = ncf.createVariable('thermal_eup_std', np.dtype('float').char, ('along_track', 'height_level'))
            thermal_eup_std.units = "W m-2"
            thermal_eup_std.long_name = "Standard deviation of thermal upward flux"
            thermal_eup_std[:, :] = self.thermal_e_std

        ncf.close()
        
    def WriteNetcdf_mc_sample_grid(self, fn, shape, verbose=True):
        if verbose:
            print("Writing libRadtran output to netcdf file: ", fn)

        ncf = Dataset(fn, 'w')

        along_tracks = self.latitude_active.shape[0]
        nHeightLevels = self.solar_e.shape[1]
        ncf.createDimension('along_track', along_tracks)
        ncf.createDimension('Nx', shape[0])
        ncf.createDimension('Ny', shape[1])

        latitude = ncf.createVariable('latitude',np.dtype('float').char,('along_track',))
        latitude.units = "degree_north" 
        latitude.long_name = "Latitude"
        latitude[:] = self.latitude_active

        # --- Flux variables: 2D (along_track, height_level) ---
        solar_eup = ncf.createVariable('solar_eup', np.dtype('float').char, ('along_track', 'Nx', 'Ny'))
        solar_eup.units = "W m-2"
        solar_eup.long_name = "Solar upward flux"
        solar_eup[:, :] = self.solar_e

        solar_eup_std = ncf.createVariable('solar_eup_std', np.dtype('float').char, ('along_track', 'Nx', 'Ny'))
        solar_eup_std.units = "W m-2"
        solar_eup_std.long_name = "Standard deviation of solar upward flux"
        solar_eup_std[:, :] = self.solar_e_std

        thermal_eup = ncf.createVariable('thermal_eup', np.dtype('float').char, ('along_track', 'Nx', 'Ny'))
        thermal_eup.units = "W m-2"
        thermal_eup.long_name = "Thermal upward flux"
        thermal_eup[:, :] = self.thermal_e

        thermal_eup_std = ncf.createVariable('thermal_eup_std', np.dtype('float').char, ('along_track', 'Nx', 'Ny'))
        thermal_eup_std.units = "W m-2"
        thermal_eup_std.long_name = "Standard deviation of thermal upward flux"
        thermal_eup_std[:, :] = self.thermal_e_std

        ncf.close()
        



















###########################################################################################################################        
if __name__ == "__main__":
    start_time = datetime.now(timezone.utc) 
    
    

    # BG: ----- Things to remember ---------
    #   1. WANT_3D
    #   2. idx_scene
    #   3. fig_index
    # --------------------------------------

     
    WANT_3D  = False      # BG: used in flux plot (DISORT or MYSTIC)
    WANT_SUR = True


    ToDo_idx = 0
    ToDo = [
        'ExtractData',
        'plot_all_levels',
        'plot_temporal_pyranometer',
    ][ToDo_idx]
    
    
    # BG: additional settings
    AssDomainSize       = [(1,1), (3,3), (11,11)][0]   # Size of assessment domain (across-track, along-track) in libRadtran 3D MC
    want_quality_status = False                 # If want ACM-COM and BMA-FLX quality status
    want_EarthCARE_info = False                 # Sets Scene3 = ACM3D (in PlotLien)
    want_product2       = False                 # Sets Scene4 = librad2 (in PlotLine)
    want_average_line   = False                 # Average line of DISORT-flux-values
    want_ps             = False                 # BG: if want DISORT pseudospherical (WANT_3D overwrite want_ps)
    want_info           = False                 # Prints out SZA, Prints out CF
    if want_info: want_2D = False               # CF of 2D swat, or 1D nadir column
    stacked             = True                  # If add quanteties to plot, if should get own figure below
    verbose             = False

    # Notes:
    # Run with verbose -> printing -> 0.12h
    # Run without verbose -> no printing -> 0.11h



    # BG: Chose idx to select additional info on solar_both and thermal_both plot 
    #   Chose from: 
    #   [None, 'elevation', 'lwp', 'iwp', 'tot_wp', 'albedo', 'aerosols', 'surface_temperature', 'CF']
    #           NOTE: do not put 'CF' at the end of quantity_list nor with tot_wc 
    #           NOTE: put tot_wp before tot_wc
    quantity_list = False
    # quantity_list = ['tot_wp']
    # quantity_list = ['tot_wc']
    # quantity_list = ['tot_wp', 'tot_wc']
 


    idx_source = 0
    sources = [['solar'], 
              ['thermal'],
              ['solar', 'thermal']][idx_source] 

    OrbitIDs = []
    StationList = []
    sites = [
        # (Station,       lat,    lon) 
        ("Oslo",        59.942,	10.720),
        ("Karasjok",    69.464,	25.502),
        ("Gratangen",   68.732,	17.241), 
        ("Beitostølen", 61.251,	8.923),
        ("Værnes",      63.460,	10.931), 
        ("Østerås",     59.948,	10.603), 
        ("Trondheim",   63.415,	10.407), 
        ("Tromsø",      69.654,	18.937), 
        ("Bergen",      60.383,	5.333),  
        ("Hopen",       76.510,	25.013), 
        ("Brusdalen",   62.485,	6.480),  
        ("Jan-Mayen",   70.939,	-8.669), 
        ("Flesland",    60.289,	5.227),  
        ("Iskoras",     69.300,	25.346), 
        ("Rena",        61.376,	11.499), 
        ("Korgåsen",    69.936,	28.377), 
        ("Bjørnøya",    74.504,	18.998), 
        ("Filefjell",   61.178,	8.113),  
        ("Juvvasshøe",  61.678,	8.369),  
    ]

    idx_range = np.arange(0,len(sites))
    idx_range = np.arange(0,1)
    sites = [sites[i] for i in idx_range]


    DATA_FILES = Path("/homevip/bgre/Download/Frames_SurfaceOverpasses")

    # Locate start-index to start extracting observations from SUR_Observations.txt
    start_idx = 0 
    for i in range(idx_range[0]):
        data_files = DATA_FILES / sites[i][0]
        with open(data_files / "OrbitIDs.txt", "r") as f:
            for line in f:
                start_idx += 1
    print(f"Start-Index to extract observations: {start_idx}") 


    # Make list of OrbitIDs and Stations to loop through
    for site, _, _ in sites:
        data_files = DATA_FILES / site
        with open(data_files / "OrbitIDs.txt", "r") as f:
            print (f"\nOrbitsIDs for {site}:")
            for line in f:
                OrbitIDs.append(line.strip())
                StationList.append(site)
                print(line.strip(), end=" ")
            print()   

                           


            





    # BG: To distinguish minor modifications to .nc files
    additional_spesifications = '' 
    # additional_spesifications += '_TEST'
            # New mc_sample_grid




    if WANT_SUR:    additional_spesifications += '_21x21_SUR' if WANT_3D else '_SUR'
    else:           additional_spesifications += '_21x21_TOA' if WANT_3D else '_TOA'

    if 'plot_all_levels' in ToDo:
        additional_spesifications = '_AllLevels'
        # additional_spesifications = '_AllLevels_eup'

        # plot_type = 'all_levels_solar' 
        # plot_type = 'all_levels_solar_diff'; want_product2 = True
        plot_type = 'all_levels_solar_subplots'; want_product2 = True
        
        want_cloud_on_plot  = True
        modify_xlim         = True
        want_closeup        = True # Only change pdf-name

        if modify_xlim:
            lat_ranges = [(67.5, 75.5) ,  (68.5, 69)  , (68.6, 68.8)  , (67.95, 69.65)][-1]




  
    
    pathL2TestProducts_base = "/homevip/bgre/Download/Frames_SurfaceOverpasses" # pathL2TestProducts  = '/xnilu_wrk2/projects/NEVAR/data/EarthCARE_Real/'  
    ProductPathRTM      = './RESULTS/' # './netcdf/' 
    plotdir_base        = './figures/'   

      
    # latitude_wanted     = 40.0
    librad_type         = 'SWIA'
    version_identifier  = 'v01'
    source_str          = 'solar'



    # pick the correct model version
    
    if WANT_3D:
        librad_version = 'montecarlo_3D'
        if want_product2: 
            if want_ps:
                librad_version2 = 'disort_pseudospherical_1D' 
            else: 
                librad_version2 = 'disort_1D'
    else:
        if want_ps:
            librad_version = 'disort_pseudospherical_1D'
        else:
            librad_version = 'disort_1D'
        if want_product2: librad_version2 = 'montecarlo_3D'
   
   

    # Write data to files
    # rte_specs = 'MYSTIC_' if WANT_3D else 'DISORT_'
    rte_specs = 'MYSTIC' if WANT_3D else 'DISORT'
    obs_level = 'SUR_' if WANT_SUR else 'TOA_'
    folder    = 'DATA/'
    out_file_name1 = folder + obs_level + rte_specs + ".txt"
    # out_file_name2 = folder + obs_level + rte_specs + "output_fluxes_INFO.txt"

    pyranometer_data = np.loadtxt("DATA/SUR_Observations.txt")




    with open(out_file_name1, "w") as flux_file: #, open(out_file_name2, "w") as info_file:
        # info_file.write(f'{"SUR" if WANT_SUR else "TOA"} {rte_specs}\n'
        #                 'SceneName ial,iacr Source Flux (Product) \n'
        #                 '-----------------------------------\n')
        
        for OrbitID, Station in zip(OrbitIDs, StationList):
            pathL2TestProducts = pathL2TestProducts_base + f'/{Station}'
            
            if WANT_3D:
                mode_folder = 'MYSTIC/'
            else:
                if want_ps:
                    mode_folder = 'PSEUDOSPHERICAL/'
                else:
                    mode_folder = 'DISORT/' #'TWOSTR/' 






            # plotdir = os.path.join(plotdir_base, SceneName, mode_folder)
            # png_spesifications = ''
            

            # # Make sure it exists
            # os.makedirs(plotdir, exist_ok=True)
            # print(f'Direcotry for figures: {plotdir}')
            # -------------------------------------------------------






            # BG: Find; Baseline, Data, Time
            BB_baseline, BA_baseline, AC_baseline = '', '', ''
            
            Product ='ALL_3D_'#'ACM_COM' #'ACM_3D_'#'ACM_COM' #
            ProductPath = '*'+Product+'*'+OrbitID+'*'
            ProductFile = os.path.join(pathL2TestProducts, ProductPath, '*'+Product+'*.h5')
            try: 
                ProductFile = sorted(glob.glob(ProductFile))[0]
            except IndexError: # If do not have all products (ALL_3D here), do not have run simulations! Skipping
                print(f"\n     Skipping {OrbitID}. Do not find product {Product}") 
                if ToDo == "ExtractData": 
                    flux_file.write(f'{np.nan:8.2f} {OrbitID}. Not all data products found. Not simulated. \n')
                continue
            ACM3D = Scene(Name=OrbitID, verbose=verbose)
            ACM3D.ReadEarthCAREh5(ProductFile, verbose=verbose)
            ACM3D.SetExtent()
            
                        # indlat = find_nearest_id(ACM3D.latitude,latitude_wanted)
                        # latitude_have_ACM3D = ACM3D.latitude.flatten()[indlat]
                        # val = latitude_wanted
                        # e=ACM3D.latitude
                        # nearest = np.unravel_index(np.argmin(np.abs(e - val), axis=None), e.shape)
            # Extract Baseline ---------------------------   
            parts = ProductFile.split("ECA_EX", 1)
            out = parts[1][:2] if len(parts) > 1 else None
            if   out == 'BA': BA_baseline += ' ALL_3D'
            elif out == 'AC': AC_baseline += ' ALL_3D'
            elif out == 'BB': BB_baseline += ' ALL_3D'
            # print(out)
            #---------------------------------------------

            # Product ='ACM_RT_'#'ACM_COM' #
                # #ProductPath = 'ECA_EXAB_'+Product+'*'  #'ECA_EXAA_'+Product+'*'
                # #ProductPath = 'ECA_EXAB_'+Product+'*' #BG: marked out this for Orbit_05926C
                # # ProductPath = 'ECA_EXAC_'+Product+'*'
                # ProductPath = '*'+Product+'*'
                # ProductFile = os.path.join(pathL2TestProducts, SceneName, 'output', ProductPath, '*'+Product+'*.h5')
                # ProductFile = sorted(glob.glob(ProductFile))[0]
                # ACMRT = Scene(Name=SceneName, verbose=verbose)
                # ACMRT.ReadEarthCAREh5(ProductFile, verbose=verbose)
                # ACMRT.SetExtent()
                # # Extract Baseline ---------------------------   
                # parts = ProductFile.split("ECA_EX", 1)
                # out = parts[1][:2] if len(parts) > 1 else None
                # if   out == 'BA': BA_baseline += ' ACM_RT'
                # elif out == 'AC': AC_baseline += ' ACM_RT'
                # elif out == 'BB': BB_baseline += ' ACM_RT'
                # # print(out)
                #---------------------------------------------
            
            Product ='BMA_FLX'
            ProductPath = '*'+Product+'*'+OrbitID+'*'
            ProductFile = os.path.join(pathL2TestProducts, ProductPath, '*'+Product+'*.h5')
            ProductFile = sorted(glob.glob(ProductFile))[0]
            BMAFLX = Scene(Name=OrbitID, verbose=verbose)
            BMAFLX.ReadEarthCAREh5(ProductFile, Resolution='StandardResolution', verbose=verbose)
            BMAFLX.SetExtent()
                        # indlat = find_nearest_id(BMAFLX.latitude,latitude_wanted)
                        # latitude_have_BMAFLX = BMAFLX.latitude.flatten()[indlat]
                        # val = latitude_wanted
                        # e=BMAFLX.latitude
                        # nearest = np.unravel_index(np.argmin(np.abs(e - val), axis=None), e.shape)

            # if len(plot_types_librad)>0 and len(plot_types_flx)>0:
            #     indlats=[]
            #     for latitude_wanted in libRad.latitude:
            #         nearest = np.unravel_index(np.argmin(np.abs(BMAFLX.latitude - latitude_wanted), axis=None), BMAFLX.latitude.shape)
            #         indlats.append(nearest[0])
            #     libRad.BMAFLXindlats = indlats
            # Extract Baseline ---------------------------   
            parts = ProductFile.split("ECA_EX", 1)
            out = parts[1][:2] if len(parts) > 1 else None
            if   out == 'BA': BA_baseline += ' ' + Product
            elif out == 'AC': AC_baseline += ' ' + Product
            elif out == 'BB': BB_baseline += ' ' + Product
            

            # print(out)
            #---------------------------------------------


            Product ='ACM_COM'
            ProductPath = '*'+Product+'*'+OrbitID+'*'
            ProductFile = os.path.join(pathL2TestProducts, ProductPath, '*'+Product+'*.h5')
            ProductFile = sorted(glob.glob(ProductFile))[0]
            ACMCOM = Scene(Name=OrbitID, verbose=verbose)
            ACMCOM.ReadEarthCAREh5(ProductFile, verbose=verbose, ACM3D=ACM3D)
            ACMCOM.SetExtent()
                            # setE=False #True
                            # if setE:
                            #     ACMCOM.extent_left = -56.4
                            #     ACMCOM.extent_right = -56.1
                            #     ACMCOM.extent_bottom = 60.95
                            #     ACMCOM.extent_top = 61.35

            
                            # indlat = find_nearest_id(ACMCOM.latitude,latitude_wanted)
                            # latitude_have_ACMCOM = ACMCOM.latitude.flatten()[indlat]
            # Extract Baseline ---------------------------   
            parts = ProductFile.split("ECA_EX", 1)
            out = parts[1][:2] if len(parts) > 1 else None
            if   out == 'BA': BA_baseline += ' ' + Product
            elif out == 'AC': AC_baseline += ' ' + Product
            elif out == 'BB': BB_baseline += ' ' + Product
            # print(out)
            #---------------------------------------------

            # Find overpass time and closes across- and along-track indices for overpass
            ########################################################################################################################################################
            # Find latitude and longitude for the station
            for station, lat, lon in sites:
                if station == Station:
                    lat_, lon_ = str(lat), str(lon)
                    break
            out = Find_Overpass_Info.find_track_values(ACMCOM.fn, lat_, lon_)
            iacr  = out["across_index"] if WANT_SUR else 151
            ial   = out["along_index"]
            time_overpass = out["time_iso_utc"]





            # Fix Baseline
            AC_baseline = ", ".join(AC_baseline.split())
            BA_baseline = ", ".join(BA_baseline.split())
            BB_baseline = ", ".join(BB_baseline.split())
            baseline_str = "Baselines: "
            if AC_baseline != '':   baseline_str += f'AC = ({AC_baseline})  '
            if BA_baseline != '':   baseline_str += f'BA = ({BA_baseline})  '
            if BB_baseline != '':   baseline_str += f'BB = ({BB_baseline})  '
            


            # Extract Date
            date_num = '20' + ProductFile.split('20', 1)[1].split('T', 1)[0]
            date = date_num[6:8] + "." + date_num[4:6] + "." + date_num[0:4]

            # Extract Start-Time
            time = ProductFile.split('T')[1].split('Z')[0]
            time = time[:2] + ":" + time[2:4]
            # print(  f'\nAC_baseline = {AC_baseline}\nBA_baseline = {BA_baseline}\n' 
            #         f'{date} - {time}\n')
                        
    



            # Download libRadtran runs
            if source_str:
                Product = (
                    'libRad_' + 
                    version_identifier  + '_' +
                    librad_version      + '_' +
                    librad_type         + '_' +
                    source_str          + '_' +
                    OrbitID             + '_' +
                    Station             +
                    additional_spesifications
                                        + '.nc'
                )
                if want_product2: 
                    additional_spesifications2 = additional_spesifications
                    Product2 = (
                        'libRad_' + 
                        version_identifier  + '_' +
                        librad_version2     + '_' +
                        librad_type         + '_' +
                        source_str          + '_' +
                        OrbitID             + 
                        additional_spesifications2
                                            + '.nc'
                    )
            
            ProductPath = ProductPathRTM
            ProductFile = os.path.join(ProductPath, Product)
            # print('ProductFile1', ProductFile)
            ProductFile = sorted(glob.glob(ProductFile))[0]    
            libRad = Scene(Name=OrbitID, verbose=verbose)
            libRad.ReadEarthCAREh5(ProductFile, verbose=verbose)
            libRad.SetExtent()
            
            PrintStuff=False
            if PrintStuff:
                ia=0
                for lat, eup, std in zip(libRad.latitude, libRad.solar_eup, libRad.solar_eup_std ):
                    print(ia, lat, eup, std)
                    ia=ia+1
                exit()


            libRad2=None
            if want_product2:      
                ProductPath = ProductPathRTM
                ProductFile = os.path.join(ProductPath, Product2) # BG: removed this [+'*'+SceneName+'*.nc')] and changed Product -> Product2
                # print('ProductFile2', ProductFile)
                ProductFile = sorted(glob.glob(ProductFile))[0]
                libRad2 = Scene(Name=OrbitID, verbose=verbose)
                libRad2.ReadEarthCAREh5(ProductFile, verbose=verbose)
                libRad2.SetExtent()





                

            
                        
            print(
                 "\n\n\n================================================================================================================="
                 "\n    Productfile= ", ProductFile,
                 "\n    OrbitID    = ", OrbitID,
                 "\n    Specs      = ", ('SUR' if WANT_SUR else 'TOA'),
                        # Station location for EarthCARE overpass  
                f"\n    Across-track index : {iacr}"
                f"\n    Along-track index  : {ial}")
            print(f"    Overpass time (UTC): {time_overpass[:16].replace('T', ' ')}")
            print("=================================================================================================================")
            ##########################################################################################################################################################
                
            # NOTE ----------------------------------------------------
            # Decided to not use the following products for now, 
            # as ångstrom exponent lack most of places 
            # and this analysis only rely on small regions of interrest.  
            # ----------------------------------------------------------    
            # Product = 'AM__ACD'
            # ProductPath = '*'+Product+'*'+OrbitID+'*'  
            # ProductFile = os.path.join(pathL2TestProducts, ProductPath, '*'+Product+'*.h5')
            # ProductFile = sorted(glob.glob(ProductFile))[0]
            # # Extract Baseline ---------------------------
            # parts = ProductFile.split("ECA_EX", 1)
            # out = parts[1][:2] if len(parts) > 1 else None
            # if   out == 'BA': BA_baseline += ' ' + Product
            # elif out == 'AC': AC_baseline += ' ' + Product
            # elif out == 'BB': BB_baseline += ' ' + Product
            # print(out)
            #---------------------------------------------


            

            # Print INFO: --------------------------------------------------------------------------
            if want_info: 
                SZA_min, SZA_max, PHI_mean, zout = get_property(BMAFLX, ACMCOM)
                print("------------------------------ INFO -----------------------------")
                # SZA
                print(f"SZA (min) = {SZA_min}, SZA (max) = {SZA_max}")  
                # CF:
                dim = '2D' if want_2D else '1D'
                cf  = calculate_cloud_fraction(ACMCOM, ACM3D, want_2D=want_2D)
                icf = calculate_cloud_fraction(ACMCOM, ACM3D, want_2D=want_2D, want_ice=True)
                print(f"Cloud Fraction ({dim}) = {cf}\nIce-Cloud Fraction ({dim}) = {icf}")
                print("------------------------------------------------------------------")
            # --------------------------------------------------------------------------------------
                


            if 'ExtractData' in ToDo:
                libRad.ExtractData(ia=ial, shape=AssDomainSize, idx_scene=start_idx, BMAFLX=BMAFLX, pyranometer_data=pyranometer_data, want_info=True)
                start_idx += 1


            if 'plot_all_levels' in ToDo:
                libRad.plot_all_levels(libRad2, OrbitID, ia=ial)

            if 'plot_temporal_pyranometer' in ToDo:
                libRad.plot_temporal_pyranometer()

    tt = datetime.now(timezone.utc) - start_time
    print("Run finished. It took {:.3f} hours \n\n\n\n\n\n\n".format(tt.total_seconds()/3600))























