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


def AssDomain(data, shape=(11,11)):
    if shape is not None:
        Nx, Ny = shape
   
        # print('\n\n---------------------- AssDomain -------------------')
        # print('     AssDomainSize', Nx, Ny)
        # print('     Initial data.shape:          ', data.shape)
                # start_x = int(data.shape[-1]/2 - Nx/2); start_y = int(data.shape[-2]/2 - Ny/2)
                # end_x   = int(data.shape[-1]/2 + Nx/2); end_y   = int(data.shape[-2]/2 + Ny/2)
        
        start_x = int(data.shape[0]/2 - Nx/2); start_y = int(data.shape[1]/2 - Ny/2)
        end_x   = int(data.shape[0]/2 + Nx/2); end_y   = int(data.shape[1]/2 + Ny/2)

        data = data[start_x:end_x, start_y:end_y]

        # print('     start, end:', start_x, end_x)
        # print('     Removed Buffer -> data.shape:', data.shape)
        # print('----------------------------------------------------\n\n')

    if np.all(np.isnan(data)):
        return np.nan
    
    data = np.nanmean(data, axis=(0,1))
    return data




class Scene:
    def __init__(self,  Name='',  verbose=False):
        self.Name=Name
        self.verbose=verbose
        return

    def ExtractData(self, ia, shape=(1,1), idx_scene=None, pyranometer_data=None, want_info=False):
        """ 
        CHANGES MADE
        1. Do not apply QS to fluxes (apply only when plotting)
        """


        data = self.solar_eup[ia]
        
        if 'montecarlo' in self.fn:
            data = AssDomain(data, shape=None)
        elif 'disort' in self.fn and want_average_line:
            w_size = 10 # Match BBR footprint
            data = np.convolve(data, np.ones(w_size), 'same') / w_size # moving average


                                                    # # ACM-COM QS Filtering to libRadtran data: 
                                                    # qs = int(np.asarray(ACMCOM.quality_status[ia]).squeeze())
                                                    # # F_libRad = data if qs in (0, 1) else np.nan
                                                    # F_libRad = data if qs == 0 else np.nan 
                                                    #             # 0: Good
                                                    #             # 1: Good with warnings
                                                    #             # 2: Not confident

        F_libRad = data
        # Write RTM result to file
        flux_file.write(f'{F_libRad:8.2f} {libRad.fn} \n')



        if want_info:
            print(
                #f"Extracted data from {self.fn}:\n\n"
                # f"shape {shape}\n\n"
                 " |------------------------|\n"
                f" |     data  = {F_libRad:6.1f}     |\n" 
                f" |------------------------|"
            )


            if WANT_SUR:
                # Extract SUR data
                obs_val = pyranometer_data[idx_scene]
                print(f" |  pyranometer = {obs_val:6.1f}  |\n"
                       " |------------------------|\n")
            else: 
                # Extract TOA data
                # Find ia for BMAFLX (not same rasolution as ACMCOM)
                latitude_wanted = ACMCOM.latitude_active[ia]
                longitude_wanted = ACMCOM.longitude_active[ia]
                dist = np.sqrt((BMAFLX.latitude - latitude_wanted)**2 + (BMAFLX.longitude - longitude_wanted)**2)
                indlatBMAFLX = np.unravel_index(np.argmin(dist, axis=None), dist.shape)

                flux = float(np.asarray(BMAFLX.solar_combined_top_of_atmosphere_flux[indlatBMAFLX]).squeeze())
                                                    # qs = int(np.asarray(BMAFLX.quality_status[indlatBMAFLX]).squeeze())
                                                    # BMAFLX_solar_eup = flux if qs in (0, 2) else np.nan
                                                    #                                         #               Quality status BMA-FLX 
                                                    #                                         #     0: Both thermal and solar fluxes are fully trustful; 
                                                    #                                         #     1: Only thermal flux is fully trustful; 
                                                    #                                         #     2: Only solar flux is fully trustful; 
                                                    #                                         #     3: Both thermal and solar fluxes are valid but not fully trustful; 
                                                    #                                         #     4: Neither thermal nor solar fluxes are valid 
                BMAFLX_solar_eup = flux
                if write_BMAFLX_to_file:
                    # Specs: SmallResolution, QualityStatus 0 or 2 -> solar fully trustfull
                    with open("DATA/TOA_Observations.txt", "a") as BMAFLX_file:
                        BMAFLX_file.write(f"{BMAFLX_solar_eup:.2f}\n")

                print(f" |    BMAFLX = {BMAFLX_solar_eup:6.1f}     |\n"
                       " |------------------------|\n")

            # Access ACM-RT 1D data
            if write_ACMRT_to_file:
                flux = np.nan
                filename = "DATA/SUR_ACMRT1D.txt" if WANT_SUR else "DATA/TOA_ACMRT1D.txt"
                if ACMRT:
                    if WANT_SUR:
                        irec_surface_pixel = ACM3D.index_construction[iacr, ia] 
                        h = ACMCOM.height_level[:, irec_surface_pixel]
                        p = ACMCOM.pressure_level[:, irec_surface_pixel]
                        valid = np.where((h > 0) & (p < 1e10))[0]
                        if len(valid) > 0:
                            SUR_idx = valid[-1]
                            flux = ACMRT.flux_down_solar_1d_all_sky[irec_surface_pixel, SUR_idx]
                            # flux = ACMRT.flux_down_solar_1d_clear_sky[irec_surface_pixel, SUR_idx]

                                                    # qs = int(np.asarray(ACMRT.quality_status[irec_surface_pixel]).squeeze())
                                                    # print(f"     QS = {qs}")
                                                    # flux = flux if qs in (0,1) else np.nan
                            flux = flux if flux < 1e10 else np.nan
                    else:
                        TOA_idx = 0
                        flux = ACMRT.flux_up_solar_1d_all_sky[ia, TOA_idx]

                                                    # qs = int(np.asarray(ACMRT.quality_status[ia]).squeeze())
                                                    # print(f"     QS = {qs}")
                                                    # flux = flux if qs in (0,1) else np.nan
                        flux = flux if flux < 1e10 else np.nan

                print(f" |    ACMRT  = {flux:6.1f}     |\n"
                        " |------------------------|\n")
                
                with open(filename, "a") as ACMRT_file:
                    ACMRT_file.write(f"{flux:.2f}\n")
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
            self.quality_status=SD['index_construction_quality_status'][()]
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
            self.flux_down_solar_1d_all_sky=SD['flux_down_solar_1d_all_sky'][()]
            self.flux_down_solar_1d_clear_sky=SD['flux_down_solar_1d_clear_sky'][()]
            
            self.flux_up_thermal_1d_all_sky=SD['flux_up_thermal_1d_all_sky'][()]
            self.flux_up_thermal_3d_reference_height_all_sky=SD['flux_up_thermal_3d_reference_height_all_sky'][()]

            # BG: only single atmosphere here!
            self.flux_up_solar_1d_all_sky=self.flux_up_solar_1d_all_sky[0,:,:]
            self.flux_up_solar_3d_all_sky=self.flux_up_solar_3d_all_sky[0,:,:]
            self.flux_down_solar_1d_all_sky=self.flux_down_solar_1d_all_sky[0,:,:]
            self.flux_down_solar_1d_clear_sky=self.flux_down_solar_1d_clear_sky[0,:,:]
            self.flux_up_thermal_1d_all_sky=self.flux_up_thermal_1d_all_sky[0,:,:]
            self.flux_up_thermal_3d_reference_height_all_sky=self.flux_up_thermal_3d_reference_height_all_sky[0,:]   

            # BG: add quality-status
            self.quality_status = SD['quality_status'][()][0,:]
         
            
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


        # Shape: 
        # (along-track, Nx/across-track buffer, Ny/along-track buffer)
        ncf.close()
        



















###########################################################################################################################        
if __name__ == "__main__":
    start_time = datetime.now(timezone.utc) 
    
    



     
    WANT_3D  = True      
    WANT_SUR = True

    # Write fluxes to files:
    write_BMAFLX_to_file    = False                
    write_ACMRT_to_file     = False







    ToDo_idx = 0
    ToDo = [
        'ExtractData',
    ][ToDo_idx]
    




    
    # BG: additional settings
    AssDomainSize       = [(1,1), (3,3), (11,11)][0]   # Size of assessment domain (across-track, along-track) in libRadtran 3D MC
    want_quality_status = False                 # If want ACM-COM and BMA-FLX quality status
    want_EarthCARE_info = False                 # Sets Scene3 = ACM3D (in PlotLien)
    want_product2       = False                 # Sets Scene4 = librad2 (in PlotLine)
    want_average_line   = False                 # Average line of DISORT-flux-values
    want_ps             = False                 # BG: if want DISORT pseudospherical (WANT_3D overwrite want_ps)
    stacked             = True                  # If add quanteties to plot, if should get own figure below
    verbose             = False
    want_info           = False                 # Prints out SZA, Prints out CF
                                                # if want_info: want_2D = False               # CF of 2D swat, or 1D nadir column
    
    

   



    # BG: Chose idx to select additional info on solar_both and thermal_both plot 
    #   Chose from: 
    #   [None, 'elevation', 'lwp', 'iwp', 'tot_wp', 'albedo', 'aerosols', 'surface_temperature', 'CF']
    #           NOTE: do not put 'CF' at the end of quantity_list nor with tot_wc 
    #           NOTE: put tot_wp before tot_wc
    quantity_list = False

 


    idx_source = 0
    sources = ['solar']

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
    # idx_range = np.arange(0,1)
    sites = [sites[i] for i in idx_range]


    DATA_FILES = Path("/xnilu_wrk2/projects/NEVAR/data/CalVal/SurfaceOverpasses/") # OLD Path("/homevip/bgre/Download/Frames_SurfaceOverpasses")

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
            print(f"\nOrbitsIDs for {site}:")
            for line in f:
                OrbitIDs.append(line.strip())
                StationList.append(site)
                print(line.strip(), end=" ")
            print()   

                           


            





    # BG: To distinguish minor modifications to .nc files
    additional_spesifications = '' 
    # additional_spesifications += '_TEST'
            # New mc_sample_grid


    if WANT_SUR:    additional_spesifications += '_SUR'          #'_21x21_SUR' if WANT_3D else '_SUR'
    else:           additional_spesifications += '_TOA'          #'_21x21_TOA' if WANT_3D else '_TOA'



  
    
    pathL2TestProducts_base = "/xnilu_wrk2/projects/NEVAR/data/CalVal/SurfaceOverpasses/" 
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












    ####################################################
    WANT_11x11_ASSESSMENT_DOMAIN    = False
    WANT_CLEAR_SKY                  = False

    if WANT_11x11_ASSESSMENT_DOMAIN:
        AssDomainSize = (11, 11)
        rte_specs = 'MYSTIC_11x11'
        if WANT_CLEAR_SKY: 
            rte_specs   = 'MYSTIC_ClearSky_11x11'
            librad_type = 'SNNA'






    WANT_FULL_DOMAIN    = True
    WANT_CLEAR_SKY      = True

    if WANT_FULL_DOMAIN:
        AssDomainSize = None
        rte_specs = 'MYSTIC_CompDomain'
        if WANT_CLEAR_SKY: 
            rte_specs   = 'MYSTIC_ClearSky_CompDomain'
            librad_type = 'SNNA'
    ########################################################
















    out_file_name1 = folder + obs_level + rte_specs + ".txt"

    pyranometer_data = np.loadtxt("DATA/SUR_Observations.txt", usecols=0)
  

    with open(out_file_name1, "w") as flux_file: 
        for OrbitID, Station in zip(OrbitIDs, StationList):
            pathL2TestProducts = pathL2TestProducts_base + f'/{Station}'



            # BG: Find; Baseline, Data, Time
            BB_baseline, BA_baseline, AC_baseline = '', '', ''
            
            Product ='ALL_3D_'#'ACM_COM' #'ACM_3D_'#'ACM_COM' #
            ProductPath = '*'+Product+'*'+OrbitID+'*'
            ProductFile = os.path.join(pathL2TestProducts, ProductPath, '*'+Product+'*.h5')
            ProductFile = sorted(glob.glob(ProductFile))[0]
            ACM3D = Scene(Name=OrbitID, verbose=verbose)
            ACM3D.ReadEarthCAREh5(ProductFile, verbose=verbose)
            ACM3D.SetExtent()
            

            # Extract Baseline ---------------------------   
            parts = ProductFile.split("ECA_EX", 1)
            out = parts[1][:2] if len(parts) > 1 else None
            if   out == 'BA': BA_baseline += ' ALL_3D'
            elif out == 'AC': AC_baseline += ' ALL_3D'
            elif out == 'BB': BB_baseline += ' ALL_3D'
            #---------------------------------------------
            
            Product ='BMA_FLX'
            ProductPath = '*'+Product+'*'+OrbitID+'*'
            ProductFile = os.path.join(pathL2TestProducts, ProductPath, '*'+Product+'*.h5')
            try: 
                ProductFile = sorted(glob.glob(ProductFile))[0]
            except IndexError: 
                # If do not have all products (BMAFLX here), did not simulate frame! Skipping
                print(f"\n     Skipping {OrbitID}. Do not find product {Product}") 
                continue
            
            BMAFLX = Scene(Name=OrbitID, verbose=verbose)
            # BMAFLX.ReadEarthCAREh5(ProductFile, Resolution='StandardResolution', verbose=verbose)
            BMAFLX.ReadEarthCAREh5(ProductFile, Resolution='SmallResolution', verbose=verbose) 
            """
            SmallResolution =  5x10 km (along x across track)
            FullResolution  = 10x10 km
            """
            BMAFLX.SetExtent()
                        
            # Extract Baseline ---------------------------   
            parts = ProductFile.split("ECA_EX", 1)
            out = parts[1][:2] if len(parts) > 1 else None
            if   out == 'BA': BA_baseline += ' ' + Product
            elif out == 'AC': AC_baseline += ' ' + Product
            elif out == 'BB': BB_baseline += ' ' + Product
            #---------------------------------------------


            Product ='ACM_RT_'      #1D and 3D RTM fluxes
            ProductPath = '*'+Product+'*'+OrbitID+'*'
            ProductFile = os.path.join(pathL2TestProducts, ProductPath, '*'+Product+'*.h5')
            try: 
                ProductFile = sorted(glob.glob(ProductFile))[0]
                ACMRT = Scene(Name=OrbitID)
                ACMRT.ReadEarthCAREh5(ProductFile)
                ACMRT.SetExtent()
            except IndexError: 
                # If do not find ACM-RT
                print(f"\n     Do not find product {Product}") 
                ACMRT = None


            Product ='ACM_COM'
            ProductPath = '*'+Product+'*'+OrbitID+'*'
            ProductFile = os.path.join(pathL2TestProducts, ProductPath, '*'+Product+'*.h5')
            ProductFile = sorted(glob.glob(ProductFile))[0]
            ACMCOM = Scene(Name=OrbitID, verbose=verbose)
            ACMCOM.ReadEarthCAREh5(ProductFile, ACM3D=ACM3D)
            ACMCOM.SetExtent()
                           
            # Extract Baseline ---------------------------   
            parts = ProductFile.split("ECA_EX", 1)
            out = parts[1][:2] if len(parts) > 1 else None
            if   out == 'BA': BA_baseline += ' ' + Product
            elif out == 'AC': AC_baseline += ' ' + Product
            elif out == 'BB': BB_baseline += ' ' + Product
            #---------------------------------------------

            # Find overpass time and closes across- and along-track indices for overpass
            ########################################################################################################################################################
            # Find latitude and longitude for the station
            for station, lat, lon in sites:
                if station == Station:
                    lat_, lon_ = str(lat), str(lon)
                    break
            out = Find_Overpass_Info.find_track_values(ACMCOM.fn, lat_, lon_)
            iacr  = out["across_index"] if WANT_SUR else 150
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


            

            if 'ExtractData' in ToDo:
                libRad.ExtractData(ia=ial, shape=AssDomainSize, idx_scene=start_idx, pyranometer_data=pyranometer_data, want_info=True)
                start_idx += 1




    tt = datetime.now(timezone.utc) - start_time
    print("Run finished. It took {:.3f} hours \n\n\n\n\n\n\n".format(tt.total_seconds()/3600))























