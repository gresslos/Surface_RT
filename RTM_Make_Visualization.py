import numpy as np
import pickle
import os
import sys
import multiprocessing
import glob
import scipy
import scipy.constants
from scipy.interpolate import griddata
import dted
import pdb # For debugging
from mpi4py import MPI # Use MPI for multiprocessing
from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess



#import ReadEarthCAREL2 as ReadEC
import Read_RTM as ReadEC
import UVspec as UVspec
import Find_Overpass_Info as Find_Overpass_Info




# BG: overflow encouner - Warnings from reading ACM_COM files
# BG: Hiding the 'overflow encountered in multiply' warnings:
np.seterr(over='ignore')




# Where are we
myhost = os.uname()[1]
home = os.environ['HOME']

# initialize MPI
comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
np_mpi = comm.Get_size()


def pdeg2km(p):
    latfact = 111.32
    p0 = latfact*p[0]/1000
    lonfact = 40075 * np.cos(np.deg2rad( p[1] )) / 360
    p1 = latfact*p[1]/1000
    return (p0,p1)


def Check3DBufferSize(ia, iacr, which_buffer):

    Nx, Ny, iacrosses, ialongs = Calc3DBufferSize(ia, iacr, which_buffer)
    if Nx>0 and Ny>0:
        BufferOK=True
    else:
        BufferOK=False
        
    return BufferOK


def Calc3DBufferSize(ia, iacr, which_buffer):
    WANT_PRINTS = False
    # BG: Decide to use my buffer, not dynamic buffer!
    if which_buffer == 0:   along_dim, across_dim =  0,  0 # ( 1, 1) comp-domain  MCIPA 
    elif which_buffer == 1: along_dim, across_dim =  2,  2 # ( 5, 5) comp-domain  Test Buffer
    elif which_buffer == 2: along_dim, across_dim = 10, 10 # (21,21) comp-domain  ~BBR +  5 pixels-buffer
    elif which_buffer == 3: along_dim, across_dim = 20, 20 # (41,41) comp-domain  ~BBR + 35 pixels-buffer
    
    ialongs = np.arange(ia-along_dim, ia+along_dim+1)
    iacrosses = np.arange(iacr-across_dim, iacr+across_dim+1)
    
    Nx = iacrosses.shape[0]
    Ny = ialongs.shape[0]


    return Nx, Ny, iacrosses, ialongs

def SetRTM(UVS, ia, iacr, sza, phi0, input_dir='', mc_basename_path='', RTdimension = '1D',
           mol_abs_param='kato2', data_dir='', source='solar'):

    
    UVS.status = 'OK'
    # First some general input
    UVS.inp['data_files_path']=data_dir
    
    libpath='/xnilu_wrk2/projects/NEVAR/libRadtran/'
    UVS.inp['atmosphere_file'] = f"{libpath}/data/atmmod/afglss.dat"
    UVS.inp['albedo'] = 0.05

    UVS.inp['mol_abs_param']=mol_abs_param
    UVS.inp['source']=source
    UVS.inp['output_process']='sum'

    UVS.inp['output_user']='sza zout edir edn eup'
    UVS.inp['quiet']=''
    

    # Solar geometry
    # Set solar geometry here to make sure azimuth shift due to 3D geometry is included
    UVS.inp['sza']=sza
    UVS.inp['phi0']=phi0  # Azimuth is only needed for 3D, but it does not hurt for 1D.

    Nx, Ny, iacrosses, ialongs = Calc3DBufferSize(ia, iacr, which_buffer)  #calculates along and cross track buffer size 
    if RTdimension == '3D':       
        if UVS.inp['rte_solver'] != 'montecarlo':
            print('3D geometry only available with MYSTIC. You specified:', UVS.inp['rte_solver'])
            exit()
                
        UVS.inp['mc_std']=''
        # UVS.mc_basename=mc_basename_path+'mc_{:00004d}'.format(ia)+'_'+source # Old->made race-conditions errors 
        UVS.mc_basename=mc_basename_path+'mc_'+f"{ia:04d}_"+source
        print("MC basename set to:", UVS.mc_basename)
        UVS.inp['mc_basename']=UVS.mc_basename
        UVS.inp['mc_randomseed'] = (ia+1)*72341//11
        
        # Set SAMPLE-GRID
        UVS.inp['mc_sample_grid']=f'{Nx} {Ny}' # 3D computation domain
        UVS.inp['mc_reference_to_nn']='' # The sampled pixels correspond to the surface pixels.
     
        
    

    ############### SPESIFICATIONS TO SURFACE-VERSION ##################
    if 'TOA' in additional_spesifications: UVS.inp['zout']='TOA'
    elif 'SUR' in additional_spesifications: 
        # Set "zout sur" for DISORT
        # Do not spesify zout for MYSTIC -> give "zout sur" (which do not work for MYSTIC)
        if UVS.inp['rte_solver'] != 'montecarlo': UVS.inp['zout'] = 'sur'  
    ####################################################################

    

    if source=='thermal':
        UVS.inp['wavelength']='4500 100000'  
        if RTdimension == '3D':
            mc_photons = int(1e2)
            UVS.inp['mc_photons']=mc_photons 
            UVS.inp['mc_minphotons']=mc_photons # Note: MC_MINPHOTONS = 1e3 in uvspec.h so need to modify!
    else:
        UVS.inp['wavelength']='295 2800'     # Changed from '250 4000' 20.01.2026   
        if RTdimension == '3D':
            mc_photons = int(1e7)  
            UVS.inp['mc_photons']=mc_photons 
      
    
    

    # Water cloud
    UVS.inp['wc_properties']='mie'
    Nz = 100 
    wc_value, wcreff_value = 1e-2, 10.0
    h_min, h_max = 2, 3
    heights = np.linspace(0, 10, Nz) # km
    # Make a 2x2 cloud in the center of the domain
    cloud_start_X = Nx//2 #+ Nx//8
    cloud_start_Y = Ny//2 #+ Ny//8
    idx_inside_cloud = []

    if RTdimension == '3D':  
        
        # iz=0
        # last_h=99999
        # for h,p in zip(ACMCOM.height_level[1:,irec_surface_pixel],ACMCOM.pressure_level[1:,irec_surface_pixel]):
        #     if h > -1e-4 and p < 1e+10:
        #         last_h=h
        #         last_p=p
        #         iz=iz+1

        # if np.abs(last_h) > 1e-4 and last_p < 1e+10:
        #     iz=iz+1

        # Nz = iz-1 #ACMCOM.height_level[1:,ia].shape[0]
        # dx = 1.0 # 1 km horizontal resolution
        # dy = 1.0

        flag = 3 # mystic 3D format# mystic 3D format# mystic 3D format# mystic 3D format
        dx, dy = 1.0, 1.0 # 1km horizontal resolution
        heights = heights[1:]
        Nz -= 1
        wc_file_3D=input_dir+f'tmp_{ia:04d}_wc3D.dat'
        f = open(wc_file_3D,'w')
        f.write('{:d} {:d} {:d} {:d}\n'.format(Nx, Ny, Nz, flag))
        f.write('{:f} {:f} '.format(dx, dy))

        
       
        for h in heights:
            f.write('{:8.3f} '.format(np.abs(h)))
            
        f.write('\n')
        wclines=0

        ix=1
        for iac in iacrosses:
            iy=1 
            for ial in ialongs:
                iz = 1
                inside_cloud_region = (
                        cloud_start_X <= ix < cloud_start_X + 2
                    and cloud_start_Y <= iy < cloud_start_Y + 2
                )

                # Define indecies for cloud-vertical height
                if ix == cloud_start_X and iy == cloud_start_Y:
                    h_valid = np.asarray(heights)[heights > 0]

                    iz_h_min = np.argmin(np.abs(h_valid - h_min)) + 1
                    iz_h_max = np.argmin(np.abs(h_valid - h_max)) + 1

                    iz_min = min(iz_h_min, iz_h_max)
                    iz_max = max(iz_h_min, iz_h_max)

                    print("Cloud vertical indices:", iz_min, iz_max)
                    print("Actual heights:", h_valid[iz_min-1], h_valid[iz_max-1])

                

                # BG: made changes
                # - iz += 1 for each realistic h > 0 measurement by EarthCARE
                # - previosly iz += 1 only for realistic h > 0 and wc > 0 (this was the error)
                for h in heights:
                    if inside_cloud_region and iz_min <= iz < iz_max: # h_min < h < h_max:
                        # set wc
                        f.write('{:d} {:d} {:d} {:f} {:f}\n'.format(ix, iy, iz, wc_value, wcreff_value))
                        print(f'Writing wc={wc_value} at ix={ix}, iy={iy}, iz={iz}')
            
                    
                        wclines=wclines+1
                    iz=iz+1 
                iy=iy+1
            ix=ix+1

        # Add dummy line to make mystic run
        if wclines==0:
            f.write('{:d} {:d} {:d} {:f} {:f}\n'.format(1, 1, 1, 0, 10))
                
        f.close()
        UVS.inp['wc_file 3D']=wc_file_3D
        UVS.inp['mc_std']=''
    else:       
        wc_file_1D=input_dir+f'tmp_{ia:04d}_wc1D.dat'
        UVS.inp['wc_file 1D']=wc_file_1D
        f = open(wc_file_1D,'w')
    
        ix = 1
        for iac in iacrosses:
            iy = 1
            for ial in ialongs:
                inside_cloud_region = (
                    cloud_start_X <= ix < cloud_start_X + 2
                    and cloud_start_Y <= iy < cloud_start_Y + 2
                )
                if inside_cloud_region:
                    idx_inside_cloud.append(Nx* (iy-1) + (ix-1))
                iy += 1
            ix += 1
        print(idx_inside_cloud)
    
                
        for h in heights[::-1]:
            if h_min <= h <= h_max and ia in idx_inside_cloud:
                wc, wcreff = wc_value, wcreff_value
            else:
                wc, wcreff = 0.0, 0.0

            f.write('{:8.3f} {:10.5f} {:8.3f}\n'.format(h, wc, wcreff))
        f.close()
    
    return UVS


def tmp_output_filename(iia, scene_name):
    """
    Return the filename to be used for temporary output files.

    """
    pass
    












if __name__ == "__main__":
    start_time = datetime.now(timezone.utc) 
    # print("Run started at", start_time.isoformat())



    ##############################################################################################
    WANT_SUR = True         # zout at SUR or TOA
    WANT_3D  = False        # MYSTIC or DISORT

    verbose  = False        # Want verbose output from RTM
    want_ps  = False        # Psudospherical solver
        
                            
    which_buffer = 1
    # 0: MCIPA   1:Test   2: 21x21   3: 41x41  
    ################################################################################################

   
  

    # BG: To distinguish minor modifications to .nc files
    additional_spesifications = 'SINGLE_CLOUD' 
    sza = 60
    phi0 = 60


    if WANT_SUR: additional_spesifications += "_SUR"
    else:        additional_spesifications += "_TOA"



    idx_source = 0
    sources = [['solar'], 
              ['thermal'],
              ['solar', 'thermal']][idx_source] 


       
    if WANT_3D:
        rte_solver = 'montecarlo' #'mystic' 
        RTdimension = '3D'
    else:
        if want_ps:
            rte_solver = 'disort \npseudospherical'
        else: 
            rte_solver = 'disort'  # 'twostr'        
        RTdimension ='1D' 
    print(f"{RTdimension} RTM - {rte_solver}")


    # ---------------------------------- Paths -----------------------------------------------
    RTInpOutPath = './tmpRTIO/'    # Folder to store the final results
    RTOutNetcdfPath = './RESULTS_TEST/'  # Folder name for netcdf result files
    uvspecpath = '/xnilu_wrk2/projects/NEVAR/libRadtran/bin/' # Folder to RTM code
    RTMdata = '/xnilu_wrk2/projects/NEVAR/libRadtran/data/' #RTM input data

    libRad_version= 'v01_'+rte_solver+'_'+RTdimension
    # --------------------------------------------------------------------------------------------



    Nx, Ny = 1, 1 # Default Horizontal Computaion-Domain Size                                                                                     # min / pixel   |     solar-std     |     thermal-std  
    if   which_buffer == 0 :                  buffer_str = 'MCIPA'
    elif which_buffer == 1 : Nx, Ny =  5,  5; buffer_str = 'TEST   Buffer (5 x 5)'                      #   1.5         |                   |
    elif which_buffer == 2 : Nx, Ny = 21, 21; buffer_str = '~BBR +  5 pixels-buffer  (21 x 21)'         #    4          |       ~5-8        |       ~0.5-2
    elif which_buffer == 3 : Nx, Ny = 41, 41; buffer_str = '~BBR + 35 pixels-buffer  (41 x 41)'         #    5          |       ~17         |       ~1.6
    else:buffer_str = ''
    

                                                                                            

        
    SceneNamelibRad='libRad'
    libRad = ReadEC.Scene(Name=SceneNamelibRad, verbose=verbose)    
    if 'montecarlo' in rte_solver:
        shape = (1, Nx, Ny)
    else: 
        shape = Nx*Ny
    libRad.latitude_active = np.zeros(shape)

    # print(shape)
    libRad.solar_e = np.zeros(shape)       
    libRad.solar_e_std = np.zeros(shape)
    libRad.thermal_e = np.zeros(shape)
    libRad.thermal_e_std = np.zeros(shape)


    if WANT_3D:
        ialongs = [Nx//2]
        iacr = Nx // 2
    else:
        ialongs = np.arange(Nx*Ny)
        iacr = 0

    



    # Parallelize over OrbitIDs instead of along-track points.
    # Each MPI rank will process all ialongs for its assigned orbit in sequence.
    
    for ia in ialongs:
        # --- BG: ADD THIS BLOCK ---
        e_solar = np.nan
        e_solar_std = np.nan
        e_thermal = np.nan
        e_thermal_std = np.nan
        # ----------------------
        
        

        for source in sources:
            try: 
                RTM=True
                if source=='solar':
                    mol_abs_param = 'kato2' #'reptran course' #
                elif source=='thermal':
                    mol_abs_param = 'fu' 

                if RTM:
                    UVS = UVspec.UVspec()   #tms: UVspec.py is an external program that defines input and runs libRadtran
                    UVS.inp['rte_solver']=rte_solver
                    #tms: "SetRTM" is a function defined above - NB! Check path
                    UVS = SetRTM(UVS, ia, iacr, sza, phi0,mc_basename_path=RTInpOutPath, RTdimension = RTdimension, input_dir=RTInpOutPath,
                                mol_abs_param=mol_abs_param, data_dir=RTMdata, source=source)
                    if UVS.status == 'OK':
                        # uvspecInputFile=RTInpOutPath+'tmp'+'{:00004d}'.format(ia)+source+str(my_rank)+'.inp'  
                        # NEW TEST
                        uvspecInputFile=RTInpOutPath+f'tmp_SINGLE_CLOUD_{ia:04d}.inp'    
                        uvspecOutputFile=uvspecInputFile.replace('inp','out')
                        UVS.WriteInputFile(uvspecInputFile)

                        status = UVS.SingleRun(uvspecInputFile, uvspecOutputFile, verbose=False, uvspecpath=uvspecpath)

                        if UVS.inp['rte_solver']=='montecarlo': ######################## MYSTIC
                            try:
                                mc_flx_dir, mc_flx_dn, mc_flx_up, edir_std, edn_std, eup_std = UVspec.ReadMCOut(UVS.mc_basename, mol_abs_param=mol_abs_param) #fn = fn + '.flx.spc'  
                                if 'AllLevels' in additional_spesifications:
                                    nlev = int(len(mc_flx_dir)/(Nx*Ny))
                                
                                    mc_flx_dir  = mc_flx_dir.reshape(nlev, Nx, Ny)
                                    mc_flx_dn   = mc_flx_dn.reshape(nlev, Nx, Ny)
                                    mc_flx_up   = mc_flx_up.reshape(nlev, Nx, Ny)

                                    eup_std     = eup_std.reshape(nlev, Nx, Ny)
                                    edn_std     = edn_std.reshape(nlev, Nx, Ny)
                                    edir_std    = edir_std.reshape(nlev, Nx, Ny)


            
                                else:
                                    mc_flx_dir  = mc_flx_dir.reshape(Nx, Ny)
                                    mc_flx_dn   = mc_flx_dn.reshape(Nx, Ny)
                                    mc_flx_up   = mc_flx_up.reshape(Nx, Ny)
                                    edir_std    = edir_std.reshape(Nx, Ny)
                                    eup_std     = eup_std.reshape(Nx, Ny)
                                    edn_std     = edn_std.reshape(Nx, Ny)


                                
                            except (FileNotFoundError, ValueError) as e:
                                print(f"[WORKER {my_rank}] WARNING: Could not read MC output for ia={ia}, source={source}. Skipping. Error: {e}")
                                # continue  # Go to the next iteration
                        
                            if source=='solar':   
                                if WANT_SUR:
                                    e_solar_std = edir_std + edn_std
                                    e_solar     = mc_flx_dir + mc_flx_dn
                                    # print('direct + diffuse solar flux at SUR = ', e_solar, '\n with direct flux = ', mc_flx_dir, ' and diffuse flux = ', mc_flx_dn)
                                else:
                                    e_solar_std = eup_std
                                    e_solar     = mc_flx_up

                            elif source=='thermal':
                                if WANT_SUR:
                                    e_thermal_std = edir_std + edn_std
                                    e_thermal     = mc_flx_dir + mc_flx_dn
                                else:
                                    e_thermal_std = eup_std
                                    e_thermal     = mc_flx_up

                
                        else: ########################################################## DISORT
                            try:
                                sza, zout, edir, edn, eup = UVspec.ReadRTOut(uvspecOutputFile, mol_abs_param=mol_abs_param)
                            except ValueError as e:
                                # print(edir, edn, eup)
                                print("Error [WORKER {}]: could not read file {} with mol_abs_param {}".
                                    format(my_rank, uvspecOutputFile, mol_abs_param))
                                print(f"Error [WORKER {my_rank}]: ", e)

                                
                        

                            e_solar_std = e_thermal_std = None  # No std output from DISORT
                            if source=='solar':
                                if WANT_SUR:
                                    e_solar = edir + edn
                                else:
                                    e_solar = eup
                            elif source=='thermal':
                                if WANT_SUR:
                                    e_thermal = edir + edn
                                else:
                                    e_thermal = eup


                    else:
                        print(UVS.status)
                        print(f'WARNING: UVS.status not OK for ia={ia}, source={source}. Skipping.')
                        zout=edir=eup=eup_std=0.0
                else:
                    zout=edir_solar=eup_solar=eup_solar_std=0.0
                    edir_thermal=eup_thermal=eup_thermal_std=0.0
                    

                

                if rte_solver=='montecarlo': 
                    if not hasattr(e_solar, "shape"):
                        # raise ValueError("Simulated solar flux has no shape (simulation was not successful)")    
                        e_solar_center = np.nan
                        e_solar_std_center = np.nan
                        e_thermal_center = np.nan
                        e_thermal_std_center = np.nan                        
                
                    elif 'AllLevels' in additional_spesifications:
                        ix_center = e_solar.shape[1] // 2
                        iy_center = e_solar.shape[2] // 2
                        
                        if hasattr(e_solar_std, "shape"):   e_solar_std_center      = e_solar_std[:, ix_center, iy_center]
                        if hasattr(e_solar, "shape"):       e_solar_center          = e_solar[:, ix_center, iy_center]
                        if hasattr(e_thermal_std, "shape"): e_thermal_std_center    = e_thermal_std[:, ix_center, iy_center]
                        if hasattr(e_thermal, "shape"):     e_thermal_center        = e_thermal[:, ix_center, iy_center]
                    else: 
                        ix_center = e_solar.shape[0] // 2
                        iy_center = e_solar.shape[1] // 2

                        if hasattr(e_solar_std, "shape"):   e_solar_std_center      = float(e_solar_std[ix_center, iy_center])
                        if hasattr(e_solar, "shape"):       e_solar_center          = float(e_solar[ix_center, iy_center])
                        if hasattr(e_thermal_std, "shape"): e_thermal_std_center    = float(e_thermal_std[ix_center, iy_center])
                        if hasattr(e_thermal, "shape"):     e_thermal_center        = float(e_thermal[ix_center, iy_center])

                    # print("ix_center, iy_center", ix_center, iy_center)                              
                else:
                    e_solar_center     = e_solar
                    e_thermal_center   = e_thermal

                ####### PRINTING RESULTS ##########
                if 'AllLevels' in additional_spesifications:
                    print(e_solar_center)
                    print(e_solar_std)
                else:
                    e_str = f"{e_solar_center:7.1f}" if source=='solar' else f"{e_thermal_center:7.1f}"
                    std_str = f"+- {e_solar_std_center:.1f}" if 'montecarlo' in rte_solver else ''
                    print("\n\n\n"
                        f"      | ia={ia:4}  ----------------------------------------------------------|\n"
                        f"      |                                   Flux ({source}) = {e_str} {std_str} W/m2\n"
                            "      |--------------------------------------------------------------------------------|"
                    )
                   
            

            except Exception as e:
                print(f"\nException for ia={ia:4}: {e}\n")
    

    

    if rte_solver == "montecarlo":
        idx = (ia, slice(None), slice(None))
    else:
        idx = ia

    print(e_solar, e_solar_std, e_thermal, e_thermal_std)
    libRad.solar_e_std[idx]   = e_solar_std
    libRad.thermal_e_std[idx] = e_thermal_std
    libRad.solar_e[idx]       = e_solar
    libRad.thermal_e[idx]     = e_thermal

    # print(e_solar, libRad.solar_e[idx])

        
                        
                

    # Writing NetCDF file...")
    src = ','.join([i for i in sources])
    if 'montecarlo' in rte_solver:
        libRad.WriteNetcdf_mc_sample_grid(RTOutNetcdfPath+'libRad_'+libRad_version+'_'+src+'_'+ additional_spesifications + '.nc', shape=(Nx,Ny), verbose=False)
    else:
        libRad.WriteNetcdf(               RTOutNetcdfPath+'libRad_'+libRad_version+'_'+src+'_'+ additional_spesifications + '.nc', verbose=False)
    print("Wrote to " + RTOutNetcdfPath+'libRad_'+libRad_version+'_'+src+'_'+ additional_spesifications + '.nc')


    tt = datetime.now(timezone.utc) - start_time
    print(f"            Finished run. It took {tt.total_seconds()/3600:.2f} hours \n\n\n\n\n\n\n")

import numpy as np
import pickle
import os
import sys
import multiprocessing
import glob
import scipy
import scipy.constants
from scipy.interpolate import griddata
import dted
import pdb # For debugging
from mpi4py import MPI # Use MPI for multiprocessing
from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess



#import ReadEarthCAREL2 as ReadEC
import Read_RTM as ReadEC
import UVspec as UVspec
import Find_Overpass_Info as Find_Overpass_Info




# BG: overflow encouner - Warnings from reading ACM_COM files
# BG: Hiding the 'overflow encountered in multiply' warnings:
np.seterr(over='ignore')




# Where are we
myhost = os.uname()[1]
home = os.environ['HOME']

# initialize MPI
comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
np_mpi = comm.Get_size()


def pdeg2km(p):
    latfact = 111.32
    p0 = latfact*p[0]/1000
    lonfact = 40075 * np.cos(np.deg2rad( p[1] )) / 360
    p1 = latfact*p[1]/1000
    return (p0,p1)


def Check3DBufferSize(ia, iacr, which_buffer):

    Nx, Ny, iacrosses, ialongs = Calc3DBufferSize(ia, iacr, which_buffer)
    if Nx>0 and Ny>0:
        BufferOK=True
    else:
        BufferOK=False
        
    return BufferOK


def Calc3DBufferSize(ia, iacr, which_buffer):
    WANT_PRINTS = False
    # BG: Decide to use my buffer, not dynamic buffer!
    if which_buffer == 0:   along_dim, across_dim =  0,  0 # ( 1, 1) comp-domain  MCIPA 
    elif which_buffer == 1: along_dim, across_dim =  2,  2 # ( 5, 5) comp-domain  Test Buffer
    elif which_buffer == 2: along_dim, across_dim = 10, 10 # (21,21) comp-domain  ~BBR +  5 pixels-buffer
    elif which_buffer == 3: along_dim, across_dim = 20, 20 # (41,41) comp-domain  ~BBR + 35 pixels-buffer
    
    ialongs = np.arange(ia-along_dim, ia+along_dim+1)
    iacrosses = np.arange(iacr-across_dim, iacr+across_dim+1)
    
    Nx = iacrosses.shape[0]
    Ny = ialongs.shape[0]


    return Nx, Ny, iacrosses, ialongs

def SetRTM(UVS, ia, iacr, sza, phi0, input_dir='', mc_basename_path='', RTdimension = '1D',
           mol_abs_param='kato2', data_dir='', source='solar'):

    
    UVS.status = 'OK'
    # First some general input
    UVS.inp['data_files_path']=data_dir
    
    libpath='/xnilu_wrk2/projects/NEVAR/libRadtran/'
    UVS.inp['atmosphere_file'] = f"{libpath}/data/atmmod/afglss.dat"
    UVS.inp['albedo'] = 0.05

    UVS.inp['mol_abs_param']=mol_abs_param
    UVS.inp['source']=source
    UVS.inp['output_process']='sum'

    UVS.inp['output_user']='sza zout edir edn eup'
    UVS.inp['quiet']=''
    

    # Solar geometry
    # Set solar geometry here to make sure azimuth shift due to 3D geometry is included
    UVS.inp['sza']=sza
    UVS.inp['phi0']=phi0  # Azimuth is only needed for 3D, but it does not hurt for 1D.

    Nx, Ny, iacrosses, ialongs = Calc3DBufferSize(ia, iacr, which_buffer)  #calculates along and cross track buffer size 
    if RTdimension == '3D':       
        if UVS.inp['rte_solver'] != 'montecarlo':
            print('3D geometry only available with MYSTIC. You specified:', UVS.inp['rte_solver'])
            exit()
                
        UVS.inp['mc_std']=''
        # UVS.mc_basename=mc_basename_path+'mc_{:00004d}'.format(ia)+'_'+source # Old->made race-conditions errors 
        UVS.mc_basename=mc_basename_path+'mc_'+f"{ia:04d}_"+source
        print("MC basename set to:", UVS.mc_basename)
        UVS.inp['mc_basename']=UVS.mc_basename
        UVS.inp['mc_randomseed'] = (ia+1)*72341//11
        
        # Set SAMPLE-GRID
        UVS.inp['mc_sample_grid']=f'{Nx} {Ny}' # 3D computation domain
        UVS.inp['mc_reference_to_nn']='' # The sampled pixels correspond to the surface pixels.
     
        
    

    ############### SPESIFICATIONS TO SURFACE-VERSION ##################
    if 'TOA' in additional_spesifications: UVS.inp['zout']='TOA'
    elif 'SUR' in additional_spesifications: 
        # Set "zout sur" for DISORT
        # Do not spesify zout for MYSTIC -> give "zout sur" (which do not work for MYSTIC)
        if UVS.inp['rte_solver'] != 'montecarlo': UVS.inp['zout'] = 'sur'  
    ####################################################################

    

    if source=='thermal':
        UVS.inp['wavelength']='4500 100000'  
        if RTdimension == '3D':
            mc_photons = int(1e2)
            UVS.inp['mc_photons']=mc_photons 
            UVS.inp['mc_minphotons']=mc_photons # Note: MC_MINPHOTONS = 1e3 in uvspec.h so need to modify!
    else:
        UVS.inp['wavelength']='295 2800'     # Changed from '250 4000' 20.01.2026   
        if RTdimension == '3D':
            mc_photons = int(1e7)  
            UVS.inp['mc_photons']=mc_photons 
      
    
    

    # Water cloud
    UVS.inp['wc_properties']='mie'
    Nz = 100 
    wc_value, wcreff_value = 1e-2, 10.0
    h_min, h_max = 2, 3
    heights = np.linspace(0, 10, Nz) # km
    # Make a 2x2 cloud in the center of the domain
    cloud_start_X = Nx//2 #+ Nx//8
    cloud_start_Y = Ny//2 #+ Ny//8
    idx_inside_cloud = []

    if RTdimension == '3D':  
        
        # iz=0
        # last_h=99999
        # for h,p in zip(ACMCOM.height_level[1:,irec_surface_pixel],ACMCOM.pressure_level[1:,irec_surface_pixel]):
        #     if h > -1e-4 and p < 1e+10:
        #         last_h=h
        #         last_p=p
        #         iz=iz+1

        # if np.abs(last_h) > 1e-4 and last_p < 1e+10:
        #     iz=iz+1

        # Nz = iz-1 #ACMCOM.height_level[1:,ia].shape[0]
        # dx = 1.0 # 1 km horizontal resolution
        # dy = 1.0

        flag = 3 # mystic 3D format# mystic 3D format# mystic 3D format# mystic 3D format
        dx, dy = 1.0, 1.0 # 1km horizontal resolution
        heights = heights[1:]
        Nz -= 1
        wc_file_3D=input_dir+f'tmp_{ia:04d}_wc3D.dat'
        f = open(wc_file_3D,'w')
        f.write('{:d} {:d} {:d} {:d}\n'.format(Nx, Ny, Nz, flag))
        f.write('{:f} {:f} '.format(dx, dy))

        
       
        for h in heights:
            f.write('{:8.3f} '.format(np.abs(h)))
            
        f.write('\n')
        wclines=0

        ix=1
        for iac in iacrosses:
            iy=1 
            for ial in ialongs:
                iz = 1
                inside_cloud_region = (
                        cloud_start_X <= ix < cloud_start_X + 2
                    and cloud_start_Y <= iy < cloud_start_Y + 2
                )

                # Define indecies for cloud-vertical height
                if ix == cloud_start_X and iy == cloud_start_Y:
                    h_valid = np.asarray(heights)[heights > 0]

                    iz_h_min = np.argmin(np.abs(h_valid - h_min)) + 1
                    iz_h_max = np.argmin(np.abs(h_valid - h_max)) + 1

                    iz_min = min(iz_h_min, iz_h_max)
                    iz_max = max(iz_h_min, iz_h_max)

                    print("Cloud vertical indices:", iz_min, iz_max)
                    print("Actual heights:", h_valid[iz_min-1], h_valid[iz_max-1])

                

                # BG: made changes
                # - iz += 1 for each realistic h > 0 measurement by EarthCARE
                # - previosly iz += 1 only for realistic h > 0 and wc > 0 (this was the error)
                for h in heights:
                    if inside_cloud_region and iz_min <= iz < iz_max: # h_min < h < h_max:
                        # set wc
                        f.write('{:d} {:d} {:d} {:f} {:f}\n'.format(ix, iy, iz, wc_value, wcreff_value))
                        print(f'Writing wc={wc_value} at ix={ix}, iy={iy}, iz={iz}')
            
                    
                        wclines=wclines+1
                    iz=iz+1 
                iy=iy+1
            ix=ix+1

        # Add dummy line to make mystic run
        if wclines==0:
            f.write('{:d} {:d} {:d} {:f} {:f}\n'.format(1, 1, 1, 0, 10))
                
        f.close()
        UVS.inp['wc_file 3D']=wc_file_3D
        UVS.inp['mc_std']=''
    else:       
        wc_file_1D=input_dir+f'tmp_{ia:04d}_wc1D.dat'
        UVS.inp['wc_file 1D']=wc_file_1D
        f = open(wc_file_1D,'w')
    
        ix = 1
        for iac in iacrosses:
            iy = 1
            for ial in ialongs:
                inside_cloud_region = (
                    cloud_start_X <= ix < cloud_start_X + 2
                    and cloud_start_Y <= iy < cloud_start_Y + 2
                )
                if inside_cloud_region:
                    idx_inside_cloud.append(Nx* (iy-1) + (ix-1))
                iy += 1
            ix += 1
        print(idx_inside_cloud)
    
                
        for h in heights[::-1]:
            if h_min <= h <= h_max and ia in idx_inside_cloud:
                wc, wcreff = wc_value, wcreff_value
            else:
                wc, wcreff = 0.0, 0.0

            f.write('{:8.3f} {:10.5f} {:8.3f}\n'.format(h, wc, wcreff))
        f.close()
    
    return UVS


def tmp_output_filename(iia, scene_name):
    """
    Return the filename to be used for temporary output files.

    """
    pass
    












if __name__ == "__main__":
    start_time = datetime.now(timezone.utc) 
    # print("Run started at", start_time.isoformat())



    ##############################################################################################
    WANT_SUR = True         # zout at SUR or TOA
    WANT_3D  = False        # MYSTIC or DISORT

    verbose  = False        # Want verbose output from RTM
    want_ps  = False        # Psudospherical solver
        
                            
    which_buffer = 1
    # 0: MCIPA   1:Test   2: 21x21   3: 41x41  
    ################################################################################################

   
  

    # BG: To distinguish minor modifications to .nc files
    additional_spesifications = 'SINGLE_CLOUD' 
    sza = 60
    phi0 = 60


    if WANT_SUR: additional_spesifications += "_SUR"
    else:        additional_spesifications += "_TOA"



    idx_source = 0
    sources = [['solar'], 
              ['thermal'],
              ['solar', 'thermal']][idx_source] 


       
    if WANT_3D:
        rte_solver = 'montecarlo' #'mystic' 
        RTdimension = '3D'
    else:
        if want_ps:
            rte_solver = 'disort \npseudospherical'
        else: 
            rte_solver = 'disort'  # 'twostr'        
        RTdimension ='1D' 
    print(f"{RTdimension} RTM - {rte_solver}")


    # ---------------------------------- Paths -----------------------------------------------
    RTInpOutPath = './tmpRTIO/'    # Folder to store the final results
    RTOutNetcdfPath = './RESULTS_TEST/'  # Folder name for netcdf result files
    uvspecpath = '/xnilu_wrk2/projects/NEVAR/libRadtran/bin/' # Folder to RTM code
    RTMdata = '/xnilu_wrk2/projects/NEVAR/libRadtran/data/' #RTM input data

    libRad_version= 'v01_'+rte_solver+'_'+RTdimension
    # --------------------------------------------------------------------------------------------



    Nx, Ny = 1, 1 # Default Horizontal Computaion-Domain Size                                                                                     # min / pixel   |     solar-std     |     thermal-std  
    if   which_buffer == 0 :                  buffer_str = 'MCIPA'
    elif which_buffer == 1 : Nx, Ny =  5,  5; buffer_str = 'TEST   Buffer (5 x 5)'                      #   1.5         |                   |
    elif which_buffer == 2 : Nx, Ny = 21, 21; buffer_str = '~BBR +  5 pixels-buffer  (21 x 21)'         #    4          |       ~5-8        |       ~0.5-2
    elif which_buffer == 3 : Nx, Ny = 41, 41; buffer_str = '~BBR + 35 pixels-buffer  (41 x 41)'         #    5          |       ~17         |       ~1.6
    else:buffer_str = ''
    

                                                                                            

        
    SceneNamelibRad='libRad'
    libRad = ReadEC.Scene(Name=SceneNamelibRad, verbose=verbose)    
    if 'montecarlo' in rte_solver:
        shape = (1, Nx, Ny)
    else: 
        shape = Nx*Ny
    libRad.latitude_active = np.zeros(shape)

    # print(shape)
    libRad.solar_e = np.zeros(shape)       
    libRad.solar_e_std = np.zeros(shape)
    libRad.thermal_e = np.zeros(shape)
    libRad.thermal_e_std = np.zeros(shape)


    if WANT_3D:
        ialongs = [Nx//2]
        iacr = Nx // 2
    else:
        ialongs = np.arange(Nx*Ny)
        iacr = 0

    



    # Parallelize over OrbitIDs instead of along-track points.
    # Each MPI rank will process all ialongs for its assigned orbit in sequence.
    
    for ia in ialongs:
        # --- BG: ADD THIS BLOCK ---
        e_solar = np.nan
        e_solar_std = np.nan
        e_thermal = np.nan
        e_thermal_std = np.nan
        # ----------------------
        
        

        for source in sources:
            try: 
                RTM=True
                if source=='solar':
                    mol_abs_param = 'kato2' #'reptran course' #
                elif source=='thermal':
                    mol_abs_param = 'fu' 

                if RTM:
                    UVS = UVspec.UVspec()   #tms: UVspec.py is an external program that defines input and runs libRadtran
                    UVS.inp['rte_solver']=rte_solver
                    #tms: "SetRTM" is a function defined above - NB! Check path
                    UVS = SetRTM(UVS, ia, iacr, sza, phi0,mc_basename_path=RTInpOutPath, RTdimension = RTdimension, input_dir=RTInpOutPath,
                                mol_abs_param=mol_abs_param, data_dir=RTMdata, source=source)
                    if UVS.status == 'OK':
                        # uvspecInputFile=RTInpOutPath+'tmp'+'{:00004d}'.format(ia)+source+str(my_rank)+'.inp'  
                        # NEW TEST
                        uvspecInputFile=RTInpOutPath+f'tmp_SINGLE_CLOUD_{ia:04d}.inp'    
                        uvspecOutputFile=uvspecInputFile.replace('inp','out')
                        UVS.WriteInputFile(uvspecInputFile)

                        status = UVS.SingleRun(uvspecInputFile, uvspecOutputFile, verbose=False, uvspecpath=uvspecpath)

                        if UVS.inp['rte_solver']=='montecarlo': ######################## MYSTIC
                            try:
                                mc_flx_dir, mc_flx_dn, mc_flx_up, edir_std, edn_std, eup_std = UVspec.ReadMCOut(UVS.mc_basename, mol_abs_param=mol_abs_param) #fn = fn + '.flx.spc'  
                                if 'AllLevels' in additional_spesifications:
                                    nlev = int(len(mc_flx_dir)/(Nx*Ny))
                                
                                    mc_flx_dir  = mc_flx_dir.reshape(nlev, Nx, Ny)
                                    mc_flx_dn   = mc_flx_dn.reshape(nlev, Nx, Ny)
                                    mc_flx_up   = mc_flx_up.reshape(nlev, Nx, Ny)

                                    eup_std     = eup_std.reshape(nlev, Nx, Ny)
                                    edn_std     = edn_std.reshape(nlev, Nx, Ny)
                                    edir_std    = edir_std.reshape(nlev, Nx, Ny)


            
                                else:
                                    mc_flx_dir  = mc_flx_dir.reshape(Nx, Ny)
                                    mc_flx_dn   = mc_flx_dn.reshape(Nx, Ny)
                                    mc_flx_up   = mc_flx_up.reshape(Nx, Ny)
                                    edir_std    = edir_std.reshape(Nx, Ny)
                                    eup_std     = eup_std.reshape(Nx, Ny)
                                    edn_std     = edn_std.reshape(Nx, Ny)


                                
                            except (FileNotFoundError, ValueError) as e:
                                print(f"[WORKER {my_rank}] WARNING: Could not read MC output for ia={ia}, source={source}. Skipping. Error: {e}")
                                # continue  # Go to the next iteration
                        
                            if source=='solar':   
                                if WANT_SUR:
                                    e_solar_std = edir_std + edn_std
                                    e_solar     = mc_flx_dir + mc_flx_dn
                                    # print('direct + diffuse solar flux at SUR = ', e_solar, '\n with direct flux = ', mc_flx_dir, ' and diffuse flux = ', mc_flx_dn)
                                else:
                                    e_solar_std = eup_std
                                    e_solar     = mc_flx_up

                            elif source=='thermal':
                                if WANT_SUR:
                                    e_thermal_std = edir_std + edn_std
                                    e_thermal     = mc_flx_dir + mc_flx_dn
                                else:
                                    e_thermal_std = eup_std
                                    e_thermal     = mc_flx_up

                
                        else: ########################################################## DISORT
                            try:
                                sza, zout, edir, edn, eup = UVspec.ReadRTOut(uvspecOutputFile, mol_abs_param=mol_abs_param)
                            except ValueError as e:
                                # print(edir, edn, eup)
                                print("Error [WORKER {}]: could not read file {} with mol_abs_param {}".
                                    format(my_rank, uvspecOutputFile, mol_abs_param))
                                print(f"Error [WORKER {my_rank}]: ", e)

                                
                        

                            e_solar_std = e_thermal_std = None  # No std output from DISORT
                            if source=='solar':
                                if WANT_SUR:
                                    e_solar = edir + edn
                                else:
                                    e_solar = eup
                            elif source=='thermal':
                                if WANT_SUR:
                                    e_thermal = edir + edn
                                else:
                                    e_thermal = eup


                    else:
                        print(UVS.status)
                        print(f'WARNING: UVS.status not OK for ia={ia}, source={source}. Skipping.')
                        zout=edir=eup=eup_std=0.0
                else:
                    zout=edir_solar=eup_solar=eup_solar_std=0.0
                    edir_thermal=eup_thermal=eup_thermal_std=0.0
                    

                

                if rte_solver=='montecarlo': 
                    if not hasattr(e_solar, "shape"):
                        # raise ValueError("Simulated solar flux has no shape (simulation was not successful)")    
                        e_solar_center = np.nan
                        e_solar_std_center = np.nan
                        e_thermal_center = np.nan
                        e_thermal_std_center = np.nan                        
                
                    elif 'AllLevels' in additional_spesifications:
                        ix_center = e_solar.shape[1] // 2
                        iy_center = e_solar.shape[2] // 2
                        
                        if hasattr(e_solar_std, "shape"):   e_solar_std_center      = e_solar_std[:, ix_center, iy_center]
                        if hasattr(e_solar, "shape"):       e_solar_center          = e_solar[:, ix_center, iy_center]
                        if hasattr(e_thermal_std, "shape"): e_thermal_std_center    = e_thermal_std[:, ix_center, iy_center]
                        if hasattr(e_thermal, "shape"):     e_thermal_center        = e_thermal[:, ix_center, iy_center]
                    else: 
                        ix_center = e_solar.shape[0] // 2
                        iy_center = e_solar.shape[1] // 2

                        if hasattr(e_solar_std, "shape"):   e_solar_std_center      = float(e_solar_std[ix_center, iy_center])
                        if hasattr(e_solar, "shape"):       e_solar_center          = float(e_solar[ix_center, iy_center])
                        if hasattr(e_thermal_std, "shape"): e_thermal_std_center    = float(e_thermal_std[ix_center, iy_center])
                        if hasattr(e_thermal, "shape"):     e_thermal_center        = float(e_thermal[ix_center, iy_center])

                    # print("ix_center, iy_center", ix_center, iy_center)                              
                else:
                    e_solar_center     = e_solar
                    e_thermal_center   = e_thermal

                ####### PRINTING RESULTS ##########
                if 'AllLevels' in additional_spesifications:
                    print(e_solar_center)
                    print(e_solar_std)
                else:
                    e_str = f"{e_solar_center:7.1f}" if source=='solar' else f"{e_thermal_center:7.1f}"
                    std_str = f"+- {e_solar_std_center:.1f}" if 'montecarlo' in rte_solver else ''
                    print("\n\n\n"
                        f"      | ia={ia:4}  ----------------------------------------------------------|\n"
                        f"      |                                   Flux ({source}) = {e_str} {std_str} W/m2\n"
                            "      |--------------------------------------------------------------------------------|"
                    )
                   
            

            except Exception as e:
                print(f"\nException for ia={ia:4}: {e}\n")
    

    

    if rte_solver == "montecarlo":
        idx = (ia, slice(None), slice(None))
    else:
        idx = ia

    print(e_solar, e_solar_std, e_thermal, e_thermal_std)
    libRad.solar_e_std[idx]   = e_solar_std
    libRad.thermal_e_std[idx] = e_thermal_std
    libRad.solar_e[idx]       = e_solar
    libRad.thermal_e[idx]     = e_thermal

    # print(e_solar, libRad.solar_e[idx])

        
                        
                

    # Writing NetCDF file...")
    src = ','.join([i for i in sources])
    if 'montecarlo' in rte_solver:
        libRad.WriteNetcdf_mc_sample_grid(RTOutNetcdfPath+'libRad_'+libRad_version+'_'+src+'_'+ additional_spesifications + '.nc', shape=(Nx,Ny), verbose=False)
    else:
        libRad.WriteNetcdf(               RTOutNetcdfPath+'libRad_'+libRad_version+'_'+src+'_'+ additional_spesifications + '.nc', verbose=False)
    print("Wrote to " + RTOutNetcdfPath+'libRad_'+libRad_version+'_'+src+'_'+ additional_spesifications + '.nc')


    tt = datetime.now(timezone.utc) - start_time
    print(f"            Finished run. It took {tt.total_seconds()/3600:.2f} hours \n\n\n\n\n\n\n")

import numpy as np
import pickle
import os
import sys
import multiprocessing
import glob
import scipy
import scipy.constants
from scipy.interpolate import griddata
import dted
import pdb # For debugging
from mpi4py import MPI # Use MPI for multiprocessing
from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess



#import ReadEarthCAREL2 as ReadEC
import Read_RTM as ReadEC
import UVspec as UVspec
import Find_Overpass_Info as Find_Overpass_Info




# BG: overflow encouner - Warnings from reading ACM_COM files
# BG: Hiding the 'overflow encountered in multiply' warnings:
np.seterr(over='ignore')




# Where are we
myhost = os.uname()[1]
home = os.environ['HOME']

# initialize MPI
comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
np_mpi = comm.Get_size()


def pdeg2km(p):
    latfact = 111.32
    p0 = latfact*p[0]/1000
    lonfact = 40075 * np.cos(np.deg2rad( p[1] )) / 360
    p1 = latfact*p[1]/1000
    return (p0,p1)


def Check3DBufferSize(ia, iacr, which_buffer):

    Nx, Ny, iacrosses, ialongs = Calc3DBufferSize(ia, iacr, which_buffer)
    if Nx>0 and Ny>0:
        BufferOK=True
    else:
        BufferOK=False
        
    return BufferOK


def Calc3DBufferSize(ia, iacr, which_buffer):
    WANT_PRINTS = False
    # BG: Decide to use my buffer, not dynamic buffer!
    if which_buffer == 0:   along_dim, across_dim =  0,  0 # ( 1, 1) comp-domain  MCIPA 
    elif which_buffer == 1: along_dim, across_dim =  2,  2 # ( 5, 5) comp-domain  Test Buffer
    elif which_buffer == 2: along_dim, across_dim = 10, 10 # (21,21) comp-domain  ~BBR +  5 pixels-buffer
    elif which_buffer == 3: along_dim, across_dim = 20, 20 # (41,41) comp-domain  ~BBR + 35 pixels-buffer
    
    ialongs = np.arange(ia-along_dim, ia+along_dim+1)
    iacrosses = np.arange(iacr-across_dim, iacr+across_dim+1)
    
    Nx = iacrosses.shape[0]
    Ny = ialongs.shape[0]


    return Nx, Ny, iacrosses, ialongs

def SetRTM(UVS, ia, iacr, sza, phi0, input_dir='', mc_basename_path='', RTdimension = '1D',
           mol_abs_param='kato2', data_dir='', source='solar'):

    
    UVS.status = 'OK'
    # First some general input
    UVS.inp['data_files_path']=data_dir
    
    libpath='/xnilu_wrk2/projects/NEVAR/libRadtran/'
    UVS.inp['atmosphere_file'] = f"{libpath}/data/atmmod/afglss.dat"
    UVS.inp['albedo'] = 0.05

    UVS.inp['mol_abs_param']=mol_abs_param
    UVS.inp['source']=source
    UVS.inp['output_process']='sum'

    UVS.inp['output_user']='sza zout edir edn eup'
    UVS.inp['quiet']=''
    

    # Solar geometry
    # Set solar geometry here to make sure azimuth shift due to 3D geometry is included
    UVS.inp['sza']=sza
    UVS.inp['phi0']=phi0  # Azimuth is only needed for 3D, but it does not hurt for 1D.

    Nx, Ny, iacrosses, ialongs = Calc3DBufferSize(ia, iacr, which_buffer)  #calculates along and cross track buffer size 
    if RTdimension == '3D':       
        if UVS.inp['rte_solver'] != 'montecarlo':
            print('3D geometry only available with MYSTIC. You specified:', UVS.inp['rte_solver'])
            exit()
                
        UVS.inp['mc_std']=''
        # UVS.mc_basename=mc_basename_path+'mc_{:00004d}'.format(ia)+'_'+source # Old->made race-conditions errors 
        UVS.mc_basename=mc_basename_path+'mc_'+f"{ia:04d}_"+source
        print("MC basename set to:", UVS.mc_basename)
        UVS.inp['mc_basename']=UVS.mc_basename
        UVS.inp['mc_randomseed'] = (ia+1)*72341//11
        
        # Set SAMPLE-GRID
        UVS.inp['mc_sample_grid']=f'{Nx} {Ny}' # 3D computation domain
        UVS.inp['mc_reference_to_nn']='' # The sampled pixels correspond to the surface pixels.
     
        
    

    ############### SPESIFICATIONS TO SURFACE-VERSION ##################
    if 'TOA' in additional_spesifications: UVS.inp['zout']='TOA'
    elif 'SUR' in additional_spesifications: 
        # Set "zout sur" for DISORT
        # Do not spesify zout for MYSTIC -> give "zout sur" (which do not work for MYSTIC)
        if UVS.inp['rte_solver'] != 'montecarlo': UVS.inp['zout'] = 'sur'  
    ####################################################################

    

    if source=='thermal':
        UVS.inp['wavelength']='4500 100000'  
        if RTdimension == '3D':
            mc_photons = int(1e2)
            UVS.inp['mc_photons']=mc_photons 
            UVS.inp['mc_minphotons']=mc_photons # Note: MC_MINPHOTONS = 1e3 in uvspec.h so need to modify!
    else:
        UVS.inp['wavelength']='295 2800'     # Changed from '250 4000' 20.01.2026   
        if RTdimension == '3D':
            mc_photons = int(1e7)  
            UVS.inp['mc_photons']=mc_photons 
      
    
    

    # Water cloud
    UVS.inp['wc_properties']='mie'
    Nz = 100 
    wc_value, wcreff_value = 1e-2, 10.0
    h_min, h_max = 2, 3
    heights = np.linspace(0, 10, Nz) # km
    # Make a 2x2 cloud in the center of the domain
    cloud_start_X = Nx//2 #+ Nx//8
    cloud_start_Y = Ny//2 #+ Ny//8
    idx_inside_cloud = []

    if RTdimension == '3D':  
        
        # iz=0
        # last_h=99999
        # for h,p in zip(ACMCOM.height_level[1:,irec_surface_pixel],ACMCOM.pressure_level[1:,irec_surface_pixel]):
        #     if h > -1e-4 and p < 1e+10:
        #         last_h=h
        #         last_p=p
        #         iz=iz+1

        # if np.abs(last_h) > 1e-4 and last_p < 1e+10:
        #     iz=iz+1

        # Nz = iz-1 #ACMCOM.height_level[1:,ia].shape[0]
        # dx = 1.0 # 1 km horizontal resolution
        # dy = 1.0

        flag = 3 # mystic 3D format# mystic 3D format# mystic 3D format# mystic 3D format
        dx, dy = 1.0, 1.0 # 1km horizontal resolution
        heights = heights[1:]
        Nz -= 1
        wc_file_3D=input_dir+f'tmp_{ia:04d}_wc3D.dat'
        f = open(wc_file_3D,'w')
        f.write('{:d} {:d} {:d} {:d}\n'.format(Nx, Ny, Nz, flag))
        f.write('{:f} {:f} '.format(dx, dy))

        
       
        for h in heights:
            f.write('{:8.3f} '.format(np.abs(h)))
            
        f.write('\n')
        wclines=0

        ix=1
        for iac in iacrosses:
            iy=1 
            for ial in ialongs:
                iz = 1
                inside_cloud_region = (
                        cloud_start_X <= ix < cloud_start_X + 2
                    and cloud_start_Y <= iy < cloud_start_Y + 2
                )

                # Define indecies for cloud-vertical height
                if ix == cloud_start_X and iy == cloud_start_Y:
                    h_valid = np.asarray(heights)[heights > 0]

                    iz_h_min = np.argmin(np.abs(h_valid - h_min)) + 1
                    iz_h_max = np.argmin(np.abs(h_valid - h_max)) + 1

                    iz_min = min(iz_h_min, iz_h_max)
                    iz_max = max(iz_h_min, iz_h_max)

                    print("Cloud vertical indices:", iz_min, iz_max)
                    print("Actual heights:", h_valid[iz_min-1], h_valid[iz_max-1])

                

                # BG: made changes
                # - iz += 1 for each realistic h > 0 measurement by EarthCARE
                # - previosly iz += 1 only for realistic h > 0 and wc > 0 (this was the error)
                for h in heights:
                    if inside_cloud_region and iz_min <= iz < iz_max: # h_min < h < h_max:
                        # set wc
                        f.write('{:d} {:d} {:d} {:f} {:f}\n'.format(ix, iy, iz, wc_value, wcreff_value))
                        print(f'Writing wc={wc_value} at ix={ix}, iy={iy}, iz={iz}')
            
                    
                        wclines=wclines+1
                    iz=iz+1 
                iy=iy+1
            ix=ix+1

        # Add dummy line to make mystic run
        if wclines==0:
            f.write('{:d} {:d} {:d} {:f} {:f}\n'.format(1, 1, 1, 0, 10))
                
        f.close()
        UVS.inp['wc_file 3D']=wc_file_3D
        UVS.inp['mc_std']=''
    else:       
        wc_file_1D=input_dir+f'tmp_{ia:04d}_wc1D.dat'
        UVS.inp['wc_file 1D']=wc_file_1D
        f = open(wc_file_1D,'w')
    
        ix = 1
        for iac in iacrosses:
            iy = 1
            for ial in ialongs:
                inside_cloud_region = (
                    cloud_start_X <= ix < cloud_start_X + 2
                    and cloud_start_Y <= iy < cloud_start_Y + 2
                )
                if inside_cloud_region:
                    idx_inside_cloud.append(Nx* (iy-1) + (ix-1))
                iy += 1
            ix += 1
        print(idx_inside_cloud)
    
                
        for h in heights[::-1]:
            if h_min <= h <= h_max and ia in idx_inside_cloud:
                wc, wcreff = wc_value, wcreff_value
            else:
                wc, wcreff = 0.0, 0.0

            f.write('{:8.3f} {:10.5f} {:8.3f}\n'.format(h, wc, wcreff))
        f.close()
    
    return UVS


def tmp_output_filename(iia, scene_name):
    """
    Return the filename to be used for temporary output files.

    """
    pass
    












if __name__ == "__main__":
    start_time = datetime.now(timezone.utc) 
    # print("Run started at", start_time.isoformat())



    ##############################################################################################
    WANT_SUR = True         # zout at SUR or TOA
    WANT_3D  = True        # MYSTIC or DISORT

    verbose  = False        # Want verbose output from RTM
    want_ps  = False        # Psudospherical solver
        
                            
    which_buffer = 1
    # 0: MCIPA   1:Test   2: 21x21   3: 41x41  
    ################################################################################################

   
  

    # BG: To distinguish minor modifications to .nc files
    additional_spesifications = 'SINGLE_CLOUD' 
    sza = 60
    phi0 = 60


    if WANT_SUR: additional_spesifications += "_SUR"
    else:        additional_spesifications += "_TOA"



    idx_source = 0
    sources = [['solar'], 
              ['thermal'],
              ['solar', 'thermal']][idx_source] 


       
    if WANT_3D:
        rte_solver = 'montecarlo' #'mystic' 
        RTdimension = '3D'
    else:
        if want_ps:
            rte_solver = 'disort \npseudospherical'
        else: 
            rte_solver = 'disort'  # 'twostr'        
        RTdimension ='1D' 
    print(f"{RTdimension} RTM - {rte_solver}")


    # ---------------------------------- Paths -----------------------------------------------
    RTInpOutPath = './tmpRTIO/'    # Folder to store the final results
    RTOutNetcdfPath = './RESULTS_TEST/'  # Folder name for netcdf result files
    uvspecpath = '/xnilu_wrk2/projects/NEVAR/libRadtran/bin/' # Folder to RTM code
    RTMdata = '/xnilu_wrk2/projects/NEVAR/libRadtran/data/' #RTM input data

    libRad_version= 'v01_'+rte_solver+'_'+RTdimension
    # --------------------------------------------------------------------------------------------



    Nx, Ny = 1, 1 # Default Horizontal Computaion-Domain Size                                                                                     # min / pixel   |     solar-std     |     thermal-std  
    if   which_buffer == 0 :                  buffer_str = 'MCIPA'
    elif which_buffer == 1 : Nx, Ny =  5,  5; buffer_str = 'TEST   Buffer (5 x 5)'                      #   1.5         |                   |
    elif which_buffer == 2 : Nx, Ny = 21, 21; buffer_str = '~BBR +  5 pixels-buffer  (21 x 21)'         #    4          |       ~5-8        |       ~0.5-2
    elif which_buffer == 3 : Nx, Ny = 41, 41; buffer_str = '~BBR + 35 pixels-buffer  (41 x 41)'         #    5          |       ~17         |       ~1.6
    else:buffer_str = ''
    

                                                                                            

        
    SceneNamelibRad='libRad'
    libRad = ReadEC.Scene(Name=SceneNamelibRad, verbose=verbose)    
    if 'montecarlo' in rte_solver:
        shape = (1, Nx, Ny)
    else: 
        shape = Nx*Ny
    libRad.latitude_active = np.zeros(shape)

    # print(shape)
    libRad.solar_e = np.zeros(shape)       
    libRad.solar_e_std = np.zeros(shape)
    libRad.thermal_e = np.zeros(shape)
    libRad.thermal_e_std = np.zeros(shape)


    if WANT_3D:
        ialongs = [Nx//2]
        iacr = Nx // 2
    else:
        ialongs = np.arange(Nx*Ny)
        iacr = 0

    



    # Parallelize over OrbitIDs instead of along-track points.
    # Each MPI rank will process all ialongs for its assigned orbit in sequence.
    
    for ia in ialongs:
        # --- BG: ADD THIS BLOCK ---
        e_solar = np.nan
        e_solar_std = np.nan
        e_thermal = np.nan
        e_thermal_std = np.nan
        # ----------------------
        
        

        for source in sources:
            try: 
                RTM=True
                if source=='solar':
                    mol_abs_param = 'kato2' #'reptran course' #
                elif source=='thermal':
                    mol_abs_param = 'fu' 

                if RTM:
                    UVS = UVspec.UVspec()   #tms: UVspec.py is an external program that defines input and runs libRadtran
                    UVS.inp['rte_solver']=rte_solver
                    #tms: "SetRTM" is a function defined above - NB! Check path
                    UVS = SetRTM(UVS, ia, iacr, sza, phi0,mc_basename_path=RTInpOutPath, RTdimension = RTdimension, input_dir=RTInpOutPath,
                                mol_abs_param=mol_abs_param, data_dir=RTMdata, source=source)
                    if UVS.status == 'OK':
                        # uvspecInputFile=RTInpOutPath+'tmp'+'{:00004d}'.format(ia)+source+str(my_rank)+'.inp'  
                        # NEW TEST
                        uvspecInputFile=RTInpOutPath+f'tmp_SINGLE_CLOUD_{ia:04d}.inp'    
                        uvspecOutputFile=uvspecInputFile.replace('inp','out')
                        UVS.WriteInputFile(uvspecInputFile)

                        status = UVS.SingleRun(uvspecInputFile, uvspecOutputFile, verbose=False, uvspecpath=uvspecpath)

                        if UVS.inp['rte_solver']=='montecarlo': ######################## MYSTIC
                            try:
                                mc_flx_dir, mc_flx_dn, mc_flx_up, edir_std, edn_std, eup_std = UVspec.ReadMCOut(UVS.mc_basename, mol_abs_param=mol_abs_param) #fn = fn + '.flx.spc'  
                                if 'AllLevels' in additional_spesifications:
                                    nlev = int(len(mc_flx_dir)/(Nx*Ny))
                                
                                    mc_flx_dir  = mc_flx_dir.reshape(nlev, Nx, Ny)
                                    mc_flx_dn   = mc_flx_dn.reshape(nlev, Nx, Ny)
                                    mc_flx_up   = mc_flx_up.reshape(nlev, Nx, Ny)

                                    eup_std     = eup_std.reshape(nlev, Nx, Ny)
                                    edn_std     = edn_std.reshape(nlev, Nx, Ny)
                                    edir_std    = edir_std.reshape(nlev, Nx, Ny)


            
                                else:
                                    mc_flx_dir  = mc_flx_dir.reshape(Nx, Ny)
                                    mc_flx_dn   = mc_flx_dn.reshape(Nx, Ny)
                                    mc_flx_up   = mc_flx_up.reshape(Nx, Ny)
                                    edir_std    = edir_std.reshape(Nx, Ny)
                                    eup_std     = eup_std.reshape(Nx, Ny)
                                    edn_std     = edn_std.reshape(Nx, Ny)


                                
                            except (FileNotFoundError, ValueError) as e:
                                print(f"[WORKER {my_rank}] WARNING: Could not read MC output for ia={ia}, source={source}. Skipping. Error: {e}")
                                # continue  # Go to the next iteration
                        
                            if source=='solar':   
                                if WANT_SUR:
                                    e_solar_std = edir_std + edn_std
                                    e_solar     = mc_flx_dir + mc_flx_dn
                                    # print('direct + diffuse solar flux at SUR = ', e_solar, '\n with direct flux = ', mc_flx_dir, ' and diffuse flux = ', mc_flx_dn)
                                else:
                                    e_solar_std = eup_std
                                    e_solar     = mc_flx_up

                            elif source=='thermal':
                                if WANT_SUR:
                                    e_thermal_std = edir_std + edn_std
                                    e_thermal     = mc_flx_dir + mc_flx_dn
                                else:
                                    e_thermal_std = eup_std
                                    e_thermal     = mc_flx_up

                
                        else: ########################################################## DISORT
                            try:
                                sza, zout, edir, edn, eup = UVspec.ReadRTOut(uvspecOutputFile, mol_abs_param=mol_abs_param)
                            except ValueError as e:
                                # print(edir, edn, eup)
                                print("Error [WORKER {}]: could not read file {} with mol_abs_param {}".
                                    format(my_rank, uvspecOutputFile, mol_abs_param))
                                print(f"Error [WORKER {my_rank}]: ", e)

                                
                        

                            e_solar_std = e_thermal_std = None  # No std output from DISORT
                            if source=='solar':
                                if WANT_SUR:
                                    e_solar = edir + edn
                                else:
                                    e_solar = eup
                            elif source=='thermal':
                                if WANT_SUR:
                                    e_thermal = edir + edn
                                else:
                                    e_thermal = eup


                    else:
                        print(UVS.status)
                        print(f'WARNING: UVS.status not OK for ia={ia}, source={source}. Skipping.')
                        zout=edir=eup=eup_std=0.0
                else:
                    zout=edir_solar=eup_solar=eup_solar_std=0.0
                    edir_thermal=eup_thermal=eup_thermal_std=0.0
                    

                

                if rte_solver=='montecarlo': 
                    if not hasattr(e_solar, "shape"):
                        # raise ValueError("Simulated solar flux has no shape (simulation was not successful)")    
                        e_solar_center = np.nan
                        e_solar_std_center = np.nan
                        e_thermal_center = np.nan
                        e_thermal_std_center = np.nan                        
                
                    elif 'AllLevels' in additional_spesifications:
                        ix_center = e_solar.shape[1] // 2
                        iy_center = e_solar.shape[2] // 2
                        
                        if hasattr(e_solar_std, "shape"):   e_solar_std_center      = e_solar_std[:, ix_center, iy_center]
                        if hasattr(e_solar, "shape"):       e_solar_center          = e_solar[:, ix_center, iy_center]
                        if hasattr(e_thermal_std, "shape"): e_thermal_std_center    = e_thermal_std[:, ix_center, iy_center]
                        if hasattr(e_thermal, "shape"):     e_thermal_center        = e_thermal[:, ix_center, iy_center]
                    else: 
                        ix_center = e_solar.shape[0] // 2
                        iy_center = e_solar.shape[1] // 2

                        if hasattr(e_solar_std, "shape"):   e_solar_std_center      = float(e_solar_std[ix_center, iy_center])
                        if hasattr(e_solar, "shape"):       e_solar_center          = float(e_solar[ix_center, iy_center])
                        if hasattr(e_thermal_std, "shape"): e_thermal_std_center    = float(e_thermal_std[ix_center, iy_center])
                        if hasattr(e_thermal, "shape"):     e_thermal_center        = float(e_thermal[ix_center, iy_center])

                    # print("ix_center, iy_center", ix_center, iy_center)                              
                else:
                    e_solar_center     = e_solar
                    e_thermal_center   = e_thermal

                ####### PRINTING RESULTS ##########
                if 'AllLevels' in additional_spesifications:
                    print(e_solar_center)
                    print(e_solar_std)
                else:
                    e_str = f"{e_solar_center:7.1f}" if source=='solar' else f"{e_thermal_center:7.1f}"
                    std_str = f"+- {e_solar_std_center:.1f}" if 'montecarlo' in rte_solver else ''
                    print("\n\n\n"
                        f"      | ia={ia:4}  ----------------------------------------------------------|\n"
                        f"      |                                   Flux ({source}) = {e_str} {std_str} W/m2\n"
                            "      |--------------------------------------------------------------------------------|"
                    )
                   
            

            except Exception as e:
                print(f"\nException for ia={ia:4}: {e}\n")
    

    

    if rte_solver == "montecarlo":
        idx = (ia, slice(None), slice(None))
    else:
        idx = ia

    print(e_solar, e_solar_std, e_thermal, e_thermal_std)
    libRad.solar_e_std[idx]   = e_solar_std
    libRad.thermal_e_std[idx] = e_thermal_std
    libRad.solar_e[idx]       = e_solar
    libRad.thermal_e[idx]     = e_thermal

    # print(e_solar, libRad.solar_e[idx])

        
                        
                

    # Writing NetCDF file...")
    src = ','.join([i for i in sources])
    if 'montecarlo' in rte_solver:
        libRad.WriteNetcdf_mc_sample_grid(RTOutNetcdfPath+'libRad_'+libRad_version+'_'+src+'_'+ additional_spesifications + '.nc', shape=(Nx,Ny), verbose=False)
    else:
        libRad.WriteNetcdf(               RTOutNetcdfPath+'libRad_'+libRad_version+'_'+src+'_'+ additional_spesifications + '.nc', verbose=False)
    print("Wrote to " + RTOutNetcdfPath+'libRad_'+libRad_version+'_'+src+'_'+ additional_spesifications + '.nc')


    tt = datetime.now(timezone.utc) - start_time
    print(f"            Finished run. It took {tt.total_seconds()/3600:.2f} hours \n\n\n\n\n\n\n")

import numpy as np
import pickle
import os
import sys
import multiprocessing
import glob
import scipy
import scipy.constants
from scipy.interpolate import griddata
import dted
import pdb # For debugging
from mpi4py import MPI # Use MPI for multiprocessing
from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess



#import ReadEarthCAREL2 as ReadEC
import Read_RTM as ReadEC
import UVspec as UVspec
import Find_Overpass_Info as Find_Overpass_Info




# BG: overflow encouner - Warnings from reading ACM_COM files
# BG: Hiding the 'overflow encountered in multiply' warnings:
np.seterr(over='ignore')




# Where are we
myhost = os.uname()[1]
home = os.environ['HOME']

# initialize MPI
comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
np_mpi = comm.Get_size()


def pdeg2km(p):
    latfact = 111.32
    p0 = latfact*p[0]/1000
    lonfact = 40075 * np.cos(np.deg2rad( p[1] )) / 360
    p1 = latfact*p[1]/1000
    return (p0,p1)


def Check3DBufferSize(ia, iacr, which_buffer):

    Nx, Ny, iacrosses, ialongs = Calc3DBufferSize(ia, iacr, which_buffer)
    if Nx>0 and Ny>0:
        BufferOK=True
    else:
        BufferOK=False
        
    return BufferOK


def Calc3DBufferSize(ia, iacr, which_buffer):
    WANT_PRINTS = False
    # BG: Decide to use my buffer, not dynamic buffer!
    if which_buffer == 0:   along_dim, across_dim =  0,  0 # ( 1, 1) comp-domain  MCIPA 
    elif which_buffer == 1: along_dim, across_dim =  2,  2 # ( 5, 5) comp-domain  Test Buffer
    elif which_buffer == 2: along_dim, across_dim = 10, 10 # (21,21) comp-domain  ~BBR +  5 pixels-buffer
    elif which_buffer == 3: along_dim, across_dim = 20, 20 # (41,41) comp-domain  ~BBR + 35 pixels-buffer
    
    ialongs = np.arange(ia-along_dim, ia+along_dim+1)
    iacrosses = np.arange(iacr-across_dim, iacr+across_dim+1)
    
    Nx = iacrosses.shape[0]
    Ny = ialongs.shape[0]


    return Nx, Ny, iacrosses, ialongs

def SetRTM(UVS, ia, iacr, sza, phi0, input_dir='', mc_basename_path='', RTdimension = '1D',
           mol_abs_param='kato2', data_dir='', source='solar'):

    
    UVS.status = 'OK'
    # First some general input
    UVS.inp['data_files_path']=data_dir
    
    libpath='/xnilu_wrk2/projects/NEVAR/libRadtran/'
    UVS.inp['atmosphere_file'] = f"{libpath}/data/atmmod/afglss.dat"
    UVS.inp['albedo'] = 0.05

    UVS.inp['mol_abs_param']=mol_abs_param
    UVS.inp['source']=source
    UVS.inp['output_process']='sum'

    UVS.inp['output_user']='sza zout edir edn eup'
    UVS.inp['quiet']=''
    

    # Solar geometry
    # Set solar geometry here to make sure azimuth shift due to 3D geometry is included
    UVS.inp['sza']=sza
    UVS.inp['phi0']=phi0  # Azimuth is only needed for 3D, but it does not hurt for 1D.

    Nx, Ny, iacrosses, ialongs = Calc3DBufferSize(ia, iacr, which_buffer)  #calculates along and cross track buffer size 
    if RTdimension == '3D':       
        if UVS.inp['rte_solver'] != 'montecarlo':
            print('3D geometry only available with MYSTIC. You specified:', UVS.inp['rte_solver'])
            exit()
                
        UVS.inp['mc_std']=''
        # UVS.mc_basename=mc_basename_path+'mc_{:00004d}'.format(ia)+'_'+source # Old->made race-conditions errors 
        UVS.mc_basename=mc_basename_path+'mc_'+f"{ia:04d}_"+source
        print("MC basename set to:", UVS.mc_basename)
        UVS.inp['mc_basename']=UVS.mc_basename
        UVS.inp['mc_randomseed'] = (ia+1)*72341//11
        
        # Set SAMPLE-GRID
        UVS.inp['mc_sample_grid']=f'{Nx} {Ny}' # 3D computation domain
        UVS.inp['mc_reference_to_nn']='' # The sampled pixels correspond to the surface pixels.
     
        
    

    ############### SPESIFICATIONS TO SURFACE-VERSION ##################
    if 'TOA' in additional_spesifications: UVS.inp['zout']='TOA'
    elif 'SUR' in additional_spesifications: 
        # Set "zout sur" for DISORT
        # Do not spesify zout for MYSTIC -> give "zout sur" (which do not work for MYSTIC)
        if UVS.inp['rte_solver'] != 'montecarlo': UVS.inp['zout'] = 'sur'  
    ####################################################################

    

    if source=='thermal':
        UVS.inp['wavelength']='4500 100000'  
        if RTdimension == '3D':
            mc_photons = int(1e2)
            UVS.inp['mc_photons']=mc_photons 
            UVS.inp['mc_minphotons']=mc_photons # Note: MC_MINPHOTONS = 1e3 in uvspec.h so need to modify!
    else:
        UVS.inp['wavelength']='295 2800'     # Changed from '250 4000' 20.01.2026   
        if RTdimension == '3D':
            mc_photons = int(1e7)  
            UVS.inp['mc_photons']=mc_photons 
      
    
    

    # Water cloud
    UVS.inp['wc_properties']='mie'
    Nz = 100 
    wc_value, wcreff_value = 1e-2, 10.0
    h_min, h_max = 2, 3
    heights = np.linspace(0, 10, Nz) # km
    # Make a 2x2 cloud in the center of the domain
    cloud_start_X = Nx//2 #+ Nx//8
    cloud_start_Y = Ny//2 #+ Ny//8
    idx_inside_cloud = []

    if RTdimension == '3D':  
        
        # iz=0
        # last_h=99999
        # for h,p in zip(ACMCOM.height_level[1:,irec_surface_pixel],ACMCOM.pressure_level[1:,irec_surface_pixel]):
        #     if h > -1e-4 and p < 1e+10:
        #         last_h=h
        #         last_p=p
        #         iz=iz+1

        # if np.abs(last_h) > 1e-4 and last_p < 1e+10:
        #     iz=iz+1

        # Nz = iz-1 #ACMCOM.height_level[1:,ia].shape[0]
        # dx = 1.0 # 1 km horizontal resolution
        # dy = 1.0

        flag = 3 # mystic 3D format# mystic 3D format# mystic 3D format# mystic 3D format
        dx, dy = 1.0, 1.0 # 1km horizontal resolution
        heights = heights[1:]
        Nz -= 1
        wc_file_3D=input_dir+f'tmp_{ia:04d}_wc3D.dat'
        f = open(wc_file_3D,'w')
        f.write('{:d} {:d} {:d} {:d}\n'.format(Nx, Ny, Nz, flag))
        f.write('{:f} {:f} '.format(dx, dy))

        
       
        for h in heights:
            f.write('{:8.3f} '.format(np.abs(h)))
            
        f.write('\n')
        wclines=0

        ix=1
        for iac in iacrosses:
            iy=1 
            for ial in ialongs:
                iz = 1
                inside_cloud_region = (
                        cloud_start_X <= ix < cloud_start_X + 2
                    and cloud_start_Y <= iy < cloud_start_Y + 2
                )

                # Define indecies for cloud-vertical height
                if ix == cloud_start_X and iy == cloud_start_Y:
                    h_valid = np.asarray(heights)[heights > 0]

                    iz_h_min = np.argmin(np.abs(h_valid - h_min)) + 1
                    iz_h_max = np.argmin(np.abs(h_valid - h_max)) + 1

                    iz_min = min(iz_h_min, iz_h_max)
                    iz_max = max(iz_h_min, iz_h_max)

                    print("Cloud vertical indices:", iz_min, iz_max)
                    print("Actual heights:", h_valid[iz_min-1], h_valid[iz_max-1])

                

                # BG: made changes
                # - iz += 1 for each realistic h > 0 measurement by EarthCARE
                # - previosly iz += 1 only for realistic h > 0 and wc > 0 (this was the error)
                for h in heights:
                    if inside_cloud_region and iz_min <= iz < iz_max: # h_min < h < h_max:
                        # set wc
                        f.write('{:d} {:d} {:d} {:f} {:f}\n'.format(ix, iy, iz, wc_value, wcreff_value))
                        print(f'Writing wc={wc_value} at ix={ix}, iy={iy}, iz={iz}')
            
                    
                        wclines=wclines+1
                    iz=iz+1 
                iy=iy+1
            ix=ix+1

        # Add dummy line to make mystic run
        if wclines==0:
            f.write('{:d} {:d} {:d} {:f} {:f}\n'.format(1, 1, 1, 0, 10))
                
        f.close()
        UVS.inp['wc_file 3D']=wc_file_3D
        UVS.inp['mc_std']=''
    else:       
        wc_file_1D=input_dir+f'tmp_{ia:04d}_wc1D.dat'
        UVS.inp['wc_file 1D']=wc_file_1D
        f = open(wc_file_1D,'w')
    
        ix = 1
        for iac in iacrosses:
            iy = 1
            for ial in ialongs:
                inside_cloud_region = (
                    cloud_start_X <= ix < cloud_start_X + 2
                    and cloud_start_Y <= iy < cloud_start_Y + 2
                )
                if inside_cloud_region:
                    idx_inside_cloud.append(Nx* (iy-1) + (ix-1))
                iy += 1
            ix += 1
        print(idx_inside_cloud)
    
                
        for h in heights[::-1]:
            if h_min <= h <= h_max and ia in idx_inside_cloud:
                wc, wcreff = wc_value, wcreff_value
            else:
                wc, wcreff = 0.0, 0.0

            f.write('{:8.3f} {:10.5f} {:8.3f}\n'.format(h, wc, wcreff))
        f.close()
    
    return UVS


def tmp_output_filename(iia, scene_name):
    """
    Return the filename to be used for temporary output files.

    """
    pass
    












if __name__ == "__main__":
    start_time = datetime.now(timezone.utc) 
    # print("Run started at", start_time.isoformat())



    ##############################################################################################
    WANT_SUR = True         # zout at SUR or TOA
    WANT_3D  = False        # MYSTIC or DISORT

    verbose  = False        # Want verbose output from RTM
    want_ps  = False        # Psudospherical solver
        
                            
    which_buffer = 1
    # 0: MCIPA   1:Test   2: 21x21   3: 41x41  
    ################################################################################################

   
  

    # BG: To distinguish minor modifications to .nc files
    additional_spesifications = 'SINGLE_CLOUD' 
    sza = 60
    phi0 = 60


    if WANT_SUR: additional_spesifications += "_SUR"
    else:        additional_spesifications += "_TOA"



    idx_source = 0
    sources = [['solar'], 
              ['thermal'],
              ['solar', 'thermal']][idx_source] 


       
    if WANT_3D:
        rte_solver = 'montecarlo' #'mystic' 
        RTdimension = '3D'
    else:
        if want_ps:
            rte_solver = 'disort \npseudospherical'
        else: 
            rte_solver = 'disort'  # 'twostr'        
        RTdimension ='1D' 
    print(f"{RTdimension} RTM - {rte_solver}")


    # ---------------------------------- Paths -----------------------------------------------
    RTInpOutPath = './tmpRTIO/'    # Folder to store the final results
    RTOutNetcdfPath = './RESULTS_TEST/'  # Folder name for netcdf result files
    uvspecpath = '/xnilu_wrk2/projects/NEVAR/libRadtran/bin/' # Folder to RTM code
    RTMdata = '/xnilu_wrk2/projects/NEVAR/libRadtran/data/' #RTM input data

    libRad_version= 'v01_'+rte_solver+'_'+RTdimension
    # --------------------------------------------------------------------------------------------



    Nx, Ny = 1, 1 # Default Horizontal Computaion-Domain Size                                                                                     # min / pixel   |     solar-std     |     thermal-std  
    if   which_buffer == 0 :                  buffer_str = 'MCIPA'
    elif which_buffer == 1 : Nx, Ny =  5,  5; buffer_str = 'TEST   Buffer (5 x 5)'                      #   1.5         |                   |
    elif which_buffer == 2 : Nx, Ny = 21, 21; buffer_str = '~BBR +  5 pixels-buffer  (21 x 21)'         #    4          |       ~5-8        |       ~0.5-2
    elif which_buffer == 3 : Nx, Ny = 41, 41; buffer_str = '~BBR + 35 pixels-buffer  (41 x 41)'         #    5          |       ~17         |       ~1.6
    else:buffer_str = ''
    

                                                                                            

        
    SceneNamelibRad='libRad'
    libRad = ReadEC.Scene(Name=SceneNamelibRad, verbose=verbose)    
    if 'montecarlo' in rte_solver:
        shape = (1, Nx, Ny)
    else: 
        shape = Nx*Ny
    libRad.latitude_active = np.zeros(shape)

    # print(shape)
    libRad.solar_e = np.zeros(shape)       
    libRad.solar_e_std = np.zeros(shape)
    libRad.thermal_e = np.zeros(shape)
    libRad.thermal_e_std = np.zeros(shape)


    if WANT_3D:
        ialongs = [Nx//2]
        iacr = Nx // 2
    else:
        ialongs = np.arange(Nx*Ny)
        iacr = 0

    



    # Parallelize over OrbitIDs instead of along-track points.
    # Each MPI rank will process all ialongs for its assigned orbit in sequence.
    
    for ia in ialongs:
        # --- BG: ADD THIS BLOCK ---
        e_solar = np.nan
        e_solar_std = np.nan
        e_thermal = np.nan
        e_thermal_std = np.nan
        # ----------------------
        
        

        for source in sources:
            try: 
                RTM=True
                if source=='solar':
                    mol_abs_param = 'kato2' #'reptran course' #
                elif source=='thermal':
                    mol_abs_param = 'fu' 

                if RTM:
                    UVS = UVspec.UVspec()   #tms: UVspec.py is an external program that defines input and runs libRadtran
                    UVS.inp['rte_solver']=rte_solver
                    #tms: "SetRTM" is a function defined above - NB! Check path
                    UVS = SetRTM(UVS, ia, iacr, sza, phi0,mc_basename_path=RTInpOutPath, RTdimension = RTdimension, input_dir=RTInpOutPath,
                                mol_abs_param=mol_abs_param, data_dir=RTMdata, source=source)
                    if UVS.status == 'OK':
                        # uvspecInputFile=RTInpOutPath+'tmp'+'{:00004d}'.format(ia)+source+str(my_rank)+'.inp'  
                        # NEW TEST
                        uvspecInputFile=RTInpOutPath+f'tmp_SINGLE_CLOUD_{ia:04d}.inp'    
                        uvspecOutputFile=uvspecInputFile.replace('inp','out')
                        UVS.WriteInputFile(uvspecInputFile)

                        status = UVS.SingleRun(uvspecInputFile, uvspecOutputFile, verbose=False, uvspecpath=uvspecpath)

                        if UVS.inp['rte_solver']=='montecarlo': ######################## MYSTIC
                            try:
                                mc_flx_dir, mc_flx_dn, mc_flx_up, edir_std, edn_std, eup_std = UVspec.ReadMCOut(UVS.mc_basename, mol_abs_param=mol_abs_param) #fn = fn + '.flx.spc'  
                                if 'AllLevels' in additional_spesifications:
                                    nlev = int(len(mc_flx_dir)/(Nx*Ny))
                                
                                    mc_flx_dir  = mc_flx_dir.reshape(nlev, Nx, Ny)
                                    mc_flx_dn   = mc_flx_dn.reshape(nlev, Nx, Ny)
                                    mc_flx_up   = mc_flx_up.reshape(nlev, Nx, Ny)

                                    eup_std     = eup_std.reshape(nlev, Nx, Ny)
                                    edn_std     = edn_std.reshape(nlev, Nx, Ny)
                                    edir_std    = edir_std.reshape(nlev, Nx, Ny)


            
                                else:
                                    mc_flx_dir  = mc_flx_dir.reshape(Nx, Ny)
                                    mc_flx_dn   = mc_flx_dn.reshape(Nx, Ny)
                                    mc_flx_up   = mc_flx_up.reshape(Nx, Ny)
                                    edir_std    = edir_std.reshape(Nx, Ny)
                                    eup_std     = eup_std.reshape(Nx, Ny)
                                    edn_std     = edn_std.reshape(Nx, Ny)


                                
                            except (FileNotFoundError, ValueError) as e:
                                print(f"[WORKER {my_rank}] WARNING: Could not read MC output for ia={ia}, source={source}. Skipping. Error: {e}")
                                # continue  # Go to the next iteration
                        
                            if source=='solar':   
                                if WANT_SUR:
                                    e_solar_std = edir_std + edn_std
                                    e_solar     = mc_flx_dir + mc_flx_dn
                                    # print('direct + diffuse solar flux at SUR = ', e_solar, '\n with direct flux = ', mc_flx_dir, ' and diffuse flux = ', mc_flx_dn)
                                else:
                                    e_solar_std = eup_std
                                    e_solar     = mc_flx_up

                            elif source=='thermal':
                                if WANT_SUR:
                                    e_thermal_std = edir_std + edn_std
                                    e_thermal     = mc_flx_dir + mc_flx_dn
                                else:
                                    e_thermal_std = eup_std
                                    e_thermal     = mc_flx_up

                
                        else: ########################################################## DISORT
                            try:
                                sza, zout, edir, edn, eup = UVspec.ReadRTOut(uvspecOutputFile, mol_abs_param=mol_abs_param)
                            except ValueError as e:
                                # print(edir, edn, eup)
                                print("Error [WORKER {}]: could not read file {} with mol_abs_param {}".
                                    format(my_rank, uvspecOutputFile, mol_abs_param))
                                print(f"Error [WORKER {my_rank}]: ", e)

                                
                        

                            e_solar_std = e_thermal_std = None  # No std output from DISORT
                            if source=='solar':
                                if WANT_SUR:
                                    e_solar = edir + edn
                                else:
                                    e_solar = eup
                            elif source=='thermal':
                                if WANT_SUR:
                                    e_thermal = edir + edn
                                else:
                                    e_thermal = eup


                    else:
                        print(UVS.status)
                        print(f'WARNING: UVS.status not OK for ia={ia}, source={source}. Skipping.')
                        zout=edir=eup=eup_std=0.0
                else:
                    zout=edir_solar=eup_solar=eup_solar_std=0.0
                    edir_thermal=eup_thermal=eup_thermal_std=0.0
                    

                

                if rte_solver=='montecarlo': 
                    if not hasattr(e_solar, "shape"):
                        # raise ValueError("Simulated solar flux has no shape (simulation was not successful)")    
                        e_solar_center = np.nan
                        e_solar_std_center = np.nan
                        e_thermal_center = np.nan
                        e_thermal_std_center = np.nan                        
                
                    elif 'AllLevels' in additional_spesifications:
                        ix_center = e_solar.shape[1] // 2
                        iy_center = e_solar.shape[2] // 2
                        
                        if hasattr(e_solar_std, "shape"):   e_solar_std_center      = e_solar_std[:, ix_center, iy_center]
                        if hasattr(e_solar, "shape"):       e_solar_center          = e_solar[:, ix_center, iy_center]
                        if hasattr(e_thermal_std, "shape"): e_thermal_std_center    = e_thermal_std[:, ix_center, iy_center]
                        if hasattr(e_thermal, "shape"):     e_thermal_center        = e_thermal[:, ix_center, iy_center]
                    else: 
                        ix_center = e_solar.shape[0] // 2
                        iy_center = e_solar.shape[1] // 2

                        if hasattr(e_solar_std, "shape"):   e_solar_std_center      = float(e_solar_std[ix_center, iy_center])
                        if hasattr(e_solar, "shape"):       e_solar_center          = float(e_solar[ix_center, iy_center])
                        if hasattr(e_thermal_std, "shape"): e_thermal_std_center    = float(e_thermal_std[ix_center, iy_center])
                        if hasattr(e_thermal, "shape"):     e_thermal_center        = float(e_thermal[ix_center, iy_center])

                    # print("ix_center, iy_center", ix_center, iy_center)                              
                else:
                    e_solar_center     = e_solar
                    e_thermal_center   = e_thermal

                ####### PRINTING RESULTS ##########
                if 'AllLevels' in additional_spesifications:
                    print(e_solar_center)
                    print(e_solar_std)
                else:
                    e_str = f"{e_solar_center:7.1f}" if source=='solar' else f"{e_thermal_center:7.1f}"
                    std_str = f"+- {e_solar_std_center:.1f}" if 'montecarlo' in rte_solver else ''
                    print("\n\n\n"
                        f"      | ia={ia:4}  ----------------------------------------------------------|\n"
                        f"      |                                   Flux ({source}) = {e_str} {std_str} W/m2\n"
                            "      |--------------------------------------------------------------------------------|"
                    )
                   
            

            except Exception as e:
                print(f"\nException for ia={ia:4}: {e}\n")
    

    

    if rte_solver == "montecarlo":
        idx = (ia, slice(None), slice(None))
    else:
        idx = ia

    print(e_solar, e_solar_std, e_thermal, e_thermal_std)
    libRad.solar_e_std[idx]   = e_solar_std
    libRad.thermal_e_std[idx] = e_thermal_std
    libRad.solar_e[idx]       = e_solar
    libRad.thermal_e[idx]     = e_thermal

    # print(e_solar, libRad.solar_e[idx])

        
                        
                

    # Writing NetCDF file...")
    src = ','.join([i for i in sources])
    if 'montecarlo' in rte_solver:
        libRad.WriteNetcdf_mc_sample_grid(RTOutNetcdfPath+'libRad_'+libRad_version+'_'+src+'_'+ additional_spesifications + '.nc', shape=(Nx,Ny), verbose=False)
    else:
        libRad.WriteNetcdf(               RTOutNetcdfPath+'libRad_'+libRad_version+'_'+src+'_'+ additional_spesifications + '.nc', verbose=False)
    print("Wrote to " + RTOutNetcdfPath+'libRad_'+libRad_version+'_'+src+'_'+ additional_spesifications + '.nc')


    tt = datetime.now(timezone.utc) - start_time
    print(f"            Finished run. It took {tt.total_seconds()/3600:.2f} hours \n\n\n\n\n\n\n")

import numpy as np
import pickle
import os
import sys
import multiprocessing
import glob
import scipy
import scipy.constants
from scipy.interpolate import griddata
import dted
import pdb # For debugging
from mpi4py import MPI # Use MPI for multiprocessing
from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess



#import ReadEarthCAREL2 as ReadEC
import Read_RTM as ReadEC
import UVspec as UVspec
import Find_Overpass_Info as Find_Overpass_Info




# BG: overflow encouner - Warnings from reading ACM_COM files
# BG: Hiding the 'overflow encountered in multiply' warnings:
np.seterr(over='ignore')




# Where are we
myhost = os.uname()[1]
home = os.environ['HOME']

# initialize MPI
comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
np_mpi = comm.Get_size()


def pdeg2km(p):
    latfact = 111.32
    p0 = latfact*p[0]/1000
    lonfact = 40075 * np.cos(np.deg2rad( p[1] )) / 360
    p1 = latfact*p[1]/1000
    return (p0,p1)


def Check3DBufferSize(ia, iacr, which_buffer):

    Nx, Ny, iacrosses, ialongs = Calc3DBufferSize(ia, iacr, which_buffer)
    if Nx>0 and Ny>0:
        BufferOK=True
    else:
        BufferOK=False
        
    return BufferOK


def Calc3DBufferSize(ia, iacr, which_buffer):
    WANT_PRINTS = False
    # BG: Decide to use my buffer, not dynamic buffer!
    if which_buffer == 0:   along_dim, across_dim =  0,  0 # ( 1, 1) comp-domain  MCIPA 
    elif which_buffer == 1: along_dim, across_dim =  2,  2 # ( 5, 5) comp-domain  Test Buffer
    elif which_buffer == 2: along_dim, across_dim = 10, 10 # (21,21) comp-domain  ~BBR +  5 pixels-buffer
    elif which_buffer == 3: along_dim, across_dim = 20, 20 # (41,41) comp-domain  ~BBR + 35 pixels-buffer
    
    ialongs = np.arange(ia-along_dim, ia+along_dim+1)
    iacrosses = np.arange(iacr-across_dim, iacr+across_dim+1)
    
    Nx = iacrosses.shape[0]
    Ny = ialongs.shape[0]


    return Nx, Ny, iacrosses, ialongs

def SetRTM(UVS, ia, iacr, sza, phi0, input_dir='', mc_basename_path='', RTdimension = '1D',
           mol_abs_param='kato2', data_dir='', source='solar'):

    
    UVS.status = 'OK'
    # First some general input
    UVS.inp['data_files_path']=data_dir
    
    libpath='/xnilu_wrk2/projects/NEVAR/libRadtran/'
    UVS.inp['atmosphere_file'] = f"{libpath}/data/atmmod/afglss.dat"
    UVS.inp['albedo'] = 0.05

    UVS.inp['mol_abs_param']=mol_abs_param
    UVS.inp['source']=source
    UVS.inp['output_process']='sum'

    UVS.inp['output_user']='sza zout edir edn eup'
    UVS.inp['quiet']=''
    

    # Solar geometry
    # Set solar geometry here to make sure azimuth shift due to 3D geometry is included
    UVS.inp['sza']=sza
    UVS.inp['phi0']=phi0  # Azimuth is only needed for 3D, but it does not hurt for 1D.

    Nx, Ny, iacrosses, ialongs = Calc3DBufferSize(ia, iacr, which_buffer)  #calculates along and cross track buffer size 
    if RTdimension == '3D':       
        if UVS.inp['rte_solver'] != 'montecarlo':
            print('3D geometry only available with MYSTIC. You specified:', UVS.inp['rte_solver'])
            exit()
                
        UVS.inp['mc_std']=''
        # UVS.mc_basename=mc_basename_path+'mc_{:00004d}'.format(ia)+'_'+source # Old->made race-conditions errors 
        UVS.mc_basename=mc_basename_path+'mc_'+f"{ia:04d}_"+source
        print("MC basename set to:", UVS.mc_basename)
        UVS.inp['mc_basename']=UVS.mc_basename
        UVS.inp['mc_randomseed'] = (ia+1)*72341//11
        
        # Set SAMPLE-GRID
        UVS.inp['mc_sample_grid']=f'{Nx} {Ny}' # 3D computation domain
        UVS.inp['mc_reference_to_nn']='' # The sampled pixels correspond to the surface pixels.
     
        
    

    ############### SPESIFICATIONS TO SURFACE-VERSION ##################
    if 'TOA' in additional_spesifications: UVS.inp['zout']='TOA'
    elif 'SUR' in additional_spesifications: 
        # Set "zout sur" for DISORT
        # Do not spesify zout for MYSTIC -> give "zout sur" (which do not work for MYSTIC)
        if UVS.inp['rte_solver'] != 'montecarlo': UVS.inp['zout'] = 'sur'  
    ####################################################################

    

    if source=='thermal':
        UVS.inp['wavelength']='4500 100000'  
        if RTdimension == '3D':
            mc_photons = int(1e2)
            UVS.inp['mc_photons']=mc_photons 
            UVS.inp['mc_minphotons']=mc_photons # Note: MC_MINPHOTONS = 1e3 in uvspec.h so need to modify!
    else:
        UVS.inp['wavelength']='295 2800'     # Changed from '250 4000' 20.01.2026   
        if RTdimension == '3D':
            mc_photons = int(1e7)  
            UVS.inp['mc_photons']=mc_photons 
      
    
    

    # Water cloud
    UVS.inp['wc_properties']='mie'
    Nz = 100 
    wc_value, wcreff_value = 1e-2, 10.0
    h_min, h_max = 2, 3
    heights = np.linspace(0, 10, Nz) # km
    # Make a 2x2 cloud in the center of the domain
    cloud_start_X = Nx//2 #+ Nx//8
    cloud_start_Y = Ny//2 #+ Ny//8
    idx_inside_cloud = []

    if RTdimension == '3D':  
        
        # iz=0
        # last_h=99999
        # for h,p in zip(ACMCOM.height_level[1:,irec_surface_pixel],ACMCOM.pressure_level[1:,irec_surface_pixel]):
        #     if h > -1e-4 and p < 1e+10:
        #         last_h=h
        #         last_p=p
        #         iz=iz+1

        # if np.abs(last_h) > 1e-4 and last_p < 1e+10:
        #     iz=iz+1

        # Nz = iz-1 #ACMCOM.height_level[1:,ia].shape[0]
        # dx = 1.0 # 1 km horizontal resolution
        # dy = 1.0

        flag = 3 # mystic 3D format# mystic 3D format# mystic 3D format# mystic 3D format
        dx, dy = 1.0, 1.0 # 1km horizontal resolution
        heights = heights[1:]
        Nz -= 1
        wc_file_3D=input_dir+f'tmp_{ia:04d}_wc3D.dat'
        f = open(wc_file_3D,'w')
        f.write('{:d} {:d} {:d} {:d}\n'.format(Nx, Ny, Nz, flag))
        f.write('{:f} {:f} '.format(dx, dy))

        
       
        for h in heights:
            f.write('{:8.3f} '.format(np.abs(h)))
            
        f.write('\n')
        wclines=0

        ix=1
        for iac in iacrosses:
            iy=1 
            for ial in ialongs:
                iz = 1
                inside_cloud_region = (
                        cloud_start_X <= ix < cloud_start_X + 2
                    and cloud_start_Y <= iy < cloud_start_Y + 2
                )

                # Define indecies for cloud-vertical height
                if ix == cloud_start_X and iy == cloud_start_Y:
                    h_valid = np.asarray(heights)[heights > 0]

                    iz_h_min = np.argmin(np.abs(h_valid - h_min)) + 1
                    iz_h_max = np.argmin(np.abs(h_valid - h_max)) + 1

                    iz_min = min(iz_h_min, iz_h_max)
                    iz_max = max(iz_h_min, iz_h_max)

                    print("Cloud vertical indices:", iz_min, iz_max)
                    print("Actual heights:", h_valid[iz_min-1], h_valid[iz_max-1])

                

                # BG: made changes
                # - iz += 1 for each realistic h > 0 measurement by EarthCARE
                # - previosly iz += 1 only for realistic h > 0 and wc > 0 (this was the error)
                for h in heights:
                    if inside_cloud_region and iz_min <= iz < iz_max: # h_min < h < h_max:
                        # set wc
                        f.write('{:d} {:d} {:d} {:f} {:f}\n'.format(ix, iy, iz, wc_value, wcreff_value))
                        print(f'Writing wc={wc_value} at ix={ix}, iy={iy}, iz={iz}')
            
                    
                        wclines=wclines+1
                    iz=iz+1 
                iy=iy+1
            ix=ix+1

        # Add dummy line to make mystic run
        if wclines==0:
            f.write('{:d} {:d} {:d} {:f} {:f}\n'.format(1, 1, 1, 0, 10))
                
        f.close()
        UVS.inp['wc_file 3D']=wc_file_3D
        UVS.inp['mc_std']=''
    else:       
        wc_file_1D=input_dir+f'tmp_{ia:04d}_wc1D.dat'
        UVS.inp['wc_file 1D']=wc_file_1D
        f = open(wc_file_1D,'w')
    
        ix = 1
        for iac in iacrosses:
            iy = 1
            for ial in ialongs:
                inside_cloud_region = (
                    cloud_start_X <= ix < cloud_start_X + 2
                    and cloud_start_Y <= iy < cloud_start_Y + 2
                )
                if inside_cloud_region:
                    idx_inside_cloud.append(Nx* (iy-1) + (ix-1))
                iy += 1
            ix += 1
        print(idx_inside_cloud)
    
                
        for h in heights[::-1]:
            if h_min <= h <= h_max and ia in idx_inside_cloud:
                wc, wcreff = wc_value, wcreff_value
            else:
                wc, wcreff = 0.0, 0.0

            f.write('{:8.3f} {:10.5f} {:8.3f}\n'.format(h, wc, wcreff))
        f.close()
    
    return UVS


def tmp_output_filename(iia, scene_name):
    """
    Return the filename to be used for temporary output files.

    """
    pass
    












if __name__ == "__main__":
    start_time = datetime.now(timezone.utc) 
    # print("Run started at", start_time.isoformat())



    ##############################################################################################
    WANT_SUR = True         # zout at SUR or TOA
    WANT_3D  = True        # MYSTIC or DISORT

    verbose  = False        # Want verbose output from RTM
    want_ps  = False        # Psudospherical solver
        
                            
    which_buffer = 1
    # 0: MCIPA   1:Test   2: 21x21   3: 41x41  
    ################################################################################################

   
  

    # BG: To distinguish minor modifications to .nc files
    additional_spesifications = 'SINGLE_CLOUD' 
    sza = 60
    phi0 = 60


    if WANT_SUR: additional_spesifications += "_SUR"
    else:        additional_spesifications += "_TOA"



    idx_source = 0
    sources = [['solar'], 
              ['thermal'],
              ['solar', 'thermal']][idx_source] 


       
    if WANT_3D:
        rte_solver = 'montecarlo' #'mystic' 
        RTdimension = '3D'
    else:
        if want_ps:
            rte_solver = 'disort \npseudospherical'
        else: 
            rte_solver = 'disort'  # 'twostr'        
        RTdimension ='1D' 
    print(f"{RTdimension} RTM - {rte_solver}")


    # ---------------------------------- Paths -----------------------------------------------
    RTInpOutPath = './tmpRTIO/'    # Folder to store the final results
    RTOutNetcdfPath = './RESULTS_TEST/'  # Folder name for netcdf result files
    uvspecpath = '/xnilu_wrk2/projects/NEVAR/libRadtran/bin/' # Folder to RTM code
    RTMdata = '/xnilu_wrk2/projects/NEVAR/libRadtran/data/' #RTM input data

    libRad_version= 'v01_'+rte_solver+'_'+RTdimension
    # --------------------------------------------------------------------------------------------



    Nx, Ny = 1, 1 # Default Horizontal Computaion-Domain Size                                                                                     # min / pixel   |     solar-std     |     thermal-std  
    if   which_buffer == 0 :                  buffer_str = 'MCIPA'
    elif which_buffer == 1 : Nx, Ny =  5,  5; buffer_str = 'TEST   Buffer (5 x 5)'                      #   1.5         |                   |
    elif which_buffer == 2 : Nx, Ny = 21, 21; buffer_str = '~BBR +  5 pixels-buffer  (21 x 21)'         #    4          |       ~5-8        |       ~0.5-2
    elif which_buffer == 3 : Nx, Ny = 41, 41; buffer_str = '~BBR + 35 pixels-buffer  (41 x 41)'         #    5          |       ~17         |       ~1.6
    else:buffer_str = ''
    

                                                                                            

        
    SceneNamelibRad='libRad'
    libRad = ReadEC.Scene(Name=SceneNamelibRad, verbose=verbose)    
    if 'montecarlo' in rte_solver:
        shape = (1, Nx, Ny)
    else: 
        shape = Nx*Ny
    libRad.latitude_active = np.zeros(shape)

    # print(shape)
    libRad.solar_e = np.zeros(shape)       
    libRad.solar_e_std = np.zeros(shape)
    libRad.thermal_e = np.zeros(shape)
    libRad.thermal_e_std = np.zeros(shape)


    if WANT_3D:
        ialongs = [Nx//2]
        iacr = Nx // 2
    else:
        ialongs = np.arange(Nx*Ny)
        iacr = 0

    



    # Parallelize over OrbitIDs instead of along-track points.
    # Each MPI rank will process all ialongs for its assigned orbit in sequence.
    
    for ia in ialongs:
        # --- BG: ADD THIS BLOCK ---
        e_solar = np.nan
        e_solar_std = np.nan
        e_thermal = np.nan
        e_thermal_std = np.nan
        # ----------------------
        
        

        for source in sources:
            try: 
                RTM=True
                if source=='solar':
                    mol_abs_param = 'kato2' #'reptran course' #
                elif source=='thermal':
                    mol_abs_param = 'fu' 

                if RTM:
                    UVS = UVspec.UVspec()   #tms: UVspec.py is an external program that defines input and runs libRadtran
                    UVS.inp['rte_solver']=rte_solver
                    #tms: "SetRTM" is a function defined above - NB! Check path
                    UVS = SetRTM(UVS, ia, iacr, sza, phi0,mc_basename_path=RTInpOutPath, RTdimension = RTdimension, input_dir=RTInpOutPath,
                                mol_abs_param=mol_abs_param, data_dir=RTMdata, source=source)
                    if UVS.status == 'OK':
                        # uvspecInputFile=RTInpOutPath+'tmp'+'{:00004d}'.format(ia)+source+str(my_rank)+'.inp'  
                        # NEW TEST
                        uvspecInputFile=RTInpOutPath+f'tmp_SINGLE_CLOUD_{ia:04d}.inp'    
                        uvspecOutputFile=uvspecInputFile.replace('inp','out')
                        UVS.WriteInputFile(uvspecInputFile)

                        status = UVS.SingleRun(uvspecInputFile, uvspecOutputFile, verbose=False, uvspecpath=uvspecpath)

                        if UVS.inp['rte_solver']=='montecarlo': ######################## MYSTIC
                            try:
                                mc_flx_dir, mc_flx_dn, mc_flx_up, edir_std, edn_std, eup_std = UVspec.ReadMCOut(UVS.mc_basename, mol_abs_param=mol_abs_param) #fn = fn + '.flx.spc'  
                                if 'AllLevels' in additional_spesifications:
                                    nlev = int(len(mc_flx_dir)/(Nx*Ny))
                                
                                    mc_flx_dir  = mc_flx_dir.reshape(nlev, Nx, Ny)
                                    mc_flx_dn   = mc_flx_dn.reshape(nlev, Nx, Ny)
                                    mc_flx_up   = mc_flx_up.reshape(nlev, Nx, Ny)

                                    eup_std     = eup_std.reshape(nlev, Nx, Ny)
                                    edn_std     = edn_std.reshape(nlev, Nx, Ny)
                                    edir_std    = edir_std.reshape(nlev, Nx, Ny)


            
                                else:
                                    mc_flx_dir  = mc_flx_dir.reshape(Nx, Ny)
                                    mc_flx_dn   = mc_flx_dn.reshape(Nx, Ny)
                                    mc_flx_up   = mc_flx_up.reshape(Nx, Ny)
                                    edir_std    = edir_std.reshape(Nx, Ny)
                                    eup_std     = eup_std.reshape(Nx, Ny)
                                    edn_std     = edn_std.reshape(Nx, Ny)


                                
                            except (FileNotFoundError, ValueError) as e:
                                print(f"[WORKER {my_rank}] WARNING: Could not read MC output for ia={ia}, source={source}. Skipping. Error: {e}")
                                # continue  # Go to the next iteration
                        
                            if source=='solar':   
                                if WANT_SUR:
                                    e_solar_std = edir_std + edn_std
                                    e_solar     = mc_flx_dir + mc_flx_dn
                                    # print('direct + diffuse solar flux at SUR = ', e_solar, '\n with direct flux = ', mc_flx_dir, ' and diffuse flux = ', mc_flx_dn)
                                else:
                                    e_solar_std = eup_std
                                    e_solar     = mc_flx_up

                            elif source=='thermal':
                                if WANT_SUR:
                                    e_thermal_std = edir_std + edn_std
                                    e_thermal     = mc_flx_dir + mc_flx_dn
                                else:
                                    e_thermal_std = eup_std
                                    e_thermal     = mc_flx_up

                
                        else: ########################################################## DISORT
                            try:
                                sza, zout, edir, edn, eup = UVspec.ReadRTOut(uvspecOutputFile, mol_abs_param=mol_abs_param)
                            except ValueError as e:
                                # print(edir, edn, eup)
                                print("Error [WORKER {}]: could not read file {} with mol_abs_param {}".
                                    format(my_rank, uvspecOutputFile, mol_abs_param))
                                print(f"Error [WORKER {my_rank}]: ", e)

                                
                        

                            e_solar_std = e_thermal_std = None  # No std output from DISORT
                            if source=='solar':
                                if WANT_SUR:
                                    e_solar = edir + edn
                                else:
                                    e_solar = eup
                            elif source=='thermal':
                                if WANT_SUR:
                                    e_thermal = edir + edn
                                else:
                                    e_thermal = eup


                    else:
                        print(UVS.status)
                        print(f'WARNING: UVS.status not OK for ia={ia}, source={source}. Skipping.')
                        zout=edir=eup=eup_std=0.0
                else:
                    zout=edir_solar=eup_solar=eup_solar_std=0.0
                    edir_thermal=eup_thermal=eup_thermal_std=0.0
                    

                

                if rte_solver=='montecarlo': 
                    if not hasattr(e_solar, "shape"):
                        # raise ValueError("Simulated solar flux has no shape (simulation was not successful)")    
                        e_solar_center = np.nan
                        e_solar_std_center = np.nan
                        e_thermal_center = np.nan
                        e_thermal_std_center = np.nan                        
                
                    elif 'AllLevels' in additional_spesifications:
                        ix_center = e_solar.shape[1] // 2
                        iy_center = e_solar.shape[2] // 2
                        
                        if hasattr(e_solar_std, "shape"):   e_solar_std_center      = e_solar_std[:, ix_center, iy_center]
                        if hasattr(e_solar, "shape"):       e_solar_center          = e_solar[:, ix_center, iy_center]
                        if hasattr(e_thermal_std, "shape"): e_thermal_std_center    = e_thermal_std[:, ix_center, iy_center]
                        if hasattr(e_thermal, "shape"):     e_thermal_center        = e_thermal[:, ix_center, iy_center]
                    else: 
                        ix_center = e_solar.shape[0] // 2
                        iy_center = e_solar.shape[1] // 2

                        if hasattr(e_solar_std, "shape"):   e_solar_std_center      = float(e_solar_std[ix_center, iy_center])
                        if hasattr(e_solar, "shape"):       e_solar_center          = float(e_solar[ix_center, iy_center])
                        if hasattr(e_thermal_std, "shape"): e_thermal_std_center    = float(e_thermal_std[ix_center, iy_center])
                        if hasattr(e_thermal, "shape"):     e_thermal_center        = float(e_thermal[ix_center, iy_center])

                    # print("ix_center, iy_center", ix_center, iy_center)                              
                else:
                    e_solar_center     = e_solar
                    e_thermal_center   = e_thermal

                ####### PRINTING RESULTS ##########
                if 'AllLevels' in additional_spesifications:
                    print(e_solar_center)
                    print(e_solar_std)
                else:
                    e_str = f"{e_solar_center:7.1f}" if source=='solar' else f"{e_thermal_center:7.1f}"
                    std_str = f"+- {e_solar_std_center:.1f}" if 'montecarlo' in rte_solver else ''
                    print("\n\n\n"
                        f"      | ia={ia:4}  ----------------------------------------------------------|\n"
                        f"      |                                   Flux ({source}) = {e_str} {std_str} W/m2\n"
                            "      |--------------------------------------------------------------------------------|"
                    )
                   
            

            except Exception as e:
                print(f"\nException for ia={ia:4}: {e}\n")
    

    

    if rte_solver == "montecarlo":
        idx = (ia, slice(None), slice(None))
    else:
        idx = ia

    print(e_solar, e_solar_std, e_thermal, e_thermal_std)
    libRad.solar_e_std[idx]   = e_solar_std
    libRad.thermal_e_std[idx] = e_thermal_std
    libRad.solar_e[idx]       = e_solar
    libRad.thermal_e[idx]     = e_thermal

    # print(e_solar, libRad.solar_e[idx])

        
                        
                

    # Writing NetCDF file...")
    src = ','.join([i for i in sources])
    if 'montecarlo' in rte_solver:
        libRad.WriteNetcdf_mc_sample_grid(RTOutNetcdfPath+'libRad_'+libRad_version+'_'+src+'_'+ additional_spesifications + '.nc', shape=(Nx,Ny), verbose=False)
    else:
        libRad.WriteNetcdf(               RTOutNetcdfPath+'libRad_'+libRad_version+'_'+src+'_'+ additional_spesifications + '.nc', verbose=False)
    print("Wrote to " + RTOutNetcdfPath+'libRad_'+libRad_version+'_'+src+'_'+ additional_spesifications + '.nc')


    tt = datetime.now(timezone.utc) - start_time
    print(f"            Finished run. It took {tt.total_seconds()/3600:.2f} hours \n\n\n\n\n\n\n")

import numpy as np
import pickle
import os
import sys
import multiprocessing
import glob
import scipy
import scipy.constants
from scipy.interpolate import griddata
import dted
import pdb # For debugging
from mpi4py import MPI # Use MPI for multiprocessing
from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess



#import ReadEarthCAREL2 as ReadEC
import Read_RTM as ReadEC
import UVspec as UVspec
import Find_Overpass_Info as Find_Overpass_Info




# BG: overflow encouner - Warnings from reading ACM_COM files
# BG: Hiding the 'overflow encountered in multiply' warnings:
np.seterr(over='ignore')




# Where are we
myhost = os.uname()[1]
home = os.environ['HOME']

# initialize MPI
comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
np_mpi = comm.Get_size()


def pdeg2km(p):
    latfact = 111.32
    p0 = latfact*p[0]/1000
    lonfact = 40075 * np.cos(np.deg2rad( p[1] )) / 360
    p1 = latfact*p[1]/1000
    return (p0,p1)


def Check3DBufferSize(ia, iacr, which_buffer):

    Nx, Ny, iacrosses, ialongs = Calc3DBufferSize(ia, iacr, which_buffer)
    if Nx>0 and Ny>0:
        BufferOK=True
    else:
        BufferOK=False
        
    return BufferOK


def Calc3DBufferSize(ia, iacr, which_buffer):
    WANT_PRINTS = False
    # BG: Decide to use my buffer, not dynamic buffer!
    if which_buffer == 0:   along_dim, across_dim =  0,  0 # ( 1, 1) comp-domain  MCIPA 
    elif which_buffer == 1: along_dim, across_dim =  2,  2 # ( 5, 5) comp-domain  Test Buffer
    elif which_buffer == 2: along_dim, across_dim = 10, 10 # (21,21) comp-domain  ~BBR +  5 pixels-buffer
    elif which_buffer == 3: along_dim, across_dim = 20, 20 # (41,41) comp-domain  ~BBR + 35 pixels-buffer
    
    ialongs = np.arange(ia-along_dim, ia+along_dim+1)
    iacrosses = np.arange(iacr-across_dim, iacr+across_dim+1)
    
    Nx = iacrosses.shape[0]
    Ny = ialongs.shape[0]


    return Nx, Ny, iacrosses, ialongs

def SetRTM(UVS, ia, iacr, sza, phi0, input_dir='', mc_basename_path='', RTdimension = '1D',
           mol_abs_param='kato2', data_dir='', source='solar'):

    
    UVS.status = 'OK'
    # First some general input
    UVS.inp['data_files_path']=data_dir
    
    libpath='/xnilu_wrk2/projects/NEVAR/libRadtran/'
    UVS.inp['atmosphere_file'] = f"{libpath}/data/atmmod/afglss.dat"
    UVS.inp['albedo'] = 0.05

    UVS.inp['mol_abs_param']=mol_abs_param
    UVS.inp['source']=source
    UVS.inp['output_process']='sum'

    UVS.inp['output_user']='sza zout edir edn eup'
    UVS.inp['quiet']=''
    

    # Solar geometry
    # Set solar geometry here to make sure azimuth shift due to 3D geometry is included
    UVS.inp['sza']=sza
    UVS.inp['phi0']=phi0  # Azimuth is only needed for 3D, but it does not hurt for 1D.

    Nx, Ny, iacrosses, ialongs = Calc3DBufferSize(ia, iacr, which_buffer)  #calculates along and cross track buffer size 
    if RTdimension == '3D':       
        if UVS.inp['rte_solver'] != 'montecarlo':
            print('3D geometry only available with MYSTIC. You specified:', UVS.inp['rte_solver'])
            exit()
                
        UVS.inp['mc_std']=''
        # UVS.mc_basename=mc_basename_path+'mc_{:00004d}'.format(ia)+'_'+source # Old->made race-conditions errors 
        UVS.mc_basename=mc_basename_path+'mc_'+f"{ia:04d}_"+source
        print("MC basename set to:", UVS.mc_basename)
        UVS.inp['mc_basename']=UVS.mc_basename
        UVS.inp['mc_randomseed'] = (ia+1)*72341//11
        
        # Set SAMPLE-GRID
        UVS.inp['mc_sample_grid']=f'{Nx} {Ny}' # 3D computation domain
        UVS.inp['mc_reference_to_nn']='' # The sampled pixels correspond to the surface pixels.
     
        
    

    ############### SPESIFICATIONS TO SURFACE-VERSION ##################
    if 'TOA' in additional_spesifications: UVS.inp['zout']='TOA'
    elif 'SUR' in additional_spesifications: 
        # Set "zout sur" for DISORT
        # Do not spesify zout for MYSTIC -> give "zout sur" (which do not work for MYSTIC)
        if UVS.inp['rte_solver'] != 'montecarlo': UVS.inp['zout'] = 'sur'  
    ####################################################################

    

    if source=='thermal':
        UVS.inp['wavelength']='4500 100000'  
        if RTdimension == '3D':
            mc_photons = int(1e2)
            UVS.inp['mc_photons']=mc_photons 
            UVS.inp['mc_minphotons']=mc_photons # Note: MC_MINPHOTONS = 1e3 in uvspec.h so need to modify!
    else:
        UVS.inp['wavelength']='295 2800'     # Changed from '250 4000' 20.01.2026   
        if RTdimension == '3D':
            mc_photons = int(1e7)  
            UVS.inp['mc_photons']=mc_photons 
      
    
    

    # Water cloud
    UVS.inp['wc_properties']='mie'
    Nz = 100 
    wc_value, wcreff_value = 1e-2, 10.0
    h_min, h_max = 2, 3
    heights = np.linspace(0, 10, Nz) # km
    # Make a 2x2 cloud in the center of the domain
    cloud_start_X = Nx//2 #+ Nx//8
    cloud_start_Y = Ny//2 #+ Ny//8
    idx_inside_cloud = []

    if RTdimension == '3D':  
        
        # iz=0
        # last_h=99999
        # for h,p in zip(ACMCOM.height_level[1:,irec_surface_pixel],ACMCOM.pressure_level[1:,irec_surface_pixel]):
        #     if h > -1e-4 and p < 1e+10:
        #         last_h=h
        #         last_p=p
        #         iz=iz+1

        # if np.abs(last_h) > 1e-4 and last_p < 1e+10:
        #     iz=iz+1

        # Nz = iz-1 #ACMCOM.height_level[1:,ia].shape[0]
        # dx = 1.0 # 1 km horizontal resolution
        # dy = 1.0

        flag = 3 # mystic 3D format# mystic 3D format# mystic 3D format# mystic 3D format
        dx, dy = 1.0, 1.0 # 1km horizontal resolution
        heights = heights[1:]
        Nz -= 1
        wc_file_3D=input_dir+f'tmp_{ia:04d}_wc3D.dat'
        f = open(wc_file_3D,'w')
        f.write('{:d} {:d} {:d} {:d}\n'.format(Nx, Ny, Nz, flag))
        f.write('{:f} {:f} '.format(dx, dy))

        
       
        for h in heights:
            f.write('{:8.3f} '.format(np.abs(h)))
            
        f.write('\n')
        wclines=0

        ix=1
        for iac in iacrosses:
            iy=1 
            for ial in ialongs:
                iz = 1
                inside_cloud_region = (
                        cloud_start_X <= ix < cloud_start_X + 2
                    and cloud_start_Y <= iy < cloud_start_Y + 2
                )

                # Define indecies for cloud-vertical height
                if ix == cloud_start_X and iy == cloud_start_Y:
                    h_valid = np.asarray(heights)[heights > 0]

                    iz_h_min = np.argmin(np.abs(h_valid - h_min)) + 1
                    iz_h_max = np.argmin(np.abs(h_valid - h_max)) + 1

                    iz_min = min(iz_h_min, iz_h_max)
                    iz_max = max(iz_h_min, iz_h_max)

                    print("Cloud vertical indices:", iz_min, iz_max)
                    print("Actual heights:", h_valid[iz_min-1], h_valid[iz_max-1])

                

                # BG: made changes
                # - iz += 1 for each realistic h > 0 measurement by EarthCARE
                # - previosly iz += 1 only for realistic h > 0 and wc > 0 (this was the error)
                for h in heights:
                    if inside_cloud_region and iz_min <= iz < iz_max: # h_min < h < h_max:
                        # set wc
                        f.write('{:d} {:d} {:d} {:f} {:f}\n'.format(ix, iy, iz, wc_value, wcreff_value))
                        print(f'Writing wc={wc_value} at ix={ix}, iy={iy}, iz={iz}')
            
                    
                        wclines=wclines+1
                    iz=iz+1 
                iy=iy+1
            ix=ix+1

        # Add dummy line to make mystic run
        if wclines==0:
            f.write('{:d} {:d} {:d} {:f} {:f}\n'.format(1, 1, 1, 0, 10))
                
        f.close()
        UVS.inp['wc_file 3D']=wc_file_3D
        UVS.inp['mc_std']=''
    else:       
        wc_file_1D=input_dir+f'tmp_{ia:04d}_wc1D.dat'
        UVS.inp['wc_file 1D']=wc_file_1D
        f = open(wc_file_1D,'w')
    
        ix = 1
        for iac in iacrosses:
            iy = 1
            for ial in ialongs:
                inside_cloud_region = (
                    cloud_start_X <= ix < cloud_start_X + 2
                    and cloud_start_Y <= iy < cloud_start_Y + 2
                )
                if inside_cloud_region:
                    idx_inside_cloud.append(Nx* (iy-1) + (ix-1))
                iy += 1
            ix += 1
        print(idx_inside_cloud)
    
                
        for h in heights[::-1]:
            if h_min <= h <= h_max and ia in idx_inside_cloud:
                wc, wcreff = wc_value, wcreff_value
            else:
                wc, wcreff = 0.0, 0.0

            f.write('{:8.3f} {:10.5f} {:8.3f}\n'.format(h, wc, wcreff))
        f.close()
    
    return UVS


def tmp_output_filename(iia, scene_name):
    """
    Return the filename to be used for temporary output files.

    """
    pass
    












if __name__ == "__main__":
    start_time = datetime.now(timezone.utc) 
    # print("Run started at", start_time.isoformat())



    ##############################################################################################
    WANT_SUR = True         # zout at SUR or TOA
    WANT_3D  = False        # MYSTIC or DISORT

    verbose  = False        # Want verbose output from RTM
    want_ps  = False        # Psudospherical solver
        
                            
    which_buffer = 1
    # 0: MCIPA   1:Test   2: 21x21   3: 41x41  
    ################################################################################################

   
  

    # BG: To distinguish minor modifications to .nc files
    additional_spesifications = 'SINGLE_CLOUD' 
    sza = 60
    phi0 = 60


    if WANT_SUR: additional_spesifications += "_SUR"
    else:        additional_spesifications += "_TOA"



    idx_source = 0
    sources = [['solar'], 
              ['thermal'],
              ['solar', 'thermal']][idx_source] 


       
    if WANT_3D:
        rte_solver = 'montecarlo' #'mystic' 
        RTdimension = '3D'
    else:
        if want_ps:
            rte_solver = 'disort \npseudospherical'
        else: 
            rte_solver = 'disort'  # 'twostr'        
        RTdimension ='1D' 
    print(f"{RTdimension} RTM - {rte_solver}")


    # ---------------------------------- Paths -----------------------------------------------
    RTInpOutPath = './tmpRTIO/'    # Folder to store the final results
    RTOutNetcdfPath = './RESULTS_TEST/'  # Folder name for netcdf result files
    uvspecpath = '/xnilu_wrk2/projects/NEVAR/libRadtran/bin/' # Folder to RTM code
    RTMdata = '/xnilu_wrk2/projects/NEVAR/libRadtran/data/' #RTM input data

    libRad_version= 'v01_'+rte_solver+'_'+RTdimension
    # --------------------------------------------------------------------------------------------



    Nx, Ny = 1, 1 # Default Horizontal Computaion-Domain Size                                                                                     # min / pixel   |     solar-std     |     thermal-std  
    if   which_buffer == 0 :                  buffer_str = 'MCIPA'
    elif which_buffer == 1 : Nx, Ny =  5,  5; buffer_str = 'TEST   Buffer (5 x 5)'                      #   1.5         |                   |
    elif which_buffer == 2 : Nx, Ny = 21, 21; buffer_str = '~BBR +  5 pixels-buffer  (21 x 21)'         #    4          |       ~5-8        |       ~0.5-2
    elif which_buffer == 3 : Nx, Ny = 41, 41; buffer_str = '~BBR + 35 pixels-buffer  (41 x 41)'         #    5          |       ~17         |       ~1.6
    else:buffer_str = ''
    

                                                                                            

        
    SceneNamelibRad='libRad'
    libRad = ReadEC.Scene(Name=SceneNamelibRad, verbose=verbose)    
    if 'montecarlo' in rte_solver:
        shape = (1, Nx, Ny)
    else: 
        shape = Nx*Ny
    libRad.latitude_active = np.zeros(shape)

    # print(shape)
    libRad.solar_e = np.zeros(shape)       
    libRad.solar_e_std = np.zeros(shape)
    libRad.thermal_e = np.zeros(shape)
    libRad.thermal_e_std = np.zeros(shape)


    if WANT_3D:
        ialongs = [Nx//2]
        iacr = Nx // 2
    else:
        ialongs = np.arange(Nx*Ny)
        iacr = 0

    



    # Parallelize over OrbitIDs instead of along-track points.
    # Each MPI rank will process all ialongs for its assigned orbit in sequence.
    
    for ia in ialongs:
        # --- BG: ADD THIS BLOCK ---
        e_solar = np.nan
        e_solar_std = np.nan
        e_thermal = np.nan
        e_thermal_std = np.nan
        # ----------------------
        
        

        for source in sources:
            try: 
                RTM=True
                if source=='solar':
                    mol_abs_param = 'kato2' #'reptran course' #
                elif source=='thermal':
                    mol_abs_param = 'fu' 

                if RTM:
                    UVS = UVspec.UVspec()   #tms: UVspec.py is an external program that defines input and runs libRadtran
                    UVS.inp['rte_solver']=rte_solver
                    #tms: "SetRTM" is a function defined above - NB! Check path
                    UVS = SetRTM(UVS, ia, iacr, sza, phi0,mc_basename_path=RTInpOutPath, RTdimension = RTdimension, input_dir=RTInpOutPath,
                                mol_abs_param=mol_abs_param, data_dir=RTMdata, source=source)
                    if UVS.status == 'OK':
                        # uvspecInputFile=RTInpOutPath+'tmp'+'{:00004d}'.format(ia)+source+str(my_rank)+'.inp'  
                        # NEW TEST
                        uvspecInputFile=RTInpOutPath+f'tmp_SINGLE_CLOUD_{ia:04d}.inp'    
                        uvspecOutputFile=uvspecInputFile.replace('inp','out')
                        UVS.WriteInputFile(uvspecInputFile)

                        status = UVS.SingleRun(uvspecInputFile, uvspecOutputFile, verbose=False, uvspecpath=uvspecpath)

                        if UVS.inp['rte_solver']=='montecarlo': ######################## MYSTIC
                            try:
                                mc_flx_dir, mc_flx_dn, mc_flx_up, edir_std, edn_std, eup_std = UVspec.ReadMCOut(UVS.mc_basename, mol_abs_param=mol_abs_param) #fn = fn + '.flx.spc'  
                                if 'AllLevels' in additional_spesifications:
                                    nlev = int(len(mc_flx_dir)/(Nx*Ny))
                                
                                    mc_flx_dir  = mc_flx_dir.reshape(nlev, Nx, Ny)
                                    mc_flx_dn   = mc_flx_dn.reshape(nlev, Nx, Ny)
                                    mc_flx_up   = mc_flx_up.reshape(nlev, Nx, Ny)

                                    eup_std     = eup_std.reshape(nlev, Nx, Ny)
                                    edn_std     = edn_std.reshape(nlev, Nx, Ny)
                                    edir_std    = edir_std.reshape(nlev, Nx, Ny)


            
                                else:
                                    mc_flx_dir  = mc_flx_dir.reshape(Nx, Ny)
                                    mc_flx_dn   = mc_flx_dn.reshape(Nx, Ny)
                                    mc_flx_up   = mc_flx_up.reshape(Nx, Ny)
                                    edir_std    = edir_std.reshape(Nx, Ny)
                                    eup_std     = eup_std.reshape(Nx, Ny)
                                    edn_std     = edn_std.reshape(Nx, Ny)


                                
                            except (FileNotFoundError, ValueError) as e:
                                print(f"[WORKER {my_rank}] WARNING: Could not read MC output for ia={ia}, source={source}. Skipping. Error: {e}")
                                # continue  # Go to the next iteration
                        
                            if source=='solar':   
                                if WANT_SUR:
                                    e_solar_std = edir_std + edn_std
                                    e_solar     = mc_flx_dir + mc_flx_dn
                                    # print('direct + diffuse solar flux at SUR = ', e_solar, '\n with direct flux = ', mc_flx_dir, ' and diffuse flux = ', mc_flx_dn)
                                else:
                                    e_solar_std = eup_std
                                    e_solar     = mc_flx_up

                            elif source=='thermal':
                                if WANT_SUR:
                                    e_thermal_std = edir_std + edn_std
                                    e_thermal     = mc_flx_dir + mc_flx_dn
                                else:
                                    e_thermal_std = eup_std
                                    e_thermal     = mc_flx_up

                
                        else: ########################################################## DISORT
                            try:
                                sza, zout, edir, edn, eup = UVspec.ReadRTOut(uvspecOutputFile, mol_abs_param=mol_abs_param)
                            except ValueError as e:
                                # print(edir, edn, eup)
                                print("Error [WORKER {}]: could not read file {} with mol_abs_param {}".
                                    format(my_rank, uvspecOutputFile, mol_abs_param))
                                print(f"Error [WORKER {my_rank}]: ", e)

                                
                        

                            e_solar_std = e_thermal_std = None  # No std output from DISORT
                            if source=='solar':
                                if WANT_SUR:
                                    e_solar = edir + edn
                                else:
                                    e_solar = eup
                            elif source=='thermal':
                                if WANT_SUR:
                                    e_thermal = edir + edn
                                else:
                                    e_thermal = eup


                    else:
                        print(UVS.status)
                        print(f'WARNING: UVS.status not OK for ia={ia}, source={source}. Skipping.')
                        zout=edir=eup=eup_std=0.0
                else:
                    zout=edir_solar=eup_solar=eup_solar_std=0.0
                    edir_thermal=eup_thermal=eup_thermal_std=0.0
                    

                

                if rte_solver=='montecarlo': 
                    if not hasattr(e_solar, "shape"):
                        # raise ValueError("Simulated solar flux has no shape (simulation was not successful)")    
                        e_solar_center = np.nan
                        e_solar_std_center = np.nan
                        e_thermal_center = np.nan
                        e_thermal_std_center = np.nan                        
                
                    elif 'AllLevels' in additional_spesifications:
                        ix_center = e_solar.shape[1] // 2
                        iy_center = e_solar.shape[2] // 2
                        
                        if hasattr(e_solar_std, "shape"):   e_solar_std_center      = e_solar_std[:, ix_center, iy_center]
                        if hasattr(e_solar, "shape"):       e_solar_center          = e_solar[:, ix_center, iy_center]
                        if hasattr(e_thermal_std, "shape"): e_thermal_std_center    = e_thermal_std[:, ix_center, iy_center]
                        if hasattr(e_thermal, "shape"):     e_thermal_center        = e_thermal[:, ix_center, iy_center]
                    else: 
                        ix_center = e_solar.shape[0] // 2
                        iy_center = e_solar.shape[1] // 2

                        if hasattr(e_solar_std, "shape"):   e_solar_std_center      = float(e_solar_std[ix_center, iy_center])
                        if hasattr(e_solar, "shape"):       e_solar_center          = float(e_solar[ix_center, iy_center])
                        if hasattr(e_thermal_std, "shape"): e_thermal_std_center    = float(e_thermal_std[ix_center, iy_center])
                        if hasattr(e_thermal, "shape"):     e_thermal_center        = float(e_thermal[ix_center, iy_center])

                    # print("ix_center, iy_center", ix_center, iy_center)                              
                else:
                    e_solar_center     = e_solar
                    e_thermal_center   = e_thermal

                ####### PRINTING RESULTS ##########
                if 'AllLevels' in additional_spesifications:
                    print(e_solar_center)
                    print(e_solar_std)
                else:
                    e_str = f"{e_solar_center:7.1f}" if source=='solar' else f"{e_thermal_center:7.1f}"
                    std_str = f"+- {e_solar_std_center:.1f}" if 'montecarlo' in rte_solver else ''
                    print("\n\n\n"
                        f"      | ia={ia:4}  ----------------------------------------------------------|\n"
                        f"      |                                   Flux ({source}) = {e_str} {std_str} W/m2\n"
                            "      |--------------------------------------------------------------------------------|"
                    )
                   
            

            except Exception as e:
                print(f"\nException for ia={ia:4}: {e}\n")
    

    

    if rte_solver == "montecarlo":
        idx = (ia, slice(None), slice(None))
    else:
        idx = ia

    print(e_solar, e_solar_std, e_thermal, e_thermal_std)
    libRad.solar_e_std[idx]   = e_solar_std
    libRad.thermal_e_std[idx] = e_thermal_std
    libRad.solar_e[idx]       = e_solar
    libRad.thermal_e[idx]     = e_thermal

    # print(e_solar, libRad.solar_e[idx])

        
                        
                

    # Writing NetCDF file...")
    src = ','.join([i for i in sources])
    if 'montecarlo' in rte_solver:
        libRad.WriteNetcdf_mc_sample_grid(RTOutNetcdfPath+'libRad_'+libRad_version+'_'+src+'_'+ additional_spesifications + '.nc', shape=(Nx,Ny), verbose=False)
    else:
        libRad.WriteNetcdf(               RTOutNetcdfPath+'libRad_'+libRad_version+'_'+src+'_'+ additional_spesifications + '.nc', verbose=False)
    print("Wrote to " + RTOutNetcdfPath+'libRad_'+libRad_version+'_'+src+'_'+ additional_spesifications + '.nc')


    tt = datetime.now(timezone.utc) - start_time
    print(f"            Finished run. It took {tt.total_seconds()/3600:.2f} hours \n\n\n\n\n\n\n")

import numpy as np
import pickle
import os
import sys
import multiprocessing
import glob
import scipy
import scipy.constants
from scipy.interpolate import griddata
import dted
import pdb # For debugging
from mpi4py import MPI # Use MPI for multiprocessing
from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess



#import ReadEarthCAREL2 as ReadEC
import Read_RTM as ReadEC
import UVspec as UVspec
import Find_Overpass_Info as Find_Overpass_Info




# BG: overflow encouner - Warnings from reading ACM_COM files
# BG: Hiding the 'overflow encountered in multiply' warnings:
np.seterr(over='ignore')




# Where are we
myhost = os.uname()[1]
home = os.environ['HOME']

# initialize MPI
comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
np_mpi = comm.Get_size()


def pdeg2km(p):
    latfact = 111.32
    p0 = latfact*p[0]/1000
    lonfact = 40075 * np.cos(np.deg2rad( p[1] )) / 360
    p1 = latfact*p[1]/1000
    return (p0,p1)


def Check3DBufferSize(ia, iacr, which_buffer):

    Nx, Ny, iacrosses, ialongs = Calc3DBufferSize(ia, iacr, which_buffer)
    if Nx>0 and Ny>0:
        BufferOK=True
    else:
        BufferOK=False
        
    return BufferOK


def Calc3DBufferSize(ia, iacr, which_buffer):
    WANT_PRINTS = False
    # BG: Decide to use my buffer, not dynamic buffer!
    if which_buffer == 0:   along_dim, across_dim =  0,  0 # ( 1, 1) comp-domain  MCIPA 
    elif which_buffer == 1: along_dim, across_dim =  2,  2 # ( 5, 5) comp-domain  Test Buffer
    elif which_buffer == 2: along_dim, across_dim = 10, 10 # (21,21) comp-domain  ~BBR +  5 pixels-buffer
    elif which_buffer == 3: along_dim, across_dim = 20, 20 # (41,41) comp-domain  ~BBR + 35 pixels-buffer
    
    ialongs = np.arange(ia-along_dim, ia+along_dim+1)
    iacrosses = np.arange(iacr-across_dim, iacr+across_dim+1)
    
    Nx = iacrosses.shape[0]
    Ny = ialongs.shape[0]


    return Nx, Ny, iacrosses, ialongs

def SetRTM(UVS, ia, iacr, sza, phi0, input_dir='', mc_basename_path='', RTdimension = '1D',
           mol_abs_param='kato2', data_dir='', source='solar'):

    
    UVS.status = 'OK'
    # First some general input
    UVS.inp['data_files_path']=data_dir
    
    libpath='/xnilu_wrk2/projects/NEVAR/libRadtran/'
    UVS.inp['atmosphere_file'] = f"{libpath}/data/atmmod/afglss.dat"
    UVS.inp['albedo'] = 0.05

    UVS.inp['mol_abs_param']=mol_abs_param
    UVS.inp['source']=source
    UVS.inp['output_process']='sum'

    UVS.inp['output_user']='sza zout edir edn eup'
    UVS.inp['quiet']=''
    

    # Solar geometry
    # Set solar geometry here to make sure azimuth shift due to 3D geometry is included
    UVS.inp['sza']=sza
    UVS.inp['phi0']=phi0  # Azimuth is only needed for 3D, but it does not hurt for 1D.

    Nx, Ny, iacrosses, ialongs = Calc3DBufferSize(ia, iacr, which_buffer)  #calculates along and cross track buffer size 
    if RTdimension == '3D':       
        if UVS.inp['rte_solver'] != 'montecarlo':
            print('3D geometry only available with MYSTIC. You specified:', UVS.inp['rte_solver'])
            exit()
                
        UVS.inp['mc_std']=''
        # UVS.mc_basename=mc_basename_path+'mc_{:00004d}'.format(ia)+'_'+source # Old->made race-conditions errors 
        UVS.mc_basename=mc_basename_path+'mc_'+f"{ia:04d}_"+source
        print("MC basename set to:", UVS.mc_basename)
        UVS.inp['mc_basename']=UVS.mc_basename
        UVS.inp['mc_randomseed'] = (ia+1)*72341//11
        
        # Set SAMPLE-GRID
        UVS.inp['mc_sample_grid']=f'{Nx} {Ny}' # 3D computation domain
        UVS.inp['mc_reference_to_nn']='' # The sampled pixels correspond to the surface pixels.
     
        
    

    ############### SPESIFICATIONS TO SURFACE-VERSION ##################
    if 'TOA' in additional_spesifications: UVS.inp['zout']='TOA'
    elif 'SUR' in additional_spesifications: 
        # Set "zout sur" for DISORT
        # Do not spesify zout for MYSTIC -> give "zout sur" (which do not work for MYSTIC)
        if UVS.inp['rte_solver'] != 'montecarlo': UVS.inp['zout'] = 'sur'  
    ####################################################################

    

    if source=='thermal':
        UVS.inp['wavelength']='4500 100000'  
        if RTdimension == '3D':
            mc_photons = int(1e2)
            UVS.inp['mc_photons']=mc_photons 
            UVS.inp['mc_minphotons']=mc_photons # Note: MC_MINPHOTONS = 1e3 in uvspec.h so need to modify!
    else:
        UVS.inp['wavelength']='295 2800'     # Changed from '250 4000' 20.01.2026   
        if RTdimension == '3D':
            mc_photons = int(1e7)  
            UVS.inp['mc_photons']=mc_photons 
      
    
    

    # Water cloud
    UVS.inp['wc_properties']='mie'
    Nz = 100 
    wc_value, wcreff_value = 1e-2, 10.0
    h_min, h_max = 2, 3
    heights = np.linspace(0, 10, Nz) # km
    # Make a 2x2 cloud in the center of the domain
    cloud_start_X = Nx//2 #+ Nx//8
    cloud_start_Y = Ny//2 #+ Ny//8
    idx_inside_cloud = []

    if RTdimension == '3D':  
        
        # iz=0
        # last_h=99999
        # for h,p in zip(ACMCOM.height_level[1:,irec_surface_pixel],ACMCOM.pressure_level[1:,irec_surface_pixel]):
        #     if h > -1e-4 and p < 1e+10:
        #         last_h=h
        #         last_p=p
        #         iz=iz+1

        # if np.abs(last_h) > 1e-4 and last_p < 1e+10:
        #     iz=iz+1

        # Nz = iz-1 #ACMCOM.height_level[1:,ia].shape[0]
        # dx = 1.0 # 1 km horizontal resolution
        # dy = 1.0

        flag = 3 # mystic 3D format# mystic 3D format# mystic 3D format# mystic 3D format
        dx, dy = 1.0, 1.0 # 1km horizontal resolution
        heights = heights[1:]
        Nz -= 1
        wc_file_3D=input_dir+f'tmp_{ia:04d}_wc3D.dat'
        f = open(wc_file_3D,'w')
        f.write('{:d} {:d} {:d} {:d}\n'.format(Nx, Ny, Nz, flag))
        f.write('{:f} {:f} '.format(dx, dy))

        
       
        for h in heights:
            f.write('{:8.3f} '.format(np.abs(h)))
            
        f.write('\n')
        wclines=0

        ix=1
        for iac in iacrosses:
            iy=1 
            for ial in ialongs:
                iz = 1
                inside_cloud_region = (
                        cloud_start_X <= ix < cloud_start_X + 2
                    and cloud_start_Y <= iy < cloud_start_Y + 2
                )

                # Define indecies for cloud-vertical height
                if ix == cloud_start_X and iy == cloud_start_Y:
                    h_valid = np.asarray(heights)[heights > 0]

                    iz_h_min = np.argmin(np.abs(h_valid - h_min)) + 1
                    iz_h_max = np.argmin(np.abs(h_valid - h_max)) + 1

                    iz_min = min(iz_h_min, iz_h_max)
                    iz_max = max(iz_h_min, iz_h_max)

                    print("Cloud vertical indices:", iz_min, iz_max)
                    print("Actual heights:", h_valid[iz_min-1], h_valid[iz_max-1])

                

                # BG: made changes
                # - iz += 1 for each realistic h > 0 measurement by EarthCARE
                # - previosly iz += 1 only for realistic h > 0 and wc > 0 (this was the error)
                for h in heights:
                    if inside_cloud_region and iz_min <= iz < iz_max: # h_min < h < h_max:
                        # set wc
                        f.write('{:d} {:d} {:d} {:f} {:f}\n'.format(ix, iy, iz, wc_value, wcreff_value))
                        print(f'Writing wc={wc_value} at ix={ix}, iy={iy}, iz={iz}')
            
                    
                        wclines=wclines+1
                    iz=iz+1 
                iy=iy+1
            ix=ix+1

        # Add dummy line to make mystic run
        if wclines==0:
            f.write('{:d} {:d} {:d} {:f} {:f}\n'.format(1, 1, 1, 0, 10))
                
        f.close()
        UVS.inp['wc_file 3D']=wc_file_3D
        UVS.inp['mc_std']=''
    else:       
        wc_file_1D=input_dir+f'tmp_{ia:04d}_wc1D.dat'
        UVS.inp['wc_file 1D']=wc_file_1D
        f = open(wc_file_1D,'w')
    
        ix = 1
        for iac in iacrosses:
            iy = 1
            for ial in ialongs:
                inside_cloud_region = (
                    cloud_start_X <= ix < cloud_start_X + 2
                    and cloud_start_Y <= iy < cloud_start_Y + 2
                )
                if inside_cloud_region:
                    idx_inside_cloud.append(Nx* (iy-1) + (ix-1))
                iy += 1
            ix += 1
        print(idx_inside_cloud)
    
                
        for h in heights[::-1]:
            if h_min <= h <= h_max and ia in idx_inside_cloud:
                wc, wcreff = wc_value, wcreff_value
            else:
                wc, wcreff = 0.0, 0.0

            f.write('{:8.3f} {:10.5f} {:8.3f}\n'.format(h, wc, wcreff))
        f.close()
    
    return UVS


def tmp_output_filename(iia, scene_name):
    """
    Return the filename to be used for temporary output files.

    """
    pass
    












if __name__ == "__main__":
    start_time = datetime.now(timezone.utc) 
    # print("Run started at", start_time.isoformat())



    ##############################################################################################
    WANT_SUR = True         # zout at SUR or TOA
    WANT_3D  = True        # MYSTIC or DISORT

    verbose  = False        # Want verbose output from RTM
    want_ps  = False        # Psudospherical solver
        
                            
    which_buffer = 1
    # 0: MCIPA   1:Test   2: 21x21   3: 41x41  
    ################################################################################################

   
  

    # BG: To distinguish minor modifications to .nc files
    additional_spesifications = 'SINGLE_CLOUD' 
    sza = 60
    phi0 = 60


    if WANT_SUR: additional_spesifications += "_SUR"
    else:        additional_spesifications += "_TOA"



    idx_source = 0
    sources = [['solar'], 
              ['thermal'],
              ['solar', 'thermal']][idx_source] 


       
    if WANT_3D:
        rte_solver = 'montecarlo' #'mystic' 
        RTdimension = '3D'
    else:
        if want_ps:
            rte_solver = 'disort \npseudospherical'
        else: 
            rte_solver = 'disort'  # 'twostr'        
        RTdimension ='1D' 
    print(f"{RTdimension} RTM - {rte_solver}")


    # ---------------------------------- Paths -----------------------------------------------
    RTInpOutPath = './tmpRTIO/'    # Folder to store the final results
    RTOutNetcdfPath = './RESULTS_TEST/'  # Folder name for netcdf result files
    uvspecpath = '/xnilu_wrk2/projects/NEVAR/libRadtran/bin/' # Folder to RTM code
    RTMdata = '/xnilu_wrk2/projects/NEVAR/libRadtran/data/' #RTM input data

    libRad_version= 'v01_'+rte_solver+'_'+RTdimension
    # --------------------------------------------------------------------------------------------



    Nx, Ny = 1, 1 # Default Horizontal Computaion-Domain Size                                                                                     # min / pixel   |     solar-std     |     thermal-std  
    if   which_buffer == 0 :                  buffer_str = 'MCIPA'
    elif which_buffer == 1 : Nx, Ny =  5,  5; buffer_str = 'TEST   Buffer (5 x 5)'                      #   1.5         |                   |
    elif which_buffer == 2 : Nx, Ny = 21, 21; buffer_str = '~BBR +  5 pixels-buffer  (21 x 21)'         #    4          |       ~5-8        |       ~0.5-2
    elif which_buffer == 3 : Nx, Ny = 41, 41; buffer_str = '~BBR + 35 pixels-buffer  (41 x 41)'         #    5          |       ~17         |       ~1.6
    else:buffer_str = ''
    

                                                                                            

        
    SceneNamelibRad='libRad'
    libRad = ReadEC.Scene(Name=SceneNamelibRad, verbose=verbose)    
    if 'montecarlo' in rte_solver:
        shape = (1, Nx, Ny)
    else: 
        shape = Nx*Ny
    libRad.latitude_active = np.zeros(shape)

    # print(shape)
    libRad.solar_e = np.zeros(shape)       
    libRad.solar_e_std = np.zeros(shape)
    libRad.thermal_e = np.zeros(shape)
    libRad.thermal_e_std = np.zeros(shape)


    if WANT_3D:
        ialongs = [Nx//2]
        iacr = Nx // 2
    else:
        ialongs = np.arange(Nx*Ny)
        iacr = 0

    



    # Parallelize over OrbitIDs instead of along-track points.
    # Each MPI rank will process all ialongs for its assigned orbit in sequence.
    
    for ia in ialongs:
        # --- BG: ADD THIS BLOCK ---
        e_solar = np.nan
        e_solar_std = np.nan
        e_thermal = np.nan
        e_thermal_std = np.nan
        # ----------------------
        
        

        for source in sources:
            try: 
                RTM=True
                if source=='solar':
                    mol_abs_param = 'kato2' #'reptran course' #
                elif source=='thermal':
                    mol_abs_param = 'fu' 

                if RTM:
                    UVS = UVspec.UVspec()   #tms: UVspec.py is an external program that defines input and runs libRadtran
                    UVS.inp['rte_solver']=rte_solver
                    #tms: "SetRTM" is a function defined above - NB! Check path
                    UVS = SetRTM(UVS, ia, iacr, sza, phi0,mc_basename_path=RTInpOutPath, RTdimension = RTdimension, input_dir=RTInpOutPath,
                                mol_abs_param=mol_abs_param, data_dir=RTMdata, source=source)
                    if UVS.status == 'OK':
                        # uvspecInputFile=RTInpOutPath+'tmp'+'{:00004d}'.format(ia)+source+str(my_rank)+'.inp'  
                        # NEW TEST
                        uvspecInputFile=RTInpOutPath+f'tmp_SINGLE_CLOUD_{ia:04d}.inp'    
                        uvspecOutputFile=uvspecInputFile.replace('inp','out')
                        UVS.WriteInputFile(uvspecInputFile)

                        status = UVS.SingleRun(uvspecInputFile, uvspecOutputFile, verbose=False, uvspecpath=uvspecpath)

                        if UVS.inp['rte_solver']=='montecarlo': ######################## MYSTIC
                            try:
                                mc_flx_dir, mc_flx_dn, mc_flx_up, edir_std, edn_std, eup_std = UVspec.ReadMCOut(UVS.mc_basename, mol_abs_param=mol_abs_param) #fn = fn + '.flx.spc'  
                                if 'AllLevels' in additional_spesifications:
                                    nlev = int(len(mc_flx_dir)/(Nx*Ny))
                                
                                    mc_flx_dir  = mc_flx_dir.reshape(nlev, Nx, Ny)
                                    mc_flx_dn   = mc_flx_dn.reshape(nlev, Nx, Ny)
                                    mc_flx_up   = mc_flx_up.reshape(nlev, Nx, Ny)

                                    eup_std     = eup_std.reshape(nlev, Nx, Ny)
                                    edn_std     = edn_std.reshape(nlev, Nx, Ny)
                                    edir_std    = edir_std.reshape(nlev, Nx, Ny)


            
                                else:
                                    mc_flx_dir  = mc_flx_dir.reshape(Nx, Ny)
                                    mc_flx_dn   = mc_flx_dn.reshape(Nx, Ny)
                                    mc_flx_up   = mc_flx_up.reshape(Nx, Ny)
                                    edir_std    = edir_std.reshape(Nx, Ny)
                                    eup_std     = eup_std.reshape(Nx, Ny)
                                    edn_std     = edn_std.reshape(Nx, Ny)


                                
                            except (FileNotFoundError, ValueError) as e:
                                print(f"[WORKER {my_rank}] WARNING: Could not read MC output for ia={ia}, source={source}. Skipping. Error: {e}")
                                # continue  # Go to the next iteration
                        
                            if source=='solar':   
                                if WANT_SUR:
                                    e_solar_std = edir_std + edn_std
                                    e_solar     = mc_flx_dir + mc_flx_dn
                                    # print('direct + diffuse solar flux at SUR = ', e_solar, '\n with direct flux = ', mc_flx_dir, ' and diffuse flux = ', mc_flx_dn)
                                else:
                                    e_solar_std = eup_std
                                    e_solar     = mc_flx_up

                            elif source=='thermal':
                                if WANT_SUR:
                                    e_thermal_std = edir_std + edn_std
                                    e_thermal     = mc_flx_dir + mc_flx_dn
                                else:
                                    e_thermal_std = eup_std
                                    e_thermal     = mc_flx_up

                
                        else: ########################################################## DISORT
                            try:
                                sza, zout, edir, edn, eup = UVspec.ReadRTOut(uvspecOutputFile, mol_abs_param=mol_abs_param)
                            except ValueError as e:
                                # print(edir, edn, eup)
                                print("Error [WORKER {}]: could not read file {} with mol_abs_param {}".
                                    format(my_rank, uvspecOutputFile, mol_abs_param))
                                print(f"Error [WORKER {my_rank}]: ", e)

                                
                        

                            e_solar_std = e_thermal_std = None  # No std output from DISORT
                            if source=='solar':
                                if WANT_SUR:
                                    e_solar = edir + edn
                                else:
                                    e_solar = eup
                            elif source=='thermal':
                                if WANT_SUR:
                                    e_thermal = edir + edn
                                else:
                                    e_thermal = eup


                    else:
                        print(UVS.status)
                        print(f'WARNING: UVS.status not OK for ia={ia}, source={source}. Skipping.')
                        zout=edir=eup=eup_std=0.0
                else:
                    zout=edir_solar=eup_solar=eup_solar_std=0.0
                    edir_thermal=eup_thermal=eup_thermal_std=0.0
                    

                

                if rte_solver=='montecarlo': 
                    if not hasattr(e_solar, "shape"):
                        # raise ValueError("Simulated solar flux has no shape (simulation was not successful)")    
                        e_solar_center = np.nan
                        e_solar_std_center = np.nan
                        e_thermal_center = np.nan
                        e_thermal_std_center = np.nan                        
                
                    elif 'AllLevels' in additional_spesifications:
                        ix_center = e_solar.shape[1] // 2
                        iy_center = e_solar.shape[2] // 2
                        
                        if hasattr(e_solar_std, "shape"):   e_solar_std_center      = e_solar_std[:, ix_center, iy_center]
                        if hasattr(e_solar, "shape"):       e_solar_center          = e_solar[:, ix_center, iy_center]
                        if hasattr(e_thermal_std, "shape"): e_thermal_std_center    = e_thermal_std[:, ix_center, iy_center]
                        if hasattr(e_thermal, "shape"):     e_thermal_center        = e_thermal[:, ix_center, iy_center]
                    else: 
                        ix_center = e_solar.shape[0] // 2
                        iy_center = e_solar.shape[1] // 2

                        if hasattr(e_solar_std, "shape"):   e_solar_std_center      = float(e_solar_std[ix_center, iy_center])
                        if hasattr(e_solar, "shape"):       e_solar_center          = float(e_solar[ix_center, iy_center])
                        if hasattr(e_thermal_std, "shape"): e_thermal_std_center    = float(e_thermal_std[ix_center, iy_center])
                        if hasattr(e_thermal, "shape"):     e_thermal_center        = float(e_thermal[ix_center, iy_center])

                    # print("ix_center, iy_center", ix_center, iy_center)                              
                else:
                    e_solar_center     = e_solar
                    e_thermal_center   = e_thermal

                ####### PRINTING RESULTS ##########
                if 'AllLevels' in additional_spesifications:
                    print(e_solar_center)
                    print(e_solar_std)
                else:
                    e_str = f"{e_solar_center:7.1f}" if source=='solar' else f"{e_thermal_center:7.1f}"
                    std_str = f"+- {e_solar_std_center:.1f}" if 'montecarlo' in rte_solver else ''
                    print("\n\n\n"
                        f"      | ia={ia:4}  ----------------------------------------------------------|\n"
                        f"      |                                   Flux ({source}) = {e_str} {std_str} W/m2\n"
                            "      |--------------------------------------------------------------------------------|"
                    )
                   
            

            except Exception as e:
                print(f"\nException for ia={ia:4}: {e}\n")
    

    

    if rte_solver == "montecarlo":
        idx = (ia, slice(None), slice(None))
    else:
        idx = ia

    print(e_solar, e_solar_std, e_thermal, e_thermal_std)
    libRad.solar_e_std[idx]   = e_solar_std
    libRad.thermal_e_std[idx] = e_thermal_std
    libRad.solar_e[idx]       = e_solar
    libRad.thermal_e[idx]     = e_thermal

    # print(e_solar, libRad.solar_e[idx])

        
                        
                

    # Writing NetCDF file...")
    src = ','.join([i for i in sources])
    if 'montecarlo' in rte_solver:
        libRad.WriteNetcdf_mc_sample_grid(RTOutNetcdfPath+'libRad_'+libRad_version+'_'+src+'_'+ additional_spesifications + '.nc', shape=(Nx,Ny), verbose=False)
    else:
        libRad.WriteNetcdf(               RTOutNetcdfPath+'libRad_'+libRad_version+'_'+src+'_'+ additional_spesifications + '.nc', verbose=False)
    print("Wrote to " + RTOutNetcdfPath+'libRad_'+libRad_version+'_'+src+'_'+ additional_spesifications + '.nc')


    tt = datetime.now(timezone.utc) - start_time
    print(f"            Finished run. It took {tt.total_seconds()/3600:.2f} hours \n\n\n\n\n\n\n")

import numpy as np
import pickle
import os
import sys
import multiprocessing
import glob
import scipy
import scipy.constants
from scipy.interpolate import griddata
import dted
import pdb # For debugging
from mpi4py import MPI # Use MPI for multiprocessing
from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess



#import ReadEarthCAREL2 as ReadEC
import Read_RTM as ReadEC
import UVspec as UVspec
import Find_Overpass_Info as Find_Overpass_Info




# BG: overflow encouner - Warnings from reading ACM_COM files
# BG: Hiding the 'overflow encountered in multiply' warnings:
np.seterr(over='ignore')




# Where are we
myhost = os.uname()[1]
home = os.environ['HOME']

# initialize MPI
comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
np_mpi = comm.Get_size()


def pdeg2km(p):
    latfact = 111.32
    p0 = latfact*p[0]/1000
    lonfact = 40075 * np.cos(np.deg2rad( p[1] )) / 360
    p1 = latfact*p[1]/1000
    return (p0,p1)


def Check3DBufferSize(ia, iacr, which_buffer):

    Nx, Ny, iacrosses, ialongs = Calc3DBufferSize(ia, iacr, which_buffer)
    if Nx>0 and Ny>0:
        BufferOK=True
    else:
        BufferOK=False
        
    return BufferOK


def Calc3DBufferSize(ia, iacr, which_buffer):
    WANT_PRINTS = False
    # BG: Decide to use my buffer, not dynamic buffer!
    if which_buffer == 0:   along_dim, across_dim =  0,  0 # ( 1, 1) comp-domain  MCIPA 
    elif which_buffer == 1: along_dim, across_dim =  2,  2 # ( 5, 5) comp-domain  Test Buffer
    elif which_buffer == 2: along_dim, across_dim = 10, 10 # (21,21) comp-domain  ~BBR +  5 pixels-buffer
    elif which_buffer == 3: along_dim, across_dim = 20, 20 # (41,41) comp-domain  ~BBR + 35 pixels-buffer
    
    ialongs = np.arange(ia-along_dim, ia+along_dim+1)
    iacrosses = np.arange(iacr-across_dim, iacr+across_dim+1)
    
    Nx = iacrosses.shape[0]
    Ny = ialongs.shape[0]


    return Nx, Ny, iacrosses, ialongs

def SetRTM(UVS, ia, iacr, sza, phi0, input_dir='', mc_basename_path='', RTdimension = '1D',
           mol_abs_param='kato2', data_dir='', source='solar'):

    
    UVS.status = 'OK'
    # First some general input
    UVS.inp['data_files_path']=data_dir
    
    libpath='/xnilu_wrk2/projects/NEVAR/libRadtran/'
    UVS.inp['atmosphere_file'] = f"{libpath}/data/atmmod/afglss.dat"
    UVS.inp['albedo'] = 0.05

    UVS.inp['mol_abs_param']=mol_abs_param
    UVS.inp['source']=source
    UVS.inp['output_process']='sum'

    UVS.inp['output_user']='sza zout edir edn eup'
    UVS.inp['quiet']=''
    

    # Solar geometry
    # Set solar geometry here to make sure azimuth shift due to 3D geometry is included
    UVS.inp['sza']=sza
    UVS.inp['phi0']=phi0  # Azimuth is only needed for 3D, but it does not hurt for 1D.

    Nx, Ny, iacrosses, ialongs = Calc3DBufferSize(ia, iacr, which_buffer)  #calculates along and cross track buffer size 
    if RTdimension == '3D':       
        if UVS.inp['rte_solver'] != 'montecarlo':
            print('3D geometry only available with MYSTIC. You specified:', UVS.inp['rte_solver'])
            exit()
                
        UVS.inp['mc_std']=''
        # UVS.mc_basename=mc_basename_path+'mc_{:00004d}'.format(ia)+'_'+source # Old->made race-conditions errors 
        UVS.mc_basename=mc_basename_path+'mc_'+f"{ia:04d}_"+source
        print("MC basename set to:", UVS.mc_basename)
        UVS.inp['mc_basename']=UVS.mc_basename
        UVS.inp['mc_randomseed'] = (ia+1)*72341//11
        
        # Set SAMPLE-GRID
        UVS.inp['mc_sample_grid']=f'{Nx} {Ny}' # 3D computation domain
        UVS.inp['mc_reference_to_nn']='' # The sampled pixels correspond to the surface pixels.
     
        
    

    ############### SPESIFICATIONS TO SURFACE-VERSION ##################
    if 'TOA' in additional_spesifications: UVS.inp['zout']='TOA'
    elif 'SUR' in additional_spesifications: 
        # Set "zout sur" for DISORT
        # Do not spesify zout for MYSTIC -> give "zout sur" (which do not work for MYSTIC)
        if UVS.inp['rte_solver'] != 'montecarlo': UVS.inp['zout'] = 'sur'  
    ####################################################################

    

    if source=='thermal':
        UVS.inp['wavelength']='4500 100000'  
        if RTdimension == '3D':
            mc_photons = int(1e2)
            UVS.inp['mc_photons']=mc_photons 
            UVS.inp['mc_minphotons']=mc_photons # Note: MC_MINPHOTONS = 1e3 in uvspec.h so need to modify!
    else:
        UVS.inp['wavelength']='295 2800'     # Changed from '250 4000' 20.01.2026   
        if RTdimension == '3D':
            mc_photons = int(1e7)  
            UVS.inp['mc_photons']=mc_photons 
      
    
    

    # Water cloud
    UVS.inp['wc_properties']='mie'
    Nz = 100 
    wc_value, wcreff_value = 1e-2, 10.0
    h_min, h_max = 2, 3
    heights = np.linspace(0, 10, Nz) # km
    # Make a 2x2 cloud in the center of the domain
    cloud_start_X = Nx//2 #+ Nx//8
    cloud_start_Y = Ny//2 #+ Ny//8
    idx_inside_cloud = []

    if RTdimension == '3D':  
        
        # iz=0
        # last_h=99999
        # for h,p in zip(ACMCOM.height_level[1:,irec_surface_pixel],ACMCOM.pressure_level[1:,irec_surface_pixel]):
        #     if h > -1e-4 and p < 1e+10:
        #         last_h=h
        #         last_p=p
        #         iz=iz+1

        # if np.abs(last_h) > 1e-4 and last_p < 1e+10:
        #     iz=iz+1

        # Nz = iz-1 #ACMCOM.height_level[1:,ia].shape[0]
        # dx = 1.0 # 1 km horizontal resolution
        # dy = 1.0

        flag = 3 # mystic 3D format# mystic 3D format# mystic 3D format# mystic 3D format
        dx, dy = 1.0, 1.0 # 1km horizontal resolution
        heights = heights[1:]
        Nz -= 1
        wc_file_3D=input_dir+f'tmp_{ia:04d}_wc3D.dat'
        f = open(wc_file_3D,'w')
        f.write('{:d} {:d} {:d} {:d}\n'.format(Nx, Ny, Nz, flag))
        f.write('{:f} {:f} '.format(dx, dy))

        
       
        for h in heights:
            f.write('{:8.3f} '.format(np.abs(h)))
            
        f.write('\n')
        wclines=0

        ix=1
        for iac in iacrosses:
            iy=1 
            for ial in ialongs:
                iz = 1
                inside_cloud_region = (
                        cloud_start_X <= ix < cloud_start_X + 2
                    and cloud_start_Y <= iy < cloud_start_Y + 2
                )

                # Define indecies for cloud-vertical height
                if ix == cloud_start_X and iy == cloud_start_Y:
                    h_valid = np.asarray(heights)[heights > 0]

                    iz_h_min = np.argmin(np.abs(h_valid - h_min)) + 1
                    iz_h_max = np.argmin(np.abs(h_valid - h_max)) + 1

                    iz_min = min(iz_h_min, iz_h_max)
                    iz_max = max(iz_h_min, iz_h_max)

                    print("Cloud vertical indices:", iz_min, iz_max)
                    print("Actual heights:", h_valid[iz_min-1], h_valid[iz_max-1])

                

                # BG: made changes
                # - iz += 1 for each realistic h > 0 measurement by EarthCARE
                # - previosly iz += 1 only for realistic h > 0 and wc > 0 (this was the error)
                for h in heights:
                    if inside_cloud_region and iz_min <= iz < iz_max: # h_min < h < h_max:
                        # set wc
                        f.write('{:d} {:d} {:d} {:f} {:f}\n'.format(ix, iy, iz, wc_value, wcreff_value))
                        print(f'Writing wc={wc_value} at ix={ix}, iy={iy}, iz={iz}')
            
                    
                        wclines=wclines+1
                    iz=iz+1 
                iy=iy+1
            ix=ix+1

        # Add dummy line to make mystic run
        if wclines==0:
            f.write('{:d} {:d} {:d} {:f} {:f}\n'.format(1, 1, 1, 0, 10))
                
        f.close()
        UVS.inp['wc_file 3D']=wc_file_3D
        UVS.inp['mc_std']=''
    else:       
        wc_file_1D=input_dir+f'tmp_{ia:04d}_wc1D.dat'
        UVS.inp['wc_file 1D']=wc_file_1D
        f = open(wc_file_1D,'w')
    
        ix = 1
        for iac in iacrosses:
            iy = 1
            for ial in ialongs:
                inside_cloud_region = (
                    cloud_start_X <= ix < cloud_start_X + 2
                    and cloud_start_Y <= iy < cloud_start_Y + 2
                )
                if inside_cloud_region:
                    idx_inside_cloud.append(Nx* (iy-1) + (ix-1))
                iy += 1
            ix += 1
        print(idx_inside_cloud)
    
                
        for h in heights[::-1]:
            if h_min <= h <= h_max and ia in idx_inside_cloud:
                wc, wcreff = wc_value, wcreff_value
            else:
                wc, wcreff = 0.0, 0.0

            f.write('{:8.3f} {:10.5f} {:8.3f}\n'.format(h, wc, wcreff))
        f.close()
    
    return UVS


def tmp_output_filename(iia, scene_name):
    """
    Return the filename to be used for temporary output files.

    """
    pass
    












if __name__ == "__main__":
    start_time = datetime.now(timezone.utc) 
    # print("Run started at", start_time.isoformat())



    ##############################################################################################
    WANT_SUR = True         # zout at SUR or TOA
    WANT_3D  = False        # MYSTIC or DISORT

    verbose  = False        # Want verbose output from RTM
    want_ps  = False        # Psudospherical solver
        
                            
    which_buffer = 1
    # 0: MCIPA   1:Test   2: 21x21   3: 41x41  
    ################################################################################################

   
  

    # BG: To distinguish minor modifications to .nc files
    additional_spesifications = 'SINGLE_CLOUD' 
    sza = 60
    phi0 = 60


    if WANT_SUR: additional_spesifications += "_SUR"
    else:        additional_spesifications += "_TOA"



    idx_source = 0
    sources = [['solar'], 
              ['thermal'],
              ['solar', 'thermal']][idx_source] 


       
    if WANT_3D:
        rte_solver = 'montecarlo' #'mystic' 
        RTdimension = '3D'
    else:
        if want_ps:
            rte_solver = 'disort \npseudospherical'
        else: 
            rte_solver = 'disort'  # 'twostr'        
        RTdimension ='1D' 
    print(f"{RTdimension} RTM - {rte_solver}")


    # ---------------------------------- Paths -----------------------------------------------
    RTInpOutPath = './tmpRTIO/'    # Folder to store the final results
    RTOutNetcdfPath = './RESULTS_TEST/'  # Folder name for netcdf result files
    uvspecpath = '/xnilu_wrk2/projects/NEVAR/libRadtran/bin/' # Folder to RTM code
    RTMdata = '/xnilu_wrk2/projects/NEVAR/libRadtran/data/' #RTM input data

    libRad_version= 'v01_'+rte_solver+'_'+RTdimension
    # --------------------------------------------------------------------------------------------



    Nx, Ny = 1, 1 # Default Horizontal Computaion-Domain Size                                                                                     # min / pixel   |     solar-std     |     thermal-std  
    if   which_buffer == 0 :                  buffer_str = 'MCIPA'
    elif which_buffer == 1 : Nx, Ny =  5,  5; buffer_str = 'TEST   Buffer (5 x 5)'                      #   1.5         |                   |
    elif which_buffer == 2 : Nx, Ny = 21, 21; buffer_str = '~BBR +  5 pixels-buffer  (21 x 21)'         #    4          |       ~5-8        |       ~0.5-2
    elif which_buffer == 3 : Nx, Ny = 41, 41; buffer_str = '~BBR + 35 pixels-buffer  (41 x 41)'         #    5          |       ~17         |       ~1.6
    else:buffer_str = ''
    

                                                                                            

        
    SceneNamelibRad='libRad'
    libRad = ReadEC.Scene(Name=SceneNamelibRad, verbose=verbose)    
    if 'montecarlo' in rte_solver:
        shape = (1, Nx, Ny)
    else: 
        shape = Nx*Ny
    libRad.latitude_active = np.zeros(shape)

    # print(shape)
    libRad.solar_e = np.zeros(shape)       
    libRad.solar_e_std = np.zeros(shape)
    libRad.thermal_e = np.zeros(shape)
    libRad.thermal_e_std = np.zeros(shape)


    if WANT_3D:
        ialongs = [Nx//2]
        iacr = Nx // 2
    else:
        ialongs = np.arange(Nx*Ny)
        iacr = 0

    



    # Parallelize over OrbitIDs instead of along-track points.
    # Each MPI rank will process all ialongs for its assigned orbit in sequence.
    
    for ia in ialongs:
        # --- BG: ADD THIS BLOCK ---
        e_solar = np.nan
        e_solar_std = np.nan
        e_thermal = np.nan
        e_thermal_std = np.nan
        # ----------------------
        
        

        for source in sources:
            try: 
                RTM=True
                if source=='solar':
                    mol_abs_param = 'kato2' #'reptran course' #
                elif source=='thermal':
                    mol_abs_param = 'fu' 

                if RTM:
                    UVS = UVspec.UVspec()   #tms: UVspec.py is an external program that defines input and runs libRadtran
                    UVS.inp['rte_solver']=rte_solver
                    #tms: "SetRTM" is a function defined above - NB! Check path
                    UVS = SetRTM(UVS, ia, iacr, sza, phi0,mc_basename_path=RTInpOutPath, RTdimension = RTdimension, input_dir=RTInpOutPath,
                                mol_abs_param=mol_abs_param, data_dir=RTMdata, source=source)
                    if UVS.status == 'OK':
                        # uvspecInputFile=RTInpOutPath+'tmp'+'{:00004d}'.format(ia)+source+str(my_rank)+'.inp'  
                        # NEW TEST
                        uvspecInputFile=RTInpOutPath+f'tmp_SINGLE_CLOUD_{ia:04d}.inp'    
                        uvspecOutputFile=uvspecInputFile.replace('inp','out')
                        UVS.WriteInputFile(uvspecInputFile)

                        status = UVS.SingleRun(uvspecInputFile, uvspecOutputFile, verbose=False, uvspecpath=uvspecpath)

                        if UVS.inp['rte_solver']=='montecarlo': ######################## MYSTIC
                            try:
                                mc_flx_dir, mc_flx_dn, mc_flx_up, edir_std, edn_std, eup_std = UVspec.ReadMCOut(UVS.mc_basename, mol_abs_param=mol_abs_param) #fn = fn + '.flx.spc'  
                                if 'AllLevels' in additional_spesifications:
                                    nlev = int(len(mc_flx_dir)/(Nx*Ny))
                                
                                    mc_flx_dir  = mc_flx_dir.reshape(nlev, Nx, Ny)
                                    mc_flx_dn   = mc_flx_dn.reshape(nlev, Nx, Ny)
                                    mc_flx_up   = mc_flx_up.reshape(nlev, Nx, Ny)

                                    eup_std     = eup_std.reshape(nlev, Nx, Ny)
                                    edn_std     = edn_std.reshape(nlev, Nx, Ny)
                                    edir_std    = edir_std.reshape(nlev, Nx, Ny)


            
                                else:
                                    mc_flx_dir  = mc_flx_dir.reshape(Nx, Ny)
                                    mc_flx_dn   = mc_flx_dn.reshape(Nx, Ny)
                                    mc_flx_up   = mc_flx_up.reshape(Nx, Ny)
                                    edir_std    = edir_std.reshape(Nx, Ny)
                                    eup_std     = eup_std.reshape(Nx, Ny)
                                    edn_std     = edn_std.reshape(Nx, Ny)


                                
                            except (FileNotFoundError, ValueError) as e:
                                print(f"[WORKER {my_rank}] WARNING: Could not read MC output for ia={ia}, source={source}. Skipping. Error: {e}")
                                # continue  # Go to the next iteration
                        
                            if source=='solar':   
                                if WANT_SUR:
                                    e_solar_std = edir_std + edn_std
                                    e_solar     = mc_flx_dir + mc_flx_dn
                                    # print('direct + diffuse solar flux at SUR = ', e_solar, '\n with direct flux = ', mc_flx_dir, ' and diffuse flux = ', mc_flx_dn)
                                else:
                                    e_solar_std = eup_std
                                    e_solar     = mc_flx_up

                            elif source=='thermal':
                                if WANT_SUR:
                                    e_thermal_std = edir_std + edn_std
                                    e_thermal     = mc_flx_dir + mc_flx_dn
                                else:
                                    e_thermal_std = eup_std
                                    e_thermal     = mc_flx_up

                
                        else: ########################################################## DISORT
                            try:
                                sza, zout, edir, edn, eup = UVspec.ReadRTOut(uvspecOutputFile, mol_abs_param=mol_abs_param)
                            except ValueError as e:
                                # print(edir, edn, eup)
                                print("Error [WORKER {}]: could not read file {} with mol_abs_param {}".
                                    format(my_rank, uvspecOutputFile, mol_abs_param))
                                print(f"Error [WORKER {my_rank}]: ", e)

                                
                        

                            e_solar_std = e_thermal_std = None  # No std output from DISORT
                            if source=='solar':
                                if WANT_SUR:
                                    e_solar = edir + edn
                                else:
                                    e_solar = eup
                            elif source=='thermal':
                                if WANT_SUR:
                                    e_thermal = edir + edn
                                else:
                                    e_thermal = eup


                    else:
                        print(UVS.status)
                        print(f'WARNING: UVS.status not OK for ia={ia}, source={source}. Skipping.')
                        zout=edir=eup=eup_std=0.0
                else:
                    zout=edir_solar=eup_solar=eup_solar_std=0.0
                    edir_thermal=eup_thermal=eup_thermal_std=0.0
                    

                

                if rte_solver=='montecarlo': 
                    if not hasattr(e_solar, "shape"):
                        # raise ValueError("Simulated solar flux has no shape (simulation was not successful)")    
                        e_solar_center = np.nan
                        e_solar_std_center = np.nan
                        e_thermal_center = np.nan
                        e_thermal_std_center = np.nan                        
                
                    elif 'AllLevels' in additional_spesifications:
                        ix_center = e_solar.shape[1] // 2
                        iy_center = e_solar.shape[2] // 2
                        
                        if hasattr(e_solar_std, "shape"):   e_solar_std_center      = e_solar_std[:, ix_center, iy_center]
                        if hasattr(e_solar, "shape"):       e_solar_center          = e_solar[:, ix_center, iy_center]
                        if hasattr(e_thermal_std, "shape"): e_thermal_std_center    = e_thermal_std[:, ix_center, iy_center]
                        if hasattr(e_thermal, "shape"):     e_thermal_center        = e_thermal[:, ix_center, iy_center]
                    else: 
                        ix_center = e_solar.shape[0] // 2
                        iy_center = e_solar.shape[1] // 2

                        if hasattr(e_solar_std, "shape"):   e_solar_std_center      = float(e_solar_std[ix_center, iy_center])
                        if hasattr(e_solar, "shape"):       e_solar_center          = float(e_solar[ix_center, iy_center])
                        if hasattr(e_thermal_std, "shape"): e_thermal_std_center    = float(e_thermal_std[ix_center, iy_center])
                        if hasattr(e_thermal, "shape"):     e_thermal_center        = float(e_thermal[ix_center, iy_center])

                    # print("ix_center, iy_center", ix_center, iy_center)                              
                else:
                    e_solar_center     = e_solar
                    e_thermal_center   = e_thermal

                ####### PRINTING RESULTS ##########
                if 'AllLevels' in additional_spesifications:
                    print(e_solar_center)
                    print(e_solar_std)
                else:
                    e_str = f"{e_solar_center:7.1f}" if source=='solar' else f"{e_thermal_center:7.1f}"
                    std_str = f"+- {e_solar_std_center:.1f}" if 'montecarlo' in rte_solver else ''
                    print("\n\n\n"
                        f"      | ia={ia:4}  ----------------------------------------------------------|\n"
                        f"      |                                   Flux ({source}) = {e_str} {std_str} W/m2\n"
                            "      |--------------------------------------------------------------------------------|"
                    )
                   
            

            except Exception as e:
                print(f"\nException for ia={ia:4}: {e}\n")
    

    

    if rte_solver == "montecarlo":
        idx = (ia, slice(None), slice(None))
    else:
        idx = ia

    print(e_solar, e_solar_std, e_thermal, e_thermal_std)
    libRad.solar_e_std[idx]   = e_solar_std
    libRad.thermal_e_std[idx] = e_thermal_std
    libRad.solar_e[idx]       = e_solar
    libRad.thermal_e[idx]     = e_thermal

    # print(e_solar, libRad.solar_e[idx])

        
                        
                

    # Writing NetCDF file...")
    src = ','.join([i for i in sources])
    if 'montecarlo' in rte_solver:
        libRad.WriteNetcdf_mc_sample_grid(RTOutNetcdfPath+'libRad_'+libRad_version+'_'+src+'_'+ additional_spesifications + '.nc', shape=(Nx,Ny), verbose=False)
    else:
        libRad.WriteNetcdf(               RTOutNetcdfPath+'libRad_'+libRad_version+'_'+src+'_'+ additional_spesifications + '.nc', verbose=False)
    print("Wrote to " + RTOutNetcdfPath+'libRad_'+libRad_version+'_'+src+'_'+ additional_spesifications + '.nc')


    tt = datetime.now(timezone.utc) - start_time
    print(f"            Finished run. It took {tt.total_seconds()/3600:.2f} hours \n\n\n\n\n\n\n")

