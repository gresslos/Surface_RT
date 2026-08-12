import numpy as np
import pickle
import os
import sys
import multiprocessing
from subprocess import Popen,PIPE, STDOUT, call

# Where are we
myhost = os.uname()[1]
home = os.environ['HOME']

class UVspec:
    def __init__(self):
        # Set some uvspec input that most likely will stay the same
        self.IOdir = './'
        self.inp = {}
        #        self.inp["albedo"] = '0.0'

        return

    def add_mc_basename_to_input_file(self,mc_basename,fn):
        f = open(fn,'a')
        f.write('{0:s}\n'.format('mc_basename '+mc_basename))
        f.close()

    def info(self):
        print()
        print('UVspec parameters:')
        print('IOdir: {:s}'.format(self.IOdir))
        sys.stdout.flush()

        return

    def Run(self,inp, out, verbose=True, n_processes=2, uvspecpath='', RandString='gabba',
            Convolve=False, Instrument=None, OldOutFiles='',RunUvspec=True, Wait=False):
        debug=False # True #
        if verbose:
            if RunUvspec:
                print("Running uvspec with input file: ", inp)
            else:
                print("NOT running uvspec, using old output file: ", inp)
            print("Output to file                : ", out)
            print("Number of processors          : ", n_processes)
            print("Convolve                      : ", Convolve)
            sys.stdout.flush()

        tmp_out_base = 'tmp_mystic_'+RandString+'.out_'
        tmp_inp_base = 'tmp_mystic_'+RandString+'.inp_'
        # Remove all old files
        # OR NOT: Keep and remove manually in order to be able
        # inspect problems in *.err files, AK 20160526
        #FIXME
        # if RunUvspec:
        #     #        for filename in glob('gabba'+tmp_out_base+"*"):
        #     for filename in glob(tmp_out_base+"*"):
        #         if not debug:
        #             os.remove(filename)
        #     for filename in glob(tmp_inp_base+"*"):
        #         #        for filename in glob('gabba'+tmp_inp_base+"*"):
        #         if not debug:
        #             os.remove(filename)

        if RunUvspec:
            jobs = []
            tmpinputfiles=[]
            tmpoutputfiles=[]
            for i in range(n_processes):
                # Copy input file to temporary input file to be able to add different
                # mc_basenames to the file without destroying the input file
                tmp_inp = tmp_inp_base+str(i)
                tmpinputfiles.append(tmp_inp)
                cmd = 'cp '+inp+' '+tmp_inp
                Popen([r"cp",inp, tmp_inp]).wait()
                mc_basename = tmp_out_base+'NP_'+str(i)
                self.add_mc_basename_to_input_file(mc_basename,tmp_inp)
                tmp_out = tmp_out_base+str(i)
                print("tmp_out", tmp_out)
                ips = '{:d}'.format(i)
                tmpoutputfile = tmp_out.replace('out_'+ips,'out_NP_'+ips)+'.rad.spc'
                print("tmpoutputfile", tmpoutputfile)
                tmpoutputfiles.append(tmpoutputfile)
                if verbose:
                    print('Starting process:',i,' inp:',tmp_inp,' out:',tmp_out)
                    sys.stdout.flush()

                if not debug:
                    if RunUvspec:
                        p = multiprocessing.Process(target=self.worker, args=(tmp_inp,tmp_out,uvspecpath))
                        jobs.append(p)
                        p.start()
            for j in jobs:
                j.join()
        else:
            tmpoutputfiles=OldOutFiles

        if verbose:
            print('All processes done. Read output, convolve, average and calculate std.')
            sys.stdout.flush()


        if Wait:
            print("Waiting .....")
            sys.stdout.flush()
            time.sleep(60*3) # Sleep for 3 minutes to assure that files are put in right place

        if Convolve:
            finalrawoutputfiles=[]
            tmpfilestoaverage=[]
            if Instrument.Type=='Spectrometer':
                # Convolve with slit function if given.
                if verbose:
                    print('Convolving with slit function:', Instrument.slitfunction)
                    sys.stdout.flush()

                ip=0
                for tmpoutputfile in tmpoutputfiles:
                    ips = '{:d}'.format(ip)
                    rawoutputfile = inp.replace('.inp','.out_NP_'+ips+'.rad.spc')
                    print(tmpoutputfile, rawoutputfile)
                    sys.stdout.flush()
                    finalrawoutputfiles.append(rawoutputfile)
                    tmpoutconv='tmpoutconv_'+Instrument.RandString+'_'+ips
                    cmd = '/usr/bin/time -v '+self.uvspecpath+'conv '+tmpoutputfile+' '+Instrument.slitfunction+' > '+tmpoutconv+' 2> '+tmpoutconv+'.err'
                    if verbose:
                        print(cmd)
                        sys.stdout.flush()
                    p   = call(cmd,shell=True,stdin=PIPE,stdout=PIPE)
                    tmpoutspline='tmpoutspline_'+Instrument.RandString+'_'+ips
                    cmd = '/usr/bin/time -v '+self.uvspecpath+'spline '+'-q -l -b '+str(Instrument.wavelength1)+' -s '+str(Instrument.wavelengthstep)+' '+tmpoutconv+' > ' + tmpoutspline+' 2> '+tmpoutspline+'.err'
                    if verbose:
                        print(cmd)
                        sys.stdout.flush()
                    p   = call(cmd,shell=True,stdin=PIPE,stdout=PIPE)
                    tmpfilestoaverage.append(tmpoutspline)
                    # Copy MYSTIC output files to final destination
                    shutil.copy(tmpoutputfile,rawoutputfile)
                    ip=ip+1
            elif Instrument.Type=='Camera':
                nx = Instrument.h_pixels
                ny = Instrument.v_pixels
                tmpSplineXFile='tmpSplineXFile'+Instrument.RandString
                # Any output file should do to get wavelength information, there should be at least one.
                tmpdata = np.loadtxt(tmpoutputfiles[0])
                nwl = int(tmpdata.shape[0]/(nx*ny))
                tmpdata = np.reshape(tmpdata,(nwl,nx, ny, tmpdata.shape[1]))
                # Interpolate filter function to MYSTIC output wavelengths
                fx = open(tmpSplineXFile,'w')
                wvls = tmpdata[:,0,0,0]
                for wvl in wvls:
                    fx.write('{:f}\n'.format(wvl))
                fx.close()
                tmpSplineOutputFile='tmpSplineOutputFile'+Instrument.RandString
                cmd = '/usr/bin/time -v '+self.uvspecpath+'spline '+'-q -l -x '+tmpSplineXFile+' '+Instrument.filterfunction+' > ' + tmpSplineOutputFile+' 2> '+tmpSplineOutputFile+'.err'
                if verbose:
                    print(cmd)
                    sys.stdout.flush()
                p   = call(cmd,shell=True,stdin=PIPE,stdout=PIPE)
                tmpfilterfunctionwvl, tmpfilterfunction = np.loadtxt(tmpSplineOutputFile,unpack=True)

                ###
                # Include loop over all output files.
                ###
                ip=0
                for tmpoutputfile in tmpoutputfiles:
                    ips = '{:d}'.format(ip)
                    rawoutputfile = inp.replace('.inp','.out_NP_'+ips+'.rad.spc')
                    if verbose:
                        print("tmpoutputfile, rawoutputfile", tmpoutputfile, rawoutputfile)
                        sys.stdout.flush()
                    finalrawoutputfiles.append(rawoutputfile)

                    tmpdata = np.loadtxt(tmpoutputfile)
                    tmpdata = np.reshape(tmpdata,(nwl,nx, ny, tmpdata.shape[1]))

                    tmpoutputfilefilter = tmpoutputfile.replace('.out','.out_NP_'+ips+'.rad.spc')
                    if verbose:
                        print("tmpoutputfilefilter", tmpoutputfilefilter)
                    tmpfilestoaverage.append(tmpoutputfilefilter)
                    f= open(tmpoutputfilefilter,'w')
                    # For each pixel
                    ix=0
                    iz=0
                    while ix<nx:
                        iy=0
                        while iy<ny:
                            ## Multiply MYSTIC radiances with filter function
                            tmprad = tmpdata[:,ix,iy,4]*tmpfilterfunction
#                            tmpstd = tmpdata[:,ix,iy,5]*tmpfilterfunction
                            ## Integrate over wavelength
                            totrad = np.trapz(tmprad, x=wvls)
#                            totstd = np.trapz(tmpstd, x=wvls)
#                            f.write('{0:8.2f} {1:3d} {2:3d} {3:3d} {4:9.4f} {5:11.6f}\n'.format(wvls[0],ix,iy,iz,totrad,totstd))
                            f.write('{0:8.2f} {1:3d} {2:3d} {3:3d} {4:9.4f}\n'.format(wvls[0],ix,iy,iz,totrad))
                            iy=iy+1
                        ix=ix+1
                    f.flush()            # Do this to make sure all os written
                    os.fsync(f.fileno()) # to file before continuing.
                    f.close()
                    ip=ip+1

        else:
            tmpfilestoaverage=tmpoutputfiles
            finalrawoutputfiles=tmpoutputfiles

        InputFiles = tmpfilestoaverage #tmp_out_base+'NP_'+'*'+'.rad.spc'
        if n_processes==1:
            if verbose:
                print("InputFiles, OutputFileRaw", InputFiles, out, tmpoutputfiles)
                sys.stdout.flush()

            CombineSingleProcessOuput(tmpoutputfiles, out, verbose=True)
        else:
            if verbose:
                print("finalrawoutputfiles", finalrawoutputfiles)
                print("tmpoutputfiles", tmpoutputfiles)
                sys.stdout.flush()

            Average_spc_Files(InputFiles, out, verbose=True)
        return (tmpoutputfiles, finalrawoutputfiles)

    def SingleRun(self,inp, out, verbose=False, uvspecpath=''):
        if verbose:
            print("Running uvspec with input file: ", inp)
            print("Output to file                : ", out)
            sys.stdout.flush()

        uvspec='uvspec'

        cmd = '/usr/bin/time -v '+uvspecpath+uvspec+' < '+inp+' > '+out+' 2> '+out+'.err'
        if verbose:
            print(cmd)
            sys.stdout.flush()

        #(uvspec < uvspec.inp > uvspec.out) >& uvspec.err

        #FIXME
        p   = call(cmd,shell=True,stdin=PIPE,stdout=PIPE)
        return

    def removefiles(self, file):
        """
        file should have the format of 'mc_albedo_type'. So second column
        is files to be removed.
        """
        f = open(file)
        lines=f.readlines()

        for line in lines:
            itype, fnalb = line.split()
            if os.path.isfile(fnalb):
                os.remove(fnalb)
            else:
                # If it fails, inform the user.
                print("Error: %s file not found" % fnalb)

            
        f.close()
        
        return

        
    def worker(self, input,output, uvspecpath=''):
        """thread worker function"""
        verbose = True
        self.SingleRun(input,output,verbose=verbose, uvspecpath=uvspecpath)
        return

    def WriteInputFile(self, InputFile=None, verbose=False):
        if verbose:
            print("Writing uvspec input file", InputFile)
            sys.stdout.flush()

        try:
            f = open(InputFile,'w')
        except:
            print("Experiment.WriteRTFile: Not able to open uvspec input file.")
            exit()
        for key in self.inp:
            if verbose:
                sys.stdout.write( key + ' ' + str(self.inp[key]) + '\n')
            f.write( key + ' ' + str(self.inp[key]) + '\n')
        f.flush()     # Do this to make sure all input files are written
        os.fsync(f.fileno())    #  to file.
        f.close()
        return

def Average_spc_Files(InputFiles, OutputFile, verbose=False):
    # First check that all files have the same number of lines. If not
    # the files are surely different.
    i = 0
    #    for fn in glob(InputFiles):
    for fn in InputFiles:
        with open(fn) as fp:
            nlin = sum(1 for line in fp)
            if i==0:
                nlin0=nlin
            else:
                if nlin != nlin0:
                    print('nlin: ' + str(nlin) + ', not equal nlin0: ' + str(nlin0))
                    exit(0)
            i = i + 1

    # All well? Combine all the files
    wvl = np.zeros([len(InputFiles),nlin])
    ix  = np.zeros([len(InputFiles),nlin],dtype=int)
    iy  = np.zeros([len(InputFiles),nlin],dtype=int)
    iz  = np.zeros([len(InputFiles),nlin],dtype=int)
    rad = np.zeros([len(InputFiles),nlin])
    s2  = np.zeros([nlin])
    radavg = np.zeros([nlin])
    i = 0
#    for f in  glob(InputFiles):
    for f in  InputFiles:
        (wvl[i],ix[i],iy[i],iz[i],rad[i]) = read_rad_spc(f, verbose=False)
        radavg[:] = radavg[:] + rad[i,:]
        s2[:]     = s2[:] + rad[i,:]*rad[i,:]
        i = i + 1

    s0 = i
    l = 0
    f = open(OutputFile,'w')
    while l < nlin:
        s1        = radavg[l]
        arg       = s0*s2[l] - s1*s1
        if arg < 0.0:
            print(l, arg, s0, s1, s2[l], file=sys.stderr)
            arg = 0.0
        std       = (1.0/s0)*math.sqrt(arg)
        f.write('{0:8.2f} {1:3d} {2:3d} {3:3d} {4:9.4f} {5:9.4f}\n'.format(wvl[0,l], ix[0,l], iy[0,l], iz[0,l], s1/s0, std))
        l = l + 1
    f.close()
    return
    

def CombineSingleProcessOuput(InputFiles, OutputFile, verbose=False):
    fin = InputFiles[0]
    finstd = fin.replace('.spc','.std.spc')

    rad = np.loadtxt(fin)
    std = np.loadtxt(finstd)

    nwvl, ncol = rad.shape

    f = open(OutputFile,'w')

    iwvl=0
    while  iwvl < nwvl:
        f.write('{0:8.2f} {1:3d} {2:3d} {3:3d} {4:9.4f} {5:9.4f}\n'.format(rad[iwvl,0], int(rad[iwvl,1]),
                                                                           int(rad[iwvl,2]), int(rad[iwvl,3]),
                                                                           rad[iwvl,4], std[iwvl,4]))
        iwvl = iwvl + 1
    f.close()

    return

def read_rad_spc(fn, STD=False, verbose=False):
    # Read MYSTIC mc.rad.spc file
    if verbose:
        print("Reading MYSTIC mc.rad.spc file: ", fn)
        sys.stdout.flush()
    if STD:
        wvl,ix,iy,iz,rad, std = np.loadtxt(fn, unpack=True)
        return (wvl,ix,iy,iz,rad,std)
    else:
        wvl,ix,iy,iz,rad = np.loadtxt(fn, unpack=True)
        return (wvl,ix,iy,iz,rad)


def ReadRTOut(fn, verbose=False, mol_abs_param='kato2'):
    sza, zout, edir, edn, eup = np.loadtxt(fn, unpack=True)
    if verbose:
        print('ReadRTOut reading file:', fn )
    if 'reptran' in mol_abs_param:
        # reptran is in mW, conervt to W
        edir=edir/1000.
        edn=edn/1000.
    
    return (sza, zout, edir, edn, eup)

def ReadMCOut(fn, verbose=False, mol_abs_param='kato2'):
    fn = fn + '.flx.spc'
    if verbose:
        print('ReadMCOut reading file:', fn )
    wvl, ix, iy, iz, edir, edn, eup, tmp1, tmp2, tmp3 = np.loadtxt(fn, unpack=True)
    fn = fn.replace('.flx.spc','.flx.std.spc')
    wvl, ix, iy, iz, edirstd, ednstd, eupstd, tmp1, tmp2, tmp3 = np.loadtxt(fn, unpack=True)
    if 'reptran' in mol_abs_param:
        # reptran is in mW, conervt to W
        eup=eup/1000.
        eupstd=eupstd/1000.
    
    return (edir, edn, eup, edirstd, ednstd, eupstd)

    
#######################################################################




if __name__ == "__main__":

    1
