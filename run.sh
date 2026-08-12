#!/bin/bash
###SBATCH --output=output/job1.log
###SBATCH --error=output/err1.log
#SBATCH --output=output/job_%j.log
#SBATCH --error=output/err_%j.log
#SBATCH --verbose
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=54
#SBATCH --ntasks-per-node=54


###SBATCH --cpus-per-task=1
#SBATCH --time=4-10:00:00
#SBATCH --mem-per-cpu=1400Mb

#SBATCH --partition=main





# Load modules and python env
module load gcc openmpi netcdf-c gsl gdal
source /xnilu_wrk/users/eso/NEVAR/env/bin/activate

# Define cleanup actions early in the script, i.e. copy files from scratch directory to permanent directory
# cleanup "cp -R $SCRATCH/RESULTS/ $SUBMITDIR/"
# Test to only copy .nc file!
cleanup "cp -R $SCRATCH/RESULTS/*nc $SUBMITDIR/RESULTS/"

echo "Running at $SCRATCH"
echo "Submitdir is $SUBMITDIR"

# --- ADDED LINES TO PRINT NODE AND TASK INFO ---
echo "Number of nodes: $SLURM_NNODES"
echo "Number of tasks: $SLURM_NTASKS"
# -----------------------------------------------

# Copy inputs to SCRATCH disk -- note: should do a selection based on scene input
mkdir $SCRATCH/RESULTS
mkdir $SCRATCH/EarthCARE_Real/
mkdir $SCRATCH/tmpRTIO/ 



echo "Making SLURM ready at  $(date)"
cp -R /xnilu_wrk2/projects/NEVAR/data/EarthCARE_Real/* $SCRATCH/EarthCARE_Real/




# cp $SUBMITDIR/MakeRTMInputFile_bg.py $SCRATCH
# cp $SUBMITDIR/ReadEarthCAREL2_bg.py $SCRATCH
cp $SUBMITDIR/*.py $SCRATCH 




# Go to scratch dir and launch program
cd $SCRATCH

echo "Starting slurm job at " `date`


# Define your own result path
export HOME_RTM="/homevip/bgre/SurfaceRTM"
mkdir -p $HOME_RTM/RESULTS

echo "Running Python file at $(date)"
srun --unbuffered --mpi=pmix python ./MakeRTM_Surface_Version.py
# srun --unbuffered --mpi=pmix python ./AllLevelsMakeRTM.py
# srun --unbuffered --mpi=pmix python ./MakeRTMInputFile_bg.py
# srun --unbuffered --mpi=pmix python ./all_levels_MakeRTMInputFile_bg.py
echo "Finished running Python file at $(date)"

# echo "$SCRATCH/EarthCARE_Real/"
# Delete large EarthCARE_REAL files
rm -r $SCRATCH/EarthCARE_Real/


