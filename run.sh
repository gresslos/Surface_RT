#!/bin/bash
###SBATCH --output=output/job1.log
###SBATCH --error=output/err1.log
#SBATCH --output=output/job_%j.log
#SBATCH --error=output/err_%j.log
#SBATCH --verbose
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --ntasks-per-node=40
#SBATCH --time=4-10:00:00
#SBATCH --mem-per-cpu=1400Mb
#SBATCH --partition=main
#SBATCH --export=ALL




# Load modules and python env
module load gcc openmpi netcdf-c gsl gdal
source /xnilu_wrk/users/eso/NEVAR/env/bin/activate

# Define cleanup actions early in the script, i.e. copy files from scratch directory to permanent directory
# cleanup "cp -R $SCRATCH/RESULTS/ $SUBMITDIR/"
# Test to only copy .nc file!
cleanup "cp -R $SCRATCH/RESULTS/*nc $SUBMITDIR/RESULTS/"

echo "Running at $SCRATCH"
echo "Submitdir is $SUBMITDIR"
echo "Number of nodes: $SLURM_NNODES"
echo "Number of tasks: $SLURM_NTASKS"

# Copy inputs to SCRATCH disk -- note: should do a selection based on scene input
mkdir $SCRATCH/RESULTS
mkdir $SCRATCH/tmpRTIO/ 
                    
echo "Making SLURM ready at $(date)"      


                    # mkdir $SCRATCH/Frames_SurfaceOverpasses/

                    # echo "Start copy Frames: $(date)"
                    # cp -R /homevip/bgre/Download/Frames_SurfaceOverpasses/* $SCRATCH/Frames_SurfaceOverpasses/
                    # echo "End copy Frames: $(date)"



# cp $SUBMITDIR/MakeRTMInputFile_bg.py $SCRATCH
# cp $SUBMITDIR/ReadEarthCAREL2_bg.py $SCRATCH
# cp $SUBMITDIR/*.py $SCRATCH

cp "$SUBMITDIR/Make_RTM.py" "$SCRATCH"
cp "$SUBMITDIR/Read_RTM.py" "$SCRATCH"
cp "$SUBMITDIR/UVspec.py" "$SCRATCH"
cp "$SUBMITDIR/Find_Overpass_Info.py" "$SCRATCH" 


# Go to scratch dir and launch program
cd $SCRATCH

echo "Starting slurm job at " `date`

# Define your own result path
export HOME_RTM="/homevip/bgre/publication_1"
mkdir -p $HOME_RTM/RESULTS

START_TIME=$(date +%s)

echo "Running Python file at $(date)"
srun --unbuffered --mpi=pmix python ./Make_RTM.py
echo "Finished running Python file at $(date)"

# echo "$SCRATCH/EarthCARE_Real/"
# Delete large EarthCARE_REAL files
# rm -r $SCRATCH/Frames_SurfaceOverpasses/


END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
printf "Runtime: %02d:%02d:%02d\n" \
    $((ELAPSED/3600)) \
    $(((ELAPSED%3600)/60)) \
    $((ELAPSED%60))

