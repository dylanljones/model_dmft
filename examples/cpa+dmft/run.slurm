#!/bin/bash
#SBATCH --job-name=CPA+DMFT
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=dylan.jones@uni-a.de
#SBATCH --partition=epyc
#SBATCH --time=7-0
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --mem-per-cpu=512mb
#SBATCH --threads-per-core=1
#SBATCH --export=ALL

# Initialize modules and activate conda environment
module purge
module load anaconda
module load gcc openmpi/5
conda activate triqs3.3

# Limit Numpy threads
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Environment variables for parallel processes
export USE_SRUN=0

# Run the script
python -m model_dmft run inp.toml -n $SLURM_NTASKS
