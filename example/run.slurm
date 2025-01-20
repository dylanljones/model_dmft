#!/bin/bash
#SBATCH --job-name=FTPS_CPA+DMFT
#SBATCH --mail-type=FAIL,INVALID_DEPEND,TIME_LIMIT
#SBATCH --mail-user=dylan.jones@uni-a.de
#SBATCH --partition=epyc
#SBATCH --time=1-0
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=2gb
#SBATCH --cpus-per-task=1

# Initialize modules and activate conda environment
module load anaconda
module load gcc openmpi/4.1
conda activate triqs

# Limit Numpy threads
export N_TASKS=$SLURM_NTASKS
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run the script
python -m model_dmft run inp.toml -n $SLURM_NTASKS
