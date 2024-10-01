#!/bin/bash -l
#SBATCH --job-name="SKA_visibilities_comp"
#SBATCH --account="sk024"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vito.rajkovic@gmail.com
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CRAY_CUDA_MPS=1
export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64:$LD_LIBRARY_PATH
echo "LD_LIBRARY_PATH is set to: $LD_LIBRARY_PATH"

source ../../par_comp_env/bin/activate
srun python -u bare_estimator_script.py