#!/bin/bash
#SBATCH --job-name=adversarial_defence
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8                         # Allocate 16 cores per task
#SBATCH --mem=32G
#SBATCH --time 24:00:00
#SBATCH --gres=gpu:ampere:1
#SBATCH --constraint=a40
#SBATCH --mail-type=BEGIN,END,FAIL                  # Email status changes
#SBATCH --mail-user=hohoangphuoc@student.utwente.nl  # Your email address

# load anaconda3
# module load anaconda3/2019.03

# load cuda
module purge # clean the environment before loading new modules
module load nvidia/cuda-12.4
module load nvidia/nvtop

module list


#============================================================
#                       PRINT INFO
#============================================================

# Print some useful information
echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)" # log hostname
echo "Working Directory = $(pwd)"
echo "Name of nodes used          : "$SLURM_JOB_NODELIST
echo "Gpu devices                 : "$CUDA_VISIBLE_DEVICES
echo "Starting worker: "

echo "Number of CPU cores = $(nproc)"
echo "SLURM_CPUS_PER_TASK:          "$SLURM_CPUS_PER_TASK

# Print GPU info
echo "Number of GPUs:               "$CUDA_VISIBLE_DEVICES
echo "CUDA Home:                    "$CUDA_HOME
echo "CUDA Directory:               "$CUDA_DIR
echo "LD LIBRARY Path:              "$LD_LIBRARY_PATH

#============================================================

# activate environment
source activate .venv

python load_adversarial.py
#================================================================
