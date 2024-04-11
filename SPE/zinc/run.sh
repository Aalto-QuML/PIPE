#!/bin/bash
#SBATCH --job-name=TopNNs
#SBATCH --time=01-23:14:00
#SBATCH --account=project_2006852
#SBATCH --mem-per-cpu=15G
#SBATCH --output=Train_spe_zinc_standard.out
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=1


python -u runner.py --config_dirpath ../configs/zinc --config_name SPE_gine_gin_mlp_pe37.yaml --seed 0
