#!/bin/bash
#SBATCH --job-name=LMAO
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=vhchu@princeton.edu
#SBATCH --mem=128G

module purge
module load anaconda3/2021.5
conda activate cos597r

python -u lmao.py