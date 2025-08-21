#!/bin/bash
#SBATCH --job-name=gpt2_prep_data
#SBATCH --output=/gpfs/gibbs/pi/krishnaswamy_smita/arv24/gpt_midtraining/logs/gpt2_prep_data_%j.out
#SBATCH --error=/gpfs/gibbs/pi/krishnaswamy_smita/arv24/gpt_midtraining/logs/gpt2_prep_data_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00


cd /gpfs/gibbs/pi/krishnaswamy_smita/arv24/gpt_midtraining/nanoGPT

module load miniconda
conda activate gpt_env


python data/wikitext/prepare.py \
