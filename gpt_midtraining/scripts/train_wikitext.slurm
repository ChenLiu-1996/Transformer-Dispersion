#!/bin/bash
#SBATCH --job-name=gpt2_dev_1step
#SBATCH --output=/gpfs/gibbs/pi/krishnaswamy_smita/arv24/gpt_midtraining/logs/gpt2_train.out
#SBATCH --error=/gpfs/gibbs/pi/krishnaswamy_smita/arv24/gpt_midtraining/logs/gpt2_train.err
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

cd /gpfs/gibbs/pi/krishnaswamy_smita/arv24/gpt_midtraining/nanoGPT


module --force purge
module load miniconda
module load CUDAcore/11.1.1 cuDNN/8.0.5.39-CUDA-11.1.1
module load GCC/10.2.0
module load git/2.28.0-GCCcore-10.2.0-nodocs
conda activate transformer

nvidia-smi

python train.py config/train_gpt2.py