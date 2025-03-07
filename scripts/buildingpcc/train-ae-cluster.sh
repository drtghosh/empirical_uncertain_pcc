#!/bin/sh
#SBATCH --cpus-per-task=6
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-cluster
#SBATCH --output=train_pcc.out
HOME=/clusterstorage/dghosh
source $HOME/.bashrc
conda init
conda activate deb
python train.py -l proj_log_ae \
                -x buildingpcc_ae \
                -m ae \
                -n buildingpcc \
                -r data/BuildingPCC \
                -f BuildingPCC.json \
                -b 200 \
                --lr 5e-4 \
                --save_frequency 500 \
                -e 2000 \
		            --vis\
                --cont
