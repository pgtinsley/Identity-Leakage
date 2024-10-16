#!/bin/bash
#$ -q gpu@@cvrl_gpu
#$ -l gpu_card=2
#$ -l h=qa-rtx6k-044
#$ -N sg2-train-fprint-rtx

####################### Set Up ###############################

cd /project01/cvrl/ptinsley/mitigation_clean/stylegan3-fun/

module load cuda/11.6
nvcc --version

export PATH=/usr/local/cuda-11.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH

source ~/miniconda3/bin/activate stylegan3
# source ~/miniconda3/bin/activate stylegan3-a6k
conda env list

module load gcc/

export CC=/opt/crc/g/gcc/8.3.0/bin/gcc
export CXX=/opt/crc/g/gcc/8.3.0/bin/g++


######################################################################


### 256x256
# --cfg=stylegan2 --gpus=2 --batch=32 --gamma=0.4096 --map-depth=2 --glr=0.0025 --dlr=0.0025 --cbase=16384
# --cfg=stylegan2 --gpus=4 --batch=64 --gamma=0.2048 --map-depth=2 --glr=0.0025 --dlr=0.0025 --cbase=16384

### 512x512
# --cfg=stylegan2 --gpus=2 --batch=16 --gamma=3.2768 --map-depth=2 --glr=0.0025 --dlr=0.0025
# --cfg=stylegan3-t --gpus=2 --batch=32 --gamma=8 --batch-gpu=8 --snap=20
# --cfg=stylegan3-t --gpus=4 --batch=32 --gamma=8
# --cfg=stylegan3-r --gpus=2 --batch=32 --gamma=8 --batch-gpu=4 --snap=20
# --cfg=stylegan3-r --gpus=4 --batch=32 --gamma=8 --batch-gpu=4 --snap=20

# --cfg=stylegan2 --gpus=2 --batch=32 --gamma=0.4096 --map-depth=2 --glr=0.0025 --dlr=0.0025 --cbase=16384 \
# --cfg=stylegan2 --gpus=4 --batch=32 --gamma=1.6384 --map-depth=2 --glr=0.0025 --dlr=0.0025


##### TRAIN #####


python train.py --data=/project01/cvrl/ptinsley/mitigation_clean/images/fprint/authentic/samples_seen_during_training.zip \
                --outdir=/project01/cvrl/ptinsley/mitigation_clean/stylegan3-fun/training-runs/ \
                --cfg=stylegan2 --gpus=2 --batch=16 --gamma=3.2768 --map-depth=2 --glr=0.0025 --dlr=0.0025 \
                --mirror=1 --aug=noaug \
                --metrics=fid50k_full \
                --snap=20
