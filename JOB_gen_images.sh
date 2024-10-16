#!/bin/bash
#$ -q gpu@@cvrl_gpu
#$ -l gpu_card=1
#$ -l h=!qa-1080ti-01*
#$ -N gen-images-S0_full_fprint
#$ -t 1-10

####################### Set Up ###############################

cd /project01/cvrl/ptinsley/mitigation_clean/stylegan3-fun/

module load cuda/11.6
nvcc --version

export PATH=/usr/local/cuda-11.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH

source ~/miniconda3/bin/activate stylegan3
conda env list

module load gcc/

export CC=/opt/crc/g/gcc/8.3.0/bin/gcc
export CXX=/opt/crc/g/gcc/8.3.0/bin/g++

######################################################################

##### GENERATE IMAGES #####


#python gen_images_sameSeeds_diffTrunc.py    --network=./training-runs/00008-stylegan2-samples2-gpus2-batch16-gamma3.2768-no_resume/network-snapshot-002000.pkl \
#                                            --outdir=/project01/cvrl/ptinsley/mitigation_clean/images/fprint/synthetic/S0_test/ \
#                                            --seeds=0-999 \
#                                            --idx=$SGE_TASK_ID \
#                                            --color=0


python gen_images_sameTrunc_diffSeeds.py    --network=./training-runs/00008-stylegan2-samples2-gpus2-batch16-gamma3.2768-no_resume/network-snapshot-002000.pkl \
                                            --outdir=/project01/cvrl/ptinsley/mitigation_clean/images/fprint/synthetic/S0_full/ \
                                            --trunc=1.2 \
                                            --seeds=0-49999 \
                                            --idx=$SGE_TASK_ID \
                                            --color=0
