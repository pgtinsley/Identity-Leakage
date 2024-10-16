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


#python gen_images_sameTrunc_diffSeeds.py    --network=./training-runs/00004-stylegan2-training_data_2-gpus4-batch64-gamma0.2048/network-snapshot-004032.pkl \
#                                            --outdir=/project01/cvrl/ptinsley/mitigation_clean/images/iris/synthetic/S0_full/ \
#                                            --trunc=1.0 \
#                                            --seeds=0-24999 \
#                                            --idx=$SGE_TASK_ID \
#                                            --color=0


### CELEBAHQ-BASED

### S0_TEST
#python gen_images_sameSeeds_diffTrunc.py    --network=./network-snapshot-025000-celebahq.pkl \
#                                            --outdir=/project01/cvrl/ptinsley/CelebAHQ_test/ \
#                                            --seeds=0-999 \
#                                            --idx=$SGE_TASK_ID \
#                                            --color=1

### FFHQ-BASED

### S1_TEST (no_resume)
#python gen_images_sameSeeds_diffTrunc.py    --network=./training-runs/00003-stylegan2-S0_training-gpus4-batch64-gamma0.2048-no_resume/network-snapshot-005644.pkl \
#                                            --outdir=/project01/cvrl/ptinsley/face3/S1_test_no_resume \
#                                            --seeds=0-999 \
#                                            --idx=$SGE_TASK_ID \
#                                            --color=1


### S1_TEST (resume from CelebA-HQ net)
#python gen_images_sameSeeds_diffTrunc.py    --network=./training-runs/00003-stylegan2-S0_training-gpus4-batch64-gamma0.2048-no_resume/network-snapshot-005644.pkl \
#                                            --outdir=/project01/cvrl/ptinsley/face3/S1_test_resume \
#                                            --seeds=0-999 \
#                                            --idx=$SGE_TASK_ID \
#                                            --color=1

### S0_FULL

#python gen_images_sameTrunc_diffSeeds.py    --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-256x256.pkl \
#                                            --outdir=/project01/cvrl/ptinsley/face3/S0_full/ \
#                                            --trunc=0.75 \
#                                            --seeds=0-299999 \
#                                            --idx=$SGE_TASK_ID \
#                                            --color=1

### FACE ###

### S0_FULL

#python gen_images_sameTrunc_diffSeeds.py    --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-256x256.pkl \
#                                            --outdir=/project01/cvrl/ptinsley/S0_full/ \
#                                            --trunc=1.0 \
#                                            --seeds=0-299999 \
#                                            --idx=$SGE_TASK_ID \
#                                            --color=1

### S1_TEST

#python gen_images_sameSeeds_diffTrunc.py    --network=./training-runs/00002-stylegan2-S0_training-gpus4-batch64-gamma0.2048-no_resume/network-snapshot-012015.pkl \
#                                            --outdir=/project01/cvrl/ptinsley/S1_test/ \
#                                            --seeds=0-999 \
#                                            --idx=$SGE_TASK_ID \
#                                            --color=1

### S1_FULL

#python gen_images_sameTrunc_diffSeeds.py    --network=./training-runs/00002-stylegan2-S0_training-gpus4-batch64-gamma0.2048-no_resume/network-snapshot-012015.pkl \
#                                            --outdir=/project01/cvrl/ptinsley/S1_full/ \
#                                            --trunc=1.0 \
#                                            --seeds=0-299999 \
#                                            --idx=$SGE_TASK_ID \
#                                            --color=1

### IRIS ###

### S0_TEST
#python gen_images_sameSeeds_diffTrunc.py    --network=/project01/cvrl/ptinsley/mitigation_clean/stylegan3-fun/training-runs/00004-stylegan2-training_data_2-gpus4-batch64-gamma0.2048/network-snapshot-025000.pkl \
#                                            --outdir=/project01/cvrl/ptinsley/mitigation_clean/images/iris/synthetic/S0_test/ \
#                                            --seeds=500-999 \
#                                            --idx=$SGE_TASK_ID \
#                                            --color=0
                                            
### S0_FULL
#python gen_images_sameTrunc_diffSeeds.py    --network=/project01/cvrl/ptinsley/mitigation_clean/stylegan3-fun/training-runs/00004-stylegan2-training_data_2-gpus4-batch64-gamma0.2048/network-snapshot-025000.pkl \
#                                            --outdir=/project01/cvrl/ptinsley/mitigation_clean/images/iris/synthetic/S0_full/ \
#                                            --trunc=0.75 \
#                                            --seeds=0-30000 \
#                                            --idx=$SGE_TASK_ID \
#                                            --color=0

