#!/usr/bin/bash


source $HOME/.bashrc  # for bash 4.2 (CentOS 7)
source $HOME/depends/anaconda3/etc/profile.d/conda.sh  # for bash 5.1 (Ubuntu 22.04)
conda activate
cd $HOME/projects/IternVL/clip_benchmark/

GPU_DEVICES=${0:-0}
MODEL_TYPE=${1:-"internvl"}
LANGUAGE=${2:-"en"}
TASK=${3:-"zeroshot_retrieval"}
DATASET=${4:-"flickr30k"}
DATASET_ROOT=${5:-"${HOME}/datasets/flickr30k"}
MODEL=${6:-"internvl_g_retrieval_hf"}
PRETRAINED=${7:-"${HOME}/ckpts/InternVL-14B-224px"}
OUTPUT=${8:-"result_g.json"}

export CUDA_VISIBLE_DEVICES=${GPU_DEVICES}
python ./clip_benchmark/cli.py eval --model_type ${MODEL_TYPE} \
                                    --language ${LANGUAGE} \
                                    --task ${TASK} \
                                    --dataset ${DATASET} \
                                    --dataset_root ${DATASET_ROOT} \
                                    --model ${MODEL} \
                                    --pretrained ${PRETRAINED} \
                                    --output ${OUTPUT}
