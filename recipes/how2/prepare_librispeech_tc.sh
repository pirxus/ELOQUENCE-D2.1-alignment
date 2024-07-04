#!/bin/bash
#$ -N libri_tc
#$ -q all.q@supergpu*
#$ -l ram_free=20G,mem_free=20G
#$ -l matylda6=5
#$ -l scratch=2
#$ -l gpu=1,gpu_ram=20G
#$ -o /mnt/matylda6/xsedla1h/projects/job_logs/libri_tc.o
#$ -e /mnt/matylda6/xsedla1h/projects/job_logs/libri_tc.e

ulimit -t 150000

# Enable opening multiple files
ulimit -n 4096

# Enable to save bigger checkpoints
ulimit -f unlimited

# Enable more threads per process by increasing virtual memory (https://stackoverflow.com/questions/344203/maximum-number-of-threads-per-process-in-linux)
ulimit -v unlimited
ulimit -u 4096

# Initialize environment
source /mnt/matylda6/xsedla1h/miniconda3/bin/activate /mnt/matylda6/xsedla1h/envs/huggingface_asr

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME="/mnt/matylda6/xsedla1h/hugging-face"


# get the gpu
export CUDA_VISIBLE_DEVICES=$(free-gpus.sh 1) || {
  echo "Could not obtain GPU."
  exit 1
}

cd /mnt/matylda6/xsedla1h/projects/utils

echo "Preparing the dataset..."
python librispeech_tc.py
