#!/bin/bash
#$ -N how2_preprocess
#$ -q short.q@@blade
#$ -l ram_free=10G,mem_free=10G
#$ -l matylda6=0.1
#$ -l scratch=5
#$ -pe smp 8
#$ -o /mnt/matylda6/xsedla1h/projects/job_logs/how2_preprocess.o
#$ -e /mnt/matylda6/xsedla1h/projects/job_logs/how2_preprocess.e
#

# Job should finish in 1 days - 86400 seconds
ulimit -t 86400

# Enable opening multiple files
ulimit -n 4096

# Enable to save bigger checkpoints
ulimit -f unlimited

# Enable more threads per process by increasing virtual memory (https://stackoverflow.com/questions/344203/maximum-number-of-threads-per-process-in-linux)
ulimit -v unlimited

ulimit -u 4096

# Initialize environment
unset PYTHONPATH
unset PYTHONHOME
source /mnt/matylda6/xsedla1h/miniconda3/bin/activate /mnt/matylda6/xsedla1h/envs/huggingface_asr

EXPERIMENT="how2_preprocess"

WORK_DIR="/mnt/matylda6/xsedla1h/projects/huggingface_asr"
EXPERIMENT_PATH="${WORK_DIR}/exp/${EXPERIMENT}"
RECIPE_DIR="${WORK_DIR}/recipes/how2"
HOW2_PATH="/mnt/scratch/tmp/xsedla1h/how2"

cd $WORK_DIR || {
  echo "No such directory $WORK_DIR"
  exit 1
}

# set pythonpath so that python works
export PYTHONPATH="${PYTHONPATH}:${WORK_DIR}/src"

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME="/mnt/matylda6/xsedla1h/hugging-face"


args=(
  # General training arguments
  --output_dir=$EXPERIMENT_PATH
  --preprocess_dataset_only

  # Data related arguments
  --max_duration_in_seconds="20.0"
  --min_duration_in_seconds="0.2"
  --remove_unused_columns="False"
  --preprocessing_num_workers="8"
  --dataset_name="${HOW2_PATH}"
  --dataset_config="text_only"
  --writer_batch_size="200" # 1000
  --text_column_name="translation"
  --validation_split val
  #--do_lower_case
  #--lcrm
)

python src/trainers/train_enc_dec_asr.py "${args[@]}"
