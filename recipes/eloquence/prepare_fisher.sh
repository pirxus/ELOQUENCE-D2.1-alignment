#!/bin/bash
#$ -N spokenwoz_prep
#$ -q all.q@@blade
#$ -l ram_free=2G,mem_free=2G
#$ -l matylda5=0.05,matylda2=0.05,matylda3=0.05,scratch=1,matylda6=0.1
#$ -pe smp 16
#$ -o /mnt/matylda6/isedlacek/projects/job_logs/fisher_prep_2.o
#$ -e /mnt/matylda6/isedlacek/projects/job_logs/fisher_prep_2.e
N_GPUS=1
EXPERIMENT="prepare_fisher_2"

# Job should finish in about 2 days
ulimit -t 200000

# Enable opening multiple files
ulimit -n 4096

# Enable to save bigger checkpoints
ulimit -f unlimited
ulimit -v unlimited
ulimit -u 4096

# Initialize environment
unset PYTHONPATH
unset PYTHONHOME
source /mnt/matylda6/isedlacek/miniconda3/bin/activate /mnt/matylda6/isedlacek/envs/huggingface_asr

WORK_DIR="/mnt/matylda6/isedlacek/projects/huggingface_asr"
EXPERIMENT_PATH="${WORK_DIR}/exp/${EXPERIMENT}"
RECIPE_DIR="${WORK_DIR}/recipes/eloquence"
DATASETS="${RECIPE_DIR}/datasets_fisher.json"

cd $WORK_DIR || {
  echo "No such directory $WORK_DIR"
  exit 1
}

# set pythonpath so that python works
export PYTHONPATH="${PYTHONPATH}:${WORK_DIR}/src"

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME="/mnt/matylda6/isedlacek/hugging-face"

# get the gpu
export CUDA_VISIBLE_DEVICES=$(free-gpus.sh $N_GPUS) || {
  echo "Could not obtain GPU."
  exit 1
}
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

args=(
  # General training arguments
  --output_dir=$EXPERIMENT_PATH

  # Data related arguments
  --max_duration_in_seconds="30.0"
  --min_duration_in_seconds="0.2"
  --preprocessing_num_workers="16"
  --writer_batch_size="200" # 1000
  --dataset_name="/mnt/matylda6/isedlacek/projects/huggingface_asr/src/dataset_builders/kaldi_dataset/"
  --text_column_name="labels"
  #--text_transformations whisper_normalize_english clean_special_tokens_english_train
  --data_dir="/mnt/matylda5/ipoloka/projects/huggingface_asr/metadata_dirs/fisher_swbd"
  --validation_split test
)

#echo "Running prepareing.."
#if [ "$N_GPUS" -gt 1 ]; then
#  torchrun --standalone --nnodes=1 --nproc-per-node=$N_GPUS ${RECIPE_DIR}/prepare_fisher.py "${args[@]}"
#else
#  python ${RECIPE_DIR}/prepare_fisher.py "${args[@]}"
#fi

python ${RECIPE_DIR}/prepare_fisher.py "${args[@]}"
