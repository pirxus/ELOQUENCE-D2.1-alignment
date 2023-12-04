#!/usr/bin/bash
#SBATCH --job-name TED
#SBATCH --account OPEN-28-57
#SBATCH --partition qcpu
#SBATCH --nodes 1
#SBATCH --time 24:00:00
#SBATCH --output=/mnt/proj1/open-28-58/lakoc/huggingface_asr/outputs/ebranchformer_joined_english_preprocess.out

EXPERIMENT="ebranchformer_joined_english_preprocess"
PROJECT="regularizations_english_corpus"
WORK_DIR="/mnt/proj1/open-28-58/lakoc/huggingface_asr"
ENV_DIR="/mnt/proj1/open-28-58/lakoc/LoCo-ASR"
RECIPE_DIR="${WORK_DIR}/recipes/ebranchformer_joined_english"

export HF_HOME="${ENV_DIR}/huggingface_cache"
export PYTHONPATH="${PYTHONPATH}:${WORK_DIR}/src"
export OMP_NUM_THREADS=128

conda deactivate
source activate loco_asr

EXPERIMENT_PATH="${WORK_DIR}/experiments/${EXPERIMENT}"

cd $WORK_DIR

python src/trainers/train_enc_dec_asr.py \
  --max_duration_in_seconds="20.0" \
  --min_duration_in_seconds="0.0" \
  --feature_extractor_name="Lakoc/log_80mel_extractor_16k" \
  --tokenizer_name="Lakoc/ted_uni500" \
  --output_dir=$EXPERIMENT_PATH \
  --length_column_name="input_len" \
  --preprocessing_num_workers="64" \
  --fix_apostrophes \
  --remove_train_unks \
  --data_preprocessing_config="${RECIPE_DIR}/data_preprocessing.json" \
  --datasets_creation_config="${RECIPE_DIR}/datasets.json" \
  --writer_batch_size="1000" \
  --preprocess_dataset_only
