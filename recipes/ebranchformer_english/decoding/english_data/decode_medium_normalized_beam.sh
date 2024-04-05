#!/usr/bin/bash
#SBATCH --job-name TED
#SBATCH --account OPEN-28-57
#SBATCH --partition qgpu_exp
#SBATCH --time 01:00:00
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --output=/mnt/proj1/open-28-58/lakoc/huggingface_asr/outputs/english_model_medium_normalized_beam_decode_no_filer.out

EXPERIMENT="english_model_medium_normalized_beam_decode_no_filer"
PROJECT="regularizations_english_corpus"
WORK_DIR="/mnt/proj1/open-28-58/lakoc/huggingface_asr"
RECIPE_DIR="${WORK_DIR}/recipes/ebranchformer_english"
EXPERIMENT_PATH="${WORK_DIR}/experiments/${EXPERIMENT}"
HF_HOME="/scratch/project/open-28-57/lakoc/huggingface_cache"


export HF_HOME="/scratch/project/open-28-57/lakoc/huggingface_cache"
export PYTHONPATH="${PYTHONPATH}:${WORK_DIR}/src"
export OMP_NUM_THREADS=64
export WANDB_PROJECT=$PROJECT
export WANDB_RUN_ID="${EXPERIMENT}"

conda deactivate
source activate loco_asr

EXPERIMENT_PATH="${WORK_DIR}/experiments/${EXPERIMENT}"

cd $WORK_DIR

args=(
  # General training arguments
  --output_dir=$EXPERIMENT_PATH
  --per_device_eval_batch_size="16"
  --dataloader_num_workers="24"
  --do_evaluate

  # Data related arguments
  --max_duration_in_seconds="20.0"
  --min_duration_in_seconds="0.2"
  --length_column_name="input_len"
  --remove_unused_columns="False"
  --preprocessing_num_workers="32"
  --dataset_name="/scratch/project/open-28-57/lakoc/processed_dataset_full"
  --writer_batch_size="500"
  --test_splits commonvoice_en_test fleurs_test

  # Preprocessing related arguments
  --data_preprocessing_config="${RECIPE_DIR}/data_preprocessing.json"

  # Model related arguments
  --tokenizer_name="Lakoc/english_corpus_uni5000_normalized"
  --feature_extractor_name="Lakoc/log_80mel_extractor_16k"
  --from_pretrained="/mnt/proj1/open-28-58/lakoc/huggingface_asr/experiments/ebranchformer_english_medium_normalized/checkpoint-238400"
  --expect_2d_input

  # Generation related arguments
  --filter_empty_labels
  --num_beams="10"
  --max_length="512"
  --predict_with_generate
  --decoding_ctc_weight="0"
)

torchrun --standalone --nnodes=1 --nproc-per-node=$SLURM_GPUS_ON_NODE src/trainers/train_enc_dec_asr.py "${args[@]}"