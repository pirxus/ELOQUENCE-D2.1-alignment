#!/bin/bash
#$ -N s2t_short_test_run
#$ -q short.q@@gpu
#$ -l ram_free=40G,mem_free=40G
#$ -l matylda6=10
#$ -l gpu=1,gpu_ram=16G
#$ -o /mnt/matylda6/xsedla1h/projects/job_logs/s2t_training/s2t_short_test_run.o
#$ -e /mnt/matylda6/xsedla1h/projects/job_logs/s2t_training/s2t_short_test_run.e
#

# Job should finish in 2 days - 172800 seconds
#ulimit -t 172800

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

EXPERIMENT="s2t_asr_training_test"
WORK_DIR="/mnt/matylda6/xsedla1h/projects/huggingface_asr"
EXPERIMENT_PATH="${WORK_DIR}/exp/s2t_test_exp_dir"
RECIPE_DIR="${WORK_DIR}/recipes/how2"
HOW2_PATH="/mnt/matylda6/xsedla1h/data/how2"

cd $WORK_DIR || {
  echo "No such directory $WORK_DIR"
  exit 1
}

# set pythonpath so that python works
export PYTHONPATH="${PYTHONPATH}:${WORK_DIR}/src"

# get the gpu
export CUDA_VISIBLE_DEVICES=$(free_gpus.sh 1) || {
  echo "Could not obtain GPU."
  exit 1
}

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export WANDB_MODE=offline
export WANDB_RUN_ID=$EXPERIMENT
export WANDB_PROJECT="S2T-asr"


args=(
  # General training arguments
  --output_dir=$EXPERIMENT_PATH
  --per_device_train_batch_size="64" # 64
  --per_device_eval_batch_size="64"
  --dataloader_num_workers="4"
  --num_train_epochs="1"
  --group_by_length="True"
  --bf16 # FIXME
  --do_train
  --do_evaluate
  --load_best_model_at_end

  # Optimizer related arguments
  --optim="adamw_torch"
  --learning_rate="2e-3"
  --warmup_steps="20000"
  --early_stopping_patience="5"
  --weight_decay="1e-6"
  --max_grad_norm="5.0"
  --lsm_factor="0.1"
  --gradient_accumulation_steps="1"

  # Logging, saving and evaluation related arguments
  --report_to="wandb"
  --logging_steps="10"
  --save_strategy="epoch"
  --evaluation_strategy="epoch"
  --wandb_predictions_to_save=50 # 60
  --greater_is_better="False"
  --save_total_limit="5"

  # Data related arguments
  --max_duration_in_seconds="20.0"
  --min_duration_in_seconds="0.2"
  --remove_unused_columns="False"
  --preprocessing_num_workers="4"
  --dataset_name="${HOW2_PATH}"
  #--dataset_name="/home/pirx/Devel/masters/APMo-SLT/src/huggingface_asr/src/dataset_builders/how2_dataset"
  #--data_dir="/home/pirx/Devel/masters/data/how2"
  --writer_batch_size="200" # 1000
  --validation_split val
  --test_splits val dev5

  --collator_rename_features="False"
  --text_column_name="transcription"
  --do_lower_case
  --remove_listed_chars=",:;()"

  # Preprocessing related arguments
  --data_preprocessing_config="${RECIPE_DIR}/data_preprocessing.json"

  # Model related arguments
  --tokenizer_name="pirxus/how2_en_unigram5000_filtered"
  --feature_extractor_name="facebook/s2t-small-librispeech-asr"

  # Generation related arguments
  --num_beams="4"
  --max_length="200"
  --predict_with_generate
  --decoding_ctc_weight="0.0"
  --eval_beam_factor="10"
)

python src/trainers/train_speech2text.py "${args[@]}"
