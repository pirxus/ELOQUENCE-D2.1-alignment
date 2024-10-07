#!/bin/bash
#$ -N eval_whisper_all_beam2
#$ -q all.q@supergpu*
#$ -l ram_free=40G,mem_free=40G
#$ -l matylda6=0.1,matylda5=0.1,scratch=0.1
#$ -l gpu=1,gpu_ram=40G
#$ -o /mnt/matylda6/xsedla1h/projects/job_logs/asr_eval/eval_whisper_all_beam2.o
#$ -e /mnt/matylda6/xsedla1h/projects/job_logs/asr_eval/eval_whisper_all_beam2.e
N_GPUS=1
EXPERIMENT="eval_whisper_all_beam2"

# Job should finish in about 2 days
ulimit -t 200000

# Enable opening multiple files
ulimit -n 4096

# Enable to save bigger checkpoints
ulimit -f unlimited
ulimit -v unlimited
ulimit -u 4096

# Initialize environment
source /mnt/matylda6/xsedla1h/miniconda3/bin/activate /mnt/matylda6/xsedla1h/envs/huggingface_asr

WORK_DIR="/mnt/matylda6/xsedla1h/projects/huggingface_asr"
EXPERIMENT_PATH="${WORK_DIR}/exp/${EXPERIMENT}"
RECIPE_DIR="${WORK_DIR}/recipes/eloquence"
DATASETS="${RECIPE_DIR}/datasets_all.json"

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

export WANDB_MODE=offline
export WANDB_RUN_ID=$EXPERIMENT
export WANDB_PROJECT="whisper-asr"

# get the gpu
export CUDA_VISIBLE_DEVICES=$(free-gpus.sh $N_GPUS) || {
  echo "Could not obtain GPU."
  exit 1
}

args=(
  # General training arguments
  --output_dir=$EXPERIMENT_PATH
  --per_device_train_batch_size="32"
  --per_device_eval_batch_size="32"
  --dataloader_num_workers="2"
  --num_train_epochs="70"
  --group_by_length="True"
  --do_evaluate
  
  # Optimizer related arguments
  --optim="adamw_torch"
  --learning_rate="1e-3"
  --warmup_steps="20000"
  --early_stopping_patience="10"
  --weight_decay="1e-6"
  --max_grad_norm="5.0"
  --lsm_factor="0.1"
  --gradient_accumulation_steps="4"

  # Logging, saving and evaluation related arguments
  --report_to="wandb"
  --logging_steps="10"
  --save_strategy="epoch"
  --evaluation_strategy="epoch"
  --wandb_predictions_to_save=50
  --greater_is_better="False"
  #--metric_for_best_model="eval_wer"
  --save_total_limit="5"
  
  # Data related arguments
  --datasets_creation_config="${DATASETS}"
  --max_duration_in_seconds="30.0"
  --min_duration_in_seconds="0.2"
  --remove_unused_columns="False"
  --preprocessing_num_workers="16"
  --writer_batch_size="200" # 1000
  --collator_rename_features="False"
  --validation_split val
  --test_splits librispeech_test.clean librispeech_test.other how2_val how2_dev5 fleurs_validation fleurs_test slurp_dev slurp_test

  # Preprocessing related arguments
  --data_preprocessing_config="${WORK_DIR}/configs/default_data_preprocessing_whisper.json"

  # Model related arguments
  --tokenizer_name="openai/whisper-small.en"
  --feature_extractor_name="openai/whisper-small.en"
  --from_pretrained="openai/whisper-small.en"

  # Generation related arguments
  --num_beams="2"
  --max_length="448"
  --predict_with_generate
  #--post_process_predictions
)

echo "Running training.."
if [ "$N_GPUS" -gt 1 ]; then
  torchrun --standalone --nnodes=1 --nproc-per-node=$N_GPUS src/trainers/train_enc_dec_asr.py "${args[@]}"
else
  python src/trainers/train_enc_dec_asr.py "${args[@]}"
fi
