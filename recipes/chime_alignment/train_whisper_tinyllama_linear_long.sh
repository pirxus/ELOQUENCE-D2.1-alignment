#!/bin/bash
#$ -N whisper_sm_tinyllama_stlinear_long
#$ -q long.q@supergpu*
#$ -l ram_free=40G,mem_free=40G
#$ -l matylda6=1
#$ -l scratch=5
#$ -l gpu=2,gpu_ram=20G
#$ -o /mnt/matylda6/xsedla1h/projects/chime/logs/whisper_sm_tinyllama_stlinear_long.o
#$ -e /mnt/matylda6/xsedla1h/projects/chime/logs/whisper_sm_tinyllama_stlinear_long.e
N_GPUS=2

EXPERIMENT="whisper_sm_tinyllama_stlinear_long"

# Job should finish in about 1 day
ulimit -t 150000

# Enable opening multiple files
ulimit -n 4096

# Enable to save bigger checkpoints
ulimit -f unlimited

ulimit -v unlimited
ulimit -u 4096

# Initialize environment
unset PYTHONPATH
unset PYTHONHOME
source /mnt/matylda6/xsedla1h/miniconda3/bin/activate /mnt/matylda6/xsedla1h/envs/huggingface_asr

WORK_DIR="/mnt/matylda6/xsedla1h/projects/huggingface_asr"
EXPERIMENT_PATH="${WORK_DIR}/exp/${EXPERIMENT}"
RECIPE_DIR="${WORK_DIR}/recipes/chime_alignment"
DATASETS="${RECIPE_DIR}/datasets.json"

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
export WANDB_PROJECT="chime"

# get the gpu
export CUDA_VISIBLE_DEVICES=$(free-gpus.sh $N_GPUS) || {
  echo "Could not obtain GPU."
  exit 1
}

args=(
  # General training arguments
  --output_dir=$EXPERIMENT_PATH
  --per_device_train_batch_size="8"
  --per_device_eval_batch_size="8"
  --dataloader_num_workers="2"
  #--num_train_epochs="10"
  --max_steps="100000"
  --group_by_length="True"
  --bf16
  --bf16_full_eval
  --do_train
  #--do_evaluate
  --load_best_model_at_end
  --qformer_eval_callback

  # Optimizer related arguments
  --optim="adamw_torch"
  --learning_rate="2e-4"
  --warmup_steps="10000"
  --early_stopping_patience="5"
  --weight_decay="1e-6"
  --max_grad_norm="5.0"
  --lsm_factor="0.1"
  --gradient_accumulation_steps="1"

  # Logging, saving and evaluation related arguments
  --report_to="wandb"
  --logging_steps="10"
  --save_strategy="steps"
  --evaluation_strategy="steps"
  --eval_steps="5000"
  --save_steps="5000"
  --wandb_predictions_to_save=50 # 60
  --greater_is_better="False"
  --metric_for_best_model="eval_loss"
  --save_total_limit="3"

  # Data related arguments
  --datasets_creation_config="${DATASETS}"
  --max_duration_in_seconds="30.0"
  --min_duration_in_seconds="0.2"
  --remove_unused_columns="False"
  --preprocessing_num_workers="16"
  --writer_batch_size="200" # 1000
  --collator_rename_features="False"
  --test_splits librispeech_validation.clean librispeech_validation.other librispeech_test.clean librispeech_test.other 
  --validation_split librispeech_validation.clean

  # Preprocessing related arguments
  --data_preprocessing_config="${RECIPE_DIR}/data_preprocessing_whisper.json"

  # Model related arguments
  --feature_extractor_name="openai/whisper-small"
  --base_encoder_model="openai/whisper-small"

  --tokenizer_name="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
  --base_decoder_model="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
  --prompt_prefix='speech: '
  --prompt_suffix=' transcript: '

  --connector_type='linear_stacked'
  --conn_hidden_size=2048
  --qf_intermediate_size=2048

  # Generation related arguments
  --num_beams="2"
  --max_new_tokens=80
  --predict_with_generate
  --no_metrics
)

echo "Running training.."
if [ "$N_GPUS" -gt 1 ]; then
  torchrun --standalone --nnodes=1 --nproc-per-node=$N_GPUS src/trainers/alignment/train_ecd_lm.py "${args[@]}"
else
  python src/trainers/alignment/train_ecd_lm.py "${args[@]}"
fi
