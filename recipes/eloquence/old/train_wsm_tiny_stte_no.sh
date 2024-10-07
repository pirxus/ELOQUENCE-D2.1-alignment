#!/bin/bash
#$ -N wsm_tiny_stte_cos_no
#$ -q long.q@supergpu*
#$ -l ram_free=40G,mem_free=40G
#$ -l matylda6=1
#$ -l scratch=2
#$ -l gpu=1,gpu_ram=20G
#$ -o /mnt/matylda6/xsedla1h/projects/chime/logs/wsm_tiny_stte_cos_no.o
#$ -e /mnt/matylda6/xsedla1h/projects/chime/logs/wsm_tiny_stte_cos_no.e
N_GPUS=1

EXPERIMENT="wsm_tiny_stte_cos_no"

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
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

args=(
  # General training arguments
  --output_dir=$EXPERIMENT_PATH
  --per_device_train_batch_size="12"
  --per_device_eval_batch_size="12"
  --dataloader_num_workers="2"
  #--num_train_epochs="10"
  --max_steps="100000"
  --group_by_length="True"
  --bf16
  --bf16_full_eval
  --do_train
  --load_best_model_at_end
  --qformer_eval_callback
  --ddp_find_unused_parameters="False"

  # Optimizer related arguments
  --optim="adamw_torch"
  --learning_rate="1.5e-4"
  --warmup_steps="5000"
  --lr_scheduler_type="cosine"
  --early_stopping_patience="5"
  --weight_decay="1e-6"
  --max_grad_norm="5.0"
  --gradient_accumulation_steps="1"

  # Logging, saving and evaluation related arguments
  --report_to="wandb"
  --logging_steps="10"
  --save_strategy="steps"
  --evaluation_strategy="steps"
  --eval_steps="3000"
  --save_steps="3000"
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
  #--prompt_prefix='USER: '
  #--prompt_suffix=' USER: Transcribe speech to text. ASSISTANT: '

  --connector_type='encoder_stacked'
  --downsampling_factor=5
  --conn_hidden_size=1024

  --conn_layers=2
  --conn_attn_heads=16
  --qf_intermediate_size=2048

  # Generation related arguments
  --no_metrics
  --num_beams="2"
  --max_new_tokens=80
  --predict_with_generate
)

echo "Running training.."
if [ "$N_GPUS" -gt 1 ]; then
  torchrun --standalone --nnodes=1 --nproc-per-node=$N_GPUS src/trainers/alignment/train_ecd_lm.py "${args[@]}"
else
  python src/trainers/alignment/train_ecd_lm.py "${args[@]}"
fi
