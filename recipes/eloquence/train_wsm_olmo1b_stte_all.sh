#!/bin/bash
#$ -N wsm_olmo1b_stte_w2000
#$ -q long.q@supergpu18
#$ -l ram_free=40G,mem_free=40G
#$ -l matylda6=1
#$ -l gpu=2,gpu_ram=20G
#$ -o /mnt/matylda6/isedlacek/projects/job_logs/eloquence/wsm_olmo1b_stte_w2000.o
#$ -e /mnt/matylda6/isedlacek/projects/job_logs/eloquence/wsm_olmo1b_stte_w2000.e
N_GPUS=2
EXPERIMENT="wsm_olmo1b_stte_w2000"

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
export HF_HOME="/mnt/matylda6/isedlacek/hugging-face"

export WANDB_MODE=offline
export WANDB_RUN_ID=$EXPERIMENT
export WANDB_PROJECT="eloquence-asr"

# get the gpu
export CUDA_VISIBLE_DEVICES=$(free-gpus.sh $N_GPUS) || {
  echo "Could not obtain GPU."
  exit 1
}
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

args=(
  # General training arguments
  --output_dir=$EXPERIMENT_PATH
  --per_device_train_batch_size="20" # 20
  --per_device_eval_batch_size="24" # 24
  --dataloader_num_workers="4"
  --num_train_epochs="14"
  #--max_steps="150000"
  --group_by_length="True"
  --bf16
  --bf16_full_eval
  --do_train
  --do_evaluate
  --load_best_model_at_end
  --qformer_eval_callback
  --ddp_find_unused_parameters="False"

  # Optimizer related arguments
  --optim="adamw_torch"
  --learning_rate="1e-4"
  --warmup_steps="2000"
  --early_stopping_patience="3"
  --weight_decay="1e-6"
  --max_grad_norm="5.0"
  #--lsm_factor="0.1"
  --gradient_accumulation_steps="1"

  # Logging, saving and evaluation related arguments
  --report_to="wandb"
  --logging_steps="10"
  --save_strategy="steps"
  --evaluation_strategy="steps"
  --save_steps="1000"
  --eval_steps="1000"
  --wandb_predictions_to_save=100 # 60
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
  --validation_split validation
  --test_splits how2_dev5 librispeech_test.clean librispeech_test.other

  # Preprocessing related arguments
  --data_preprocessing_config="${RECIPE_DIR}/data_preprocessing_whisper.json"

  # Model related arguments
  #--from_pretrained=""

  --feature_extractor_name="openai/whisper-small.en"
  --base_encoder_model="openai/whisper-small.en"

  --tokenizer_name="allenai/OLMo-1B-hf"
  --base_decoder_model="allenai/OLMo-1B-hf"
  --prompt_prefix='Transcribe speech to text: '
  --prompt_suffix='\nTranscript: ' 
  
  --connector_type='encoder_stacked'
  --downsampling_factor=5
  --conn_hidden_size=1024
  --conn_layers=2
  --conn_attn_heads=16
  --qf_intermediate_size=4096
  
  #--connector_type='linear_stacked'
  #--downsampling_factor=5
  #--conn_hidden_size=2048
  #--qf_intermediate_size=4096

  # Generation related arguments
  --num_beams="2"
  --max_new_tokens=150
  --predict_with_generate
  #--no_metrics
)

echo "Running training.."
if [ "$N_GPUS" -gt 1 ]; then
  torchrun --standalone --nnodes=1 --nproc-per-node=$N_GPUS src/trainers/alignment/train_ecd_lm.py "${args[@]}"
else
  python src/trainers/alignment/train_ecd_lm.py "${args[@]}"
fi
