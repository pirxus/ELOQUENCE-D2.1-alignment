#!/bin/bash
#SBATCH --job-name=wll_olmo7b_instruct_stte_mix_synth_train_enc
#SBATCH --output=/gpfs/home/vut/$USER/logs/%x_%j.o
#SBATCH --error=/gpfs/home/vut/$USER/logs/%x_%j.e
#SBATCH --account=ehpc62
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=20
#SBATCH --time=2-00:00:00

. /gpfs/home/vut/vut719833/path.sh
N_GPUS=$SLURM_NTASKS_PER_NODE

#### Multi-node setup
#NUM_NODES=$SLURM_NNODES
#export host_port=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
#export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
#
#echo "WORLD_SIZE="$WORLD_SIZE
#
#host_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
##export MASTER_ADDR=$master_addr
##echo "MASTER_ADDR="$MASTER_ADDR
##export MASTER_PORT=$host_port
#HOST_NODE_ADDR="$host_addr:$host_port"
#echo "HOST_NODE_ADDR="$HOST_NODE_ADDR
#####

EXPERIMENT=$SLURM_JOB_NAME

WORK_DIR="/gpfs/home/vut/vut719833/code/huggingface_asr"
EXPS_DIR="/gpfs/projects/ehpc62/bolaji/"
EXPERIMENT_PATH="${EXPS_DIR}/exp/${EXPERIMENT}"
RECIPE_DIR="${WORK_DIR}/recipes/eloquence"
#DATASETS="${RECIPE_DIR}/datasets.json"
DATASETS="${RECIPE_DIR}/datasets_slurp_asr_bsc.json"
#DATASETS="${RECIPE_DIR}/datasets_librispeech_slurp_asr_bsc.json"

cd $WORK_DIR || {
  echo "No such directory $WORK_DIR"
  exit 1
}

# set pythonpath so that python works
export PYTHONPATH="${PYTHONPATH}:${WORK_DIR}/src"

export WANDB_MODE=offline
export WANDB_PROJECT="eloquence_asr_llm"
export WANDB_ENTITY="butspeechfit"
export WANDB_RUN_ID="${EXPERIMENT}"

args=(
  # General training arguments
  --output_dir="$EXPERIMENT_PATH"
  --per_device_train_batch_size="4"
  --per_device_eval_batch_size="4"
  --dataloader_num_workers="16"
  #--num_train_epochs="10"
  --max_steps="15000"
  --group_by_length="True"
  --bf16
  --bf16_full_eval
  --do_train
  --do_evaluate
  --load_best_model_at_end
  --qformer_eval_callback
  --ddp_find_unused_parameters="True"

  # Optimizer related arguments
  --optim="adamw_torch"
  --learning_rate="1.2e-4"
  --warmup_steps="3000"
  --early_stopping_patience="3"
  --weight_decay="1e-6"
  --max_grad_norm="5.0"
  #--lsm_factor="0.1"
  --gradient_accumulation_steps="2"

  # Logging, saving and evaluation related arguments
  --report_to="wandb"
  --logging_steps="10"
  --save_strategy="steps"
  --eval_strategy="steps"
  --save_steps="5000"
  --eval_steps="200"
  --wandb_predictions_to_save=50 # 60
  --greater_is_better="False"
  --metric_for_best_model="eval_wer"
  --save_total_limit="3"

  # Data related arguments
  --datasets_creation_config="${DATASETS}"
  --max_duration_in_seconds="30.0"
  --min_duration_in_seconds="0.2"
  --remove_unused_columns="False"
  --preprocessing_num_workers="16"
  --writer_batch_size="200" # 1000
  --collator_rename_features="False"
#  --test_splits librispeech_validation.clean librispeech_validation.other librispeech_test.clean librispeech_test.other
#  --validation_split librispeech_validation.clean

  # Preprocessing related arguments
  --data_preprocessing_config="${RECIPE_DIR}/data_preprocessing_wavlm.json"

  # Model related arguments
#  --feature_extractor_name="openai/whisper-small.en"
#  --base_encoder_model="openai/whisper-small.en"
  --feature_extractor_name="microsoft/wavlm-large"
  --base_encoder_model="microsoft/wavlm-large"
#  --layer_to_extract="15"
  --freeze_encoder="False"

#  --tokenizer_name="allenai/OLMo-1B-hf"
#  --base_decoder_model="allenai/OLMo-1B-hf"
  --tokenizer_name="allenai/OLMo-7B-Instruct-hf"
  --base_decoder_model="allenai/OLMo-7B-Instruct-hf"
  --prompt_prefix='Transcribe the following speech: '
  --prompt_suffix='\nTranscription: '

  --connector_type='encoder_stacked'
  --downsampling_factor=5
  --conn_hidden_size=1024
  --conn_layers=2
  --conn_attn_heads=16
  --qf_intermediate_size=4096

#  --decoder_lora=true
#  --quantize_decoder=4

  #--connector_type='linear_stacked'
  #--downsampling_factor=5
  #--conn_hidden_size=2048
  #--qf_intermediate_size=4096

  # Generation related arguments
  --num_beams="1"
  --max_new_tokens=200
  --predict_with_generate
  #--no_metrics
)

echo "Running training.."
if [ "$N_GPUS" -gt 1 ]; then
  torchrun --standalone --nnodes=1 --nproc-per-node=$N_GPUS src/trainers/alignment/train_ecd_lm.py "${args[@]}"
#  srun torchrun --nnodes=$NUM_NODES --nproc-per-node=$N_GPUS  --rdzv-id=$SLURM_JOB_ID --rdzv-backend=c10d --rdzv-endpoint=$HOST_NODE_ADDR src/trainers/alignment/train_ecd_lm.py "${args[@]}"
else
  python -u src/trainers/alignment/train_ecd_lm.py "${args[@]}"
fi
