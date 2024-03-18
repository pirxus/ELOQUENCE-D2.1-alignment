#!/bin/bash
#$ -N mt_marian_bpe_best_eval_sacre
#$ -q short.q@supergpu*
#$ -l ram_free=20G,mem_free=20G
#$ -l matylda6=0.5
#$ -l scratch=0.5
#$ -l gpu=1,gpu_ram=16G
#$ -o /mnt/matylda6/xsedla1h/projects/job_logs/mt_eval/mt_marian_bpe_best_eval_sacre.o
#$ -e /mnt/matylda6/xsedla1h/projects/job_logs/mt_eval/mt_marian_bpe_best_eval_sacre.e
#

EXPERIMENT="mt_marian_bpe_best_sacre"

# Job should finish in about 1 day
ulimit -t 100000

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


WORK_DIR="/mnt/matylda6/xsedla1h/projects/huggingface_asr"
EXPERIMENT_PATH="${WORK_DIR}/exp/${EXPERIMENT}"
RECIPE_DIR="${WORK_DIR}/recipes/how2"
HOW2_BASE="/mnt/scratch/tmp/kesiraju/how2"
HOW2_PATH="/mnt/ssd/xsedla1h/${EXPERIMENT}/how2"

cd $WORK_DIR || {
  echo "No such directory $WORK_DIR"
  exit 1
}

# set pythonpath so that python works
export PYTHONPATH="${PYTHONPATH}:${WORK_DIR}/src"

# get the gpu
export CUDA_VISIBLE_DEVICES=$(free-gpus.sh 1) || {
  echo "Could not obtain GPU."
  exit 1
}

export TRANSFORMERS_NO_ADVISORY_WARNINGS=1 # cause of the tokenizer..
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME="/mnt/matylda6/xsedla1h/hugging-face"

export WANDB_MODE=offline
export WANDB_RUN_ID=$EXPERIMENT
export WANDB_PROJECT="mt"

#mkdir -p /mnt/ssd/xsedla1h/$EXPERIMENT
#echo "Copying data to ssd.."
#cp -r $HOW2_BASE /mnt/ssd/xsedla1h/${EXPERIMENT}


args=(
  # General training arguments
  --output_dir=$EXPERIMENT_PATH
  --per_device_train_batch_size="64" # 64
  --per_device_eval_batch_size="64"
  --dataloader_num_workers="4"
  --num_train_epochs="50"
  --group_by_length="True"
  --bf16
  #--do_train
  --do_evaluate
  --load_best_model_at_end

  #--restart_from="/mnt/matylda6/xsedla1h/projects/huggingface_asr/exp/s2t_50ep_no_pert_cont/checkpoint-21270"

  # Optimizer related arguments
  --optim="adamw_torch"
  --learning_rate="1e-3"
  --warmup_steps="10000"
  --early_stopping_patience="5"
  --weight_decay="1e-5"
  --max_grad_norm="5.0"
  --lsm_factor="0.1"
  --gradient_accumulation_steps="2"

  # Logging, saving and evaluation related arguments
  --report_to="wandb"
  --logging_steps="10"
  --save_strategy="epoch"
  --evaluation_strategy="epoch"
  --wandb_predictions_to_save=50 # 60
  --greater_is_better="True"
  --metric_for_best_model="eval_bleu"
  --save_total_limit="5"

  # Data related arguments
  #--dataset_name="/home/pirx/Devel/masters/APMo-SLT/src/huggingface_asr/src/dataset_builders/how2_dataset"
  #--data_dir="/home/pirx/Devel/masters/data/how2"
  --dataset_name="${WORK_DIR}/src/dataset_builders/how2_dataset"
  --data_dir="${HOW2_BASE}"
  --dataset_config="text_only"
  --max_duration_in_seconds="30.0"
  --min_duration_in_seconds="0.2"
  --remove_unused_columns="False"
  --preprocessing_num_workers="4"
  --writer_batch_size="200" # 1000
  --collator_rename_features="False"
  --text_column_name="transcription"
  --validation_split val
  --test_splits val dev5
  #--do_lower_case
  #--lcrm

  # Preprocessing related arguments
  #--data_preprocessing_config="${RECIPE_DIR}/data_preprocessing.json"

  # Model related arguments
  #--tokenizer_name="pirxus/how2_pt_bpe8000_tc"
  #--feature_extractor_name="pirxus/features_fbank_80"
  #--from_config="facebook/s2t-medium-librispeech-asr"

  --from_pretrained="/mnt/matylda6/xsedla1h/projects/huggingface_asr/exp/mt_marian_bpe/checkpoint-69360"
  #--average_checkpoints

  # Generation related arguments
  --num_beams="10"
  --max_length="150"
  --predict_with_generate
  --decoding_ctc_weight="0.0"
  --eval_beam_factor="1"
)

echo "Running training.."
python src/trainers/train_mt.py "${args[@]}"

# delete the ssd directory
#echo "Cleaning the ssd directory.."
#rm -rf /mnt/ssd/xsedla1h/${EXPERIMENT}