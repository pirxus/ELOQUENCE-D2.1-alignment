#!/bin/bash
#$ -N arch1_whisper_qf_marian_dec_6l_100q
#$ -q long.q@supergpu*
#$ -l ram_free=40G,mem_free=40G
#$ -l matylda6=0.5
#$ -l ssd=1,ssd_free=100G
#$ -l gpu=1,gpu_ram=20G
#$ -o /mnt/matylda6/xsedla1h/projects/job_logs/qformer_tests/arch1_whisper_qf_marian_dec_6l_100q.o
#$ -e /mnt/matylda6/xsedla1h/projects/job_logs/qformer_tests/arch1_whisper_qf_marian_dec_6l_100q.e
#

EXPERIMENT="arch1_whisper_qf_marian_dec_6l_100q"

# Job should finish in about 1 day
ulimit -t 150000

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
HOW2_BASE="/mnt/matylda6/xsedla1h/data/how2"
HOW2_PATH="/mnt/ssd/xsedla1h/${EXPERIMENT}/how2"

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
export WANDB_PROJECT="qformer"

mkdir -p /mnt/ssd/xsedla1h/$EXPERIMENT
echo "Copying data to ssd.."
cp -r $HOW2_BASE /mnt/ssd/xsedla1h/${EXPERIMENT}

# get the gpu
export CUDA_VISIBLE_DEVICES=$(free-gpus.sh 1) || {
  echo "Could not obtain GPU."
  exit 1
}

args=(
  # General training arguments
  --output_dir=$EXPERIMENT_PATH
  --per_device_train_batch_size="64" # 64
  --per_device_eval_batch_size="32"
  --dataloader_num_workers="4"
  --num_train_epochs="70"
  --group_by_length="True"
  --bf16 # FIXME
  --do_train
  --do_evaluate
  --load_best_model_at_end
  --qformer_eval_callback

  #--qf_pretrain_epochs=15
  --eval_delay="15.0"
  #--qf_enc_unfreeze_epochs="40"
  #--qf_dec_unfreeze_epochs="40"

  #--restart_from="/mnt/matylda6/xsedla1h/projects/huggingface_asr/exp/s2t_50ep_no_pert_cont/checkpoint-21270"

  # Optimizer related arguments
  --optim="adamw_torch"
  --learning_rate="2e-4"
  --warmup_steps="12000"
  --early_stopping_patience="10"
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
  --track_ctc_loss

  # Data related arguments
  #--dataset_name="/home/pirx/Devel/masters/APMo-SLT/src/huggingface_asr/src/dataset_builders/how2_dataset"
  #--data_dir="/home/pirx/Devel/masters/data/how2"
  --dataset_name="${HOW2_PATH}"
  --max_duration_in_seconds="20.0"
  --min_duration_in_seconds="0.2"
  --remove_unused_columns="False"
  --preprocessing_num_workers="4"
  --writer_batch_size="200" # 1000
  --collator_rename_features="False"
  --text_column_name="translation"
  --validation_split val
  --test_splits val dev5

  # Preprocessing related arguments
  #--data_preprocessing_config="${RECIPE_DIR}/data_preprocessing_no_spec.json"
  --data_preprocessing_config="${RECIPE_DIR}/data_preprocessing_whisper.json"

  # Model related arguments
  #--from_pretrained="/mnt/matylda6/xsedla1h/projects/huggingface_asr/exp/qf_ebr_mt_6l_128_avg_mm_cb_mmlr/checkpoint-96458"
  --tokenizer_name="pirxus/how2_pt_bpe8000_tc"
  #--feature_extractor_name="pirxus/features_fbank_80"
  --feature_extractor_name="openai/whisper-small.en"
  #--base_encoder_model="/mnt/matylda6/xsedla1h/projects/huggingface_asr/exp/s2t_fixed_v2_1d/average_checkpoint/"
  #--base_encoder_model="/mnt/matylda6/xsedla1h/projects/huggingface_asr/exp/ebr_small_first_5k/checkpoint-85110/"
  #--base_encoder_model="BUT-FIT/EBranchRegulaFormer-medium"
  --base_encoder_model="openai/whisper-small.en"
  --base_decoder_model="/mnt/matylda6/xsedla1h/projects/huggingface_asr/exp/mt_marian_bpe/checkpoint-69360/"
  --n_queries=100
  --conn_layers=6
  --conn_hidden_size=256
  --conn_attn_heads=4
  --qf_intermediate_size=2048
  --qf_mm_loss_weight="1.0"
  --qf_mm_pooling="avg"
  #--qf_mm_micro_loss="dot"

  # Generation related arguments
  --num_beams="5"
  --max_length="150"
  --predict_with_generate
  --decoding_ctc_weight="0.0"
  --eval_beam_factor="1"
)

echo "Running training.."
python src/trainers/train_qformer_slt.py "${args[@]}"

# delete the ssd directory
echo "Cleaning the ssd directory.."
rm -rf /mnt/ssd/xsedla1h/${EXPERIMENT}
