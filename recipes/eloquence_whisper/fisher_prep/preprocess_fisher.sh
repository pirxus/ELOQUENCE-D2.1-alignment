#$ -N fisher_clean_labels
#$ -q long.q@supergpu16
#$ -l ram_free=300G,mem_free=300G
#$ -l matylda6=0.1,scratch=2
#$ -o /mnt/scratch/tmp/xsedla1h/fisher_clean_labels.o
#$ -e /mnt/scratch/tmp/xsedla1h/fisher_clean_labels.e

# Job should finish in about 2 days
ulimit -t 200000

# Enable opening multiple files and bigger checkpoints
ulimit -n 4096
ulimit -f unlimited
ulimit -v unlimited
ulimit -u 4096

# Initialize environment
source /mnt/matylda6/xsedla1h/miniconda3/bin/activate /mnt/matylda6/xsedla1h/envs/huggingface_asr

WORK_DIR="/mnt/scratch/tmp/xsedla1h"
cd $WORK_DIR

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME="/mnt/matylda6/xsedla1h/hugging-face"

args=(
 --cores 32
 --dataset_path fisher_orig
 --output_dir fisher_cleaned
)

python preprocess_fisher.py "${args[@]}"
