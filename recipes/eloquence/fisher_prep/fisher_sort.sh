#$ -N fisher_context_prep_fixed
#$ -q long.q@supergpu16
#$ -l ram_free=300G,mem_free=300G
#$ -l matylda6=0.1,scratch=2
#$ -o /mnt/scratch/tmp/xsedla1h/fisher_context_prep_fixed.o
#$ -e /mnt/scratch/tmp/xsedla1h/fisher_context_prep_fixed.e

# -o /mnt/matylda6/xsedla1h/projects/job_logs/fisher_prep/fisher_context_prep.o
# -e /mnt/matylda6/xsedla1h/projects/job_logs/fisher_prep/fisher_context_prep.e

# Job should finish in about 2 days
ulimit -t 200000

# Enable opening multiple files and bigger checkpoints
ulimit -n 4096
ulimit -f unlimited
ulimit -v unlimited
ulimit -u 4096

# Initialize environment
source /mnt/matylda6/xsedla1h/miniconda3/bin/activate /mnt/matylda6/xsedla1h/envs/huggingface_asr

TARGET_DIR="/mnt/scratch/tmp/xsedla1h"
RECIPE_DIR="/mnt/matylda6/xsedla1h/projects/huggingface_asr/recipes/eloquence/fisher_prep"

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME="/mnt/matylda6/xsedla1h/hugging-face"

args=(
 --cores 16
 --dataset_path $TARGET_DIR/fisher_cleaned
 --output_dir $TARGET_DIR/fisher_context_correct
 --shard_size 256
 --min_len 5.0
 --max_len 25.0
 --max_context 20
)

python $RECIPE_DIR/fisher_sort_mp.py "${args[@]}"
