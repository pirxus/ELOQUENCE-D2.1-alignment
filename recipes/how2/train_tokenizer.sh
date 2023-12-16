#!/usr/bin/bash

EXPERIMENT="how2_english_bpe_tokenizer"
PROJECT="regularizations_english_corpus"
WORK_DIR="/home/pirx/Devel/masters/APMo-SLT/src"
HOW2_PATH="/home/pirx/Devel/masters/data/how2"

EXPERIMENT_PATH="${WORK_DIR}/experiments/${EXPERIMENT}"

cd $WORK_DIR

python huggingface_asr/src/trainers/train_tokenizer.py \
  --output_dir=$EXPERIMENT_PATH \
  --dataset_name="$WORK_DIR/huggingface_asr/src/dataset_builders/how2_dataset" \
  --data_dir=$HOW2_PATH \
  --dataset_config="text_only" \
  --preprocessing_num_workers="4" \
  --writer_batch_size="200" \
  --tokenizer_name="pirxus/how2_en_unigram5000" \
  --vocab_size=5000 \
  --tokenizer_type="unigram" \
  --text_column_name="transcription" \
  --train_split="train" \
  --pad_token="<pad>" \
  --unk_token="<unk>" \
  --bos_token="<s>" \
  --eos_token="</s>" \
  --mask_token="<mask>" \
  --validation_split="val"\
  --do_lower_case \
  --remove_listed_chars=",:" \
  --skip_audio_processing
