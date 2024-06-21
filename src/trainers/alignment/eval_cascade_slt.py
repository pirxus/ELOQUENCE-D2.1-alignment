"""Evaluation script for the baseline cascade ST system"""
import sys

from transformers import (
    AutoFeatureExtractor,
    AutoTokenizer,
    GenerationConfig,
    HfArgumentParser,
    Seq2SeqTrainer,
    Speech2TextConfig,
    Speech2TextFeatureExtractor,
    Speech2TextForConditionalGeneration,
    TrainerCallback,
    MarianConfig,
    MarianMTModel,
    MarianForCausalLM,
    Blip2QFormerConfig,
)
from transformers.utils import logging

from utilities.callbacks import init_callbacks
from utilities.collators import SpeechCollatorWithPadding
from utilities.data_utils import get_dataset
from utilities.eval_utils import compute_metrics_translation
from utilities.model_utils import average_checkpoints as average_checkpoints
from utilities.general_utils import do_evaluate, do_generate
from utilities.training_arguments import (
    DataTrainingArguments,
    GeneralTrainingArguments,
    GenerationArguments,
    ModelArguments,
    QFormerArguments
)

from models.s2t_qformer_marian import ApmoConfig, S2TEncoderMarianDecoder
from models.cascade_slt import CascadeSLTModel, CascadeSLTModelConfig
from models.ctc_encoder_plus_autoregressive_decoder import JointCTCAttentionEncoderDecoder
from utilities.training_utils import AdditionalLossTrackerTrainer


if __name__ == "__main__":
    logging.set_verbosity_debug()
    logger = logging.get_logger("transformers")
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, GeneralTrainingArguments, GenerationArguments, QFormerArguments))

    model_args, data_args, training_args, gen_args, qformer_args = parser.parse_args_into_dataclasses()

    # 0. prepare the how2 dataset object..
    # 1. Collect, preprocess dataset and extract evaluation dataset
    dataset, training_eval_dataset = get_dataset(
        datasets_creation_config_path=data_args.datasets_creation_config,
        dataset_name=data_args.dataset_name,
        dataset_config=data_args.dataset_config,
        data_dir=data_args.data_dir,
        preprocessing_num_workers=data_args.preprocessing_num_workers,
        writer_batch_size=data_args.writer_batch_size,
        sampling_rate=data_args.sampling_rate,
        max_input_len=data_args.max_duration_in_seconds,
        min_input_len=data_args.min_duration_in_seconds,
        len_column=training_args.length_column_name,
        text_column=data_args.text_column_name,
        audio_column=data_args.audio_column_name,
        train_split=data_args.train_split,
        validation_split=data_args.validation_split,
        text_transformations=data_args.text_transformations,
        split_long_segments_to_chunks=data_args.split_long_segments_to_chunks,
        validation_slice_str=data_args.validation_slice,
        cut_validation_from_train=data_args.cut_validation_from_train,
        seed=data_args.validation_slice_seed,
        reshuffle_at_start=data_args.reshuffle_at_start,
    )

    logger.info(f"Dataset processed successfully.{dataset}")

    if training_args.preprocess_dataset_only:
        logger.info("Finished preprocessing dataset.")
        sys.exit(0)

    # 2. Create feature extractor and tokenizer
    feature_extractor = AutoFeatureExtractor.from_pretrained(training_args.feature_extractor_name)
    tokenizer_asr_n, tokenizer_mt_src_n, tokenizer_mt_tgt_n = training_args.tokenizer_name.split(';')
    tokenizer_asr = AutoTokenizer.from_pretrained(tokenizer_asr_n)
    tokenizer_mt_src = AutoTokenizer.from_pretrained(tokenizer_mt_src_n)
    tokenizer_mt_tgt = AutoTokenizer.from_pretrained(tokenizer_mt_tgt_n)

    # 3. Instantiate model
    if 's2t' in model_args.base_encoder_model:
        encoder = Speech2TextForConditionalGeneration.from_pretrained(model_args.base_encoder_model)
        d_model = encoder.config.d_model
    else:
        encoder = JointCTCAttentionEncoderDecoder.from_pretrained(model_args.base_encoder_model)
        d_model = encoder.config.encoder.hidden_size

    decoder = MarianMTModel.from_pretrained(model_args.base_decoder_model)

    config = CascadeSLTModelConfig(enc_config=encoder.config, dec_config=decoder.config)
    model = CascadeSLTModel(
            encoder=encoder,
            decoder=decoder,
            encoder_tokenizer=tokenizer_asr,
            decoder_tokenizer=tokenizer_mt_src,
            config=config,
        )

    logger.info(f"Finished loading model {model}")

    # 4. Update generation config
    gen_config = GenerationConfig(
        bos_token_id=tokenizer_mt_src.bos_token_id,
        pad_token_id=tokenizer_mt_src.pad_token_id,
        decoder_start_token_id=tokenizer_mt_tgt.bos_token_id,
        decoder_end_token_id=tokenizer_mt_tgt.eos_token_id,
        length_penalty=gen_args.length_penalty,
        early_stopping=gen_args.early_stopping,
        eos_token_id=tokenizer_mt_src.eos_token_id,
        max_length=gen_args.max_length,
        num_beams=gen_args.num_beams,
    )
    logger.info(f"Model updating generation config:\n {str(gen_config)}")
    training_args.generation_max_length = gen_args.max_length
    training_args.generation_num_beams = gen_args.num_beams
    model.generation_config = gen_config
    model.decoder.generation_config = gen_config

    # 5. Initialize callbacks
    callbacks = init_callbacks(data_args, training_args, dataset, feature_extractor)

    # 6. Initialize data collator
    data_collator = SpeechCollatorWithPadding(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer_mt_tgt,
        padding=True,
        sampling_rate=data_args.sampling_rate,
        audio_path=data_args.audio_column_name,
        text_path=data_args.text_column_name,
        model_input_name=model.main_input_name,
    )

    # 7. Initialize trainer
    trainer_class = AdditionalLossTrackerTrainer if qformer_args.qf_mm_loss_weight > 0 else Seq2SeqTrainer
    trainer = Seq2SeqTrainer
    trainer = trainer_class(
            args=training_args,
            model=model,
            callbacks=callbacks,
            train_dataset=dataset[data_args.train_split],
            eval_dataset=training_eval_dataset,
            data_collator=data_collator,
            compute_metrics=lambda pred: compute_metrics_translation(tokenizer_mt_tgt, pred, gen_args.wandb_predictions_to_save),
    )

    # 8. Train model
    if training_args.do_train:
        trainer.train(resume_from_checkpoint=training_args.restart_from or None)

    # 9. Evaluation
    if training_args.do_evaluate:
        do_evaluate(
            trainer=trainer,
            dataset=dataset,
            model=model,
            tokenizer=tokenizer_mt_tgt,
            gen_args=gen_args,
            training_args=training_args,
            data_args=data_args,
        )
    # 10. N-best generation
    if training_args.do_generate:
        do_generate(
            trainer=trainer,
            dataset=dataset,
            model=model,
            tokenizer=tokenizer_mt_tgt,
            gen_args=gen_args,
            data_args=data_args,
            gen_config=gen_config,
        )
