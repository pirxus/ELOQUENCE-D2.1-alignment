"""Main training script for training of attention based encoder decoder ASR models."""
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
)
from transformers.utils import logging
from datasets import load_dataset

from utilities.callbacks import init_callbacks
from utilities.collators import SpeechCollatorWithPadding
from utilities.data_utils import get_dataset
from utilities.eval_utils import compute_metrics
from utilities.general_utils import do_evaluate, do_generate
from utilities.generation_utils import activate_joint_decoding
from utilities.model_utils import instantiate_aed_model
from utilities.training_arguments import (
    DataTrainingArguments,
    GeneralTrainingArguments,
    GenerationArguments,
    ModelArguments,
)

if __name__ == "__main__":
    logging.set_verbosity_debug()
    logger = logging.get_logger("transformers")
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, GeneralTrainingArguments, GenerationArguments))

    model_args, data_args, training_args, gen_args = parser.parse_args_into_dataclasses()

    # 0. prepare the how2 dataset object..
    # 1. Collect, preprocess dataset and extract evaluation dataset
    dataset = get_dataset(
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
        unk_token=data_args.unk_token,
        fix_apostrophes=data_args.fix_apostrophes,
        remove_train_unks=data_args.remove_train_unks,
        do_lower_case=data_args.do_lower_case,
        remove_punctuation=data_args.remove_punctuation,
        remove_commas_stops=data_args.remove_commas_stops,
        remove_listed_chars=data_args.remove_listed_chars,
        lcrm=data_args.lcrm,
    )

    if data_args.validation_slice:
        training_eval_dataset = dataset[data_args.validation_split].shuffle().select(range(data_args.validation_slice))
        # Ensure that transformations are also attached to the sliced validation dataset
        dataset[data_args.validation_split + str(data_args.validation_slice)] = training_eval_dataset
    else:
        training_eval_dataset = dataset[data_args.validation_split]

    logger.info(f"Dataset processed successfully.{dataset}")

    if training_args.preprocess_dataset_only:
        logger.info("Finished preprocessing dataset.")
        sys.exit(0)

    # 2. Create feature extractor and tokenizer
    feature_extractor = AutoFeatureExtractor.from_pretrained(training_args.feature_extractor_name)
    tokenizer = AutoTokenizer.from_pretrained(training_args.tokenizer_name)

    # 3. Instantiate model
    base_model_config = {
        "encoder_layerdrop": 0.0,
        "pad_token_id": tokenizer.pad_token_id,
        "encoder_pad_token_id": tokenizer.pad_token_id,
        "decoder_vocab_size": len(tokenizer),
        "vocab_size": len(tokenizer), # s2t specific
        "lsm_factor": model_args.lsm_factor,
        "shared_lm_head": model_args.shared_lm_head,
        "encoder_expect_2d_input": model_args.expect_2d_input,
        "decoder_start_token_id": tokenizer.bos_token_id,
        "decoder_pos_emb_fixed": model_args.decoder_pos_emb_fixed,
    }
    if base_model_config["encoder_expect_2d_input"] and isinstance(feature_extractor, Speech2TextFeatureExtractor):
        base_model_config["encoder_second_dim_input_size"] = feature_extractor.num_mel_bins

    s2t_config = Speech2TextConfig()
    s2t_config.update(base_model_config)

    model = Speech2TextForConditionalGeneration(config=s2t_config)
    logger.info(f"Finished loading model {model}")

    # 4. Update generation config
    gen_config = GenerationConfig(
        bos_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        #decoder_start_token_id=tokenizer.bos_token_id,
        decoder_start_token_id=tokenizer.eos_token_id,
        length_penalty=gen_args.length_penalty,
        #early_stopping=gen_args.early_stopping,
        early_stopping=True,
        eos_token_id=tokenizer.eos_token_id,
        max_length=gen_args.max_length,
        num_beams=gen_args.num_beams,
    )
    logger.info(f"Model updating generation config:\n {str(gen_config)}")
    model.generation_config = gen_config


    # 5. Initialize callbacks
    callbacks = init_callbacks(data_args, training_args, dataset, feature_extractor)

    # 6. Initialize data collator
    data_collator = SpeechCollatorWithPadding(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        padding=True,
        sampling_rate=data_args.sampling_rate,
        audio_path=data_args.audio_column_name,
        text_path=data_args.text_column_name,
        model_input_name=model.main_input_name,
    )

    # 7. Initialize trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        callbacks=callbacks,
        train_dataset=dataset[data_args.train_split],
        eval_dataset=training_eval_dataset,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(tokenizer, pred, gen_args.wandb_predictions_to_save),
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
            tokenizer=tokenizer,
            gen_args=gen_args,
            training_args=training_args,
            data_args=data_args,
            eos_token_id=tokenizer.eos_token_id,
        )
    # 10. N-best generation
    if training_args.do_generate:
        do_generate(
            trainer=trainer,
            dataset=dataset,
            model=model,
            tokenizer=tokenizer,
            gen_args=gen_args,
            data_args=data_args,
            eos_token_id=tokenizer.eos_token_id,
            gen_config=gen_config,
        )
