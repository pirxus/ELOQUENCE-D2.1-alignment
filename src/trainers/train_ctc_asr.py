"""Main training script for training of CTC ASR models."""
import sys

from datasets import load_dataset, load_from_disk
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForCTC,
    AutoTokenizer,
    EarlyStoppingCallback,
    HfArgumentParser,
    Trainer,
)
from transformers.utils import logging

from models.encoders.branchformer import (
    Wav2Vec2BranchformerConfig,
    Wav2Vec2BranchformerForCTC,
)
from models.encoders.e_branchformer import (
    Wav2Vec2EBranchformerConfig,
    Wav2Vec2EBranchformerForCTC,
)
from trainers.training_arguments import (
    DataTrainingArguments,
    GeneralTrainingArguments,
    GenerationArguments,
    ModelArguments,
)
from training_utils import (
    AugmentationManagerCallback,
    Seq2SeqDataCollatorWithPadding,
    average_checkpoints,
    compute_metrics_ctc,
    do_evaluate,
    prepare_dataset,
)

AutoConfig.register("wav2vec2-branchformer", Wav2Vec2BranchformerConfig)
AutoModelForCTC.register(Wav2Vec2BranchformerConfig, Wav2Vec2BranchformerForCTC)

AutoConfig.register("wav2vec2-ebranchformer", Wav2Vec2EBranchformerConfig)
AutoModelForCTC.register(Wav2Vec2EBranchformerConfig, Wav2Vec2EBranchformerForCTC)

if __name__ == "__main__":
    logging.set_verbosity_debug()
    logger = logging.get_logger("transformers")
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, GeneralTrainingArguments, GenerationArguments))

    model_args, data_args, training_args, gen_args = parser.parse_args_into_dataclasses()

    # 1. Create feature extractor and tokenizer
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_args.feature_extractor_name)

    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)

    # 2. Load dataset
    if data_args.dataset_config is not None:
        dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config,
            keep_in_memory=False,
            num_proc=data_args.preprocessing_num_workers,
        )
    else:
        dataset = load_from_disk(data_args.dataset_name, keep_in_memory=False)

    len_column = training_args.length_column_name
    text_column = data_args.text_column_name
    audio_column = data_args.audio_column_name
    sampling_rate = model_args.sampling_rate
    max_input_len = data_args.max_duration_in_seconds
    min_input_len = data_args.min_duration_in_seconds

    # 3. Preprocess dataset
    logger.info("Preprocessing dataset...")
    dataset = prepare_dataset(
        dataset=dataset,
        dataset_name=data_args.dataset_name,
        length_column_name=len_column,
        text_column_name=text_column,
        audio_column_name=audio_column,
        preprocessing_num_workers=data_args.preprocessing_num_workers,
        writer_batch_size=data_args.writer_batch_size,
        train_split=data_args.train_split,
        fix_apostrophes=data_args.fix_apostrophes,
        remove_train_unks=data_args.remove_train_unks,
        apply_augmentations=data_args.apply_augmentations,
        unk_token=tokenizer.unk_token,
        sampling_rate=sampling_rate,
        max_input_len=max_input_len,
        min_input_len=min_input_len,
    )

    if training_args.preprocess_dataset_only:
        logger.info("Finished preprocessing dataset.")
        sys.exit(0)

    base_model_config = {
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "mask_token_id": tokenizer.mask_token_id,
        "feat_proj_dropout": 0.0,
        "layerdrop": 0.0,
        "min_length": 0,
        "no_repeat_ngram_size": 0,
        "early_stopping": False,
        "num_beams": gen_args.num_beams,
        "ctc_weight": training_args.ctc_weight,
        "ctc_loss_reduction": "mean",
        "vocab_size": len(tokenizer),
        "lsm_factor": training_args.lsm_factor,
        "shared_lm_head": training_args.shared_lm_head,
        "use_fbanks": training_args.use_fbanks,
        "apply_spec_augment": training_args.apply_spec_augment,
    }
    if hasattr(feature_extractor, "num_mel_bins"):
        base_model_config["num_mel_bins"] = feature_extractor.num_mel_bins

    # 4. Initialize seq2seq model
    if model_args.from_pretrained:
        config = AutoConfig.from_pretrained(model_args.from_pretrained)
        config.update(base_model_config)
        model_path = model_args.from_pretrained
        if model_args.average_checkpoints:
            model_path = average_checkpoints(model_path)
        model = AutoModelForCTC.from_pretrained(model_path, config=config)
    else:
        config = AutoConfig.from_pretrained(model_args.base_encoder_model)
        config.update(base_model_config)
        model = AutoModelForCTC.from_config(config)

    # 5. Init trainer
    callbacks = []
    if training_args.apply_spec_augment:
        callbacks.append(
            AugmentationManagerCallback(training_args.num_steps_to_activate_spec_augment, model_config_path="config")
        )
    if training_args.early_stopping_patience > -1:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience))

    data_collator = Seq2SeqDataCollatorWithPadding(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        padding=True,
        sampling_rate=model_args.sampling_rate,
        audio_path=data_args.audio_column_name,
        text_path=data_args.text_column_name,
        rename_features=training_args.collator_rename_features,
    )
    training_eval_dataset = (
        dataset[data_args.validation_split].select(range(data_args.validation_slice))
        if data_args.validation_slice
        else dataset[data_args.validation_split]
    )

    trainer = Trainer(
        args=training_args,
        model=model,
        callbacks=callbacks,
        train_dataset=dataset[data_args.train_split],
        eval_dataset=training_eval_dataset,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics_ctc(tokenizer, pred, gen_args.wandb_predictions_to_save),
    )

    # 6. Train
    if training_args.do_train:
        trainer.train(resume_from_checkpoint=training_args.restart_from or None)

    # 7. Evaluation
    if training_args.do_evaluate:
        do_evaluate(
            trainer=trainer,
            dataset=dataset,
            model=model,
            tokenizer=tokenizer,
            gen_args=gen_args,
            training_args=training_args,
            eos_token_id=base_model_config["eos_token_id"],
        )
