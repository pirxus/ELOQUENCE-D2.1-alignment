from dataclasses import dataclass, field
from typing import List, Optional

from transformers import Seq2SeqTrainingArguments


@dataclass
class ModelArguments:
    base_encoder_model: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    base_decoder_model: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    from_pretrained: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    feature_extractor_name: Optional[str] = field(
        default=None,
        metadata={"help": "feature extractor name or path if not the same as model_name"},
    )
    pad_token: Optional[str] = field(default="|", metadata={"help": "PAD token"})
    mask_token: Optional[str] = field(default="+", metadata={"help": "MASK token"})
    bos_token: Optional[str] = field(default="<", metadata={"help": "BOS token"})
    eos_token: Optional[str] = field(default=">", metadata={"help": "EOS token"})
    enc_adapters: Optional[bool] = field(default=False, metadata={"help": "Add adapters to the encoder."})
    dec_adapters: Optional[bool] = field(default=False, metadata={"help": "Add adapters to the decoder."})
    sampling_rate: Optional[int] = field(default=16_000, metadata={"help": "Sampling rate for the model."})
    from_encoder_decoder_config: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to create model from encoder and decoder configs."},
    )
    decoder_pos_emb_fixed: Optional[bool] = field(default=False, metadata={"help": "Whether to disable decoder WPE."})
    average_checkpoints: Optional[bool] = field(default=False, metadata={"help": "Whether to average checkpoints."})
    language: Optional[str] = field(default=None, metadata={"help": "Language of the model."})
    task: Optional[str] = field(default=None, metadata={"help": "Task of the model."})
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )


@dataclass
class GeneralTrainingArguments(Seq2SeqTrainingArguments):
    early_stopping_patience: Optional[int] = field(default=-1, metadata={"help": "Patience for early stopping."})
    decoder_cold_start: Optional[bool] = field(
        default=False, metadata={"help": "Whenever to reinitialize decoder weights"}
    )
    enc_layers_to_freeze: Optional[int] = field(default=0, metadata={"help": "Encoder layers to freeze"})
    dec_layers_to_freeze: Optional[int] = field(default=0, metadata={"help": "Decoder layers to freeze"})
    steps_to_freeze_enc: Optional[int] = field(default=0, metadata={"help": "Steps to freeze encoder"})
    steps_to_freeze_dec: Optional[int] = field(default=0, metadata={"help": "Steps to freeze decoder"})
    cross_attention_scaling_factor: Optional[float] = field(
        default=1, metadata={"help": "Custom scaling factor for cross attention weights"}
    )
    ctc_weight: Optional[float] = field(default=0, metadata={"help": "Weight of CTC loss."})
    restart_from: Optional[str] = field(
        default="", metadata={"help": "Path to checkpoint used to restart the training."}
    )
    lsm_factor: Optional[float] = field(default=0, metadata={"help": "Label smoothing coefficient for CE loss."})
    shared_lm_head: Optional[bool] = field(default=False, metadata={"help": "Whether to share LM head params."})
    use_fbanks: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use fbanks instead of raw audio signal."}
    )
    freeze_cross_attention: bool = field(default=False, metadata={"help": "Whether to freeze cross attentions"})
    preprocess_dataset_only: bool = field(default=False, metadata={"help": "Whether to preprocess dataset only"})
    skip_if_exists: Optional[str] = field(default=None, metadata={"help": "Whether to check if tokenizer exists."})
    apply_spec_augment: Optional[bool] = field(default=False, metadata={"help": "Whether to apply spec augmentations."})
    do_train: Optional[bool] = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: Optional[bool] = field(default=False, metadata={"help": "Whether to run evaluation."})
    collator_rename_features: Optional[bool] = field(
        default=True,
        metadata={"help": ("Rename input_features to input_values in collator.")},
    )
    track_ctc_loss: Optional[bool] = field(default=False, metadata={"help": "Whether to log CTC loss."})


@dataclass
class GenerationArguments:
    num_beams: Optional[int] = field(default=1, metadata={"help": "Num beams for decoding."})
    max_len: Optional[int] = field(default=200, metadata={"help": "Max number of generated tokens."})
    wandb_predictions_to_save: Optional[int] = field(
        default=100, metadata={"help": "Number of predictions to save to wandb."}
    )
    length_penalty: Optional[float] = field(default=1.0, metadata={"help": "Length penalty for generation."})
    decoding_ctc_weight: Optional[float] = field(default=0.0, metadata={"help": "CTC weight to bias hypothesis."})
    ctc_margin: Optional[float] = field(default=0, metadata={"help": "Margin to stop generation."})
    external_lm: Optional[str] = field(default=None, metadata={"help": "Path to external LM."})
    external_lm_weight: Optional[float] = field(default=0.0, metadata={"help": "Weight of external LM."})
    eval_beam_factor: Optional[int] = field(
        default=1, metadata={"help": "Factor to increase beam size for evaluation."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: str = field(metadata={"help": "The name of the dataset to use (via the datasets library)."})
    dataset_config: Optional[str] = field(
        default=None,
        metadata={"help": "The config of the dataset to use (via the datasets library)."},
    )
    audio_column_name: Optional[str] = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    text_column_name: Optional[str] = field(
        default="text",
        metadata={"help": "The name of the dataset column containing the text data. Defaults to 'text'"},
    )
    max_duration_in_seconds: Optional[float] = field(
        default=20.0,
        metadata={
            "help": (
                "Truncate audio files that are longer than `max_duration_in_seconds` seconds to"
                " 'max_duration_in_seconds`"
            )
        },
    )
    min_duration_in_seconds: Optional[float] = field(
        default=0.0,
        metadata={"help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"},
    )
    train_split: Optional[str] = field(default="train", metadata={"help": "Training split to be used."})
    validation_split: Optional[str] = field(default="validation", metadata={"help": "Validation split to be used."})
    test_split: Optional[str] = field(default="test", metadata={"help": "Test split to be used."})
    validation_slice: Optional[int] = field(default=None, metadata={"help": "Part of the validation split to be used."})
    apply_augmentations: Optional[bool] = field(
        default=False, metadata={"help": "Whether to apply on-the fly augmentations."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=1, metadata={"help": "Number of processes to use for data preprocessing."}
    )
    writer_batch_size: Optional[int] = field(default=500, metadata={"help": "Batch size to use for writing to disk."})
    remove_train_unks: Optional[bool] = field(
        default=False, metadata={"help": "Whether to remove UNKs from training data."}
    )
    fix_apostrophes: Optional[bool] = field(
        default=False, metadata={"help": "Whether to remove trailing spaces from labels."}
    )


@dataclass
class ModelArgumentsContext(ModelArguments):
    enc_memory_cells_location: List[int] = field(
        default=None, metadata={"help": "Positions where to place memory cells in encoder"}
    )
    dec_memory_cells_location: List[int] = field(
        default=None, metadata={"help": "Positions where to place memory cells in decoder"}
    )
    enc_memory_dim: int = field(default=None, metadata={"help": "Size of memory on encoder size"})
    dec_memory_dim: int = field(default=None, metadata={"help": "Size of memory on decoder size"})


@dataclass
class GeneralTrainingArgumentsContext(GeneralTrainingArguments):
    freeze_others: bool = field(default=False, metadata={"help": "Whether to freeze rest of the model"})
    conv_ids_column_name: str = field(default=None, metadata={"help": "Conv ids column."})
    turn_index_column_name: str = field(default=None, metadata={"help": "Turn index column."})


@dataclass
class TokenizerTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: str = field(metadata={"help": "The name of the dataset to use (via the datasets library)."})
    tokenizer_name: str = field(
        metadata={"help": "The name of the model to be created (via the transformers library)."}
    )
    dataset_config: Optional[str] = field(
        default=None,
        metadata={"help": "The config of the dataset to use (via the datasets library)."},
    )
    text_column_name: Optional[str] = field(
        default="text",
        metadata={"help": "The name of the dataset column containing the text data. Defaults to 'text'"},
    )
    train_split: Optional[str] = field(default="train", metadata={"help": "Training split to be used."})
    vocab_size: Optional[int] = field(default=5_000, metadata={"help": "Vocab size."})
    apply_regularization: Optional[bool] = field(default=False, metadata={"help": "Whether to apply regularization."})
    tokenizer_type: Optional[str] = field(
        default="unigram", metadata={"help": "Type of tokenizer to create if does not exists."}
    )
    additional_raw_data: Optional[str] = field(
        default=None,
        metadata={"help": "The input additional raw data file (a text file)."},
    )
    skip_if_exists: Optional[str] = field(default=None, metadata={"help": "Whether to check if tokenizer exists."})
    pad_token: Optional[str] = field(default="<pad>", metadata={"help": "PAD token"})
    mask_token: Optional[str] = field(default="<mask>", metadata={"help": "MASK token"})
    bos_token: Optional[str] = field(default="<s>", metadata={"help": "BOS token"})
    eos_token: Optional[str] = field(default="</s>", metadata={"help": "EOS token"})
    unk_token: Optional[str] = field(default="<unk>", metadata={"help": "EOS token"})
