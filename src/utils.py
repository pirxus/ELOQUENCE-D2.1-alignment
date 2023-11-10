import glob
import os
import shutil
from dataclasses import dataclass, field, make_dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import wandb
from audiomentations import Compose, TimeStretch
from datasets import Dataset
from jiwer import cer, compute_measures
from transformers import (
    AutoConfig,
    BatchFeature,
    LogitsProcessor,
    LogitsProcessorList,
    PreTrainedTokenizerBase,
    Seq2SeqTrainer,
    SpeechEncoderDecoderModel,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    Wav2Vec2FeatureExtractor,
)
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names
from transformers.utils import logging

from models.ctc_encoder_plus_autoregressive_decoder import (
    JointCTCAttentionEncoderDecoderConfig,
)

logger = logging.get_logger("transformers")


class EnforceEosIfCTCStops(LogitsProcessor):
    """Logits processor (to use with HuggingFace `generate()` method :
    https://huggingface.co/docs/transformers/v4.24.0/en/main_classes/
    text_generation#transformers.generation_utils.GenerationMixin).

    This logit processor simply ensure that after hitting logzero likelihood for all tokens eos is generated.

    Args:
        eos_token_id (int): ID of the EOS token.
        log_thr (float): Value to use for logzero.
    """

    def __init__(self, eos_token_id: int, log_thr: float = -10000000000.0):
        super().__init__()
        self.log_thr = log_thr
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.Tensor) -> torch.Tensor:
        should_enforce_stop = scores.max(dim=1).values <= self.log_thr
        mask = should_enforce_stop.unsqueeze(dim=-1).expand(scores.size())
        eos_mask = torch.zeros_like(mask, dtype=torch.bool)
        eos_mask[:, self.eos_token_id] = True
        mask = mask & eos_mask
        scores = torch.where(~mask, scores, self.log_thr / 2)
        return scores


def compute_metrics(tokenizer, pred, wandb_pred_to_save=10):
    pred_ids = pred.predictions

    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = [label if label else "-" for label in tokenizer.batch_decode(label_ids, skip_special_tokens=True)]
    metrics = compute_measures(label_str, pred_str)
    del metrics["ops"]
    del metrics["truth"]
    del metrics["hypothesis"]
    if wandb.run is not None:
        write_wandb_pred(pred_str, label_str, rows_to_log=wandb_pred_to_save)

    return {"cer": cer(label_str, pred_str), **metrics}


class FrozenLayersManager(TrainerCallback):
    def __init__(
        self,
        enc_layers_to_freeze,
        dec_layers_to_freeze,
        steps_to_freeze_enc,
        steps_to_freeze_dec,
        freeze_cross_attention=False,
        freeze_others=False,
        callbacks=None,
    ):
        super().__init__()
        self.enc_layers_to_freeze = enc_layers_to_freeze
        self.dec_layers_to_freeze = dec_layers_to_freeze
        self.steps_to_freeze_enc = steps_to_freeze_enc
        self.steps_to_freeze_dec = steps_to_freeze_dec
        self.freeze_cross_attention = freeze_cross_attention
        self.freeze_others = freeze_others
        self.callbacks = callbacks

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        curr_model: SpeechEncoderDecoderModel = kwargs["model"]
        curr_model.train()
        curr_model.encoder.train()
        curr_model.decoder.train()
        if self.enc_layers_to_freeze > 0:
            for name, param in curr_model.encoder.named_parameters():
                if name.startswith("wav2vec2.encoder.layers"):
                    layer = int(name.split(".")[3])
                    if layer < self.enc_layers_to_freeze:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
                elif self.enc_layers_to_freeze > 0 and name.startswith("wav2vec2.encoder"):
                    param.requires_grad = False
                elif "adapter" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = not self.freeze_others

        if self.dec_layers_to_freeze > 0:
            for name, param in curr_model.decoder.named_parameters():
                if name.startswith("transformer.h."):
                    if "cross" in name and not self.freeze_cross_attention:
                        param.requires_grad = True
                    elif "adapter" in name:
                        param.requires_grad = True
                    else:
                        layer = int(name.split(".")[2])
                        if layer < self.dec_layers_to_freeze:
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                else:
                    param.requires_grad = not self.freeze_others

        if self.freeze_others:
            curr_model.freeze_feature_encoder()
        curr_model.decoder.lm_head.weight.requires_grad = not self.freeze_others

        if self.callbacks:
            for callback in self.callbacks:
                callback()

        logger.debug(str([n for n, p in curr_model.named_parameters() if p.requires_grad]))
        logger.info(
            f"Model info: total parameters - {curr_model.num_parameters()}, trainable parameters - "
            f"{sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, curr_model.parameters())])}"
        )

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        curr_model: SpeechEncoderDecoderModel = kwargs["model"]
        if state.global_step == self.steps_to_freeze_enc:
            logger.info(f"Step: {state.global_step} encoder unfrozen.")
            self.reactivate_params(curr_model, curr_model.encoder.parameters())

        if state.global_step == self.steps_to_freeze_dec:
            logger.info(f"Step: {state.global_step} decoder unfrozen.")
            self.reactivate_params(curr_model, curr_model.decoder.parameters())

    @staticmethod
    def reactivate_params(curr_model, params_to_activate):
        for param in params_to_activate:
            param.requires_grad = True
        logger.debug([n for n, p in curr_model.named_parameters() if p.requires_grad])
        logger.info(
            f"Model info: total parameters - {curr_model.num_parameters()}, trainable parameters - "
            f"{sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, curr_model.parameters())])}"
        )


class AdditionalLossPrinterCallback(TrainerCallback):
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        state.__class__ = make_dataclass(
            "state_derived",
            [("additional_logs", List[List[float]], field(default_factory=list))],
            bases=(TrainerState,),
        )
        state.additional_logs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if hasattr(state, "additional_logs") and len(state.additional_logs) > 0:
            enc_loss, dec_loss = torch.tensor(state.additional_logs).mean(axis=0)
            if state.is_local_process_zero:
                logs["enc_loss"] = float(enc_loss)
                logs["dec_loss"] = float(dec_loss)
            state.additional_logs = []


class AdditionalLossTrackerTrainer(Seq2SeqTrainer):
    """Custom trainer to log both losses"""

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        MAX: Subclassed to compute training accuracy.

        How the loss is computed by Trainer. By default, all models return the loss in
        the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)

        if hasattr(self.state, "additional_logs"):
            self.state.additional_logs.append([outputs.enc_loss.mean(), outputs.dec_loss.mean()])

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of
            # ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


def audio_object_stripper(audio, key="array"):
    return audio[key] if isinstance(audio, dict) and key in audio else audio


@dataclass
class Seq2SeqDataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        feature_extractor (:class:`~transformers.Wav2Vec2FeatureExtractor`)
            The feature extractor used for processing the data.
        tokenizer (:class:`~transformers.PreTrainedTokenizerFast`)
            The processor used for processing the text data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`,
        defaults to :obj:`True`):
            Select a strategy to pad the returned sequences
            (according to the model's padding side and padding index) among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    Based upon: https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/
                /Fine_tuning_Wav2Vec2_for_English_ASR.ipynb
    """

    feature_extractor: Wav2Vec2FeatureExtractor
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    sampling_rate: Optional[int] = 16_000
    audio_path: str = None
    text_path: str = None
    rename_features: bool = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> BatchFeature:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = self.feature_extractor(
            [audio_object_stripper(feature[self.audio_path]) for feature in features],
            sampling_rate=self.sampling_rate,
        )

        labels = self.tokenizer.batch_encode_plus(
            [feature[self.text_path] for feature in features],
            return_attention_mask=True,
            padding="longest",
            return_tensors="pt",
        )

        batch = self.feature_extractor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        labels = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        batch["labels"] = labels

        if self.rename_features and "input_features" in batch:
            batch["input_values"] = batch["input_features"]
            del batch["input_features"]

        return batch


@dataclass
class Seq2SeqDataCollatorWithPaddingAndConvId(Seq2SeqDataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> BatchFeature:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = self.feature_extractor(
            [feature["input_values"] for feature in features],
            padding=True,
            sampling_rate=self.sampling_rate,
        )
        labels = self.tokenizer.batch_encode_plus(
            [feature["labels"] for feature in features],
            return_attention_mask=True,
            padding="longest",
            return_tensors="pt",
        )

        batch = self.feature_extractor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        labels = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        batch["labels"] = labels
        batch["conv_ids"] = [feature["recording"] for feature in features]

        if "input_features" in batch:
            batch["input_values"] = batch["input_features"]
            del batch["input_features"]
        return batch


def filter_sequences_in_range_batched(batch: List[int], max_input_len: int, min_input_len: int):
    arr = np.array(batch)
    return (arr <= max_input_len) & (arr >= min_input_len)


def extract_lens_batched(audios: List[List[float]], len_column: str, sampling_rate: int):
    lens = [len(audio_object_stripper(example)) / sampling_rate for example in audios]
    batch = {len_column: lens}
    return batch


def filter_wrongly_annotated_segments_batched(batch: List[str]):
    return map(lambda x: x != "ignore_time_segment_in_scoring", batch)


def remove_unks_batched(batch: List[str], unk_token: str, label_column: str):
    return {label_column: [sequence.replace(unk_token, "") for sequence in batch]}


tedlium_contractions = [" 's", " 't", " 're", " 've", " 'm", " 'll", " 'd", " 'clock", " 'all"]


def replace_contractions(text):
    for contraction in tedlium_contractions:
        text = text.replace(contraction, contraction[1:])
    return text


def fix_apostrophes_batched(batch: List[str], label_column: str):
    # replace spaced apostrophes with un-spaced (it 's -> it's)
    return {label_column: [replace_contractions(sequence).replace(r"\s+ '", r" '") for sequence in batch]}


def filter_empty_transcriptions(batch: List[str]):
    return [example != "" for example in batch]


def preprocess_cv_labels(batch: List[str], label_column: str):
    processed = []
    for transcription in batch:
        if transcription.startswith('"') and transcription.endswith('"'):
            # we can remove trailing quotation marks as they do not affect the transcription
            transcription = transcription[1:-1]

        if transcription[-1] not in [".", "?", "!"]:
            # append a full-stop to sentences that do not end in punctuation
            transcription = transcription + "."
        transcription = transcription.replace('""', '"')
        processed.append(transcription)

    return {label_column: processed}


def filter_out_sequence_from_dataset(
    df: Dataset, max_input_len: float = 5.0, min_input_len: float = 0.1, length_column="input_len"
) -> Dataset:
    """Filters out sequences form dataset which are longer than provided threshold"""
    lengths = np.array(df[length_column])
    indexes_ok = np.argwhere(np.logical_and(lengths <= max_input_len, lengths >= min_input_len))
    df = df.select(indexes_ok.flatten())
    return df


def group_params(model, weight_decay, learning_rate, cross_attention_scaling_factor):
    """Add different weight decay and lr rate for specific layers"""
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    return [
        {
            "params": [p for n, p in model.named_parameters() if (n in decay_parameters and "cross" not in n)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if (n not in decay_parameters and "cross" not in n)],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if (n in decay_parameters and "cross" in n)],
            "weight_decay": weight_decay,
            "lr": learning_rate * cross_attention_scaling_factor,
        },
        {
            "params": [p for n, p in model.named_parameters() if (n not in decay_parameters and "cross" in n)],
            "weight_decay": 0.0,
            "lr": learning_rate * cross_attention_scaling_factor,
        },
    ]


def write_wandb_pred(pred_str, label_str, rows_to_log=10):
    current_step = wandb.run.step
    columns = ["id", "label_str", "hyp_str"]
    wandb.log(
        {
            f"eval_predictions/step_{int(current_step)}": wandb.Table(
                columns=columns,
                data=[
                    [i, ref, hyp] for i, hyp, ref in zip(range(min(len(pred_str), rows_to_log)), pred_str, label_str)
                ],
            )
        },
        current_step,
    )


def fetch_AED_config(enc_config_path, dec_config_path, base_config, config_overrides):
    enc_config = AutoConfig.from_pretrained(enc_config_path)
    dec_config = AutoConfig.from_pretrained(dec_config_path)
    config = JointCTCAttentionEncoderDecoderConfig.from_encoder_decoder_configs(enc_config, dec_config)
    if config_overrides is not None:
        logger.info(f"Overriding config: {config_overrides}")
        d = dict(x.split("=") for x in config_overrides.split(","))
        base_config.update(d)
    kwargs_encoder = {
        argument[len("encoder_") :]: value for argument, value in base_config.items() if argument.startswith("encoder_")
    }
    kwargs_decoder = {
        argument[len("decoder_") :]: value
        for argument, value in base_config.items()
        if argument.startswith("decoder_") and argument != "decoder_start_token_id"
    }
    config.encoder.update(kwargs_encoder)
    config.decoder.update(kwargs_decoder)
    config.update(base_config)
    return config


def prepare_dataset(
    dataset,
    dataset_name,
    length_column_name,
    text_column_name,
    audio_column_name,
    preprocessing_num_workers,
    writer_batch_size,
    train_split,
    validation_split,
    fix_apostrophes,
    remove_train_unks,
    apply_augmentations,
    unk_token,
    sampling_rate,
    max_input_len,
    min_input_len,
    validation_slice,
):
    if length_column_name not in dataset[train_split].column_names:
        logger.info(f"Extracting audio lens.")
        dataset = dataset.map(
            extract_lens_batched,
            num_proc=preprocessing_num_workers,
            input_columns=[audio_column_name],
            batched=True,
            writer_batch_size=writer_batch_size,
            fn_kwargs={"sampling_rate": sampling_rate, "len_column": length_column_name},
        )

    logger.info(f"Filtering out too long and too short sequences from dataset.")
    dataset[train_split] = dataset[train_split].filter(
        filter_sequences_in_range_batched,
        batched=True,
        input_columns=[length_column_name],
        num_proc=preprocessing_num_workers,
        writer_batch_size=writer_batch_size,
        fn_kwargs={"max_input_len": max_input_len, "min_input_len": min_input_len},
    )

    logger.info(f"Filtering unlabeled data from dataset.")
    dataset = dataset.filter(
        filter_wrongly_annotated_segments_batched,
        batched=True,
        input_columns=[text_column_name],
        writer_batch_size=writer_batch_size,
        num_proc=preprocessing_num_workers,
    )

    dataset = dataset.filter(
        filter_empty_transcriptions,
        input_columns=[text_column_name],
        batched=True,
        writer_batch_size=writer_batch_size,
        num_proc=preprocessing_num_workers,
    )

    if remove_train_unks:
        logger.info(f"Removing UNKs from training data.")
        dataset[train_split] = dataset[train_split].map(
            remove_unks_batched,
            batched=True,
            input_columns=[text_column_name],
            num_proc=preprocessing_num_workers,
            writer_batch_size=writer_batch_size,
            fn_kwargs={"unk_token": unk_token, "label_column": text_column_name},
        )
    if fix_apostrophes:
        logger.info(f"Fixing apostrophes in dataset.")
        dataset = dataset.map(
            fix_apostrophes_batched,
            input_columns=[text_column_name],
            batched=True,
            num_proc=preprocessing_num_workers,
            writer_batch_size=writer_batch_size,
            fn_kwargs={"label_column": text_column_name},
        )

    if dataset_name == "mozilla-foundation/common_voice_13_0":
        logger.info(f"Fixing labels for commonvoice.")
        dataset = dataset.map(
            preprocess_cv_labels,
            input_columns=[text_column_name],
            batched=True,
            writer_batch_size=writer_batch_size,
            num_proc=preprocessing_num_workers,
            fn_kwargs={"label_column": text_column_name},
        )

    if apply_augmentations:
        logger.info(f"Setting augmentations transform.")
        augmenter = Compose(
            [
                #     TimeMask(max_band_part=0.05, p=0.05),
                #     TimeMask(max_band_part=0.05, p=0.05),
                #     TimeMask(max_band_part=0.05, p=0.05),
                #     TimeMask(max_band_part=0.05, p=0.05),
                #     AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.2),
                TimeStretch(min_rate=0.9, max_rate=1.1, p=0.5),
                #     PitchShift(min_semitones=-4, max_semitones=4, p=0.2),
                #     Shift(min_fraction=-0.5, max_fraction=0.5, p=0.2),
                #     TanhDistortion(min_distortion=0, max_distortion=0.2, p=0.2)
            ]
        )
        dataset[train_split].set_transform(
            lambda batch: {
                audio_column_name: [
                    augmenter(
                        np.array(audio_object_stripper(audio), dtype=np.float32),
                        sample_rate=sampling_rate,
                    )
                    for audio in batch[audio_column_name]
                ]
            },
            columns=[audio_column_name],
            output_all_columns=True,
        )

    if validation_slice:
        logger.info(f"Selecting specified part of validation indexes.")
        dataset[validation_split] = dataset[validation_split].select(range(validation_slice))
    return dataset


def activate_joint_decoding(model, ctc_weight, ctc_margin, num_tokens, eos_token, external_lm, external_lm_weight):
    def new_beam(*args, **kwargs):
        logits_processor = LogitsProcessorList(
            [
                EnforceEosIfCTCStops(
                    eos_token,
                    log_thr=-10000000000.0 * ctc_weight if ctc_weight > 0 else -10000000000.0,
                )
            ]
        )
        kwargs.update({"logits_processor": logits_processor})
        return model.joint_beam_search(
            *args,
            **kwargs,
            ctc_weight=ctc_weight,
            margin=ctc_margin,
            ctc_beam_width=num_tokens,
            external_lm=external_lm,
            external_lm_weight=external_lm_weight,
        )

    model.beam_search = new_beam


# def unpack_predictions(file_path, tokenizer_name):
#     import pickle
#     from transformers import AutoTokenizer
#     tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
#
#     with open(file_path, 'rb') as f:
#         predictions = pickle.load(f)
#     ref = tokenizer.batch_decode(predictions.label_ids, skip_special_tokens=True)
#     hyp = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)


def average_dicts(*dicts):
    result = {}

    # Count the number of dictionaries
    num_dicts = len(dicts)

    for d in dicts:
        for key, value in d.items():
            if key in result:
                result[key] += value
            else:
                result[key] = value

    return result, num_dicts


def average_checkpoints(experiment_dir):
    checkpoints = glob.glob(f"{experiment_dir}/checkpoint*/pytorch_model.bin")
    state_dicts = [torch.load(checkpoint) for checkpoint in checkpoints]
    sum_state_dict, n_checkpoints = average_dicts(*state_dicts)
    del state_dicts
    average_dict = {key: sum_state_dict[key].div(n_checkpoints) for key in sum_state_dict}
    dst_path = os.path.join(experiment_dir, "average_checkpoint")
    shutil.copytree(os.path.dirname(checkpoints[0]), dst_path, dirs_exist_ok=True)
    shutil.copytree(os.path.join(experiment_dir, "tokenizer"), dst_path, dirs_exist_ok=True)
    shutil.copytree(os.path.join(experiment_dir, "feature_extractor"), dst_path, dirs_exist_ok=True)
    torch.save(average_dict, os.path.join(dst_path, "pytorch_model.bin"))
    return dst_path
