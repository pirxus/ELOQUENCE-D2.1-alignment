import glob
import math
import os
import pickle  # nosec
import shutil
from dataclasses import dataclass, field, make_dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import tqdm
from datasets import Dataset
from jiwer import cer, compute_measures
from torchaudio.transforms import SpeedPerturbation
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
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

import wandb
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


def compute_metrics_ctc(tokenizer, pred, wandb_pred_to_save=10):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = [label if label else "-" for label in tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)]
    metrics = compute_measures(label_str, pred_str)
    del metrics["ops"]
    del metrics["truth"]
    del metrics["hypothesis"]
    if wandb.run is not None:
        write_wandb_pred(pred_str, label_str, rows_to_log=wandb_pred_to_save)

    return {"cer": cer(label_str, pred_str), **metrics}


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


class AugmentationManagerCallback(TrainerCallback):
    def __init__(self, activate_aug_after_steps, model_config_path="encoder.config"):
        super().__init__()
        if model_config_path is None:
            model_config_path = []
        self.activate_aug_after_steps = activate_aug_after_steps
        self.model_config_path = model_config_path

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs["model"]
        resolve_attribute_from_nested_class(model, self.model_config_path).apply_spec_augment = False

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs["model"]
        config = resolve_attribute_from_nested_class(model, self.model_config_path)
        if state.global_step >= self.activate_aug_after_steps and not config.apply_spec_augment:
            config.apply_spec_augment = True
            logger.info(f"Step: {state.global_step} augmentations activated.")


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


def resolve_attribute_from_nested_class(obj, attr_spec):
    for attr in attr_spec.split("."):
        try:
            obj = obj[attr]
        except (TypeError, KeyError):
            obj = getattr(obj, attr)
    return obj


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
    dataset: Dataset, max_input_len: float = 5.0, min_input_len: float = 0.1, length_column="input_len"
) -> Dataset:
    """Filters out sequences form dataset which are longer than provided threshold"""
    lengths = np.array(dataset[length_column])
    indexes_ok = np.argwhere(np.logical_and(lengths <= max_input_len, lengths >= min_input_len))
    dataset = dataset.select(indexes_ok.flatten())
    return dataset


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
        parsed_dict = dict(x.split("=") for x in config_overrides.split(","))
        base_config.update(parsed_dict)
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
    fix_apostrophes,
    remove_train_unks,
    apply_augmentations,
    unk_token,
    sampling_rate,
    max_input_len,
    min_input_len,
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
        speed_perturb = SpeedPerturbation(sampling_rate, [0.9, 1.1, 1.0])
        dataset[train_split].set_transform(
            lambda batch: {
                audio_column_name: [
                    speed_perturb(torch.tensor(audio_object_stripper(audio), dtype=torch.float32))[0].detach().numpy()
                    for audio in batch[audio_column_name]
                ]
            },
            columns=[audio_column_name],
            output_all_columns=True,
        )
    logger.info(str(dataset))
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


def save_nbests(path, nbests, scores, labels, tokenizer, group_size=1, batch_size=1, outputs=None):
    nbests = [tokenizer.decode(elem.tolist(), skip_special_tokens=True) for item in nbests for elem in item.unbind()]
    processed_labels = []
    if outputs is not None:
        for index, output in enumerate(outputs):
            with open(path + f"_utterance{index * batch_size}-{(index + 1) * batch_size - 1}.pkl", "wb") as f:
                pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

    for label in labels:
        label[label == -100] = tokenizer.pad_token_id
        processed_labels.extend(
            [
                tokenizer.decode(elem.tolist(), skip_special_tokens=True)
                for elem in label.repeat_interleave(group_size, dim=0)
            ]
        )
    scores = [float(elem) for item in scores for elem in item.unbind()]
    with open(path + "_scores.txt", "w") as f1:
        with open(path + "_hyps.txt", "w") as f2:
            with open(path + "_refs.txt", "w") as f3:
                for item, (sample, score, ref) in enumerate(zip(nbests, scores, processed_labels)):
                    utterance_id = f"utterance{item // group_size}-{item % group_size + 1}"
                    f1.write(f"{utterance_id} {score}\n")
                    f2.write(f"{utterance_id} {sample}\n")
                    f3.write(f"{utterance_id} {ref}\n")


def save_predictions(tokenizer, predictions, path):
    pred_ids = predictions.predictions

    label_ids = predictions.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = [label if label else "-" for label in tokenizer.batch_decode(label_ids, skip_special_tokens=True)]
    df = pd.DataFrame({"label": label_str, "prediction": pred_str})
    df.to_csv(path, index=False)


def check_and_activate_joint_decoding(gen_args, model, tokenizer, eos_token_id):
    if gen_args.decoding_ctc_weight > 0 or gen_args.external_lm_weight > 0:
        external_lm = None
        if gen_args.external_lm is not None:
            external_lm = AutoModelForCausalLM.from_pretrained(gen_args.external_lm)
            external_lm.eval()
        activate_joint_decoding(
            model,
            gen_args.decoding_ctc_weight,
            gen_args.ctc_margin,
            len(tokenizer),
            eos_token_id,
            external_lm,
            gen_args.external_lm_weight,
        )


def do_evaluate(trainer, dataset, model, tokenizer, gen_args, training_args, eos_token_id):
    check_and_activate_joint_decoding(gen_args, model, tokenizer, eos_token_id)

    trainer.args.per_device_eval_batch_size = math.ceil(
        trainer.args.per_device_eval_batch_size / gen_args.eval_beam_factor
    )
    for split in training_args.evaluation_splits:
        predictions = trainer.predict(
            dataset[split],
            output_hidden_states=True,
            num_beams=model.generation_config.num_beams * gen_args.eval_beam_factor,
        )
        logger.info(f"Metrics for {split} split: {predictions.metrics}")
        save_predictions(
            tokenizer,
            predictions,
            f"{training_args.output_dir}/" f'predictions_{split}_wer{100 * predictions.metrics["test_wer"]:.2f}.csv',
        )


def move_to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    elif isinstance(obj, dict):
        return {key: move_to_cpu(value) for key, value in obj.items()}
    elif isinstance(obj, tuple):
        return tuple(move_to_cpu(item) for item in obj)
    else:
        return obj


def postprocess_beam_outputs(outputs):
    for key in outputs:
        outputs[key] = move_to_cpu(outputs[key])
    outputs["joint_scores"] = outputs["scores"][::4]
    outputs["dec_scores"] = outputs["scores"][1::4]
    outputs["ctc_scores"] = outputs["scores"][2::4]
    outputs["external_lm_scores"] = outputs["scores"][3::4]
    outputs = dict(outputs)
    del outputs["scores"]
    del outputs["encoder_hidden_states"]
    del outputs["decoder_hidden_states"]
    return outputs


def do_generate(trainer, dataset, model, tokenizer, gen_args, training_args, gen_config, eos_token_id):
    check_and_activate_joint_decoding(gen_args, model, tokenizer, eos_token_id)

    gen_config.num_return_sequences = gen_args.num_predictions_to_return
    gen_config.return_dict_in_generate = True
    gen_config.num_beams = model.generation_config.num_beams * gen_args.eval_beam_factor
    gen_config.output_scores = True
    trainer.args.per_device_eval_batch_size = math.ceil(
        trainer.args.per_device_eval_batch_size / gen_args.eval_beam_factor
    )
    for split in training_args.evaluation_splits:
        logger.info(f"Generating predictions for split: {split}")
        dataloader = trainer.get_eval_dataloader(dataset[split])
        n_bests = []
        scores = []
        labels = []
        outputs_agg = []
        for sample in tqdm.tqdm(dataloader):
            outputs = model.generate(generation_config=gen_config, **sample)
            if gen_args.save_output_states:
                outputs_agg.append(postprocess_beam_outputs(outputs))
            n_bests.append(outputs.sequences)
            scores.append(outputs.sequences_scores)
            labels.append(sample["labels"])
        save_nbests(
            gen_args.nbest_path_to_save + "_" + split,
            n_bests,
            scores,
            labels,
            tokenizer,
            group_size=gen_args.num_predictions_to_return,
            outputs=outputs_agg,
            batch_size=trainer.args.per_device_eval_batch_size,
        )
