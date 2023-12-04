import importlib
import json
from dataclasses import field, make_dataclass
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    EarlyStoppingCallback,
    SequenceFeatureExtractor,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.utils import logging

from utilities.data_utils import audio_object_stripper
from utilities.general_utils import (
    FunctionReturnWrapper,
    resolve_attribute_from_nested_class,
)
from utilities.training_arguments import DataTrainingArguments, GeneralTrainingArguments

logger = logging.get_logger("transformers")


class DelayedStartWrapper:
    def __init__(self, callback: FunctionReturnWrapper, delay_steps: int):
        self.callback = callback
        self.start_at = delay_steps
        self.active = False

    def new_step(self, step: int):
        if step >= self.start_at:
            self.active = True

    def __call__(self, *args, **kwargs):
        if self.active:
            return self.callback(*args, **kwargs)
        return args[0]


class DataPreprocessingManagerCallback(TrainerCallback):
    def __init__(
        self,
        preprocessing_config: List[Dict],
        dataset: Dataset,
        audio_column_name: str,
        feature_extractor: SequenceFeatureExtractor,
    ):
        super().__init__()
        self.train_dataset = dataset
        self.audio_column_name = audio_column_name
        self.transforms = []
        for config in preprocessing_config:
            if config["name"] == "feature_extractor":
                fun = feature_extractor
            else:
                module, attribute = config["name"].rsplit(".", 1)
                fun = resolve_attribute_from_nested_class(importlib.import_module(module), attribute)(
                    **config["params"]
                )

            self.transforms.append(
                (
                    DelayedStartWrapper(
                        FunctionReturnWrapper(fun, config["return_behaviour"]), config["steps_before_activation"]
                    ),
                    config["fn_call_params"],
                )
            )

    @staticmethod
    def transformer(audio: Union[np.ndarray, torch.Tensor], transforms: List[Tuple[DelayedStartWrapper, Dict]]):
        if not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio, dtype=torch.float32)
        for augmentation, fn_call_params in transforms:
            audio = augmentation(audio, **fn_call_params)
        return audio

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.train_dataset.set_transform(
            lambda batch: {
                self.audio_column_name: [
                    self.transformer(audio_object_stripper(audio), self.transforms)
                    for audio in batch[self.audio_column_name]
                ]
            },
            columns=[self.audio_column_name],
            output_all_columns=True,
        )

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        for transform in self.transforms:
            transform[0].new_step(state.global_step)

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        for transform in self.transforms:
            transform[0].new_step(state.global_step)


class AdditionalLossPrinterCallback(TrainerCallback):
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        state.__class__ = make_dataclass(
            "state_derived",
            [("additional_logs", List[List[float]], field(default_factory=list))],
            bases=(TrainerState,),
        )
        state.additional_logs = []

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if hasattr(state, "additional_logs") and len(state.additional_logs) > 0:
            enc_loss, dec_loss = torch.tensor(state.additional_logs).mean(axis=0)
            if state.is_local_process_zero:
                logs["enc_loss"] = float(enc_loss)
                logs["dec_loss"] = float(dec_loss)
            state.additional_logs = []


def init_callbacks(
    data_args: DataTrainingArguments,
    training_args: GeneralTrainingArguments,
    dataset: Dataset,
    feature_extractor: SequenceFeatureExtractor,
):
    callbacks = []
    if data_args.data_preprocessing_config:
        with open(data_args.data_preprocessing_config) as config_handle:
            callbacks.append(
                DataPreprocessingManagerCallback(
                    preprocessing_config=json.load(config_handle),
                    dataset=dataset,
                    audio_column_name=data_args.audio_column_name,
                    feature_extractor=feature_extractor,
                )
            )
    else:
        callbacks.append(
            DataPreprocessingManagerCallback(
                preprocessing_config=[
                    {
                        "name": "feature_extractor",
                        "steps_before_activation": 0,
                        "fn_call_params": {
                            "return_attention_mask": False,
                            "sampling_rate": 16000,
                            "return_tensors": "pt",
                        },
                        "return_behaviour": ["input_features[0]"],
                    }
                ],
                dataset=dataset,
                audio_column_name=data_args.audio_column_name,
                feature_extractor=feature_extractor,
            )
        )
    if training_args.early_stopping_patience > -1:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience))
    if training_args.track_ctc_loss:
        callbacks.append(AdditionalLossPrinterCallback())
    return callbacks
