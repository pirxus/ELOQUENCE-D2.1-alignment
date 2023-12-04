"""Utilities for data loading and preprocessing."""
import json
from typing import Dict, List, Union

import numpy as np
from datasets import (
    Audio,
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)
from transformers.utils import logging

logger = logging.get_logger("transformers")

tedlium_contractions = [" 's", " 't", " 're", " 've", " 'm", " 'll", " 'd", " 'clock", " 'all"]


def audio_object_stripper(audio: Union[Dict, np.ndarray, List[float]], key="array"):
    """Strips audio object to numpy array."""
    audio_array = audio[key] if isinstance(audio, dict) and key in audio else audio
    trimmed = np.trim_zeros(audio_array)
    return trimmed


def filter_sequences_in_range_batched(batch: List[int], max_input_len: int, min_input_len: int) -> List[bool]:
    """Filters out sequences form dataset which are in bounds."""
    arr = np.array(batch)
    return (arr <= max_input_len) & (arr >= min_input_len)


def extract_lens_batched(audios: List[List[float]], len_column: str, sampling_rate: int) -> Dict[str, List[float]]:
    """Extracts audio lens from dataset."""
    lens = [len(audio_object_stripper(example)) / sampling_rate for example in audios]
    batch = {len_column: lens}
    return batch


def filter_wrongly_annotated_segments_batched(batch: List[str]) -> List[bool]:
    """Filters out segments which are wrongly annotated."""
    return list(map(lambda x: x != "ignore_time_segment_in_scoring", batch))


def remove_unks_batched(batch: List[str], unk_token: str, label_column: str) -> Dict[str, List[str]]:
    """Removes UNK tokens from dataset."""
    return {label_column: [sequence.replace(unk_token, "") for sequence in batch]}


def replace_contractions(text: str) -> str:
    """Replaces contractions in text."""
    for contraction in tedlium_contractions:
        text = text.replace(contraction, contraction[1:])
    return text


def fix_apostrophes_batched(batch: List[str], label_column: str) -> Dict[str, List[str]]:
    """Fixes apostrophes in dataset."""
    return {label_column: [replace_contractions(sequence).replace(r"\s+ '", r" '") for sequence in batch]}


def filter_empty_transcriptions(batch: List[str]) -> List[bool]:
    """Filters out empty transcriptions."""
    return [example != "" for example in batch]


def preprocess_cv_labels(batch: List[str], label_column: str) -> Dict[str, List[str]]:
    """Preprocesses labels for commonvoice dataset."""
    processed = []
    for transcription in batch:
        if transcription.startswith('"') and transcription.endswith('"'):
            # we can remove trailing quotation marks as they do not affect the transcription
            transcription = transcription[1:-1]

        transcription = transcription.replace(r"[,.?!:;]", "")
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


def prepare_dataset(
    dataset: DatasetDict,
    dataset_name: str,
    length_column_name: str,
    text_column_name: str,
    audio_column_name: str,
    preprocessing_num_workers: int,
    writer_batch_size: int,
    train_split: str,
    fix_apostrophes: bool,
    remove_train_unks: bool,
    unk_token: str,
    sampling_rate: int,
    max_input_len: float,
    min_input_len: float,
) -> DatasetDict:
    """Preprocesses dataset."""
    if train_split is not None and length_column_name not in dataset[train_split].column_names:
        logger.info(f"Extracting audio lens.")
        dataset = dataset.map(
            extract_lens_batched,
            num_proc=preprocessing_num_workers,
            input_columns=[audio_column_name],
            batched=True,
            writer_batch_size=writer_batch_size,
            fn_kwargs={"sampling_rate": sampling_rate, "len_column": length_column_name},
        )

    if train_split is not None:
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

    dataset = dataset.cast_column(audio_column_name, Audio(sampling_rate=sampling_rate))

    if train_split is not None and remove_train_unks:
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

    logger.info(str(dataset))
    return dataset


def merge_splits(dataset: DatasetDict, splits_to_merge: List[str], new_name: str) -> DatasetDict:
    """Merge splits of the provided dataset."""
    if len(splits_to_merge) > 1:
        dataset[new_name] = concatenate_datasets([dataset[split] for split in splits_to_merge])
        for split in splits_to_merge:
            if split != new_name:
                del dataset[split]
    if len(splits_to_merge) == 1 and splits_to_merge[0] != new_name:
        dataset[new_name] = dataset[splits_to_merge[0]]
        del dataset[splits_to_merge[0]]
    return dataset


def join_datasets(
    dataset1: DatasetDict,
    dataset2: DatasetDict,
    test_splits: List[str],
    local_dataset_prefix: str,
    train_split: str,
    validation_split: str,
) -> DatasetDict:
    """Add local datasets to the global dataset."""
    if train_split is not None:
        if train_split in dataset1:
            dataset1[train_split] = concatenate_datasets([dataset1[train_split], dataset2[train_split]])
        else:
            dataset1[train_split] = dataset2[train_split]
    if validation_split is not None:
        if validation_split in dataset1:
            dataset1[validation_split] = concatenate_datasets([dataset1[validation_split], dataset2[validation_split]])
        else:
            dataset1[validation_split] = dataset2[validation_split]
    for split in test_splits:
        dataset1[local_dataset_prefix.split("/")[-1] + "_" + split] = dataset2[split]
    return dataset1


def load_multiple_datasets(
    config_path: str,
    num_proc: int,
    writer_batch_size: int,
    sampling_rate: int,
    max_input_len: float,
    min_input_len: float,
    global_len_column: str,
    global_text_column: str,
    global_audio_column: str,
    global_train_split: str,
    global_validation_split: str,
) -> DatasetDict:
    """Loads multiple datasets, preprocess them and join to single dataset instance."""
    with open(config_path) as config_handle:
        config_dict = json.load(config_handle)
    dataset_merged = DatasetDict()
    for dataset_config in config_dict:
        logger.info(f"Loading dataset {dataset_config['dataset_name']}")
        if dataset_config["load_from_disk"]:
            dataset = load_from_disk(
                dataset_config["dataset_name"], keep_in_memory=False, **dataset_config["additional_args"]
            )

        else:
            dataset = load_dataset(
                dataset_config["dataset_name"],
                keep_in_memory=False,
                num_proc=num_proc,
                **dataset_config["additional_args"],
            )
        new_train_split_name = global_train_split if len(dataset_config["train_splits"]) > 0 else None
        new_dev_split_name = global_validation_split if len(dataset_config["dev_splits"]) > 0 else None
        dataset = merge_splits(dataset, dataset_config["train_splits"], new_train_split_name)
        dataset = merge_splits(dataset, dataset_config["dev_splits"], new_dev_split_name)

        logger.info(f"Preprocessing dataset {dataset_config['dataset_name']}")
        dataset_processed = prepare_dataset(
            dataset=dataset,
            dataset_name=dataset_config["dataset_name"],
            length_column_name=dataset_config["length_column_name"],
            text_column_name=dataset_config["text_column_name"],
            audio_column_name=dataset_config["audio_column_name"],
            preprocessing_num_workers=num_proc,
            writer_batch_size=writer_batch_size,
            train_split=new_train_split_name,
            fix_apostrophes=dataset_config["fix_apostrophes"],
            remove_train_unks=dataset_config["remove_train_unks"],
            unk_token=dataset_config["unk_token"],
            sampling_rate=sampling_rate,
            max_input_len=max_input_len,
            min_input_len=min_input_len,
        )
        dataset_renamed = dataset_processed.rename_columns(
            {
                dataset_config["length_column_name"]: global_len_column,
                dataset_config["text_column_name"]: global_text_column,
                dataset_config["audio_column_name"]: global_audio_column,
            }
        )
        dataset_local = dataset_renamed.remove_columns(
            list(
                set()
                .union(*dataset_renamed.column_names.values())
                .difference([global_len_column, global_text_column, global_audio_column])
            )
        )
        dataset_merged = join_datasets(
            dataset_merged,
            dataset_local,
            dataset_config["test_splits"],
            dataset_config["dataset_id"],
            new_train_split_name,
            new_dev_split_name,
        )
    return dataset_merged


def get_dataset(
    datasets_creation_config_path: str,
    dataset_name: str,
    dataset_config: str,
    preprocessing_num_workers: int,
    writer_batch_size: int,
    sampling_rate: int,
    max_input_len: float,
    min_input_len: float,
    len_column: str,
    text_column: str,
    audio_column: str,
    train_split: str,
    validation_split: str,
    unk_token: str,
    fix_apostrophes: bool,
    remove_train_unks: bool,
) -> DatasetDict:
    """Loads single or multiple datasets, preprocess, and merge them."""
    if datasets_creation_config_path is not None:
        dataset = load_multiple_datasets(
            config_path=datasets_creation_config_path,
            num_proc=preprocessing_num_workers,
            writer_batch_size=writer_batch_size,
            sampling_rate=sampling_rate,
            max_input_len=max_input_len,
            min_input_len=min_input_len,
            global_len_column=len_column,
            global_text_column=text_column,
            global_audio_column=audio_column,
            global_train_split=train_split,
            global_validation_split=validation_split,
        )
    else:
        if dataset_config is not None:
            dataset = load_dataset(
                dataset_name, dataset_config, keep_in_memory=False, num_proc=preprocessing_num_workers
            )
        else:
            dataset = load_from_disk(dataset_name, keep_in_memory=False)

        # 3. Preprocess dataset
        dataset = prepare_dataset(
            dataset=dataset,
            dataset_name=dataset_name,
            length_column_name=len_column,
            text_column_name=text_column,
            audio_column_name=audio_column,
            preprocessing_num_workers=preprocessing_num_workers,
            writer_batch_size=writer_batch_size,
            train_split=train_split,
            fix_apostrophes=fix_apostrophes,
            remove_train_unks=remove_train_unks,
            unk_token=unk_token,
            sampling_rate=sampling_rate,
            max_input_len=max_input_len,
            min_input_len=min_input_len,
        )
    return dataset
