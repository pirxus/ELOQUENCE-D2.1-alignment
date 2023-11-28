"""Kaldi dataset builder"""
import logging
import math
import os
from itertools import groupby
from typing import Iterable, List, Tuple, Union

import datasets
import kaldiio
import librosa
import numpy as np

_FILEPATHS = {
    "feats": "wav.scp",
    "segments": "segments",
    "transcripts": "text",
    "channels2recordings": "reco2file_and_channel",
}


class KaldiDataset(datasets.GeneratorBasedBuilder):
    """Dataset builder for Fisher dataset"""

    DEFAULT_WRITER_BATCH_SIZE = 50  # the default size of the batch may not fit in memory

    def __init__(self, metadata_dir: os.PathLike, splits: List[str], sampling_rate: int = 16000, **kwargs):
        super().__init__(**kwargs)
        self.sampling_rate = sampling_rate
        self.data_dir = metadata_dir
        self.splits = splits

    def _info(self):
        return datasets.DatasetInfo(
            description="",
            supervised_keys=None,
            homepage="",
        )

    def _prepare_split_single(
        self,
        gen_kwargs: dict,
        fpath: str,
        file_format: str,
        max_shard_size: int,
        split_info: datasets.SplitInfo,
        check_duplicate_keys: bool,
        job_id: int,
    ) -> Iterable[Tuple[int, bool, Union[int, tuple]]]:
        self.info.features = None  # Disable unnecessary type check and conversion that slows generation
        return super()._prepare_split_single(
            gen_kwargs, fpath, file_format, max_shard_size, split_info, check_duplicate_keys, job_id
        )

    def _split_generators(self, _):
        """Generate dataset splits"""
        splits = [
            datasets.SplitGenerator(
                name=split,
                gen_kwargs=self._fetch_split_meta(split),
            )
            for split in self.splits
        ]
        return splits

    @staticmethod
    def _split_text_string(text):
        """Split text string into uttid and transcript"""
        parts = text.strip().split(maxsplit=1)
        if len(parts) == 1:
            parts.append("")
        return parts

    def _fetch_split_meta(self, split: str):
        """Fetch split meta data from kaldi-like dataset"""

        with open(os.path.join(self.data_dir, split, _FILEPATHS["transcripts"])) as file:
            texts = dict(map(lambda line: self._split_text_string(line), file))  # creates (segment_id -> text) mapping

        with open(os.path.join(self.data_dir, split, _FILEPATHS["segments"])) as file:
            segments = dict(
                map(lambda s: self._parse_segment_info(*s.strip().split()), file)
            )  # creates (segment_id -> wav_id, start, end) mapping

        # load kaldiio feature generator
        featfile = os.path.join(self.data_dir, split, _FILEPATHS["feats"])
        feats_generator = kaldiio.load_scp(featfile)
        segments = [(*segments[uttid], uttid, transcript) for (uttid, transcript) in texts.items()]
        grouped_by_recordings = [(k, list(v)) for k, v in groupby(sorted(segments), key=lambda segment: segment[0])]
        return {
            "recordings": grouped_by_recordings,
            "features": feats_generator,
        }

    def _generate_examples(self, recordings, features):
        """Generator for split examples fetching"""
        for recording, segments in recordings:
            sampling_rate, audio = features[recording]
            if audio.dtype == np.int16:
                audio = librosa.util.buf_to_float(audio, n_bytes=audio.dtype.itemsize)
            else:
                raise ValueError("Data type of input audio is not int16.")
            if len(audio.shape) > 1:
                raise ValueError(f"Recording {recording} does not have single channel.")
            if sampling_rate != self.sampling_rate or len(audio.shape) > 1:
                logging.debug(f"Resampled {recording} from {sampling_rate} to {self.sampling_rate}")
                audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=self.sampling_rate)
            sorted_segments = sorted(segments, key=lambda x: x[1])
            for index, (_, start, end, uttid, transcript) in enumerate(sorted_segments):
                audio_cropped = self._crop_audio(audio, self.sampling_rate, start, end)
                text = self.preprocess_text(transcript)
                yield f"{recording}_{index}", {
                    "input_values": audio_cropped,
                    "labels": text,
                    "uttid": uttid,
                    "recording": recording,
                    "turn_index": index,
                    "input_len": end - start,
                }

    @staticmethod
    def _parse_segment_info(segment_key, uri, start, end):
        """Parse segment info"""
        return segment_key, (uri, float(start), float(end))

    @staticmethod
    def _crop_audio(audio, sampling_rate, start, end):
        """Crop audio"""
        return audio[math.floor(sampling_rate * start) : math.ceil(end * sampling_rate)]

    @staticmethod
    def preprocess_text(utterance_batch: List[str]):
        """Preprocess text"""
        return utterance_batch
