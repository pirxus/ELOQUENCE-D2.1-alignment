"""Dataset builder module for the raw audio version of the HOW2 dataset"""
import sys
import string
import re
import datasets
import json
import pandas as pd
from typing import Optional, List
import os
import torchaudio


class SpokenWOZ(datasets.GeneratorBasedBuilder):
    """Dataset builder for the raw audio version of the HOW2 dataset"""

    BUILDER_CONFIG_CLASS = datasets.BuilderConfig

    def __init__(self, data_dir: Optional[str], splits: List[str] = [], **kwargs):
        super().__init__(data_dir=data_dir, **kwargs)
        self.data_dir = data_dir
        self.splits = splits if splits else ['dev', 'train', 'test']

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "audio": datasets.Audio(sampling_rate=16_000),
                    "wav_id": datasets.Value('string'),
                    "turn_index": datasets.Value("int32"),
                    "text": datasets.Value("string"),
                    "span_info": datasets.Sequence(feature=datasets.Sequence(feature=datasets.Value('string'))),
                    "dialog_act": datasets.Value('string'),
                    "metadata": datasets.Value('string'),
                    "tag": datasets.ClassLabel(num_classes=2, names=['user', 'system']),
                    "start_time": datasets.Value("int32"),
                    "end_time": datasets.Value("int32"),
                    # TODO: subsequently convert to something consistent with fisher..
                    "context": datasets.Sequence(feature={
                        "turn_index": datasets.Value("int32"),
                        "text": datasets.Value("string"),
                        "span_info": datasets.Sequence(feature=datasets.Sequence(feature=datasets.Value('string'))),
                        "dialog_act": datasets.Value('string'),
                        "metadata": datasets.Value('string'),
                        "tag": datasets.ClassLabel(num_classes=2, names=['user', 'system']),
                        "start_time": datasets.Value("int32"),
                        "end_time": datasets.Value("int32"),
                    }),
                }
            ),
            supervised_keys=None,
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

    def _fetch_split_meta(self, split: str):

        split_dir = 'test' if split == 'test' else 'train_dev'
        with open(str(self.data_dir) + f'/text_5700_{split_dir}/data.json') as json_file:
            json_data = json.load(json_file)

            if split in ['train', 'dev']:
                with open(str(self.data_dir) + f'/text_5700_train_dev/valListFile.json') as val_list:
                    val_file_list = [ line.strip() for line in val_list ]
                
                if split == 'dev':
                    audio_files = { wav: str(self.data_dir) + f'/audio_5700_{split_dir}/' + wav + '.wav' for wav in val_file_list }
                    json_data = dict(filter(lambda pair: pair[0] in val_file_list, json_data.items()))

                else: # train, remove the validation split files
                    audio_files = { wav: str(self.data_dir) + f'/audio_5700_{split_dir}/' + wav + '.wav' for wav in json_data.keys() - val_file_list }

                    for key in val_file_list:
                        json_data.pop(key)
            else:
                audio_files = { wav: str(self.data_dir) + f'/audio_5700_{split_dir}/' + wav + '.wav' for wav in json_data.keys() }

            # finaly construct the feature dict
            features = list(json_data.items())

        return {
            'recordings': audio_files,
            'features': features
        }

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, recordings, features):
        for wav_id, data in features:
            audio_file = recordings[wav_id]

            if not os.path.isfile(audio_file):
                continue

            audio, sr = torchaudio.load(audio_file)
            audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=16000)
            channels = {'user': audio[0], 'system': audio[1]}

            context = []

            for i, turn in enumerate(data['log']):
                span_info = turn['span_info']
                dialog_act = json.dumps(turn['dialog_act'])
                metadata = json.dumps(turn['metadata'])
                tag = turn['tag']
                text = turn['text']
                start_time = turn['words'][0]['BeginTime']
                end_time = turn['words'][-1]['EndTime']
                
                # get the corresponding audio slice
                audio_slice = channels[tag][start_time*16:end_time*16]

                # preprocess the text
                # remove punctuation
                text = text.translate(str.maketrans('', '', string.punctuation.replace("'", "")))
                text = re.sub(' +', ' ', text)
                text = text.strip()

                return_dict = {
                    'audio': datasets.features.Audio(sampling_rate=16000).encode_example({
                            'path': None,
                            'array': audio_slice,
                            'sampling_rate': 16000,
                    }),
                    'wav_id': wav_id,
                    'turn_index': i,
                    'text': text,
                    'span_info': span_info,
                    'dialog_act': dialog_act,
                    'metadata': metadata,
                    'tag': tag,
                    'start_time': start_time,
                    'end_time': end_time,
                    'context': context,
                }

                yield wav_id + '_' + str(i), return_dict

                return_dict.pop('wav_id')
                return_dict.pop('context')
                return_dict.pop('audio')
                context.append(return_dict)

