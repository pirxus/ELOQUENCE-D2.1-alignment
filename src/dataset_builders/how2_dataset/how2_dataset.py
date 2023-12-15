"""Dataset builder module for the raw audio version of the HOW2 dataset"""

import datasets
import pandas as pd
from typing import Optional
import os

class HOW2Dataset(datasets.GeneratorBasedBuilder):
    """Dataset builder for the raw audio version of the HOW2 dataset"""

    def __init__(self, data_dir: Optional[str], **kwargs):
        super().__init__(data_dir=data_dir, **kwargs)
        self.data_dir = data_dir
        self.splits = ['dev5', 'train', 'val']

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "audio": datasets.Audio(sampling_rate=16_000),
                    "transcription": datasets.Value("string"),
                    "translation": datasets.Value("string"),
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
        en = open(str(self.data_dir) + f'/text_data/{split}/text.id.en')
        pt = open(str(self.data_dir) + f'/text_data/{split}/text.id.pt')

        # retreive the split metadata from the text files
        df_en = pd.DataFrame([ line.rstrip().partition(' ')[0::2] for line in en ]
                             ).rename(columns={0: 'utt_id', 1: 'transcription'})
        df_pt = pd.DataFrame([ line.rstrip().partition(' ')[0::2] for line in pt ]
                             ).rename(columns={0: 'utt_id', 1: 'translation'})

        en.close()
        pt.close()

        # merge on utterance name
        df = pd.merge(df_en, df_pt, on='utt_id')

        # edit the relative audio path
        df['file_path'] = df['utt_id'].map(
                lambda utt_id: str(self.data_dir) + f'/audio_data/{split}/' + utt_id + '.wav')

        recordings = list(df['utt_id'])
        df = df.set_index('utt_id')
        features = df.to_dict('index')

        return {
                'recordings': recordings,
                'features': features
        }

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, recordings, features):

        for utt_id in recordings:
            utt_feats = features[utt_id]

            # skip out the missing files, just in case..
            if not os.path.isfile(utt_feats['file_path']): continue
            yield utt_id, {
                    'audio': datasets.features.Audio(sampling_rate=16_000
                                                     ).encode_example(utt_feats['file_path']),
                    'transcription': utt_feats['transcription'],
                    'translation': utt_feats['translation']
            }
