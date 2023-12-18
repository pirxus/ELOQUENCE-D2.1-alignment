from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
from transformers import (
    BatchFeature,
    PreTrainedTokenizer,
    Speech2TextFeatureExtractor,
    Wav2Vec2FeatureExtractor,
)


@dataclass
class SpeechCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        feature_extractor (:class:`~transformers.SequenceFeatureExtractor`)
            The feature extractor used for processing the data.
        tokenizer (:class:`~transformers.PreTrainedTokenizer`)
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

    feature_extractor: Union[Wav2Vec2FeatureExtractor, Speech2TextFeatureExtractor]
    tokenizer: PreTrainedTokenizer
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    sampling_rate: Optional[int] = 16_000
    audio_path: str = None
    text_path: str = None
    model_input_name: str = True

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor, Dict[str, BatchFeature]]]]
    ) -> BatchFeature:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [
            BatchFeature({self.feature_extractor.model_input_names[0]: feature[self.audio_path].squeeze(dim=0)})
            for feature in features
        ]

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

        if self.model_input_name != self.feature_extractor.model_input_names[0]:
            batch[self.model_input_name] = batch[self.feature_extractor.model_input_names[0]]
            del batch[self.feature_extractor.model_input_names[0]]

        return batch
