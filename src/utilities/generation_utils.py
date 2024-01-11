import os
import pickle  # nosec
from typing import Dict, List, Optional

import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    LogitsProcessor,
    LogitsProcessorList,
    PreTrainedTokenizer,
    SpeechEncoderDecoderModel,
)
from transformers.generation.utils import BeamSearchEncoderDecoderOutput
from transformers.trainer_utils import PredictionOutput

from models.ctc_encoder_plus_autoregressive_decoder import (
    JointCTCAttentionEncoderDecoder,
)
from utilities.training_arguments import GenerationArguments


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


def activate_joint_decoding(
    model: SpeechEncoderDecoderModel,
    ctc_weight: float,
    ctc_margin: float,
    num_tokens: int,
    eos_token: int,
    external_lm: Optional[AutoModelForCausalLM],
    external_lm_weight: float,
):
    def new_beam(*args, **kwargs) -> BeamSearchEncoderDecoderOutput:
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

    if not isinstance(model, JointCTCAttentionEncoderDecoder):
        raise ValueError("Model must be of type JointCTCAttentionEncoderDecoder")
    model.beam_search = new_beam


def save_nbests(
    path: str,
    nbests: List[torch.Tensor],
    scores: List[torch.Tensor],
    labels: List[torch.Tensor],
    tokenizer: PreTrainedTokenizer,
    group_size: int = 1,
    batch_size: int = 1,
    outputs: List[Dict] = None,
):
    nbests = [tokenizer.decode(elem.tolist(), skip_special_tokens=True) for item in nbests for elem in item.unbind()]
    processed_labels = []
    if outputs is not None:
        for index, output in enumerate(outputs):
            with open(
                path + f"_utterance{index * batch_size}-{(index + 1) * batch_size - 1}.pkl", "wb"
            ) as file_handler:
                pickle.dump(output, file_handler, protocol=pickle.HIGHEST_PROTOCOL)

    for label in labels:
        label[label == -100] = tokenizer.pad_token_id
        processed_labels.extend(
            [
                tokenizer.decode(elem.tolist(), skip_special_tokens=True)
                for elem in label.repeat_interleave(group_size, dim=0)
            ]
        )
    scores = [float(elem) for item in scores for elem in item.unbind()]
    with open(path + "_scores.txt", "w") as file_handler_1:
        with open(path + "_hyps.txt", "w") as file_handler_2:
            with open(path + "_refs.txt", "w") as file_handler_3:
                for item, (sample, score, ref) in enumerate(zip(nbests, scores, processed_labels)):
                    utterance_id = f"utterance{item // group_size}-{item % group_size + 1}"
                    file_handler_1.write(f"{utterance_id} {score}\n")
                    file_handler_2.write(f"{utterance_id} {sample}\n")
                    file_handler_3.write(f"{utterance_id} {ref}\n")


def save_predictions(tokenizer: PreTrainedTokenizer, predictions: PredictionOutput, path: str):
    pred_ids = predictions.predictions

    label_ids = predictions.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = [label if label else "-" for label in tokenizer.batch_decode(label_ids, skip_special_tokens=True)]
    df = pd.DataFrame({"label": label_str, "prediction": pred_str})
    df.to_csv(path, index=False)

    sclite_files = [path.replace(".csv", f"_{type}.trn") for type in ["hyp", "ref"]]
    for strings, file_to_save in zip([pred_str, label_str], sclite_files):
        with open(file_to_save, "w") as file_handler:
            for index, string in enumerate(strings):
                file_handler.write(f"{string} (utterance_{index})\n")

    # evaluate wer also with sclite
    os.system(f"sclite -F -D -i wsj -r {sclite_files[1]} trn -h {sclite_files[0]} trn -o snt sum")  # nosec


def check_and_activate_joint_decoding(
    gen_args: GenerationArguments, model: SpeechEncoderDecoderModel, tokenizer: PreTrainedTokenizer, eos_token_id: int
):
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
