
from transformers import (
        Speech2TextPreTrainedModel,
        Speech2TextConfig,
        )

from transformers.models.speech_to_text.modeling_speech_to_text import Speech2TextEncoder

from transformers.modeling_outputs import CausalLMOutput

from typing import Optional

import torch
import torch.nn as nn

_HIDDEN_STATES_START_POSITION = 1

class Speech2TextForCTCConfig(Speech2TextConfig):
    model_type = 's2t-ctc-encoder'

    def __init__(self, ctc_vocab_size=32, ctc_zero_infinity=False, **kwargs):
        super().__init__(**kwargs)
        self.ctc_vocab_size=ctc_vocab_size
        self.ctc_zero_infinity=ctc_zero_infinity


# the basis for this class is the original Speech2TextEncoder class, modified for CTC
class Speech2TextEncoderForCTC(Speech2TextPreTrainedModel):
    config_class = Speech2TextForCTCConfig

    def __init__(self, config: Speech2TextForCTCConfig , target_lang: Optional['str'] = 'eng'):
        super().__init__(config)

        self.encoder = Speech2TextEncoder(config)

        # CTC stuff
        self.target_lang = target_lang

        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `Speech2TextEncoderForCTC.from_pretrained(..., ctc_vocab_size=vocab_size)`. "
                "or define `ctc_vocab_size` of your model's configuration."
            )
        output_hidden_size = (
            config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.d_model
        )

        vocab_size = 32 if not config.vocab_size else config.vocab_size

        self.lm_head = nn.Linear(output_hidden_size, vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.encoder(
                input_features,
                attention_mask,
                head_mask,
                output_attentions,
                output_hidden_states,
                return_dict=return_dict
                )

        hidden_states = outputs[0]

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_features, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
