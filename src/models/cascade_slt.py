"""
This module implements the Ebr./MarianMT cascade ST model. 
Author: Simon Sedlacek
"""

from transformers import (
        PretrainedConfig,
        PreTrainedModel,
        PreTrainedTokenizerFast,
        T5Tokenizer,
        T5ForConditionalGeneration,
        )

from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from typing import Tuple, Optional
from models.utils import shift_tokens_right
import torch

@dataclass
class CascadeSLTModelOutput(ModelOutput):
    loss: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None


class CascadeSLTModelConfig(PretrainedConfig):
    def __init__(
            self,
            enc_config,
            dec_config,
            **kwargs):
        self.enc_config = enc_config
        self.dec_config = dec_config

        super().__init__(**kwargs)

class CascadeSLTModel(PreTrainedModel):
    """Cascade ST system used as a baseline in the thesis.

    This class is designed to acommodate the baseline
    E-Branchformer small and MarianMT small models from the thesis.
    """


    config_class = CascadeSLTModelConfig
    main_input_name = "input_features"

    def __init__(
            self,
            encoder: PreTrainedModel,
            decoder: PreTrainedModel,
            encoder_tokenizer: PreTrainedTokenizerFast,
            decoder_tokenizer: PreTrainedTokenizerFast,
            config: Optional[CascadeSLTModelConfig] = None,
        ):
        super().__init__(config)

        self.encoder = encoder
        self.decoder = decoder
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.config = config

        self.tc_tokenizer = T5Tokenizer.from_pretrained('SJ-Ray/Re-Punctuate')
        self.tcaser = T5ForConditionalGeneration.from_pretrained('SJ-Ray/Re-Punctuate', from_tf=True)

        self.config.update({'pad_token_id': self.decoder.config.pad_token_id})

    def forward(
        self,
        input_features: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ):
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.decoder.config.pad_token_id, self.decoder.config.decoder_start_token_id
                )

        trans_logits = self.encoder.generate(
            input_features,
            attention_mask=attention_mask,
        )

        asr_hyp = self.encoder_tokenizer.batch_decode(trans_logits, skip_special_tokens=True)

        # truecase and add punctuation
        tc_tokenized_batch = self.tc_tokenizer.batch_encode_plus(
            ["punctuate: " + hyp for hyp in asr_hyp],
            return_attention_mask=True,
            padding="longest",
            return_tensors="pt",
        )
        tc_tokenized_batch = {k: v.cuda() for k, v in tc_tokenized_batch.items()}
        tc_hyp = self.tcaser.generate(**tc_tokenized_batch)
        tc_text = self.tc_tokenizer.batch_decode(tc_hyp, skip_special_tokens=True)

        # translate
        mt_tokenized_batch = self.decoder_tokenizer.batch_encode_plus(
            [hyp for hyp in tc_text],
            return_attention_mask=True,
            padding="longest",
            return_tensors="pt",
        )
        del mt_tokenized_batch['token_type_ids']
        mt_tokenized_batch = {k: v.cuda() for k, v in mt_tokenized_batch.items()}

        decoder_output = self.decoder(
            input_ids=mt_tokenized_batch['input_ids'],
            inputs_embeds=None,
            attention_mask=mt_tokenized_batch['attention_mask'],
            decoder_input_ids=decoder_input_ids, # input_ids
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=None,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        return decoder_output

    

    @torch.no_grad()
    def generate(
        self,
        input_features: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:

        # transcribe the audio
        trans_logits = self.encoder.generate(
            input_features,
            attention_mask=attention_mask,
            #generation_config=generation_config,
            #**generate_kwargs,
        )

        asr_hyp = self.encoder_tokenizer.batch_decode(trans_logits, skip_special_tokens=True)

        # truecase and add punctuation
        tc_tokenized_batch = self.tc_tokenizer.batch_encode_plus(
            ["punctuate: " + hyp for hyp in asr_hyp],
            return_attention_mask=True,
            padding="longest",
            return_tensors="pt",
        )
        tc_tokenized_batch = {k: v.cuda() for k, v in tc_tokenized_batch.items()}
        tc_hyp = self.tcaser.generate(**tc_tokenized_batch)
        tc_text = self.tc_tokenizer.batch_decode(tc_hyp, skip_special_tokens=True)

        # translate
        mt_tokenized_batch = self.decoder_tokenizer.batch_encode_plus(
            [hyp for hyp in tc_text],
            return_attention_mask=True,
            padding="longest",
            return_tensors="pt",
        )
        del mt_tokenized_batch['token_type_ids']
        mt_tokenized_batch = {k: v.cuda() for k, v in mt_tokenized_batch.items()}
        mt_hyp = self.decoder.generate(**mt_tokenized_batch)
        return mt_hyp
