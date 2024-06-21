"""This module implements the re-trainng model class for the T5 translation model.

Author: Simon Sedlacek
"""

from transformers import (
    BertModel,
    PreTrainedModel,
    PreTrainedTokenizer,
    PretrainedConfig,
    Blip2QFormerConfig,
    Blip2QFormerModel,
    Speech2TextModel,
    MarianMTModel,
    MarianForCausalLM,
    MarianConfig,
    T5Config,
    T5ForConditionalGeneration,
)
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput, ModelOutput
from transformers.models.marian.modeling_marian import MarianEncoder
from tslearn.metrics import SoftDTWLossPyTorch

from typing import Optional, Tuple, Union
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy

from models.utils import shift_tokens_right

from models.old_alignment import SpeechQFormerMarianOutput, ApmoConfig

class T5PlusMarian(PreTrainedModel):
    config_class = T5Config

    def __init__(
            self,
            model_path: Optional[str] = None,
            config: Optional[ApmoConfig] = None,
            encoder: Optional[T5ForConditionalGeneration] = None,
            tokenizer: Optional[PreTrainedTokenizer] = None,
            freeze_decoder: Optional[bool] = True,
        ):
        super().__init__(config)

        self.encoder = encoder.encoder

        self.language_projection = nn.Linear(768, 256)

        decoder_config = MarianConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=256,
            encoder_layers=6,
            decoder_layers=6,
            encoder_attention_heads=4,
            decoder_attention_heads=4,
            decoder_ffn_dim=2048,
            encoder_ffn_dim=2048,
            activation_function='gelu_new',
            attention_dropout=0.1,
            activation_dropout=0.1,
            scale_embedding=True,
            forced_eos_token_id=tokenizer.eos_token_id,
            share_encoder_decoder_embeddings=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            decoder_start_token_id=tokenizer.bos_token_id,
            )

        self.decoder = MarianMTModel(decoder_config)

        # freeze encoder
        for _, param in self.encoder.named_parameters():
            param.requires_grad = False

        for param in self.encoder.parameters():
            param.requires_grad = False

        # freeze decoder encoder
        for _, param in self.decoder.model.encoder.named_parameters():
            param.requires_grad = False

        if freeze_decoder:
            for _, param in self.decoder.model.decoder.named_parameters():
                param.requires_grad = False

            self.decoder.lm_head.requires_grad = False
            self.decoder.final_logits_bias.requires_grad = False

        # FIXME: the padding required in generate - rework the config class to include all
        # the token_ids on the top level ...
        self.config = config
        self.config.update({'pad_token_id': self.decoder.config.pad_token_id})

    def freeze_model(self):
        # freeze encoder
        for _, param in self.encoder.named_parameters():
            param.requires_grad = False

        for param in self.encoder.parameters():
            param.requires_grad = False

        # freeze decoder encoder
        for _, param in self.decoder.model.encoder.named_parameters():
            param.requires_grad = False

        for _, param in self.decoder.model.decoder.named_parameters():
            param.requires_grad = False

        self.decoder.lm_head.requires_grad = False
        self.decoder.final_logits_bias.requires_grad = False
    
    def get_encoder(self):
        return self.encoder

    def encoder_decoder_eval(self):
        self.encoder.eval()
        self.decoder.eval()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SpeechQFormerMarianOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.decoder.config.pad_token_id, self.decoder.config.decoder_start_token_id
                )


        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            #head_mask=head_mask,
            #output_attentions=output_attentions,
            #output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden_states = self.language_projection(encoder_outputs[0])

        hidden_states = BaseModelOutput(
            last_hidden_state=hidden_states,
        )

        decoder_output = self.decoder(
            input_ids=None,
            inputs_embeds=None,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids, # input_ids
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=hidden_states,
            labels=labels,
            return_dict=True,
        )

        return decoder_output

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.FloatTensor = None,
        inputs_embeds: torch.FloatTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:

        # 1. forward the audio through the encoder
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )

        hidden_states = self.language_projection(encoder_outputs[0])

        decoder_input = (
            torch.LongTensor([[self.decoder.config.decoder_start_token_id]])
            .repeat(hidden_states.shape[0], 1).to(hidden_states.device)
        )

        hidden_states = BaseModelOutput(
            last_hidden_state=hidden_states,
        )

        decoder_attention_mask = torch.ones_like(decoder_input)

        decoder_output = self.decoder.generate(
            input_ids=None,
            input_embeds=None,
            attention_mask=attention_mask,
            encoder_outputs=hidden_states,
            decoder_input_ids=decoder_input,
            decoder_attention_mask=decoder_attention_mask,
            generation_config=self.generation_config,
            **generate_kwargs,
        )

        return decoder_output
