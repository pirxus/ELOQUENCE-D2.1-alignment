"""Module containing all the up-to-date alignment architecture definitions.

Both the ECD and ECED architectures are confirmed to be compatible with
the E-Branchformer/Whisper encoders and T5/MarianMT decoders.

Author: Simon Sedlacek
"""

from transformers import (

    BertModel,
    PreTrainedModel,
    PretrainedConfig,
    Blip2QFormerConfig,
    Blip2QFormerModel,
    Speech2TextModel,
    MarianMTModel,
    MarianForCausalLM,
    MarianConfig,
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

from models.utils import SpeechEncoderOutputSubsampler, shift_tokens_right

from models.old_alignment import SpeechQFormerMarianOutput, ApmoConfig

class AlignmentNetwork(PreTrainedModel):
    config_class = ApmoConfig

    def __init__(
            self,
            config: Optional[ApmoConfig] = None,
            model_type: Optional[str] = 'qformer'
        ):
        super().__init__(config)
        self.model_type = model_type

        # fix the configs if needed..
        if isinstance(config.qformer_config, dict):
            config.qformer_config = Blip2QFormerConfig(**config.qformer_config)

        if hasattr(config.lm_config, 'hidden_size'):
            lm_hidden_size = config.lm_config.hidden_size
        else:
            lm_hidden_size = config.lm_config.d_model

        # first define the language projection as it's used by both alignment configurations
        self.language_projection = nn.Linear(config.qformer_config.hidden_size, lm_hidden_size)

        if self.model_type == 'qformer':

            self.query_tokens = nn.Parameter(
                    torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size)
                )

            self.qformer = Blip2QFormerModel(self.config.qformer_config)
            self.forward = self._forward_qformer

        else:
            conv_config = SpeechEncoderOutputSubsampler.default_config
            conv_config.update({
                'input_feat_per_channel': self.config.qformer_config.encoder_hidden_size,
                'd_model': self.config.qformer_config.hidden_size,
                })
            self.conv = SpeechEncoderOutputSubsampler(conv_config)

            q_conf = self.config.qformer_config
            marian_config = MarianConfig(
                    d_model=q_conf.hidden_size,
                    encoder_layers=q_conf.num_hidden_layers,
                    decoder_layers=q_conf.num_hidden_layers,
                    encoder_attention_heads=q_conf.num_attention_heads,
                    decoder_attention_heads=q_conf.num_attention_heads,
                    encoder_ffn_dim=q_conf.intermediate_size,
                    decoder_ffn_dim=q_conf.intermediate_size,
                    activation_function='gelu_new',
                    attention_dropout=0.1,
                    activation_dropout=0.1,
                    scale_embedding=True,
                    )

            self.bridge = MarianEncoder(marian_config)
            self.bridge.embed_tokens.weight.requires_grad = False
            self.forward = self._forward_conv

    def _forward_qformer(
            self,
            encoder_outputs,
            attention_mask,
        ):

        if not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                    last_hidden_state=encoder_outputs,
                )

        audio_embeds = encoder_outputs.last_hidden_state

        query_tokens = self.query_tokens.expand(audio_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=audio_embeds,
            encoder_attention_mask=attention_mask,
            return_dict=True,
        )

        query_output = BaseModelOutput(
            last_hidden_state=self.language_projection(query_outputs[0]),
        )

        return query_output, None

    def _forward_conv(
            self,
            encoder_outputs,
            attention_mask,
        ):

        if not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                    last_hidden_state=encoder_outputs,
                )

        audio_embeds = encoder_outputs.last_hidden_state

        # Downsample the speech encoder outputs 
        audio_embeds = self.conv(audio_embeds)

        # downsample encoder attention mask again..
        if attention_mask is not None:
            attention_mask = self.conv._get_feature_vector_attention_mask(
                audio_embeds.shape[1], attention_mask
            )
        else:
            attention_mask = None

        # pass the encoder outputs through the bridge network
        bridge_outputs = self.bridge(
                attention_mask=attention_mask,
                inputs_embeds=audio_embeds,
                return_dict=True,
        )

        bridge_outputs.last_hidden_state = self.language_projection(bridge_outputs.last_hidden_state)

        return bridge_outputs, attention_mask

class SpeechEncoderBridgeTextDecoder(PreTrainedModel):
    config_class = ApmoConfig
    main_input_name = "input_features"

    def __init__(
            self,
            model_path: Optional[str] = None,
            config: Optional[ApmoConfig] = None,
            encoder: Optional[PreTrainedModel] = None,
            decoder: Optional[PreTrainedModel] = None,
            freeze_decoder: Optional[bool] = True,
        ):
        super().__init__(config)

        if encoder:
            try:
                self.encoder = encoder.get_encoder()
            except:
                self.encoder = encoder.encoder
        else:
            raise ValueError("Encoder model needs to be supplied")

        if decoder:
            self.decoder = decoder

        else:
            raise ValueError("Decoder model needs to be supplied")

        if isinstance(config.qformer_config, dict):
            config.qformer_config = Blip2QFormerConfig(**config.qformer_config)
        if isinstance(config.lm_config, dict):
            config.lm_config = PretrainedConfig(**config.lm_config)

        self.bridge = AlignmentNetwork(config=self.config, model_type=self.config.bridge_type)

        # freeze encoder
        for _, param in self.encoder.named_parameters():
            param.requires_grad = False

        ## freeze whole decoder
        if hasattr(self.decoder, 'model'):
            for _, param in self.decoder.model.encoder.named_parameters():
                param.requires_grad = False
            for _, param in self.decoder.model.decoder.named_parameters():
                param.requires_grad = False
        else:
            for _, param in self.decoder.encoder.named_parameters():
                param.requires_grad = False
            for _, param in self.decoder.decoder.named_parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False

        self.decoder.lm_head.requires_grad = False
        if hasattr(self.decoder, 'final_logits_bias'):
            self.decoder.final_logits_bias.requires_grad = False

        # FIXME: the padding required in generate - rework the config class to include all
        # the token_ids on the top level ...
        self.config = config
        self.config.update({'pad_token_id': self.decoder.config.pad_token_id})

    def encoder_decoder_eval(self):
        self.encoder.eval()
        self.decoder.eval()

    def forward(
        self,
        input_features: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_prefix_ids: Optional[torch.LongTensor] = None,
        encoder_prefix_mask: Optional[torch.LongTensor] = None,
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
    ) -> Union[Tuple, SpeechQFormerMarianOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.decoder.config.pad_token_id, self.decoder.config.decoder_start_token_id
                )

        # 1. forward the audio through the encoder
        encoder_outputs = self.encoder(
            input_features,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )

        encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs.last_hidden_state,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions
                )

        # downsample encoder attention mask
        if attention_mask is not None:
            audio_attention_mask = self.encoder._get_feature_vector_attention_mask(
                encoder_outputs.last_hidden_state.shape[1], attention_mask
            )
        else:
            audio_attention_mask = None

        # pass the encoder outputs through the bridge network
        bridge_outputs, audio_attention_mask = self.bridge(
                encoder_outputs=encoder_outputs.last_hidden_state,
                attention_mask=audio_attention_mask,
            )

        # combine the bridge outputs with the encoder prefix ids
        """
        if encoder_prefix_ids is not None:

            # embed the prefix ids
            emb = self.decoder.get_encoder().get_input_embeddings()
            prefix_embeds = emb(encoder_prefix_ids)

            bridge_outputs.last_hidden_state = torch.hstack((prefix_embeds, bridge_outputs.last_hidden_state))
            if audio_attention_mask is not None:
                audio_attention_mask = torch.hstack((encoder_prefix_mask, audio_attention_mask))
        """

        decoder_output = self.decoder(
            input_ids=None,
            inputs_embeds=None,
            attention_mask=audio_attention_mask,
            decoder_input_ids=decoder_input_ids, # input_ids
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=bridge_outputs,
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
        encoder_prefix_ids: Optional[torch.LongTensor] = None,
        encoder_prefix_mask: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:

        # 1. forward the audio through the encoder
        encoder_outputs = self.encoder(
            input_features,
            attention_mask=attention_mask,
            return_dict=True
        )

        # maintain comaptability with marian
        encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs.last_hidden_state,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions
                )

        audio_embeds = encoder_outputs.last_hidden_state

        # downsample encoder attention mask
        if attention_mask is not None:
            audio_attention_mask = self.encoder._get_feature_vector_attention_mask(
                audio_embeds.shape[1], attention_mask
            )
        else:
            audio_attention_mask = None

        # pass the encoder outputs through the bridge network
        bridge_outputs, audio_attention_mask = self.bridge(
                encoder_outputs=audio_embeds,
                attention_mask=audio_attention_mask,
            )

        decoder_input = (
            torch.LongTensor([[self.decoder.config.decoder_start_token_id]])
            .repeat(audio_embeds.shape[0], 1).to(audio_embeds.device)
        )

        attention_mask = torch.ones_like(decoder_input)

        decoder_output = self.decoder.generate(
            input_ids=None, # input_ids
            attention_mask=audio_attention_mask,
            encoder_outputs=bridge_outputs,
            decoder_input_ids=decoder_input,
            decoder_attention_mask=attention_mask,
            generation_config=self.generation_config,
            **generate_kwargs,
        )
        return decoder_output


class SpeechEncoderBridgeMarianEncoderDecoder(PreTrainedModel):
    config_class = ApmoConfig
    main_input_name = "input_features"

    def __init__(
            self,
            model_path: Optional[str] = None,
            config: Optional[ApmoConfig] = None,
            encoder: Optional[PreTrainedModel] = None,
            decoder: Optional[PreTrainedModel] = None,
            freeze_decoder: Optional[bool] = True,
        ):
        super().__init__(config)

        if encoder:
            try:
                self.encoder = encoder.get_encoder()
            except:
                self.encoder = encoder.encoder
        else:
            raise ValueError("Encoder model needs to be supplied")

        if decoder:
            self.decoder = decoder

        else:
            raise ValueError("Decoder model needs to be supplied")

        if isinstance(config.qformer_config, dict):
            config.qformer_config = Blip2QFormerConfig(**config.qformer_config)
        if isinstance(config.lm_config, dict):
            config.lm_config = PretrainedConfig(**config.lm_config)

        self.bridge = AlignmentNetwork(config=self.config, model_type=self.config.bridge_type)

        # freeze encoder
        for _, param in self.encoder.named_parameters():
            param.requires_grad = False

        ## freeze whole decoder
        if hasattr(self.decoder, 'model'):
            for _, param in self.decoder.model.encoder.named_parameters():
                param.requires_grad = False
        else:
            for _, param in self.decoder.encoder.named_parameters():
                param.requires_grad = False

        self.freeze_decoder = freeze_decoder
        if freeze_decoder:
            if hasattr(self.decoder, 'model'):
                for _, param in self.decoder.model.decoder.named_parameters():
                    param.requires_grad = False
            else:
                for _, param in self.decoder.decoder.named_parameters():
                    param.requires_grad = False
                for param in self.decoder.parameters():
                    param.requires_grad = False

            if hasattr(self.decoder, 'lm_head'):
                self.decoder.lm_head.requires_grad = False
            if hasattr(self.decoder, 'final_logits_bias'):
                self.decoder.final_logits_bias.requires_grad = False

        # FIXME: the padding required in generate - rework the config class to include all
        # the token_ids on the top level ...
        self.config = config
        self.config.update({'pad_token_id': self.decoder.config.pad_token_id})

    def encoder_decoder_eval(self):
        # TODO: this will have to be modified when fine-tuning
        self.encoder.eval()
        if self.freeze_decoder:
            self.decoder.eval()
        else:
            self.decoder.model.encoder.eval()

    def forward(
        self,
        input_features: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_prefix_ids: Optional[torch.LongTensor] = None,
        encoder_prefix_mask: Optional[torch.LongTensor] = None,
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
    ) -> Union[Tuple, SpeechQFormerMarianOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.decoder.config.pad_token_id, self.decoder.config.decoder_start_token_id
                )

        # 1. forward the audio through the encoder
        encoder_outputs = self.encoder(
            input_features,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )

        encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs.last_hidden_state,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions
                )

        # downsample encoder attention mask
        if attention_mask is not None:
            audio_attention_mask = self.encoder._get_feature_vector_attention_mask(
                encoder_outputs.last_hidden_state.shape[1], attention_mask
            )
        else:
            audio_attention_mask = None

        # pass the encoder outputs through the bridge network
        bridge_outputs, audio_attention_mask = self.bridge(
                encoder_outputs=encoder_outputs.last_hidden_state,
                attention_mask=audio_attention_mask,
            )

        # combine the bridge outputs with the encoder prefix ids
        if encoder_prefix_ids is not None:


            # embed the prefix ids
            emb = self.decoder.get_encoder().get_input_embeddings()
            prefix_embeds = emb(encoder_prefix_ids)

            bridge_outputs.last_hidden_state = torch.hstack((prefix_embeds, bridge_outputs.last_hidden_state))
            if audio_attention_mask is not None:
                audio_attention_mask = torch.hstack((encoder_prefix_mask, audio_attention_mask))

        decoder_output = self.decoder(
            input_ids=None,
            inputs_embeds=bridge_outputs.last_hidden_state,
            attention_mask=audio_attention_mask,
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
        encoder_prefix_ids: Optional[torch.LongTensor] = None,
        encoder_prefix_mask: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:

        # 1. forward the audio through the encoder
        encoder_outputs = self.encoder(
            input_features,
            attention_mask=attention_mask,
            return_dict=True
        )

        # maintain comaptability with marian
        encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs.last_hidden_state,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions
                )

        audio_embeds = encoder_outputs.last_hidden_state

        # downsample encoder attention mask
        if attention_mask is not None:
            audio_attention_mask = self.encoder._get_feature_vector_attention_mask(
                audio_embeds.shape[1], attention_mask
            )
        else:
            audio_attention_mask = None

        # pass the encoder outputs through the bridge network
        bridge_outputs, audio_attention_mask = self.bridge(
                encoder_outputs=audio_embeds,
                attention_mask=audio_attention_mask,
            )

        # combine the bridge outputs with the encoder prefix ids
        if encoder_prefix_ids is not None:

            # embed the prefix ids
            emb = self.decoder.get_encoder().get_input_embeddings()
            prefix_embeds = emb(encoder_prefix_ids)

            bridge_outputs.last_hidden_state = torch.hstack((prefix_embeds, bridge_outputs.last_hidden_state))
            if audio_attention_mask is not None:
                audio_attention_mask = torch.hstack((encoder_prefix_mask, audio_attention_mask))

        decoder_output = self.decoder.generate(
            input_ids=None, # input_ids
            inputs_embeds=bridge_outputs.last_hidden_state,
            attention_mask=audio_attention_mask,
            **generate_kwargs,
        )
        return decoder_output
