"""
This module implements the ASR encoder + connector + decoder-only LM alignment model

"""

from dataclasses import dataclass
from transformers import (

    PreTrainedModel,
    PretrainedConfig,
    Blip2QFormerConfig,
)
from transformers.modeling_outputs import ModelOutput, BaseModelOutput

from torch.nn import CrossEntropyLoss

from typing import Optional, Tuple, Union, Any

import torch
from torch import nn
import torch.nn.functional as F
from models.utils import shift_tokens_right
from models.aligned import AlignmentNetwork
from models.old_alignment import AlignmentConfig


@dataclass
class SpeechEncoderConnectorLMDecoderModelOuput(ModelOutput):
    """
    Model output class for the ASR + conn + LM aligned models.

    Args:
        TODO
        loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Language modeling loss from the language model.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head of the language model.
        qformer_outputs (`BaseModelOutputWithPoolingAndCrossAttentions`):
            Outputs of the Q-Former (Querying Transformer).
        language_model_outputs (`CausalLMOutputWithPast` or `Seq2SeqLMOutput`):
            Outputs of the language model.
    """

    loss: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    audio_outputs: Optional[torch.FloatTensor] = None
    connector_outputs: Optional[Tuple[torch.FloatTensor]] = None
    lm_outputs: Optional[Tuple[torch.FloatTensor]] = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k]
            if k not in ["vision_outputs", "qformer_outputs", "language_model_outputs"]
            else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class SpeechEncoderConnectorLMDecoderConfig(PretrainedConfig):
    def __init__(
        self,
        encoder_config=None,
        qformer_config=None,
        connector_type='qformer',
        lm_config=None,
        num_query_tokens=80,
        modality_matching=True,
        mm_pooling='avg',
        mm_micro_loss='dot',
        mm_loss_weight=1.0,
        ce_loss_weight=1.0,
        num_pretrain_epochs=0,
        **kwargs
    ):

        super().__init__(**kwargs)

        self.encoder_config = encoder_config
        if qformer_config is None:
            self.qformer_config = Blip2QFormerConfig()
        else:
            self.qformer_config = qformer_config

        self.lm_config = lm_config
        self.modality_matching = modality_matching
        self.mm_pooling = mm_pooling
        self.num_query_tokens = num_query_tokens
        self.mm_loss_weight = mm_loss_weight
        self.ce_loss_weight = ce_loss_weight
        self.num_pretrain_epochs = num_pretrain_epochs
        self.mm_micro_loss = mm_micro_loss
        self.connector_type = connector_type


class SpeechEncoderConnectorLMDecoder(PreTrainedModel):
    config_class = AlignmentConfig
    main_input_name = "input_features"

    # TODO: refactor the model building methods
    # - fix the model saving -> save only the connector module
    # - create an model builder method aside from init that requires the connector as well

    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[AlignmentConfig] = None,
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

        self.connector = AlignmentNetwork(
            config=self.config, model_type=self.config.connector_type)

        # freeze encoder and decoder
        self.do_freeze_decoder = freeze_decoder
        self.freeze_encoder()
        if self.do_freeze_decoder:
            self.freeze_decoder() # NOTE: the freezing should be done when the model is initialized

        # FIXME: the padding required in generate - rework the config class to include all
        # the token_ids on the top level ...
        self.config = config
        self.config.update({'pad_token_id': self.decoder.config.eos_token_id})
        self.decoder.config.update({'pad_token_id': self.decoder.config.eos_token_id})

    def encoder_decoder_eval(self):
        self.encoder.eval()
        if self.freeze_decoder:
            self.decoder.eval()

    def freeze_encoder(self):
        for _, param in self.encoder.named_parameters():
            param.requires_grad = False

    def freeze_decoder(self):
        if not self.freeze_decoder: return

        if hasattr(self.decoder, 'model'):
            for _, param in self.decoder.model.named_parameters():
                param.requires_grad = False
            for param in self.decoder.model.parameters():
                param.requires_grad = False
        else:
            for _, param in self.decoder.named_parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False

        for param in self.decoder.parameters():
            param.requires_grad = False

        self.decoder.lm_head.requires_grad = False
        if hasattr(self.decoder, 'final_logits_bias'):
            self.decoder.final_logits_bias.requires_grad = False

    def forward(
        self,
        input_features: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        prompt_prefix_ids: Optional[torch.LongTensor] = None,
        prompt_prefix_mask: Optional[torch.LongTensor] = None,
        prompt_suffix_ids: Optional[torch.LongTensor] = None,
        prompt_suffix_mask: Optional[torch.LongTensor] = None,
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
    ) -> Union[Tuple, BaseModelOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            # we don't want a bos token at the beginning of the labels
            if labels[0, 0] == self.decoder.config.bos_token_id:
                labels = labels[:, 1:]

            if decoder_input_ids is None and decoder_inputs_embeds is None:
                # TODO: fix this.. probably just always pad the labels with -100?
                if self.decoder.config.model_type == 'llama':
                    sep = 29871
                else:
                    sep = self.decoder.config.bos_token_id

                decoder_input_ids = shift_tokens_right(
                    labels, self.decoder.config.pad_token_id, sep
                )

                # because of the way we compute loss, we don't need the shifted decoder_input_ids
                # NOTE: this may not be ideal, think about what this means for the prompt
                # suffix -- perhaps we need to enforce a space at the end of it?
                decoder_input_ids = decoder_input_ids[:,1:]

        # 1. forward the audio through the encoder
        encoder_outputs = self.encoder(
            input_features,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )

        # FIXME: just for whisper compatibility...
        attention_mask = None

        encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_outputs.last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions
        )

        audio_embeds = encoder_outputs.last_hidden_state

        # downsample encoder attention mask
        if attention_mask is not None:
            audio_attention_mask = self.encoder._get_feature_vector_attention_mask(
                encoder_outputs.last_hidden_state.shape[1], attention_mask
            )
        else:
            audio_attention_mask = None

        # pass the encoder outputs through the bridge network
        connector_outputs, audio_attention_mask = self.connector(
            encoder_outputs=encoder_outputs.last_hidden_state,
            attention_mask=audio_attention_mask,
        )

        batch_size = audio_embeds.shape[0]

        if audio_attention_mask is None:
            audio_attention_mask = torch.ones(
                (
                    connector_outputs.last_hidden_state.shape[0],
                    connector_outputs.last_hidden_state.shape[1],
                ),
                device = audio_embeds.device
            )

        # prepend the prompt prefix to the connector output
        if prompt_prefix_ids is None:

            prompt_prefix_ids = (
                torch.LongTensor([[self.config.decoder_bos_token_id]])
                .repeat(batch_size, 1)
                .to(audio_embeds.device)
            )
            prompt_prefix_mask = torch.ones_like(
                prompt_prefix_ids, device=audio_embeds.device)

        else:
            # cut off the prefix eos token id
            if prompt_prefix_ids[0, -1] == self.decoder.config.eos_token_id:
                print("WARNING: there was a trailing prompt prefix eos",
                      prompt_suffix_ids)
                prompt_prefix_ids = prompt_prefix_ids[..., :-1]
                prompt_prefix_mask = prompt_prefix_mask[..., :-1]

        # embed the prefix ids
        prefix_embeds = self.decoder.get_input_embeddings()(prompt_prefix_ids)

        connector_outputs.last_hidden_state = torch.hstack(
            (prefix_embeds, connector_outputs.last_hidden_state))
        if audio_attention_mask is not None:
            audio_attention_mask = torch.hstack(
                (prompt_prefix_mask, audio_attention_mask))

        # append the prompt suffix
        if prompt_suffix_ids is not None:
            # cut off the bos token
            if prompt_suffix_ids[0, 0] == self.decoder.config.bos_token_id:
                print("WARNING: there was a trailing prompt suffix bos",
                      prompt_suffix_ids)
                prompt_suffix_ids = prompt_suffix_ids[..., 1:]
                prompt_suffix_mask = prompt_suffix_mask[..., 1:]

            # cut off the eos token
            if prompt_suffix_ids[0, -1] == self.decoder.config.eos_token_id:
                print("WARNING: there was a trailing prompt suffix eos",
                      prompt_suffix_ids)
                prompt_suffix_ids = prompt_suffix_ids[..., :-1]
                prompt_suffix_mask = prompt_suffix_mask[..., :-1]

            # embed the suffix ids
            suffix_embeds = self.decoder.get_input_embeddings()(prompt_suffix_ids)

            connector_outputs.last_hidden_state = torch.hstack(
                (connector_outputs.last_hidden_state, suffix_embeds))
            if audio_attention_mask is not None:
                audio_attention_mask = torch.hstack(
                    (audio_attention_mask, prompt_suffix_mask))

        device = connector_outputs.last_hidden_state.device

        decoder_inputs_embeds = self.decoder.get_input_embeddings()(decoder_input_ids)
        decoder_inputs_attn_mask = torch.ones_like(
            decoder_input_ids, device=device)

        decoder_inputs_embeds = torch.hstack(
            (connector_outputs.last_hidden_state, decoder_inputs_embeds))

        attention_mask = torch.hstack(
            (audio_attention_mask, decoder_inputs_attn_mask))

        decoder_outputs = self.decoder(
            inputs_embeds=decoder_inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )

        logits = decoder_outputs.logits
        loss = None

        if labels is not None:
            labels = labels.to(logits.device)
            logits = logits[:, -labels.size(1):, :]
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(logits.device)

            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction="mean")

            loss = loss_fct(
                shift_logits.view(-1, self.decoder.config.vocab_size), shift_labels.view(-1))


        # NOTE: here we're only returning the final cut logits, as there's no need for us to
        # return the whole sequence..
        return SpeechEncoderConnectorLMDecoderModelOuput(
            #logits=decoder_outputs.logits,
            logits=logits,
            loss=loss,
        )

    @torch.no_grad()
    def generate(
        self,
        input_features: torch.FloatTensor,
        prompt_prefix_ids: Optional[torch.LongTensor] = None,
        prompt_prefix_mask: Optional[torch.LongTensor] = None,
        prompt_suffix_ids: Optional[torch.LongTensor] = None,
        prompt_suffix_mask: Optional[torch.LongTensor] = None,
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

        # FIXME: just for whisper compatibility...
        attention_mask = None

        # downsample encoder attention mask
        if attention_mask is not None:
            audio_attention_mask = self.encoder._get_feature_vector_attention_mask(
                audio_embeds.shape[1], attention_mask
            )
        else:
            audio_attention_mask = None

        # pass the encoder outputs through the bridge network
        connector_outputs, audio_attention_mask = self.connector(
            encoder_outputs=audio_embeds,
            attention_mask=audio_attention_mask,
        )

        batch_size = audio_embeds.shape[0]

        # prepend the prompt prefix to the connector output
        if prompt_prefix_ids is None:

            prompt_prefix_ids = (
                torch.LongTensor([[self.config.decoder_bos_token_id]])
                .repeat(batch_size, 1)
                .to(audio_embeds.device)
            )
            prompt_prefix_mask = torch.ones_like(
                prompt_prefix_ids, device=audio_embeds.device)

        else:
            # cut off the prefix eos token id
            if prompt_prefix_ids[0, -1] == self.decoder.config.eos_token_id:
                print("WARNING: there was a trailing prompt prefix eos",
                      prompt_suffix_ids)
                prompt_prefix_ids = prompt_prefix_ids[..., :-1]
                prompt_prefix_mask = prompt_prefix_mask[..., :-1]

        # embed the prefix ids
        prefix_embeds = self.decoder.get_input_embeddings()(prompt_prefix_ids)

        connector_outputs.last_hidden_state = torch.hstack(
            (prefix_embeds, connector_outputs.last_hidden_state))
        if audio_attention_mask is not None:
            audio_attention_mask = torch.hstack(
                (prompt_prefix_mask, audio_attention_mask))

        # append the prompt suffix
        if prompt_suffix_ids is not None:

            # cut off the bos token
            if prompt_suffix_ids[0, 0] == self.decoder.config.bos_token_id:
                print("WARNING: there was a trailing prompt suffix bos",
                      prompt_suffix_ids)
                prompt_suffix_ids = prompt_suffix_ids[..., 1:]
                prompt_suffix_mask = prompt_suffix_mask[..., 1:]

            # cut off the eos token
            if prompt_suffix_ids[0, -1] == self.decoder.config.eos_token_id:
                print("WARNING: there was a trailing prompt suffix eos",
                      prompt_suffix_ids)
                prompt_suffix_ids = prompt_suffix_ids[..., :-1]
                prompt_suffix_mask = prompt_suffix_mask[..., :-1]

            # embed the suffix ids
            suffix_embeds = self.decoder.get_input_embeddings()(prompt_suffix_ids)

            connector_outputs.last_hidden_state = torch.hstack(
                (connector_outputs.last_hidden_state, suffix_embeds))
            if audio_attention_mask is not None:
                audio_attention_mask = torch.hstack(
                    (audio_attention_mask, prompt_suffix_mask))

        # remove the labels just to be sure..
        generate_kwargs.pop('labels')
        decoder_outputs = self.decoder.generate(
            inputs_embeds=connector_outputs.last_hidden_state,
            attention_mask=audio_attention_mask,
            **generate_kwargs,
        )

        return decoder_outputs