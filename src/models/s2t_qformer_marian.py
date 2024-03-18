from transformers import (

    PreTrainedModel,
    PretrainedConfig,
    Blip2QFormerConfig,
    Blip2QFormerModel,
    Speech2TextModel,
    MarianMTModel,
    MarianForCausalLM,
)
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput, ModelOutput

from typing import Optional, Tuple, Union
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy


# Copied from transformers.models.bart.modeling_bart.shift_tokens_right
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

@dataclass
class SpeechQFormerMarianOutput(ModelOutput):
    """
    TODO: rewrite
    Class defining the outputs of [`Blip2ForConditionalGeneration`].

    Args:
        loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Language modeling loss from the language model.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head of the language model.
        vision_outputs (`BaseModelOutputWithPooling`):
            Outputs of the vision encoder.
        qformer_outputs (`BaseModelOutputWithPoolingAndCrossAttentions`):
            Outputs of the Q-Former (Querying Transformer).
        language_model_outputs (`CausalLMOutputWithPast` or `Seq2SeqLMOutput`):
            Outputs of the language model.
    """

    loss: Optional[Tuple[torch.FloatTensor]] = None
    enc_loss: Optional[Tuple[torch.FloatTensor]] = None
    dec_loss: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    audio_outputs: Optional[torch.FloatTensor] = None
    qformer_outputs: Optional[Tuple[torch.FloatTensor]] = None
    language_model_outputs: Optional[Tuple[torch.FloatTensor]] = None


class  ApmoConfig(PretrainedConfig):
    def __init__(
            self,
            encoder_config=None,
            qformer_config=None,
            lm_config=None,
            num_query_tokens=80,
            modality_matching=True,
            mm_pooling='avg',
            mm_loss_weight=1.0,
            ce_loss_weight=1.0,
            **kwargs
        ):

        self.encoder_config = encoder_config
        if qformer_config is None:
            self.qformer_config = Blip2QFormerConfig()
        else:
            self.qformer_config = qformer_config

        self.lm_config = lm_config
        self.modality_matching = modality_matching
        self.mm_pooling=mm_pooling
        self.num_query_tokens = num_query_tokens
        self.mm_loss_weight = mm_loss_weight
        self.ce_loss_weight = ce_loss_weight
        super().__init__(**kwargs)


class ApmoModel(PreTrainedModel):
    config_class = ApmoConfig
    main_input_name = "input_features"

    def __init__(
            self,
            config: Optional[ApmoConfig] = None,
            encoder: Optional[PreTrainedModel] = None,
            decoder: Optional[PreTrainedModel] = None,
        ):
        super().__init__(config)

        if encoder:
            #self.encoder = Speech2TextModel.from_pretrained(encoder).get_encoder()
            self.encoder = encoder.get_encoder()
        else:
            raise ValueError("Encoder model needs to be supplied")
            #self.encoder = Speech2TextModel(config=config.encoder_config)

        if decoder:

            self.mt_encoder = decoder.model.encoder

            self.decoder = MarianForCausalLM(decoder.config)
            #self.decoder = decoder

            self.decoder.model.decoder = deepcopy(decoder.model.decoder)
            self.decoder.lm_head = deepcopy(decoder.lm_head)

        else:
            raise ValueError("Decoder model needs to be supplied")
            #self.decoder = MarianMTModel(config=config.lm_config)

        self.query_tokens = nn.Parameter(
                torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size)
            )

        self.qformer = Blip2QFormerModel(config.qformer_config)
        self.language_projection = nn.Linear(config.qformer_config.hidden_size, config.lm_config.hidden_size)

        # freeze encoder and decoder
        for _, param in self.encoder.named_parameters():
            param.requires_grad = False

        for _, param in self.mt_encoder.named_parameters():
            param.requires_grad = False

        for _, param in self.decoder.model.named_parameters():
            param.requires_grad = False
        self.decoder.lm_head.requires_grad = False

        # FIXME: the padding required in generate - rework the config class to include all
        # the token_ids on the top level ...
        self.config = config
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
        mm_input_ids: Optional[torch.LongTensor] = None,
        mm_attention_mask: Optional[torch.LongTensor] = None,
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
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        audio_embeds = encoder_outputs[0]

        # downsample encoder attention mask
        if attention_mask is not None:
            audio_attention_mask = self.encoder._get_feature_vector_attention_mask(
                encoder_outputs[0].shape[1], attention_mask
            )
        else:
            audio_attention_mask = None

        # 2. pass the encoder output to the qformer
        query_tokens = self.query_tokens.expand(audio_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=audio_embeds,
            encoder_attention_mask=audio_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        query_output = BaseModelOutput(
            last_hidden_state=self.language_projection(query_outputs[0]),
        )

        # compute the modality matching loss
        mm_loss = None
        if mm_input_ids is not None:
            mt_encoder_output = self.mt_encoder(
                input_ids=mm_input_ids,
                attention_mask=mm_attention_mask,
                return_dict=return_dict,
            )
            # TODO:

            if self.config.mm_pooling == 'avg':
                query_pooled = torch.mean(query_output.last_hidden_state, 1)
                enc_pooled = torch.mean(mt_encoder_output[0], 1)

                mm_loss = F.mse_loss(query_pooled, enc_pooled)

        decoder_output = self.decoder(
            #None,
            input_ids=decoder_input_ids, # input_ids
            #attention_mask=None,
            attention_mask=decoder_attention_mask,
            #decoder_input_ids=decoder_input_ids,
            #encoder_outputs=query_output,
            encoder_hidden_states=query_output.last_hidden_state,
            #decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            #decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=None,
            #decoder_inputs_embeds=None,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        loss = decoder_output.loss
        if mm_loss:
            if return_dict:
                loss = self.config.ce_loss_weight * decoder_output['loss'] + self.config.mm_loss_weight * mm_loss
                        
            else:
                loss = self.config.ce_loss_weight * decoder_output[0] + self.config.mm_loss_weight * mm_loss


        return SpeechQFormerMarianOutput(
                loss=loss,
                enc_loss=mm_loss,
                dec_loss=decoder_output.loss,
                logits=decoder_output.logits,
                audio_outputs=audio_embeds,
                qformer_outputs=None, #FIXME
                language_model_outputs=None, #FIXME
            )

    @torch.no_grad()
    def generate(
        self,
        input_features: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        mm_input_ids: Optional[torch.LongTensor] = None,
        mm_attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:

        batch_size = input_features.shape[0]

        # 1. forward the audio through the encoder
        encoder_outputs = self.encoder(
            input_features,
            attention_mask=attention_mask,
            return_dict=True
        )

        audio_embeds = encoder_outputs[0]

        # downsample encoder attention mask
        if attention_mask is not None:
            audio_attention_mask = self.encoder._get_feature_vector_attention_mask(
                audio_embeds.shape[1], attention_mask
            )
        else:
            audio_attention_mask = None

        # 2. pass the encoder output to the qformer
        query_tokens = self.query_tokens.expand(audio_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=audio_embeds,
            encoder_attention_mask=audio_attention_mask,
            return_dict=True,
        )
        query_outputs = self.language_projection(query_outputs[0])

        #decoder_output = self.decoder(
        #    None, # input_ids
        #    attention_mask=None,
        #    encoder_outputs=query_output,
        #    inputs_embeds=None,
        #    decoder_inputs_embeds=None,
        #    return_dict=True,
        #)

        decoder_input = (
            torch.LongTensor([[self.decoder.config.decoder_start_token_id]])
            .repeat(batch_size, 1)
            .to(query_outputs.device)
        )
        attention_mask = torch.ones_like(decoder_input)
        decoder_output = self.decoder.generate(
            input_ids=decoder_input, # input_ids
            attention_mask=attention_mask,
            encoder_hidden_states=query_outputs,
            **generate_kwargs,
        )
        return decoder_output
