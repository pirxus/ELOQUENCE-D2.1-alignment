from dataclasses import dataclass
from transformers import (

    PreTrainedModel,
    PretrainedConfig,
    Blip2QFormerConfig,
    Blip2QFormerModel,
)
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput, ModelOutput

from torch.nn import CrossEntropyLoss

from typing import Optional, Tuple, Union, Any

import torch
from torch import nn


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
class SpeechQFormerEncoderDecoderModelOutput(ModelOutput):
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
    logits: Optional[Tuple[torch.FloatTensor]] = None
    audio_outputs: Optional[torch.FloatTensor] = None
    qformer_outputs: Optional[Tuple[torch.FloatTensor]] = None
    language_model_outputs: Optional[Tuple[torch.FloatTensor]] = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k]
            if k not in ["vision_outputs", "qformer_outputs", "language_model_outputs"]
            else getattr(self, k).to_tuple()
            for k in self.keys()
        )



class  SpeechQFormerEncoderDecoderConfig(PretrainedConfig):
    def __init__(
            self,
            encoder=None,
            qformer=None,
            decoder=None,
            num_query_tokens=80,
            decoder_pad_token_id=None,
            bos_token_id=None,
            eos_token_id=None,
            pad_token_id=None,
            **kwargs
        ):

        self.encoder = encoder
        if encoder is None:
            self.qformer = Blip2QFormerConfig()
        else:
            self.qformer = qformer

        self.decoder = decoder
        self.decoder_pad_token_id = pad_token_id
        if self.decoder:
            self.decoder.pad_token_id = decoder_pad_token_id

        self.num_query_tokens = num_query_tokens

        self.decoder_bos_token_id = bos_token_id
        self.decoder_eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        super().__init__(**kwargs)


class SpeechQFormerEncoderDecoder(PreTrainedModel):
    """
    This class has been modified from it's original version taken
    from: https://github.com/huggingface/transformers/blob/main/src/transformers/models/blip_2/modeling_blip_2.py
    """

    config_class = SpeechQFormerEncoderDecoderConfig
    main_input_name = "input_features"

    def __init__(
            self,
            config: Optional[SpeechQFormerEncoderDecoderConfig] = None,
            encoder: Optional[PreTrainedModel] = None,
            decoder: Optional[PreTrainedModel] = None,
        ):
        super().__init__(config)


        if encoder:
            self.encoder = encoder.get_encoder()
        else:
            raise ValueError("Encoder model needs to be supplied")
            #self.encoder = Speech2TextModel(config=config.encoder_config)

        if decoder:
            self.decoder = decoder

        else:
            raise ValueError("Decoder language model needs to be supplied")

        self.query_tokens = nn.Parameter(
                torch.zeros(1, config.num_query_tokens, config.qformer.hidden_size)
            )

        self.qformer = Blip2QFormerModel(config.qformer)
        self.language_projection = nn.Linear(config.qformer.hidden_size, config.decoder.hidden_size)

        # freeze encoder and decoder
        for _, param in self.encoder.named_parameters():
            param.requires_grad = False

        for _, param in self.decoder.named_parameters():
            param.requires_grad = False

        # FIXME: the padding required in generate - rework the config class to include all
        # the token_ids on the top level ...

        self.config = config
        self.config.update({'pad_token_id': self.config.decoder_pad_token_id})
        #self.decoder.config.update({'pad_token_id': self.config.decoder_pad_token_id})
        self.decoder_bos_token_id = self.config.decoder_bos_token_id
        self.decoder_eos_token_id = self.config.decoder_eos_token_id
        self.decoder_pad_token_id = self.config.decoder_pad_token_id
        self.decoder.config.pad_token_id = self.config.decoder_pad_token_id
        self.decoder.config.bos_token_id = self.config.decoder_bos_token_id
        self.decoder.config.eos_token_id = self.config.decoder_eos_token_id

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
    ) -> Union[Tuple, SpeechQFormerEncoderDecoderModelOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.decoder_pad_token_id, self.config.decoder_bos_token_id
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

        query_output = query_outputs[0]

        # step 3: use the language model, conditioned on the query outputs and the prompt
        language_model_inputs = self.language_projection(query_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )
        inputs_embeds = language_model_inputs

        batch_size = input_features.shape[0]

        inputs_embeds = self.decoder.get_input_embeddings()(decoder_input_ids)
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)

        attention_mask = torch.ones_like(decoder_input_ids)
        expected_device = language_model_attention_mask.device
        attention_mask = torch.cat([language_model_attention_mask, attention_mask.to(expected_device)], dim=1)

        #attention_mask = language_model_attention_mask

        outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits if return_dict else outputs[0]
        loss = None
        # we compute the loss here since we need to take into account the sequence length of the query embeds
        if labels is not None:
            labels = labels.to(logits.device)
            logits = logits[:, -labels.size(1) :, :]
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(logits.device)

            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction="mean")

            loss = loss_fct(shift_logits.view(-1, self.config.decoder.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits, encoder_outputs, query_outputs, outputs)
            return ((loss,) + output) if loss is not None else output

        return SpeechQFormerEncoderDecoderModelOutput(
            loss=loss,
            logits=logits,
            audio_outputs=encoder_outputs,
            qformer_outputs=query_outputs,
            language_model_outputs=outputs,
        )

    @torch.no_grad()
    def generate(
        self,
        input_features: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
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
        query_output = query_outputs[0]

        language_model_inputs = self.language_projection(query_output)
        language_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )
        #if input_ids is None:
        input_ids = (
            torch.LongTensor([[self.config.decoder_bos_token_id]])
            .repeat(batch_size, 1)
            .to(audio_embeds.device)
        )

        attention_mask = torch.ones_like(input_ids)
        attention_mask = torch.cat([language_attention_mask, attention_mask.to(language_attention_mask.device)], dim=1)

        # concatenate query embeddings with prompt embeddings
        inputs_embeds = self.decoder.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)

        outputs = self.decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        return outputs