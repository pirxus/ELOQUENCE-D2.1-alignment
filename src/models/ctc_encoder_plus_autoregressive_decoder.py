import inspect
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
    SpeechEncoderDecoderConfig,
    SpeechEncoderDecoderModel,
)
from transformers.generation.beam_search import BeamScorer, BeamSearchScorer
from transformers.generation.logits_process import (
    LogitsProcessorList,
    MinLengthLogitsProcessor,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.generation.utils import (
    BeamSearchDecoderOnlyOutput,
    BeamSearchEncoderDecoderOutput,
    BeamSearchOutput,
)
from transformers.modeling_outputs import CausalLMOutput, ModelOutput, Seq2SeqLMOutput
from transformers.models.speech_encoder_decoder.modeling_speech_encoder_decoder import (
    shift_tokens_right,
)
from transformers.utils import logging

from decoding.ctc_scorer import CTCPrefixScoreTH
from models.auto_wrappers import CustomAutoModelForCTC
from models.decoders.multi_head_gpt2 import GPT2LMMultiHeadModel, GPT2MultiHeadConfig
from models.decoders.residual_clasiffier_gpt2 import (
    GPT2ResidualsLMHeadConfig,
    GPT2ResidualsLMHeadModel,
)
from models.encoders.e_branchformer import (
    Wav2Vec2EBranchformerConfig,
    Wav2Vec2EBranchformerForCTC,
)

from models.encoders.s2t_ctc import (
        Speech2TextEncoderForCTC,
        Speech2TextForCTCConfig
    )

logger = logging.get_logger("transformers")

AutoConfig.register("gpt2-multi-head", GPT2MultiHeadConfig)
AutoModelForCausalLM.register(GPT2MultiHeadConfig, GPT2LMMultiHeadModel)

AutoConfig.register("gpt2-residuals-head", GPT2ResidualsLMHeadConfig)
AutoModelForCausalLM.register(GPT2ResidualsLMHeadConfig, GPT2ResidualsLMHeadModel)

AutoConfig.register("wav2vec2-ebranchformer", Wav2Vec2EBranchformerConfig)
CustomAutoModelForCTC.register(Wav2Vec2EBranchformerConfig, Wav2Vec2EBranchformerForCTC)

AutoConfig.register("s2t-ctc-encoder", Speech2TextForCTCConfig)
CustomAutoModelForCTC.register(Speech2TextForCTCConfig, Speech2TextEncoderForCTC)


class JointCTCAttentionEncoderDecoderConfig(SpeechEncoderDecoderConfig):
    model_type = "joint_aed_ctc_speech-encoder-decoder"
    is_composition = True


@dataclass
class Seq2SeqLMOutputLosses(Seq2SeqLMOutput):
    enc_loss: Optional[torch.FloatTensor] = None
    dec_loss: Optional[torch.FloatTensor] = None
    encoder_logits: Optional[torch.FloatTensor] = None


def wav2vec2_for_ctc_forward_hook(model: CustomAutoModelForCTC, input: Any, output: CausalLMOutput):
    if "hidden_states" in output:
        output.last_hidden_state = output.hidden_states[-1]


class JointCTCAttentionEncoderDecoder(SpeechEncoderDecoderModel):
    """Custom model for CTC+Attention loss based on the ESPNet architecture"""

    config_class = JointCTCAttentionEncoderDecoderConfig
    base_model_prefix = "joint_aed_ctc_speech-encoder-decoder"

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[PreTrainedModel] = None,
    ):
        if config is None and (encoder is None or decoder is None):
            raise ValueError("Either a configuration or an encoder and a decoder has to be provided.")
        if config is None:
            config = SpeechEncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config)
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(f"Config: {config} has to be of type {self.config_class}")

        if config.decoder.cross_attention_hidden_size is not None:
            if config.decoder.cross_attention_hidden_size != config.encoder.hidden_size:
                raise ValueError(
                    "If `cross_attention_hidden_size` is specified in the decoder's configuration, it has to be equal"
                    f" to the encoder's `hidden_size`. Got {config.decoder.cross_attention_hidden_size} for"
                    f" `config.decoder.cross_attention_hidden_size` and {config.encoder.hidden_size} for"
                    " `config.encoder.hidden_size`."
                )

            # initialize with config
            # make sure input & output embeddings is not tied
        config.tie_word_embeddings = False
        super(SpeechEncoderDecoderModel, self).__init__(config)

        if encoder is None:
            encoder = CustomAutoModelForCTC.from_config(config.encoder)
            encoder.register_forward_hook(wav2vec2_for_ctc_forward_hook)
        if decoder is None:
            decoder = AutoModelForCausalLM.from_config(config.decoder)

        self.encoder = encoder
        self.decoder = decoder

        if self.encoder.config.to_dict() != self.config.encoder.to_dict():
            logger.warning(
                f"Config of the encoder: {self.encoder.__class__} is overwritten by shared encoder config:"
                f" {self.config.encoder}"
            )
        if self.decoder.config.to_dict() != self.config.decoder.to_dict():
            logger.warning(
                f"Config of the decoder: {self.decoder.__class__} is overwritten by shared decoder config:"
                f" {self.config.decoder}"
            )

        # make sure that the individual model's config refers to the shared config
        # so that the updates to the config will be synced
        self.encoder.config = self.config.encoder
        self.decoder.config = self.config.decoder

        # get encoder output hidden size
        self.encoder_output_dim = getattr(config.encoder, "output_hidden_size", config.encoder.hidden_size)
        if (
            self.encoder_output_dim != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            # encoder outputs might need to be projected to different dimension for decoder
            self.enc_to_dec_proj = nn.Linear(self.encoder.config.hidden_size, self.decoder.config.hidden_size)

        if self.encoder.get_output_embeddings() is not None:
            raise ValueError(
                f"The encoder {self.encoder} should not have a LM Head. Please use a model without LM Head"
            )
        self.enc_loss_weight = config.ctc_weight
        self.dec_loss_weight = 1 - config.ctc_weight
        self.lsm_factor = config.lsm_factor

        if config.shared_lm_head:
            self.encoder.lm_head.weight = self.decoder.lm_head.weight

        if (hasattr(config, "decoder_pos_emb_fixed") and config.decoder_pos_emb_fixed) or (
            hasattr(config.decoder, "pos_emb_fixed") and config.decoder.pos_emb_fixed
        ):
            from transformers.models.transfo_xl.modeling_transfo_xl import (
                AdaptiveEmbedding,
                PositionalEmbedding,
            )

            self.decoder.transformer.wte = AdaptiveEmbedding(
                n_token=config.decoder.vocab_size,
                d_embed=config.decoder.hidden_size,
                d_proj=config.decoder.hidden_size,
                cutoffs=[],
            )

            class PositionalEmbeddingM(PositionalEmbedding):
                def forward(self, pos_seq, bsz=None):
                    return super().forward(pos_seq.squeeze(0), bsz=bsz).squeeze(1)

            self.decoder.transformer.wpe = PositionalEmbeddingM(demb=config.decoder.hidden_size)
            self.decoder.post_init()

    @classmethod
    def from_encoder_decoder_pretrained(
        cls,
        encoder_pretrained_model_name_or_path: str = None,
        decoder_pretrained_model_name_or_path: str = None,
        *model_args,
        **kwargs,
    ) -> PreTrainedModel:

        kwargs_encoder = {
            argument[len("encoder_") :]: value for argument, value in kwargs.items() if argument.startswith("encoder_")
        }

        kwargs_decoder = {
            argument[len("decoder_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("decoder_") and argument != "decoder_start_token_id"
        }

        # remove encoder, decoder kwargs from kwargs
        for key in kwargs_encoder.keys():
            del kwargs["encoder_" + key]
        for key in kwargs_decoder.keys():
            del kwargs["decoder_" + key]

        # Load and initialize the encoder and decoder
        # The distinction between encoder and decoder at the model level is made
        # by the value of the flag `is_decoder` that we need to set correctly.
        encoder = kwargs_encoder.pop("model", None)
        if encoder is None:
            if encoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `encoder_model` is not defined as an argument, a `encoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_encoder:
                encoder_config, kwargs_encoder = AutoConfig.from_pretrained(
                    encoder_pretrained_model_name_or_path, **kwargs_encoder, return_unused_kwargs=True
                )

                if encoder_config.is_decoder is True or encoder_config.add_cross_attention is True:
                    logger.info(
                        f"Initializing {encoder_pretrained_model_name_or_path} as a encoder model "
                        "from a decoder model. Cross-attention and casual mask are disabled."
                    )
                    encoder_config.is_decoder = False
                    encoder_config.add_cross_attention = False

                kwargs_encoder["config"] = encoder_config

            encoder = CustomAutoModelForCTC.from_pretrained(
                encoder_pretrained_model_name_or_path, *model_args, **kwargs_encoder
            )
            encoder.register_forward_hook(wav2vec2_for_ctc_forward_hook)

        decoder = kwargs_decoder.pop("model", None)
        if decoder is None:
            if decoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `decoder_model` is not defined as an argument, a `decoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_decoder:
                decoder_config, kwargs_decoder = AutoConfig.from_pretrained(
                    decoder_pretrained_model_name_or_path, **kwargs_decoder, return_unused_kwargs=True
                )

                if decoder_config.is_decoder is False or decoder_config.add_cross_attention is False:
                    logger.info(
                        f"Initializing {decoder_pretrained_model_name_or_path} as a decoder model. Cross attention"
                        f" layers are added to {decoder_pretrained_model_name_or_path} and randomly initialized if"
                        f" {decoder_pretrained_model_name_or_path}'s architecture allows for cross attention layers."
                    )
                    decoder_config.is_decoder = True
                    decoder_config.add_cross_attention = True

                kwargs_decoder["config"] = decoder_config

            if kwargs_decoder["config"].is_decoder is False or kwargs_decoder["config"].add_cross_attention is False:
                logger.warning(
                    f"Decoder model {decoder_pretrained_model_name_or_path} is not initialized as a decoder. "
                    f"In order to initialize {decoder_pretrained_model_name_or_path} as a decoder, "
                    "make sure that the attributes `is_decoder` and `add_cross_attention` of `decoder_config` "
                    "passed to `.from_encoder_decoder_pretrained(...)` are set to `True` or do not pass a "
                    "`decoder_config` to `.from_encoder_decoder_pretrained(...)`"
                )

            decoder = AutoModelForCausalLM.from_pretrained(decoder_pretrained_model_name_or_path, **kwargs_decoder)

        # instantiate config with corresponding kwargs
        config = JointCTCAttentionEncoderDecoderConfig.from_encoder_decoder_configs(
            encoder.config, decoder.config, **kwargs
        )

        # make sure input & output embeddings is not tied
        config.tie_word_embeddings = False
        return cls(encoder=encoder, decoder=decoder, config=config)

    def forward(
        self,
        inputs: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        input_values: Optional[torch.FloatTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutputLosses]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        if encoder_outputs is None:
            if inputs is None:
                if input_values is not None and input_features is not None:
                    raise ValueError("You cannot specify both input_values and input_features at the same time")
                elif input_values is not None:
                    inputs = input_values
                elif input_features is not None:
                    inputs = input_features
                else:
                    raise ValueError("You have to specify either input_values or input_features")

            encoder_outputs = self.encoder(
                inputs,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict,
                labels=labels,
                **kwargs_encoder,
            )
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = CausalLMOutput(*encoder_outputs)

        encoder_hidden_states = encoder_outputs.last_hidden_state

        # optionally project encoder_hidden_states
        if (
            self.encoder_output_dim != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        # compute correct encoder attention mask
        if attention_mask is not None:
            encoder_attention_mask = self.encoder._get_feature_vector_attention_mask(
                encoder_hidden_states.shape[1], attention_mask
            )
        else:
            encoder_attention_mask = None

        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True
            if hasattr(self.decoder, "head_weights") and len(self.decoder.head_weights) > 1
            else output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        # Compute loss independent from decoder (as some shift the logits inside them)
        loss = enc_loss = dec_loss = None

        if labels is not None:

            loss_fct = CrossEntropyLoss(label_smoothing=self.lsm_factor)
            enc_loss = encoder_outputs.loss if return_dict else encoder_outputs[0]
            if isinstance(self.decoder, GPT2LMMultiHeadModel) and len(self.decoder.head_weights) > 1:
                dec_loss = torch.zeros_like(enc_loss)
                lm_logits_per_layer = []
                for index, lm_head, lm_weight in zip(
                    [*self.decoder.head_locations, -1],
                    [*self.decoder.additional_lm_heads, self.decoder.lm_head],
                    self.decoder.head_weights,
                ):
                    lm_logits = lm_head(decoder_outputs.hidden_states[index])
                    dec_loss += lm_weight * loss_fct(
                        lm_logits.reshape(-1, self.decoder.config.vocab_size), labels.reshape(-1)
                    )
                    lm_logits_per_layer.append(lm_logits)
                if self.decoder.config.average_logits:
                    decoder_outputs.logits = torch.matmul(
                        torch.stack(lm_logits_per_layer).T,
                        torch.tensor(self.decoder.head_weights, device=lm_logits_per_layer[-1].device),
                    ).T

            else:
                dec_logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
                dec_loss = loss_fct(dec_logits.reshape(-1, self.decoder.config.vocab_size), labels.reshape(-1))
            loss = self.enc_loss_weight * enc_loss + self.dec_loss_weight * dec_loss

        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs

        return Seq2SeqLMOutputLosses(
            loss=loss,
            enc_loss=enc_loss,
            dec_loss=dec_loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_hidden_states,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            encoder_logits=encoder_outputs.logits,
        )

    def activate_memory_params(self):
        self.encoder.activate_memory_params()
        self.decoder.activate_memory_params()

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def connect_context_container(self, context_container):
        self.encoder.connect_context_container(context_container)
        self.decoder.connect_context_container(context_container)

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        if model_kwargs.get("token_type_ids") is not None:
            model_kwargs["token_type_ids"] = model_kwargs["token_type_ids"].repeat_interleave(expand_size, dim=0)

        if model_kwargs.get("attention_mask") is not None:
            model_kwargs["attention_mask"] = model_kwargs["attention_mask"].repeat_interleave(expand_size, dim=0)

        if is_encoder_decoder:
            encoder_outputs = model_kwargs.get("encoder_outputs")
            if encoder_outputs is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.repeat_interleave(
                expand_size, dim=0
            )
            if encoder_outputs.get("logits") is not None:
                encoder_outputs["logits"] = encoder_outputs.logits.repeat_interleave(expand_size, dim=0)
            if hasattr(self.decoder, "expand_context_states"):
                self.decoder.expand_context_states(expand_size)

            model_kwargs["encoder_outputs"] = encoder_outputs
            decoder_attention_mask = model_kwargs.get("decoder_attention_mask")
            if decoder_attention_mask is not None:
                model_kwargs["decoder_attention_mask"] = decoder_attention_mask.repeat_interleave(expand_size, dim=0)

        return input_ids, model_kwargs

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ) -> Dict[str, Any]:
        # 1. get encoder
        encoder = self.get_encoder()
        # Compatibility with Accelerate big model inference: we need the encoder to outputs stuff on the same device
        # as the inputs.
        if hasattr(encoder, "_hf_hook"):
            encoder._hf_hook.io_same_device = True

        # 2. Prepare encoder args and encoder kwargs from model kwargs.
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }
        encoder_signature = set(inspect.signature(encoder.forward).parameters)
        encoder_accepts_wildcard = "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
        if not encoder_accepts_wildcard:
            encoder_kwargs = {
                argument: value for argument, value in encoder_kwargs.items() if argument in encoder_signature
            }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs["output_hidden_states"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)

        return model_kwargs

    def joint_beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        **model_kwargs,
    ) -> Union[BeamSearchOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **beam search decoding** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        <Tip warning={true}>

        In most cases, you do not need to call [`~generation.GenerationMixin.beam_search`] directly. Use generate()
        instead. For an overview of generation strategies and code examples, check the [following
        guide](./generation_strategies).

        </Tip>

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            beam_scorer (`BeamScorer`):
                An derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
                sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`int`, *optional*):
                The id of the *end-of-sequence* token.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`generation.BeamSearchDecoderOnlyOutput`], [`~generation.BeamSearchEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.BeamSearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.BeamSearchEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.


        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForSeq2SeqLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     BeamSearchScorer,
        ... )
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

        >>> encoder_input_str = "translate English to German: How old are you?"
        >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids


        >>> # lets run beam search using 3 beams
        >>> num_beams = 3
        >>> # define decoder start token ids
        >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
        >>> input_ids = input_ids * model.config.decoder_start_token_id

        >>> # add encoder_outputs to model keyword arguments
        >>> model_kwargs = {
        ...     "encoder_outputs": model.get_encoder()(
        ...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
        ...     )
        ... }

        >>> # instantiate beam scorer
        >>> beam_scorer = BeamSearchScorer(
        ...     batch_size=1,
        ...     num_beams=num_beams,
        ...     device=model.device,
        ... )

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
        ...     ]
        ... )

        >>> outputs = model.beam_search(input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs)

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ['Wie alt bist du?']
        ```"""

        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        """
        CTC Prefix scorer to bias model predictions with CTC scores (lower number of insertions and deletions)
        """
        non_repeated_beam_mask = torch.zeros((batch_size, num_beams), dtype=torch.bool)
        non_repeated_beam_mask[:, 0] = True
        non_repeated_beam_mask = non_repeated_beam_mask.view(-1)
        ctc_prefix_scorer = CTCPrefixScoreTH(
            nn.functional.log_softmax(model_kwargs["encoder_outputs"].get("logits")[non_repeated_beam_mask], dim=-1),
            self.encoder._get_feat_extract_output_lengths(
                model_kwargs["attention_mask"][non_repeated_beam_mask].sum(dim=1)
            ),
            self.generation_config.pad_token_id,
            self.generation_config.eos_token_id,
            model_kwargs["margin"],
        )
        ctc_weight = model_kwargs["ctc_weight"]
        ctc_beam_width = model_kwargs["ctc_beam_width"]
        ctc_states = None
        att_w = None
        external_lm = model_kwargs["external_lm"]
        if external_lm is not None:
            external_lm = external_lm.to(self.device)
            # from transformers import AutoTokenizer
            # tokenizer = AutoTokenizer.from_pretrained("Lakoc/ted_uni500")
        external_lm_weight = model_kwargs["external_lm_weight"]

        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))
        # print("\n", "-" * 100, "\n", "-" * 100, "\n")
        # print("Reference:", tokenizer.batch_decode(model_kwargs["labels"].tolist()))

        this_peer_finished = False  # used by synced_gpus only
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if outputs.get("cross_attentions") is not None:
                att_w = outputs["cross_attentions"][-1]

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]
            # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
            # cannot be generated both before and after the `nn.functional.log_softmax` operation.
            next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
            next_token_scores_dec = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            """
            Sample next tokens that will be used for CTC prefix computation and compute CTC cumulative prefix scores
            """
            # print("\n\n")
            # print("Input:", tokenizer.batch_decode(input_ids.tolist()))
            # print("Acoustic:",
            #       torch.topk(next_token_scores, 5).values, "\n",
            #       tokenizer.batch_decode(torch.topk(next_token_scores, 5).indices.tolist()))
            next_token_scores_dec[:, self.generation_config.pad_token_id] = ctc_prefix_scorer.logzero
            local_best_scores, local_best_ids = torch.topk(next_token_scores_dec, ctc_beam_width, dim=1)

            ctc_scores, ctc_states = ctc_prefix_scorer(input_ids, ctc_states, local_best_ids, att_w)
            next_token_scores = (1 - ctc_weight) * next_token_scores_dec + ctc_weight * ctc_scores

            # print("CTC:",
            #       torch.topk(ctc_scores, 5).values, "\n",
            #       tokenizer.batch_decode(torch.topk(ctc_scores, 5).indices.tolist()))

            if external_lm is not None:
                external_lm_logits = external_lm(
                    input_ids, labels=None, output_hidden_states=False, return_dict=True
                ).logits
                external_lm_logits = external_lm_logits[:, -1, :]
                external_lm_logits = self.adjust_logits_during_generation(external_lm_logits, cur_len=cur_len)
                external_lm_scores = nn.functional.log_softmax(external_lm_logits, dim=-1)
                next_token_scores = next_token_scores + external_lm_weight * external_lm_scores
            else:
                external_lm_scores = torch.zeros_like(next_token_scores)

            #     print("LM:",
            #           torch.topk(external_lm_scores, 5).values, "\n",
            #           tokenizer.batch_decode(torch.topk(external_lm_scores, 5).indices.tolist()))
            # print("LM+Acoustic:",
            #       torch.topk(next_token_scores, 5).values, "\n",
            #       tokenizer.batch_decode(torch.topk(next_token_scores, 5).indices.tolist()))
            next_token_scores_processed = logits_processor(input_ids, next_token_scores)

            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed, next_token_scores_dec, ctc_scores, external_lm_scores)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,) if self.config.is_encoder_decoder else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            # Sample 2 next tokens for each beam (so we have some spare tokens and match output of beam search)
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            """

            Changed part


            """
            ctc_states = ctc_prefix_scorer.index_select_state(ctc_states, beam_next_tokens.view(batch_size, -1))
            """

            /Changed part


            """

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past_key_values"] is not None:
                model_kwargs["past_key_values"] = self._reorder_cache(model_kwargs["past_key_values"], beam_idx)

            if return_dict_in_generate and output_scores:
                beam_indices: Tuple
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
        )
        # ref = tokenizer.decode(model_kwargs['labels'].tolist()[0], skip_special_tokens=True)
        # hyp = tokenizer.decode(sequence_outputs['sequences'][0].tolist(), skip_special_tokens=True)
        # print(f"Ref: {ref}\nHyp: {hyp}")
        # import jiwer
        # print(jiwer.compute_measures(ref, hyp))

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None

            if self.config.is_encoder_decoder:
                return BeamSearchEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=sequence_outputs["beam_indices"],
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return BeamSearchDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=sequence_outputs["beam_indices"],
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequence_outputs["sequences"]
