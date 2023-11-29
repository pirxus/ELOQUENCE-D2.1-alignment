# coding=utf-8
# Copyright 2022 The Fairseq Authors and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch Wav2Vec2-Ebranchformer model."""

import functools
import math
import operator
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.modeling_outputs import (
    BaseModelOutput,
    CausalLMOutput,
    Wav2Vec2BaseModelOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import (
    Wav2Vec2ConformerAdapter as Wav2Vec2EBranchformerAdapter,
)
from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import (
    Wav2Vec2ConformerFeatureEncoder as Wav2Vec2EBranchformerFeatureEncoder,
)
from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import (
    Wav2Vec2ConformerFeatureProjection as Wav2Vec2EBranchformerFeatureProjection,
)
from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import (
    Wav2Vec2ConformerFeedForward as Wav2Vec2EBranchformerFeedForward,
)
from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import (
    Wav2Vec2ConformerRelPositionalEmbedding as Wav2Vec2EBranchformerRelPositionalEmbedding,
)
from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import (
    Wav2Vec2ConformerRotaryPositionalEmbedding as Wav2Vec2EBranchformerRotaryPositionalEmbedding,
)
from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import (
    Wav2Vec2ConformerSelfAttention as Wav2Vec2EBranchformerSelfAttention,
)
from transformers.utils import logging

from models.encoders.extractors import MelFeatureExtractor

logger = logging.get_logger(__name__)

_HIDDEN_STATES_START_POSITION = 2


# Copied from transformers.models.wav2vec2.modeling_wav2vec2._compute_mask_indices
def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    attention_mask: Optional[torch.LongTensor] = None,
    min_masks: int = 0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape. Used to implement [SpecAugment: A Simple Data Augmentation Method for
    ASR](https://arxiv.org/abs/1904.08779). Note that this method is not optimized to run on TPU and should be run on
    CPU as part of the preprocessing during training.

    Args:
        shape: The shape for which to compute masks. This should be of a tuple of size 2 where
               the first element is the batch size and the second element is the length of the axis to span.
        mask_prob:  The percentage of the whole axis (between 0 and 1) which will be masked. The number of
                    independently generated mask spans of length `mask_length` is computed by
                    `mask_prob*shape[1]/mask_length`. Note that due to overlaps, `mask_prob` is an upper bound and the
                    actual percentage will be smaller.
        mask_length: size of the mask
        min_masks: minimum number of masked spans
        attention_mask: A (right-padded) attention mask which independently shortens the feature axis of
                        each batch dimension.
    """
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length}"
            f" and `sequence_length`: {sequence_length}`"
        )

    # epsilon is used for probabilistic rounding
    epsilon = np.random.rand(1).item()

    def compute_num_masked_span(input_length):
        """Given input length, compute how many spans should be masked"""
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)
        num_masked_span = max(num_masked_span, min_masks)

        # make sure num masked span <= sequence_length
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length

        # make sure num_masked span is also <= input_length - (mask_length - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)

        return num_masked_span

    # compute number of masked spans in batch
    input_lengths = (
        attention_mask.sum(-1).detach().tolist()
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]
    )

    # SpecAugment mask to fill
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)
    spec_aug_mask_idxs = []

    max_num_masked_span = compute_num_masked_span(sequence_length)

    if max_num_masked_span == 0:
        return spec_aug_mask

    for input_length in input_lengths:
        # compute num of masked spans for this input
        num_masked_span = compute_num_masked_span(input_length)

        # get random indices to mask
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
        )

        # pick first sampled index that will serve as a dummy index to pad vector
        # to ensure same dimension for all batches due to probabilistic rounding
        # Picking first sample just pads those vectors twice.
        if len(spec_aug_mask_idx) == 0:
            # this case can only happen if `input_length` is strictly smaller then
            # `sequence_length` in which case the last token has to be a padding
            # token which we can use as a dummy mask id
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]

        spec_aug_mask_idx = np.concatenate(
            [
                spec_aug_mask_idx,
                np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx,
            ]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)

    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)

    # expand masked indices to masked spans
    spec_aug_mask_idxs = np.broadcast_to(spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length))
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)

    # add offset to the starting indexes so that indexes now create a span
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        batch_size, max_num_masked_span * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # ensure that we cannot have indices larger than sequence_length
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1

    # scatter indices to mask
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    return spec_aug_mask


class Wav2Vec2EBranchformerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Wav2Vec2EBranchformerModel`]. It is used to
    instantiate an Wav2Vec2EBranchformer model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the Wav2Vec2EBranchformer
    [facebook/wav2vec2-Ebranchformer-rel-pos-large](https://huggingface.co/facebook/wav2vec2-Ebranchformer-rel-pos-large)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*):
            Vocabulary size of the Wav2Vec2EBranchformer model. Defines the number of different tokens that can be
            represented by the `inputs_ids` passed when calling [`Wav2Vec2EBranchformerModel`]. Vocabulary size of the
            model. Defines the different tokens that can be represented by the *inputs_ids* passed to the forward
            method of [`Wav2Vec2EBranchformerModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        final_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for the final projection layer of [`Wav2Vec2EBranchformerForCTC`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        feat_extract_norm (`str`, *optional*, defaults to `"group"`):
            The norm to be applied to 1D convolutional layers in feature encoder. One of `"group"` for group
            normalization of only the first 1D convolutional layer or `"layer"` for layer normalization of all 1D
            convolutional layers.
        feat_proj_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for output of the feature encoder.
        feat_extract_activation (`str, `optional`, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the 1D convolutional layers of the feature
            extractor. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.
        feat_quantizer_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probabilitiy for quantized feature encoder states.
        conv_dim (`Tuple[int]` or `List[int]`, *optional*, defaults to `(512, 512, 512, 512, 512, 512, 512)`):
            A tuple of integers defining the number of input and output channels of each 1D convolutional layer in the
            feature encoder. The length of *conv_dim* defines the number of 1D convolutional layers.
        conv_stride (`Tuple[int]` or `List[int]`, *optional*, defaults to `(5, 2, 2, 2, 2, 2, 2)`):
            A tuple of integers defining the stride of each 1D convolutional layer in the feature encoder. The length
            of *conv_stride* defines the number of convolutional layers and has to match the length of *conv_dim*.
        conv_kernel (`Tuple[int]` or `List[int]`, *optional*, defaults to `(10, 3, 3, 3, 3, 3, 3)`):
            A tuple of integers defining the kernel size of each 1D convolutional layer in the feature encoder. The
            length of *conv_kernel* defines the number of convolutional layers and has to match the length of
            *conv_dim*.
        conv_bias (`bool`, *optional*, defaults to `False`):
            Whether the 1D convolutional layers have a bias.
        num_conv_pos_embeddings (`int`, *optional*, defaults to 128):
            Number of convolutional positional embeddings. Defines the kernel size of 1D convolutional positional
            embeddings layer.
        num_conv_pos_embedding_groups (`int`, *optional*, defaults to 16):
            Number of groups of 1D convolutional positional embeddings layer.
        apply_spec_augment (`bool`, *optional*, defaults to `True`):
            Whether to apply *SpecAugment* data augmentation to the outputs of the feature encoder. For reference see
            [SpecAugment: A Simple Data Augmentation Method for Automatic Speech
            Recognition](https://arxiv.org/abs/1904.08779).
        mask_time_prob (`float`, *optional*, defaults to 0.05):
            Percentage (between 0 and 1) of all feature vectors along the time axis which will be masked. The masking
            procecure generates ''mask_time_prob*len(time_axis)/mask_time_length'' independent masks over the axis. If
            reasoning from the propability of each feature vector to be chosen as the start of the vector span to be
            masked, *mask_time_prob* should be `prob_vector_start*mask_time_length`. Note that overlap may decrease the
            actual percentage of masked vectors. This is only relevant if `apply_spec_augment is True`.
        mask_time_length (`int`, *optional*, defaults to 10):
            Length of vector span along the time axis.
        mask_time_min_masks (`int`, *optional*, defaults to 2),:
            The minimum number of masks of length `mask_feature_length` generated along the time axis, each time step,
            irrespectively of `mask_feature_prob`. Only relevant if ''mask_time_prob*len(time_axis)/mask_time_length <
            mask_time_min_masks''
        mask_feature_prob (`float`, *optional*, defaults to 0.0):
            Percentage (between 0 and 1) of all feature vectors along the feature axis which will be masked. The
            masking procecure generates ''mask_feature_prob*len(feature_axis)/mask_time_length'' independent masks over
            the axis. If reasoning from the propability of each feature vector to be chosen as the start of the vector
            span to be masked, *mask_feature_prob* should be `prob_vector_start*mask_feature_length`. Note that overlap
            may decrease the actual percentage of masked vectors. This is only relevant if `apply_spec_augment is
            True`.
        mask_feature_length (`int`, *optional*, defaults to 10):
            Length of vector span along the feature axis.
        mask_feature_min_masks (`int`, *optional*, defaults to 0),:
            The minimum number of masks of length `mask_feature_length` generated along the feature axis, each time
            step, irrespectively of `mask_feature_prob`. Only relevant if
            ''mask_feature_prob*len(feature_axis)/mask_feature_length < mask_feature_min_masks''
        num_codevectors_per_group (`int`, *optional*, defaults to 320):
            Number of entries in each quantization codebook (group).
        num_codevector_groups (`int`, *optional*, defaults to 2):
            Number of codevector groups for product codevector quantization.
        contrastive_logits_temperature (`float`, *optional*, defaults to 0.1):
            The temperature *kappa* in the contrastive loss.
        feat_quantizer_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probabilitiy for the output of the feature encoder that's used by the quantizer.
        num_negatives (`int`, *optional*, defaults to 100):
            Number of negative samples for the contrastive loss.
        codevector_dim (`int`, *optional*, defaults to 256):
            Dimensionality of the quantized feature vectors.
        proj_codevector_dim (`int`, *optional*, defaults to 256):
            Dimensionality of the final projection of both the quantized and the transformer features.
        diversity_loss_weight (`int`, *optional*, defaults to 0.1):
            The weight of the codebook diversity loss component.
        ctc_loss_reduction (`str`, *optional*, defaults to `"sum"`):
            Specifies the reduction to apply to the output of `torch.nn.CTCLoss`. Only relevant when training an
            instance of [`Wav2Vec2EBranchformerForCTC`].
        ctc_zero_infinity (`bool`, *optional*, defaults to `False`):
            Whether to zero infinite losses and the associated gradients of `torch.nn.CTCLoss`. Infinite losses mainly
            occur when the inputs are too short to be aligned to the targets. Only relevant when training an instance
            of [`Wav2Vec2EBranchformerForCTC`].
        use_weighted_layer_sum (`bool`, *optional*, defaults to `False`):
            Whether to use a weighted average of layer outputs with learned weights. Only relevant when using an
            instance of [`Wav2Vec2EBranchformerForSequenceClassification`].
        classifier_proj_size (`int`, *optional*, defaults to 256):
            Dimensionality of the projection before token mean-pooling for classification.
        tdnn_dim (`Tuple[int]` or `List[int]`, *optional*, defaults to `(512, 512, 512, 512, 1500)`):
            A tuple of integers defining the number of output channels of each 1D convolutional layer in the *TDNN*
            module of the *XVector* model. The length of *tdnn_dim* defines the number of *TDNN* layers.
        tdnn_kernel (`Tuple[int]` or `List[int]`, *optional*, defaults to `(5, 3, 3, 1, 1)`):
            A tuple of integers defining the kernel size of each 1D convolutional layer in the *TDNN* module of the
            *XVector* model. The length of *tdnn_kernel* has to match the length of *tdnn_dim*.
        tdnn_dilation (`Tuple[int]` or `List[int]`, *optional*, defaults to `(1, 2, 3, 1, 1)`):
            A tuple of integers defining the dilation factor of each 1D convolutional layer in *TDNN* module of the
            *XVector* model. The length of *tdnn_dilation* has to match the length of *tdnn_dim*.
        xvector_output_dim (`int`, *optional*, defaults to 512):
            Dimensionality of the *XVector* embedding vectors.
        add_adapter (`bool`, *optional*, defaults to `False`):
            Whether a convolutional network should be stacked on top of the Wav2Vec2EBranchformer Encoder. Can be very
            useful for warm-starting Wav2Vec2EBranchformer for SpeechEncoderDecoder models.
        adapter_kernel_size (`int`, *optional*, defaults to 3):
            Kernel size of the convolutional layers in the adapter network. Only relevant if `add_adapter is True`.
        adapter_stride (`int`, *optional*, defaults to 2):
            Stride of the convolutional layers in the adapter network. Only relevant if `add_adapter is True`.
        num_adapter_layers (`int`, *optional*, defaults to 3):
            Number of convolutional layers that should be used in the adapter network. Only relevant if `add_adapter is
            True`.
        output_hidden_size (`int`, *optional*):
            Dimensionality of the encoder output layer. If not defined, this defaults to *hidden-size*. Only relevant
            if `add_adapter is True`.
        position_embeddings_type (`str`, *optional*, defaults to `"relative"`):
            Can be specified to `relative` or `rotary` for relative or rotary position embeddings respectively. If left
            `None` no relative position embedding is applied.
        rotary_embedding_base (`int`, *optional*, defaults to 10000):
            If `"rotary"` position embeddings are used, defines the size of the embedding base.
        max_source_positions (`int`, *optional*, defaults to 5000):
            if `"relative"` position embeddings are used, defines the maximum source input positions.
        conv_depthwise_kernel_size (`int`, defaults to 31):
            Kernel size of convolutional depthwise 1D layer in EBranchformer blocks.
        Ebranchformer_conv_dropout (`float`, defaults to 0.1):
            The dropout probability for all convolutional layers in EBranchformer blocks.
    """
    model_type = "wav2vec2-ebranchformer"

    def __init__(
        self,
        vocab_size=None,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout=0.1,
        activation_dropout=0.1,
        attention_dropout=0.1,
        feat_proj_dropout=0.0,
        feat_quantizer_dropout=0.0,
        final_dropout=0.1,
        layerdrop=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        feat_extract_norm="group",
        feat_extract_activation="gelu",
        conv_dim=(512, 512, 512, 512, 512, 512, 512),
        conv_stride=(5, 2, 2, 2, 2, 2, 2),
        conv_kernel=(10, 3, 3, 3, 3, 2, 2),
        conv_bias=False,
        num_conv_pos_embeddings=128,
        num_conv_pos_embedding_groups=16,
        apply_spec_augment=True,
        mask_time_prob=0.05,
        mask_time_length=10,
        mask_time_min_masks=2,
        mask_feature_prob=0.0,
        mask_feature_length=10,
        mask_feature_min_masks=0,
        num_codevectors_per_group=320,
        num_codevector_groups=2,
        contrastive_logits_temperature=0.1,
        num_negatives=100,
        codevector_dim=256,
        proj_codevector_dim=256,
        diversity_loss_weight=0.1,
        ctc_loss_reduction="sum",
        ctc_zero_infinity=False,
        use_weighted_layer_sum=False,
        classifier_proj_size=256,
        tdnn_dim=(512, 512, 512, 512, 1500),
        tdnn_kernel=(5, 3, 3, 1, 1),
        tdnn_dilation=(1, 2, 3, 1, 1),
        xvector_output_dim=512,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        add_adapter=False,
        adapter_kernel_size=3,
        adapter_stride=2,
        num_adapter_layers=3,
        output_hidden_size=None,
        position_embeddings_type="relative",
        rotary_embedding_base=10000,
        max_source_positions=5000,
        num_mel_bins=84,
        use_fbanks=False,
        ebranchformer_conv_dropout=0.1,
        csgu_activation="identity",
        csgu_kernel_size=31,
        csgu_use_linear_after_conv=False,
        merge_conv_kernel=31,
        use_macaron_ff=True,
        fe_position_embeddings=True,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )
        self.hidden_size = hidden_size
        self.feat_extract_norm = feat_extract_norm
        self.feat_extract_activation = feat_extract_activation
        self.conv_dim = list(conv_dim)
        self.conv_stride = list(conv_stride)
        self.conv_kernel = list(conv_kernel)
        self.conv_bias = conv_bias
        self.num_conv_pos_embeddings = num_conv_pos_embeddings
        self.num_conv_pos_embedding_groups = num_conv_pos_embedding_groups
        self.num_feat_extract_layers = len(self.conv_dim)
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_attention_heads = num_attention_heads
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.feat_proj_dropout = feat_proj_dropout
        self.final_dropout = final_dropout
        self.layerdrop = layerdrop
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.vocab_size = vocab_size
        self.use_weighted_layer_sum = use_weighted_layer_sum
        self.max_source_positions = max_source_positions
        self.position_embeddings_type = position_embeddings_type
        self.rotary_embedding_base = rotary_embedding_base

        if (
            (len(self.conv_stride) != self.num_feat_extract_layers)
            or (len(self.conv_kernel) != self.num_feat_extract_layers)
            or (len(self.conv_dim) != self.num_feat_extract_layers)
        ):
            raise ValueError(
                "Configuration for convolutional layers is incorrect. It is required that `len(config.conv_dim)` =="
                " `len(config.conv_stride)` == `len(config.conv_kernel)`, but is `len(config.conv_dim) ="
                f" {len(self.conv_dim)}`, `len(config.conv_stride) = {len(self.conv_stride)}`,"
                f" `len(config.conv_kernel) = {len(self.conv_kernel)}`."
            )

        # fine-tuning config parameters for SpecAugment: https://arxiv.org/abs/1904.08779
        self.apply_spec_augment = apply_spec_augment
        self.mask_time_prob = mask_time_prob
        self.mask_time_length = mask_time_length
        self.mask_time_min_masks = mask_time_min_masks
        self.mask_feature_prob = mask_feature_prob
        self.mask_feature_length = mask_feature_length
        self.mask_feature_min_masks = mask_feature_min_masks

        # parameters for pretraining with codevector quantized representations
        self.num_codevectors_per_group = num_codevectors_per_group
        self.num_codevector_groups = num_codevector_groups
        self.contrastive_logits_temperature = contrastive_logits_temperature
        self.feat_quantizer_dropout = feat_quantizer_dropout
        self.num_negatives = num_negatives
        self.codevector_dim = codevector_dim
        self.proj_codevector_dim = proj_codevector_dim
        self.diversity_loss_weight = diversity_loss_weight

        # ctc loss
        self.ctc_loss_reduction = ctc_loss_reduction
        self.ctc_zero_infinity = ctc_zero_infinity

        # adapter
        self.add_adapter = add_adapter
        self.adapter_kernel_size = adapter_kernel_size
        self.adapter_stride = adapter_stride
        self.num_adapter_layers = num_adapter_layers
        self.output_hidden_size = output_hidden_size or hidden_size

        # SequenceClassification-specific parameter. Feel free to ignore for other classes.
        self.classifier_proj_size = classifier_proj_size

        # XVector-specific parameters. Feel free to ignore for other classes.
        self.tdnn_dim = list(tdnn_dim)
        self.tdnn_kernel = list(tdnn_kernel)
        self.tdnn_dilation = list(tdnn_dilation)
        self.xvector_output_dim = xvector_output_dim

        # Custom params
        self.num_mel_bins = num_mel_bins
        self.use_fbanks = use_fbanks

        # EBranchformer related params
        self.csgu_kernel_size = csgu_kernel_size
        self.csgu_activation = csgu_activation
        self.csgu_conv_dropout = ebranchformer_conv_dropout
        self.csgu_use_linear_after_conv = csgu_use_linear_after_conv
        self.merge_conv_kernel = merge_conv_kernel
        self.use_macaron_ff = use_macaron_ff
        self.fe_position_embeddings = fe_position_embeddings

    @property
    def inputs_to_logits_ratio(self):
        return functools.reduce(operator.mul, self.conv_stride, 1)


class ConvolutionalSpatialGatingUnit(torch.nn.Module):
    """Convolutional Spatial Gating Unit (CSGU)."""

    def __init__(self, config):
        super().__init__()

        n_channels = config.intermediate_size // 2  # split input channels
        self.norm = torch.nn.LayerNorm(n_channels)
        self.conv = torch.nn.Conv1d(
            n_channels,
            n_channels,
            config.csgu_kernel_size,
            1,
            (config.csgu_kernel_size - 1) // 2,
            groups=n_channels,
        )
        if config.csgu_use_linear_after_conv:
            self.linear = torch.nn.Linear(n_channels, n_channels)
        else:
            self.linear = None

        if config.csgu_activation == "identity":
            self.act = torch.nn.Identity()
        else:
            self.act = ACT2FN[config.csgu_activation]

        self.dropout = torch.nn.Dropout(config.csgu_conv_dropout)

    def forward(self, hidden_states):
        """Forward method

        Args:
            hidden_states (torch.Tensor): (N, T, D)
            gate_add (torch.Tensor): (N, T, D/2)

        Returns:
            out (torch.Tensor): (N, T, D/2)
        """

        x_r, x_g = hidden_states.chunk(2, dim=-1)

        x_g = self.norm(x_g)  # (N, T, D/2)
        x_g = self.conv(x_g.transpose(1, 2)).transpose(1, 2)  # (N, T, D/2)
        if self.linear is not None:
            x_g = self.linear(x_g)

        x_g = self.act(x_g)
        hidden_states = x_r * x_g  # (N, T, D/2)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class ConvolutionalGatingMLP(torch.nn.Module):
    """Convolutional Gating MLP (cgMLP)."""

    def __init__(self, config):
        super().__init__()
        self.channel_proj1 = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size, config.intermediate_size), torch.nn.GELU()
        )
        self.csgu = ConvolutionalSpatialGatingUnit(config)
        self.channel_proj2 = torch.nn.Linear(config.intermediate_size // 2, config.hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.channel_proj1(hidden_states)  # hidden_size -> intermediate_size
        hidden_states = self.csgu(hidden_states)  # intermediate_size -> intermediate_size/2
        hidden_states = self.channel_proj2(hidden_states)  # intermediate_size/2 -> hidden_size
        return hidden_states


class Wav2Vec2EBranchformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        dropout = config.attention_dropout

        # Feed-forward 1
        if config.use_macaron_ff:
            self.ff1 = nn.Sequential(nn.LayerNorm(embed_dim), Wav2Vec2EBranchformerFeedForward(config))

        # Self-Attention
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.self_attn_dropout = torch.nn.Dropout(dropout)
        self.self_attn = Wav2Vec2EBranchformerSelfAttention(config)

        # cgMLP
        self.cgMLP = ConvolutionalGatingMLP(config)
        self.cgMLP_layer_norm = nn.LayerNorm(config.hidden_size)
        self.cgMLP_dropout = torch.nn.Dropout(dropout)

        # Merge
        self.final_dropout = torch.nn.Dropout(dropout)
        self.merge_proj = torch.nn.Linear(embed_dim + embed_dim, embed_dim)
        self.depthwise_conv_fusion = torch.nn.Conv1d(
            embed_dim + embed_dim,
            embed_dim + embed_dim,
            kernel_size=config.merge_conv_kernel,
            stride=1,
            padding=(config.merge_conv_kernel - 1) // 2,
            groups=embed_dim + embed_dim,
            bias=True,
        )
        self.final_layer_norm = nn.LayerNorm(embed_dim)

        # Feed-forward 2
        if config.use_macaron_ff:
            self.ff2 = nn.Sequential(nn.LayerNorm(embed_dim), Wav2Vec2EBranchformerFeedForward(config))

    def forward(
        self,
        hidden_states,
        attention_mask: Optional[torch.Tensor] = None,
        relative_position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        # 1. Optional ff1
        if self.ff1:
            residual = hidden_states
            hidden_states = residual + 0.5 * self.ff1(hidden_states)

        # 2. Split input to three branches
        residual = hidden_states
        global_branch = hidden_states
        local_branch = hidden_states

        # 3. Self-Attention branch
        global_branch = self.self_attn_layer_norm(global_branch)
        global_branch, attn_weigts = self.self_attn(
            hidden_states=global_branch,
            attention_mask=attention_mask,
            relative_position_embeddings=relative_position_embeddings,
            output_attentions=output_attentions,
        )
        global_branch = self.self_attn_dropout(global_branch)

        # 4. cgMLP Branch
        local_branch = self.cgMLP_layer_norm(local_branch)
        local_branch = self.cgMLP(local_branch)

        # 5. Merge operator
        # a, concat
        hidden_states = torch.cat([global_branch, local_branch], dim=-1)
        merge_residual = hidden_states
        # b, depth-wise conv mixing
        hidden_states = merge_residual + self.depthwise_conv_fusion(hidden_states.transpose(1, 2)).transpose(1, 2)
        # c, project back to original size and final dropout
        hidden_states = self.final_dropout(self.merge_proj(hidden_states))

        # 6. Add residual
        hidden_states = residual + hidden_states

        # 7. Optional ff2
        if self.ff2:
            residual = hidden_states
            hidden_states = residual + 0.5 * self.ff2(hidden_states)

        # 8. Final layer norm
        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states, attn_weigts


class Wav2Vec2EBranchformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if config.position_embeddings_type == "relative":
            self.embed_positions = Wav2Vec2EBranchformerRelPositionalEmbedding(config)
        elif config.position_embeddings_type == "rotary":
            self.embed_positions = Wav2Vec2EBranchformerRotaryPositionalEmbedding(config)
        else:
            self.embed_positions = None

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList(
            [Wav2Vec2EBranchformerEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            # make sure padded tokens output 0
            hidden_states[~attention_mask] = 0.0

            # extend attention_mask
            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )

        hidden_states = self.dropout(hidden_states)

        if self.embed_positions is not None:
            relative_position_embeddings = self.dropout(self.embed_positions(hidden_states))
        else:
            relative_position_embeddings = None

        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()

        for _, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = np.random.uniform(0, 1)

            skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                # under deepspeed zero3 all gpus must run in sync
                if self.gradient_checkpointing and self.training:
                    # create gradient checkpointing function
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer),
                        hidden_states,
                        attention_mask,
                        relative_position_embeddings,
                    )
                else:
                    layer_outputs = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        relative_position_embeddings=relative_position_embeddings,
                        output_attentions=output_attentions,
                    )
                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class Wav2Vec2EBranchformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Wav2Vec2EBranchformerConfig
    base_model_prefix = "wav2vec2_ebranchformer"
    main_input_name = "input_values"
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        # gumbel softmax requires special init
        if isinstance(module, ConvolutionalSpatialGatingUnit):
            torch.nn.init.normal_(module.conv.weight, std=1e-6)
            torch.nn.init.ones_(module.conv.bias)
            if module.linear is not None:
                torch.nn.init.normal_(module.linear.weight, std=1e-6)
                torch.nn.init.ones_(module.linear.bias)
        if isinstance(module, Wav2Vec2EBranchformerSelfAttention):
            if hasattr(module, "pos_bias_u"):
                nn.init.xavier_uniform_(module.pos_bias_u)
            if hasattr(module, "pos_bias_v"):
                nn.init.xavier_uniform_(module.pos_bias_v)
        # elif isinstance(module, Wav2Vec2EBranchformerPositionalConvEmbedding):
        #     nn.init.normal_(
        #         module.conv.weight,
        #         mean=0,
        #         std=2 * math.sqrt(1 / (module.conv.kernel_size[0] * module.conv.in_channels)),
        #     )
        #     nn.init.constant_(module.conv.bias, 0)
        elif isinstance(module, Wav2Vec2EBranchformerFeatureProjection):
            k = math.sqrt(1 / module.projection.in_features)
            nn.init.uniform_(module.projection.weight, a=-k, b=k)
            nn.init.uniform_(module.projection.bias, a=-k, b=k)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)

            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)

    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[torch.LongTensor, int], add_adapter: Optional[bool] = None
    ):
        """
        Computes the output length of the convolutional layers
        """

        add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        if add_adapter:
            for _ in range(self.config.num_adapter_layers):
                input_lengths = _conv_out_length(input_lengths, 1, self.config.adapter_stride)

        return input_lengths

    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: torch.LongTensor, add_adapter=None
    ):
        # Effectively attention_mask.sum(-1), but not inplace to be able to run
        # on inference mode.
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]

        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths, add_adapter=add_adapter)
        output_lengths = output_lengths.to(torch.long)

        batch_size = attention_mask.shape[0]

        attention_mask = torch.zeros(
            (batch_size, feature_vector_length),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        # these two operations makes sure that all values before the output lengths idxs are attended to
        attention_mask[
            (
                torch.arange(attention_mask.shape[0], device=attention_mask.device),
                output_lengths - 1,
            )
        ] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (Wav2Vec2EBranchformerEncoder, MelFeatureExtractor)) or isinstance(
            module, (Wav2Vec2EBranchformerEncoder, Wav2Vec2EBranchformerFeatureEncoder)
        ):
            module.gradient_checkpointing = value


class Wav2Vec2EBranchformerModel(Wav2Vec2EBranchformerPreTrainedModel):
    def __init__(self, config: Wav2Vec2EBranchformerConfig):
        super().__init__(config)
        self.config = config
        if config.use_fbanks:
            self.feature_extractor = MelFeatureExtractor(config)
        else:
            self.feature_extractor = Wav2Vec2EBranchformerFeatureEncoder(config)
        self.feature_projection = Wav2Vec2EBranchformerFeatureProjection(config)

        # model only needs masking vector if mask prob is > 0.0
        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())

        self.encoder = Wav2Vec2EBranchformerEncoder(config)

        self.adapter = Wav2Vec2EBranchformerAdapter(config) if config.add_adapter else None

        if config.apply_spec_augment:
            # TODO: Rewrite this
            from espnet2.asr.specaug.specaug import SpecAug

            self.spec_aug = SpecAug(
                apply_time_warp=True,
                time_warp_window=5,
                time_warp_mode="bicubic",
                apply_freq_mask=True,
                freq_mask_width_range=(0, 27),
                num_freq_mask=2,
                apply_time_mask=True,
                time_mask_width_ratio_range=(0, 0.05),
                num_time_mask=5,
            )
        # Initialize weights and apply final processing
        self.post_init()

    def _mask_hidden_states(
        self,
        hidden_states: torch.FloatTensor,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """

        # `config.apply_spec_augment` can set masking to False
        if not getattr(self.config, "apply_spec_augment", True):
            return hidden_states

        if self.training:
            # TODO: Rewrite this
            hidden_states, _ = self.spec_aug(hidden_states, attention_mask.sum(-1).long())

        # # generate indices & apply SpecAugment along time axis
        # batch_size, sequence_length, hidden_size = hidden_states.size()
        #
        # if mask_time_indices is not None:
        #     # apply SpecAugment along time axis with given mask_time_indices
        #     hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
        # elif self.config.mask_time_prob > 0 and self.training:
        #     mask_time_indices = _compute_mask_indices(
        #         (batch_size, sequence_length),
        #         mask_prob=self.config.mask_time_prob,
        #         mask_length=self.config.mask_time_length,
        #         attention_mask=attention_mask,
        #         min_masks=self.config.mask_time_min_masks,
        #     )
        #     mask_time_indices = torch.tensor(mask_time_indices, device=hidden_states.device, dtype=torch.bool)
        #     hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
        #
        # if self.config.mask_feature_prob > 0 and self.training:
        #     # generate indices & apply SpecAugment along feature axis
        #     mask_feature_indices = _compute_mask_indices(
        #         (batch_size, hidden_size),
        #         mask_prob=self.config.mask_feature_prob,
        #         mask_length=self.config.mask_feature_length,
        #         min_masks=self.config.mask_feature_min_masks,
        #     )
        #     mask_feature_indices = torch.tensor(mask_feature_indices, device=hidden_states.device, dtype=torch.bool)
        #     mask_feature_indices = mask_feature_indices[:, None].expand(-1, sequence_length, -1)
        #     hidden_states[mask_feature_indices] = 0

        return hidden_states

    # Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Model.freeze_feature_encoder
    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.feature_extractor._freeze_parameters()

    # Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Model.forward with wav2vec2->wav2vec2_Ebranchformer
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Wav2Vec2BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]

        return Wav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class Wav2Vec2EBranchformerForCTC(Wav2Vec2EBranchformerPreTrainedModel):
    # Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForCTC.__init__ with Wav2Vec2->Wav2Vec2EBranchformer,wav2vec2->wav2vec2_Ebranchformer
    def __init__(self, config):
        super().__init__(config)

        self.wav2vec2_Ebranchformer = Wav2Vec2EBranchformerModel(config)
        self.dropout = nn.Dropout(config.final_dropout)

        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that does not define the"
                " vocabulary size of the language model head. Please instantiate the model as follows:"
                " `Wav2Vec2EBranchformerForCTC.from_pretrained(..., vocab_size=vocab_size)`. or define `vocab_size` of"
                " your model's configuration."
            )
        output_hidden_size = (
            config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
        )
        self.lm_head = nn.Linear(output_hidden_size, config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForCTC.freeze_feature_encoder with wav2vec2->wav2vec2_Ebranchformer
    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wav2vec2_Ebranchformer.feature_extractor._freeze_parameters()

    # Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForCTC.forward with Wav2Vec2->Wav2Vec2EBranchformer,wav2vec2->wav2vec2_Ebranchformer
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.wav2vec2_Ebranchformer(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:

            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
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
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
