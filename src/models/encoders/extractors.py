import torch
from torch import nn
from transformers.activations import ACT2FN


class MelFeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = torch.nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(
                        conv_in,
                        out_channels=conv_out,
                        kernel_size=(conv_kernel, conv_kernel),
                        stride=(conv_stride, conv_stride),
                    ),
                    ACT2FN[config.feat_extract_activation],
                )
                for conv_in, conv_out, conv_kernel, conv_stride in zip(
                    [1, *config.conv_dim], config.conv_dim, config.conv_kernel, config.conv_stride
                )
            ],
        )

        linear_in_dim = config.conv_dim[-1] * (((config.num_mel_bins - 1) // 2 - 1) // 2)
        self.out = torch.nn.Linear(linear_in_dim, config.hidden_size, bias=True)

    def forward(self, input_values):
        hidden_states = self.conv(input_values[:, None, ...])
        hidden_states = self.out(hidden_states.transpose(1, 2).flatten(2, 3))
        return hidden_states.transpose(1, 2)
