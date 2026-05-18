import os
import sys

import torch
import torch.nn.functional as F
from torch.nn import Conv2d, GroupNorm, Identity, MaxPool2d, Module, Upsample, init

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

torch.manual_seed(2)


class UNet(Module):
    def __init__(self, nfilters, kernel_size, use_batch_norm=False, layer_type=Conv2d):
        super(UNet, self).__init__()
        self.nfilters = nfilters
        self.kernel_size = kernel_size
        self.use_batch_norm = use_batch_norm
        self.layer_type = layer_type

        for i in range(1, len(self.nfilters)):
            in_channels = self.nfilters[i - 1]
            out_channels = self.nfilters[i]

            if i != 1:
                setattr(self, f"encoder_maxpool_{i}", MaxPool2d(kernel_size=2, stride=2))

            setattr(
                self,
                f"encoder_conv_1_{i}",
                self.layer_type(
                    in_channels,
                    out_channels,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
                ),
            )
            conv_layer = getattr(self, f"encoder_conv_1_{i}")
            init.kaiming_normal_(conv_layer.weight, nonlinearity='relu')
            init.constant_(conv_layer.bias, 0)
            setattr(
                self,
                f"encoder_bn_1_{i}",
                self._build_norm_layer(out_channels),
            )

            setattr(
                self,
                f"encoder_conv_2_{i}",
                self.layer_type(
                    out_channels,
                    out_channels,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
                ),
            )
            conv_layer = getattr(self, f"encoder_conv_2_{i}")
            init.kaiming_normal_(conv_layer.weight, nonlinearity='relu')
            init.constant_(conv_layer.bias, 0)
            setattr(
                self,
                f"encoder_bn_2_{i}",
                self._build_norm_layer(out_channels),
            )

        for i in range(len(self.nfilters) - 1, 1, -1):
            combined_channels = self.nfilters[i] + self.nfilters[i - 1]
            out_channels = self.nfilters[i - 1]

            setattr(
                self,
                f"decoder_upsample_{i}",
                Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            )
            setattr(
                self,
                f"decoder_conv_1_{i}",
                self.layer_type(
                    combined_channels,
                    out_channels,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
                ),
            )
            conv_layer = getattr(self, f"decoder_conv_1_{i}")
            init.kaiming_normal_(conv_layer.weight, nonlinearity='relu')
            init.constant_(conv_layer.bias, 0)
            setattr(
                self,
                f"decoder_bn_1_{i}",
                self._build_norm_layer(out_channels),
            )

            setattr(
                self,
                f"decoder_conv_2_{i}",
                self.layer_type(
                    out_channels,
                    out_channels,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
                ),
            )
            conv_layer = getattr(self, f"decoder_conv_2_{i}")
            init.kaiming_normal_(conv_layer.weight, nonlinearity='relu')
            init.constant_(conv_layer.bias, 0)
            setattr(
                self,
                f"decoder_bn_2_{i}",
                self._build_norm_layer(out_channels),
            )

        self.final_conv = Conv2d(self.nfilters[1], 1, kernel_size=1, padding=0)
        init.kaiming_normal_(self.final_conv.weight, nonlinearity='linear')
        init.constant_(self.final_conv.bias, 0)

    def _build_norm_layer(self, num_channels):
        if not self.use_batch_norm:
            return Identity()

        num_groups = min(8, num_channels)
        while num_channels % num_groups != 0:
            num_groups -= 1

        return GroupNorm(num_groups=num_groups, num_channels=num_channels)

    def _match_spatial_size(self, x, ref_tensor):
        if x.shape[-2:] != ref_tensor.shape[-2:]:
            x = F.interpolate(
                x,
                size=ref_tensor.shape[-2:],
                mode='bilinear',
                align_corners=True,
            )
        return x

    def forward(self, x):
        encoder_outputs = []
        for i in range(1, len(self.nfilters)):
            x = self.optional_step_en(x, i)
            if i != 1:
                x = getattr(self, f"encoder_maxpool_{i}")(x)
            x = getattr(self, f"encoder_conv_1_{i}")(x)
            x = getattr(self, f"encoder_bn_1_{i}")(x)
            x = F.relu(x)
            x = getattr(self, f"encoder_conv_2_{i}")(x)
            x = getattr(self, f"encoder_bn_2_{i}")(x)
            x = F.relu(x)
            encoder_outputs.append(x)

        for idx, i in enumerate(range(len(self.nfilters) - 1, 1, -1)):
            x = getattr(self, f"decoder_upsample_{i}")(x)
            skip_input = encoder_outputs[-(idx + 2)]
            x = self._match_spatial_size(x, skip_input)
            x = torch.cat([x, skip_input], dim=1)
            x = self.optional_step_dec(x, i)
            x = getattr(self, f"decoder_conv_1_{i}")(x)
            x = getattr(self, f"decoder_bn_1_{i}")(x)
            x = F.relu(x)
            x = getattr(self, f"decoder_conv_2_{i}")(x)
            x = getattr(self, f"decoder_bn_2_{i}")(x)
            x = F.relu(x)

        x = self.final_conv(x)
        return x

    def optional_step_en(self, x, i):
        return x

    def optional_step_dec(self, x, i):
        return x
