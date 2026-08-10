"""Optional neuraloperator backend with the repository FNO interface."""

import torch
from torch import nn


class NeuralOperatorFNO2d(nn.Module):
    """Map [B, 2, H, W] to [B, H, W] using ``neuralop.models.FNO``."""

    def __init__(self, cfg):
        super().__init__()
        try:
            from neuralop.models import FNO
        except ImportError as error:
            raise ImportError(
                "FNO-neuraloperator requires the optional 'neuraloperator' package. "
                "Install it with: python -m pip install -r requirements_fno_neuralop.txt"
            ) from error

        self.input_channels = int(cfg.input_channels)
        self.output_channels = int(cfg.output_channels)
        self.use_coordinates = bool(cfg.use_coordinates)
        in_channels = self.input_channels + (2 if self.use_coordinates else 0)
        self.operator = FNO(
            n_modes=(int(cfg.modes1), int(cfg.modes2)),
            hidden_channels=int(cfg.width),
            in_channels=in_channels,
            out_channels=self.output_channels,
            n_layers=int(cfg.n_layers),
        )

    @staticmethod
    def _coordinate_grid(x):
        height, width = x.shape[-2:]
        grid_y = torch.linspace(0.0, 1.0, height, device=x.device, dtype=x.dtype)
        grid_x = torch.linspace(0.0, 1.0, width, device=x.device, dtype=x.dtype)
        yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")
        return torch.stack((xx, yy), dim=0).unsqueeze(0).expand(x.shape[0], -1, -1, -1)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.input_channels:
            raise ValueError(f"Expected [B, {self.input_channels}, H, W], got {tuple(x.shape)}")
        if self.use_coordinates:
            x = torch.cat((x, self._coordinate_grid(x)), dim=1)
        output = self.operator(x)
        if isinstance(output, dict):
            output = output["predictions"]
        if output.ndim != 4:
            raise ValueError(f"Expected neuraloperator output [B, C, H, W], got {tuple(output.shape)}")
        return output[:, 0] if self.output_channels == 1 else output


def build_neuralop_fno_model(cfg):
    return NeuralOperatorFNO2d(cfg)


__all__ = ["NeuralOperatorFNO2d", "build_neuralop_fno_model"]
