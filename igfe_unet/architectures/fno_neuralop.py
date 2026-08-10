"""NeuralOperator-backed FNO wrapper with the repository FNO interface."""

import torch
from torch import nn


class NeuralOperatorFNO2d(nn.Module):
    """Map [B, 2, H, W] to [B, H, W] using ``neuralop.models.FNO``."""

    def __init__(
        self,
        input_channels=2,
        output_channels=1,
        width=21,
        modes1=8,
        modes2=8,
        n_layers=4,
        use_coordinates=True,
    ):
        super().__init__()
        try:
            from neuralop.models import FNO
        except ImportError as error:
            raise ImportError(
                "FNO-neuraloperator requires the optional 'neuraloperator' package. "
                "Install it in the active environment with: "
                "python -m pip install neuraloperator"
            ) from error

        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.use_coordinates = bool(use_coordinates)
        operator_input_channels = self.input_channels + (2 if self.use_coordinates else 0)
        self.operator = FNO(
            n_modes=(int(modes1), int(modes2)),
            hidden_channels=int(width),
            in_channels=operator_input_channels,
            out_channels=self.output_channels,
            n_layers=int(n_layers),
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
            raise ValueError(
                f"Expected [B, {self.input_channels}, H, W], got {tuple(x.shape)}"
            )
        if self.use_coordinates:
            x = torch.cat((x, self._coordinate_grid(x)), dim=1)
        output = self.operator(x)
        if isinstance(output, dict):
            output = output["predictions"]
        if output.ndim != 4:
            raise ValueError(f"Expected neuraloperator output [B, C, H, W], got {tuple(output.shape)}")
        if self.output_channels == 1:
            return output[:, 0]
        return output


def build_neuralop_fno_model(cfg):
    return NeuralOperatorFNO2d(
        input_channels=cfg.input_channels,
        output_channels=cfg.output_channels,
        width=cfg.width,
        modes1=cfg.modes1,
        modes2=cfg.modes2,
        n_layers=cfg.n_layers,
        use_coordinates=cfg.use_coordinates,
    )


__all__ = ["NeuralOperatorFNO2d", "build_neuralop_fno_model"]
