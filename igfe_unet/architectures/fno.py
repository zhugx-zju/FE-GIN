"""Minimal 2-D Fourier Neural Operator for the fixed FE grid."""

import torch
from torch import nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """Fourier convolution using the positive and negative x-frequency bands."""

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        scale = 1.0 / (self.in_channels * self.out_channels)
        shape = (self.in_channels, self.out_channels, self.modes1, self.modes2)
        self.weights1 = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))

    @staticmethod
    def _complex_multiply(inputs, weights):
        return torch.einsum("bixy,ioxy->boxy", inputs, weights)

    def forward(self, x):
        height, width = x.shape[-2:]
        max_modes1 = height // 2
        max_modes2 = width // 2 + 1
        modes1 = min(self.modes1, max_modes1)
        modes2 = min(self.modes2, max_modes2)
        if modes1 < 1 or modes2 < 1:
            raise ValueError(f"Grid {height}x{width} is too small for Fourier modes")

        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(
            x.shape[0],
            self.out_channels,
            height,
            width // 2 + 1,
            dtype=x_ft.dtype,
            device=x.device,
        )
        out_ft[:, :, :modes1, :modes2] = self._complex_multiply(
            x_ft[:, :, :modes1, :modes2], self.weights1[:, :, :modes1, :modes2]
        )
        out_ft[:, :, -modes1:, :modes2] = self._complex_multiply(
            x_ft[:, :, -modes1:, :modes2], self.weights2[:, :, :modes1, :modes2]
        )
        return torch.fft.irfft2(out_ft, s=(height, width), norm="ortho")


class FourierBlock(nn.Module):
    def __init__(self, width, modes1, modes2):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.pointwise = nn.Conv2d(width, width, kernel_size=1)
        self.norm = nn.GroupNorm(1, width)

    def forward(self, x):
        x = self.spectral(x) + self.pointwise(x)
        return F.gelu(self.norm(x))


class FNO2d(nn.Module):
    """Map displacement channels [B, 2, H, W] to modulus [B, H, W]."""

    def __init__(
        self,
        input_channels=2,
        output_channels=1,
        width=24,
        modes1=8,
        modes2=8,
        n_layers=4,
        use_coordinates=True,
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be positive")
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.width = int(width)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        self.n_layers = int(n_layers)
        self.use_coordinates = bool(use_coordinates)
        lifted_channels = self.input_channels + (2 if self.use_coordinates else 0)
        self.lifting = nn.Conv2d(lifted_channels, self.width, kernel_size=1)
        self.layers = nn.ModuleList(
            FourierBlock(self.width, self.modes1, self.modes2)
            for _ in range(self.n_layers)
        )
        self.projection = nn.Sequential(
            nn.Conv2d(self.width, self.width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(self.width, self.output_channels, kernel_size=1),
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
        x = self.lifting(x)
        for layer in self.layers:
            x = layer(x)
        output = self.projection(x)
        if self.output_channels == 1:
            return output[:, 0]
        return output


__all__ = ["SpectralConv2d", "FourierBlock", "FNO2d"]
