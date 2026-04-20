from __future__ import annotations

import torch
from torch import nn


class PreNormResidual(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fn(self.norm(x))


def feed_forward(dim: int, hidden_dim: int, dropout: float = 0.0) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, dim),
        nn.Dropout(dropout),
    )


class MixerBlock(nn.Module):
    def __init__(
        self,
        num_patches: int,
        dim: int,
        tokens_mlp_dim: int,
        channels_mlp_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.token_norm = nn.LayerNorm(dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(num_patches, tokens_mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(tokens_mlp_dim, num_patches),
            nn.Dropout(dropout),
        )
        self.channel_mlp = PreNormResidual(dim, feed_forward(dim, channels_mlp_dim, dropout=dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.token_norm(x).transpose(1, 2)
        y = self.token_mlp(y).transpose(1, 2)
        x = x + y
        return self.channel_mlp(x)


class MLPMixer(nn.Module):
    def __init__(
        self,
        image_size: int,
        channels: int,
        patch_size: int,
        dim: int,
        depth: int,
        num_classes: int,
        token_expansion: float = 2.0,
        channel_expansion: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        tokens_mlp_dim = int(num_patches * token_expansion)
        channels_mlp_dim = int(dim * channel_expansion)

        self.patch_embedding = nn.Conv2d(
            channels,
            dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.mixer_blocks = nn.ModuleList(
            [
                MixerBlock(
                    num_patches=num_patches,
                    dim=dim,
                    tokens_mlp_dim=tokens_mlp_dim,
                    channels_mlp_dim=channels_mlp_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.layer_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(x).flatten(2).transpose(1, 2)
        for block in self.mixer_blocks:
            x = block(x)
        x = self.layer_norm(x)
        x = x.mean(dim=1)
        return self.head(x)
