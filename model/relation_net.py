from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 64, use_maxpool: bool = True):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if use_maxpool:
            layers.append(nn.MaxPool2d(kernel_size=2))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class EmbeddingModule(nn.Module):
    def __init__(self, in_channels: int = 3, hidden_channels: int = 64):
        super().__init__()
        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hidden_channels, use_maxpool=True),
            ConvBlock(hidden_channels, hidden_channels, use_maxpool=True),
            ConvBlock(hidden_channels, hidden_channels, use_maxpool=False),
            ConvBlock(hidden_channels, hidden_channels, use_maxpool=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class RelationModule(nn.Module):
    def __init__(self, input_channels: int, spatial_size: int, hidden_dim: int = 8):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, spatial_size, spatial_size)
            features = self.conv_block2(self.conv_block1(dummy))
            flattened_dim = features.flatten(1).shape[1]

        self.fc_layers = nn.Sequential(
            nn.Linear(flattened_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.flatten(1)
        return self.fc_layers(x)


class RelationNet(nn.Module):
    def __init__(
        self,
        n_way: int,
        k_shot: int,
        image_size: int = 84,
        in_channels: int = 3,
        hidden_channels: int = 64,
    ):
        super().__init__()
        self.n_way = n_way
        self.k_shot = k_shot
        self.embedding = EmbeddingModule(in_channels=in_channels, hidden_channels=hidden_channels)

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, image_size, image_size)
            feat = self.embedding(dummy)
        self.feature_channels = feat.shape[1]
        self.feature_size = feat.shape[2]
        self.relation_module = RelationModule(2 * self.feature_channels, self.feature_size)

    def compute_scores(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, support_set_size = support_x.shape[:2]
        query_set_size = query_x.shape[1]
        channels, height, width = support_x.shape[2:]

        support_features = self.embedding(support_x.view(-1, channels, height, width))
        support_features = support_features.view(
            batch_size,
            support_set_size,
            self.feature_channels,
            self.feature_size,
            self.feature_size,
        )

        query_features = self.embedding(query_x.view(-1, channels, height, width))
        query_features = query_features.view(
            batch_size,
            query_set_size,
            self.feature_channels,
            self.feature_size,
            self.feature_size,
        )

        class_features = []
        for batch_index in range(batch_size):
            batch_class_features = []
            for class_index in range(self.n_way):
                class_mask = support_y[batch_index] == class_index
                class_feature = support_features[batch_index, class_mask].sum(dim=0)
                batch_class_features.append(class_feature)
            class_features.append(torch.stack(batch_class_features, dim=0))
        class_features = torch.stack(class_features, dim=0)

        class_features = class_features.unsqueeze(1).expand(-1, query_set_size, -1, -1, -1, -1)
        query_features = query_features.unsqueeze(2).expand(-1, -1, self.n_way, -1, -1, -1)
        relations = torch.cat([query_features, class_features], dim=3)

        scores = self.relation_module(
            relations.contiguous().view(
                -1,
                2 * self.feature_channels,
                self.feature_size,
                self.feature_size,
            )
        )
        return scores.view(batch_size, query_set_size, self.n_way)

    def episode_loss(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.compute_scores(support_x, support_y, query_x)
        targets = torch.nn.functional.one_hot(query_y, num_classes=self.n_way).float()
        loss = ((targets - scores) ** 2).sum() / support_x.shape[0]
        return loss, scores

    def predict(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        scores = self.compute_scores(support_x, support_y, query_x)
        predictions = torch.argmax(scores, dim=-1)
        correct = None
        if query_y is not None:
            correct = (predictions == query_y).sum()
        return predictions, correct
