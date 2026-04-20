from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .fewshot_backbones import CosineDistLinear


class MetaTemplate(nn.Module):
    def __init__(self, model_func, n_way: int, k_shot: int, n_query: int, change_way: bool = True):
        super().__init__()
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.change_way = change_way
        self.feature = model_func()
        self.feat_dim = self.feature.final_feat_dim

    def set_forward(self, x: torch.Tensor, is_feature: bool = False) -> torch.Tensor:
        pass

    def set_forward_loss(self, x: torch.Tensor) -> tuple[float, torch.Tensor]:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature(x)

    @property
    def episode_size(self) -> int:
        return self.k_shot + self.n_query

    def episode_targets(self, device: torch.device) -> torch.Tensor:
        return torch.arange(self.n_way, device=device).repeat_interleave(self.n_query)

    def encode_episode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous().view(self.n_way * self.episode_size, *x.shape[2:])
        x = self.feature(x)
        return x.view(self.n_way, self.episode_size, *x.shape[1:])

    def split_episode(self, x: torch.Tensor, is_feature: bool = False):
        features = x if is_feature else self.encode_episode(x)
        return features[:, : self.k_shot], features[:, self.k_shot :]

    def correct(self, x: torch.Tensor) -> tuple[float, int]:
        scores = self.set_forward(x)
        targets = self.episode_targets(scores.device)
        predictions = scores.argmax(dim=1)
        correct = (predictions == targets).sum().item()
        return float(correct), int(targets.numel())


def cosine_distance(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return torch.matmul(F.normalize(x1, dim=-1), F.normalize(x2, dim=-2))


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, variant: str):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.variant = variant
        self.softmax = nn.Softmax(dim=-1)
        self.input_linear = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, inner_dim, bias=False))
        self.output_linear = nn.Linear(inner_dim, dim) if not (heads == 1 and dim_head == dim) else nn.Identity()

    def _project(self, tensor: torch.Tensor) -> torch.Tensor:
        projected = self.input_linear(tensor)
        q_dim, n_dim, _ = projected.shape
        projected = projected.view(q_dim, n_dim, self.heads, self.dim_head)
        return projected.permute(2, 0, 1, 3)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        f_q, f_k, f_v = map(self._project, (q, k, v))
        if self.variant == "cosine":
            dots = cosine_distance(f_q, f_k.transpose(-1, -2))
            out = torch.matmul(dots, f_v)
        else:
            dots = torch.matmul(f_q, f_k.transpose(-1, -2)) * self.scale
            out = torch.matmul(self.softmax(dots), f_v)

        out = out.permute(1, 2, 0, 3).contiguous()
        out = out.view(out.shape[0], out.shape[1], -1)
        return self.output_linear(out)


class FewShotTransformer(MetaTemplate):
    def __init__(
        self,
        model_func,
        n_way: int,
        k_shot: int,
        n_query: int,
        variant: str = "softmax",
        depth: int = 1,
        heads: int = 8,
        dim_head: int = 64,
        mlp_dim: int = 512,
    ):
        super().__init__(model_func, n_way, k_shot, n_query)
        self.loss_fn = nn.CrossEntropyLoss()
        self.depth = depth
        self.variant = variant
        self.attention = Attention(self.feat_dim, heads=heads, dim_head=dim_head, variant=variant)
        self.prototype_softmax = nn.Softmax(dim=-2)
        self.prototype_weight = nn.Parameter(torch.ones(n_way, k_shot, 1))
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, self.feat_dim),
        )
        final_linear: nn.Module
        if variant == "cosine":
            final_linear = CosineDistLinear(dim_head, 1)
        else:
            final_linear = nn.Linear(dim_head, 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, dim_head),
            final_linear,
        )

    def set_forward(self, x: torch.Tensor, is_feature: bool = False) -> torch.Tensor:
        z_support, z_query = self.split_episode(x, is_feature=is_feature)
        z_support = z_support.contiguous().view(self.n_way, self.k_shot, -1)
        z_proto = (z_support * self.prototype_softmax(self.prototype_weight)).sum(dim=1).unsqueeze(0)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1).unsqueeze(1)

        state = z_proto
        for _ in range(self.depth):
            state = self.attention(q=state, k=z_query, v=z_query) + state
            state = self.feed_forward(state) + state

        return self.classifier(state).squeeze(-1)

    def set_forward_loss(self, x: torch.Tensor) -> tuple[float, torch.Tensor]:
        scores = self.set_forward(x)
        targets = self.episode_targets(scores.device)
        loss = self.loss_fn(scores, targets)
        predictions = scores.argmax(dim=1)
        accuracy = (predictions == targets).float().mean().item()
        return accuracy, loss


class CTX(MetaTemplate):
    def __init__(
        self,
        model_func,
        n_way: int,
        k_shot: int,
        n_query: int,
        variant: str = "softmax",
        input_dim: int = 64,
        dim_attn: int = 128,
    ):
        super().__init__(model_func, n_way, k_shot, n_query)
        self.loss_fn = nn.CrossEntropyLoss()
        self.variant = variant
        self.dim_attn = dim_attn
        self.softmax = nn.Softmax(dim=-1)
        self.linear_attn = nn.Conv2d(input_dim, dim_attn, kernel_size=1, bias=False)

    def set_forward(self, x: torch.Tensor, is_feature: bool = False) -> torch.Tensor:
        z_support, z_query = self.split_episode(x, is_feature=is_feature)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, *z_query.shape[2:])
        z_support = z_support.contiguous().view(self.n_way * self.k_shot, *z_support.shape[2:])

        query_q = self.linear_attn(z_query).flatten(1).unsqueeze(1)
        query_v = self.linear_attn(z_query).flatten(1).unsqueeze(1)
        support_k = self.linear_attn(z_support).flatten(1).view(self.n_way, self.k_shot, -1).transpose(1, 2)
        support_v = self.linear_attn(z_support).flatten(1).view(self.n_way, self.k_shot, -1).transpose(1, 2)
        query_q = query_q.permute(1, 0, 2)

        if self.variant == "softmax":
            dots = torch.matmul(query_q, support_k) / (self.dim_attn ** 0.5)
            attn_weights = self.softmax(dots)
        else:
            dots = torch.matmul(query_q, support_k)
            query_norm = torch.norm(query_q, p=2, dim=-1, keepdim=True)
            support_norm = torch.norm(support_k, p=2, dim=-2, keepdim=True)
            attn_weights = dots / (query_norm * support_norm + 1e-8)

        attn_weights = attn_weights.squeeze(0)
        out = torch.einsum("nqk,ndk->qnd", attn_weights, support_v)
        query_v = query_v.squeeze(1).unsqueeze(1)
        scores = -((query_v - out) ** 2).sum(dim=-1) / (self.feat_dim[1] * self.feat_dim[2])
        return scores

    def set_forward_loss(self, x: torch.Tensor) -> tuple[float, torch.Tensor]:
        scores = self.set_forward(x)
        targets = self.episode_targets(scores.device)
        loss = self.loss_fn(scores, targets)
        predictions = scores.argmax(dim=1)
        accuracy = (predictions == targets).float().mean().item()
        return accuracy, loss
