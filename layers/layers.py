from typing import Union, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from skfp.fingerprints import ECFPFingerprint


# Usage is mostly the same as the keras version, except now you must pass key_padding_mask from
# Embeddings.forward() to EncoderLayer.forward().


class FPGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = ECFPFingerprint(include_chirality=True, n_jobs=-1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Get fingerprints given SMILES string.
        """
        # TODO: better to precompute fingerprints offline?
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.detach().cpu().numpy().astype(str)

        x = self.transformer.transform(inputs)
        x = torch.tensor(x, dtype=torch.long)
        return x


class Encodings(nn.Module):
    def __init__(self, depth: int, max_length: int):
        super().__init__()
        self.depth = depth
        self.register_buffer("encodings", self._pos_encodings(depth, max_length))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Scale
        scale = torch.sqrt(torch.tensor(self.depth, dtype=torch.float32))
        x = inputs * scale
        # Add encodings
        x = x + self.encodings[: inputs.size(1), :].unsqueeze(0)
        return x

    @staticmethod
    def _pos_encodings(depth: int, max_length: int) -> torch.Tensor:
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        idx = torch.arange(depth).unsqueeze(0)
        power = (2 * (idx // 2)).float() / depth
        angles = 1.0 / (10000.0**power)
        radians = positions * angles

        sin = torch.sin(radians[:, 0::2])
        cos = torch.cos(radians[:, 1::2])
        encodings = torch.cat([sin, cos], dim=-1)
        return encodings


class Embeddings(nn.Module):
    def __init__(self, max_length: int, depth: int, input_dim: int):
        super().__init__()
        self.depth = depth
        self.input_dim = input_dim
        self.embeddings = nn.Embedding(
            num_embeddings=input_dim, embedding_dim=depth, padding_idx=0
        )
        self.encodings = Encodings(depth=depth, max_length=max_length)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        key_padding_mask = inputs == 0

        x = self.embeddings(inputs)
        x = self.encodings(x)
        return x, key_padding_mask


class FeedForward(nn.Module):
    def __init__(
        self,
        activation: Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]],
        depth: int,
        dropout_rate: float,
        epsilon: float,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(depth, eps=epsilon)
        self.dense1 = nn.Linear(depth, depth * 2)
        self.dense2 = nn.Linear(depth * 2, depth)
        self.dropout = nn.Dropout(dropout_rate)

        if isinstance(activation, nn.Module):
            self.activation_fn = activation
        elif callable(activation):
            self.activation_fn = activation
        else:
            raise ValueError("activation must be callable or nn.Module")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.norm(inputs)
        x = self.dense1(x)
        x = self.activation_fn(x)
        x = self.dense2(x)
        x = self.dropout(x)
        return x + inputs  # residual


class SelfAttention(nn.Module):
    def __init__(
        self,
        causal: bool,
        depth: int,
        dropout_rate: float,
        epsilon: float,
        num_heads: int,
    ):
        super().__init__()
        self.causal = causal
        self.norm = nn.LayerNorm(depth, eps=epsilon)
        self.mha = nn.MultiheadAttention(
            embed_dim=depth, num_heads=num_heads, dropout=dropout_rate, batch_first=True
        )

    def forward(
        self,
        inputs: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # inputs: [B, L, D]
        x = self.norm(inputs)

        attn_mask = None
        if self.causal:
            L = inputs.size(1)
            attn_mask = torch.triu(torch.ones(L, L, device=inputs.device), diagonal=1).bool()

        out, _ = self.mha(
            x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )
        return out + inputs  # residual


class EncoderLayer(nn.Module):
    def __init__(
        self,
        activation: Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]],
        depth: int,
        dropout_rate: float,
        epsilon: float,
        num_heads: int,
    ):
        super().__init__()
        self.self_attention = SelfAttention(
            causal=False,
            depth=depth,
            dropout_rate=dropout_rate,
            epsilon=epsilon,
            num_heads=num_heads,
        )
        self.ffn = FeedForward(
            activation=activation,
            depth=depth,
            dropout_rate=dropout_rate,
            epsilon=epsilon,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ):
        x = self.self_attention(inputs, key_padding_mask=key_padding_mask)
        x = self.ffn(x)
        return x
