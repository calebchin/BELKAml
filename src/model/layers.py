from typing import Union, Callable, Optional, Tuple

import torch
import torch.nn as nn
from skfp.fingerprints import ECFPFingerprint


# This makes more sense as a util
class FPGenerator(nn.Module):
    """Fingerprint generator module using ECFP (Extended-Connectivity Fingerprint).

    This module transforms SMILES strings into molecular fingerprints
    using `skfp.fingerprints.ECFPFingerprint`.

    Attributes
    ----------
    transformer : ECFPFingerprint
        Transformer object that computes fingerprints from SMILES strings.

    """

    def __init__(self):
        """Initialize the fingerprint generator with chirality included."""
        super().__init__()
        self.transformer = ECFPFingerprint(include_chirality=True, n_jobs=-1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute molecular fingerprints from SMILES strings.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of SMILES strings (dtype should be convertible to str).

        Returns
        -------
        torch.Tensor
            Fingerprint tensor of dtype `torch.long` with shape [B, F],
            where B is batch size and F is fingerprint dimension.

        """
        # TODO: do this offline?
        x = self.transformer.transform(
            inputs.detach().cpu().numpy().astype(str).tolist()
        )
        x = torch.tensor(x, dtype=torch.long)
        return x


class Encodings(nn.Module):
    """Positional encoding layer.

    Implements sinusoidal positional encodings as described in the
    "Attention is All You Need" paper.

    Attributes
    ----------
    depth : int
        Dimensionality of the encoding vectors.
    encodings : torch.Tensor
        Buffer storing precomputed positional encodings of shape [max_length, depth].

    """

    encodings: torch.Tensor

    def __init__(self, depth: int, max_length: int):
        """Initialize positional encodings.

        Parameters
        ----------
        depth : int
            Dimensionality of the encoding vectors.
        max_length : int
            Maximum sequence length supported.

        """
        super().__init__()
        self.depth = depth
        self.register_buffer("encodings", self._pos_encodings(depth, max_length))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply positional encodings to input embeddings.

        Parameters
        ----------
        inputs : torch.Tensor
            Input embeddings of shape [B, L, D].

        Returns
        -------
        torch.Tensor
            Encoded tensor of shape [B, L, D].

        """
        scale = torch.sqrt(torch.tensor(self.depth, dtype=torch.float32))
        x = inputs * scale
        x = x + self.encodings[: inputs.size(1), :].unsqueeze(0)
        return x

    @staticmethod
    def _pos_encodings(depth: int, max_length: int) -> torch.Tensor:
        """Compute sinusoidal positional encodings.

        Parameters
        ----------
        depth : int
            Dimensionality of the encoding vectors.
        max_length : int
            Maximum sequence length supported.

        Returns
        -------
        torch.Tensor
            Positional encodings of shape [max_length, depth].

        """
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
    """Token embedding layer with positional encodings.

    Attributes
    ----------
    depth : int
        Dimensionality of embedding vectors.
    input_dim : int
        Vocabulary size.
    embeddings : nn.Embedding
        Learnable embedding matrix.
    encodings : Encodings
        Positional encoding layer.

    """

    def __init__(self, max_length: int, depth: int, input_dim: int):
        """Initialize embedding and positional encoding layers.

        Parameters
        ----------
        max_length : int
            Maximum sequence length.
        depth : int
            Dimensionality of embedding vectors.
        input_dim : int
            Vocabulary size.

        """
        super().__init__()
        self.depth = depth
        self.input_dim = input_dim
        self.embeddings = nn.Embedding(
            num_embeddings=input_dim, embedding_dim=depth, padding_idx=0
        )
        self.encodings = Encodings(depth=depth, max_length=max_length)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed tokens and apply positional encodings.

        Parameters
        ----------
        inputs : torch.Tensor
            Input token IDs of shape [B, L].

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            - Encoded embeddings of shape [B, L, D].
            - Key padding mask of shape [B, L], where True indicates padding tokens.

        """
        key_padding_mask = inputs == 0
        x = self.embeddings(inputs)
        x = self.encodings(x)
        return x, key_padding_mask


class FeedForward(nn.Module):
    """Feed-forward block with pre-layer normalization and residual connection.

    Architecture:
    LayerNorm -> Dense (2*depth) -> Activation -> Dense (depth) -> Dropout -> Residual
    """

    def __init__(
        self,
        activation: Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]],
        depth: int,
        dropout_rate: float,
        epsilon: float,
    ):
        """Initialize feed-forward block.

        Parameters
        ----------
        activation : nn.Module or Callable
            Activation function to apply.
        depth : int
            Dimensionality of input/output.
        dropout_rate : float
            Dropout probability.
        epsilon : float
            Epsilon for layer normalization.

        """
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
        """Apply feed-forward transformation.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape [B, L, D].

        Returns
        -------
        torch.Tensor
            Output tensor of shape [B, L, D].

        """
        x = self.norm(inputs)
        x = self.dense1(x)
        x = self.activation_fn(x)
        x = self.dense2(x)
        x = self.dropout(x)
        return x + inputs  # residual


class SelfAttention(nn.Module):
    """Self-attention block with pre-layer normalization and residual connection.

    Architecture:
    LayerNorm -> MultiHeadAttention -> Residual

    Supports optional causal masking for autoregressive decoding.
    """

    def __init__(
        self,
        causal: bool,
        depth: int,
        dropout_rate: float,
        epsilon: float,
        num_heads: int,
    ):
        """Initialize self-attention block.

        Parameters
        ----------
        causal : bool
            Whether to apply causal masking (no attending to future positions).
        depth : int
            Dimensionality of embeddings.
        dropout_rate : float
            Dropout probability in attention.
        epsilon : float
            Epsilon for layer normalization.
        num_heads : int
            Number of attention heads.

        """
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
        """Apply self-attention with residual connection.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape [B, L, D].
        key_padding_mask : Optional[torch.Tensor], default=None
            Boolean mask of shape [B, L], where True marks padded tokens.

        Returns
        -------
        torch.Tensor
            Output tensor of shape [B, L, D].

        """
        x = self.norm(inputs)

        attn_mask = None
        if self.causal:
            L = inputs.size(1)
            attn_mask = torch.triu(
                torch.ones(L, L, device=inputs.device), diagonal=1
            ).bool()

        out, _ = self.mha(
            x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )
        return out + inputs  # residual


class EncoderLayer(nn.Module):
    """Transformer encoder layer with pre-layer normalization.

    Architecture:
    LayerNorm -> SelfAttention -> Residual -> FeedForward -> Residual
    """

    def __init__(
        self,
        activation: Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]],
        depth: int,
        dropout_rate: float,
        epsilon: float,
        num_heads: int,
    ):
        """Initialize encoder layer.

        Parameters
        ----------
        activation : nn.Module or Callable
            Activation function for feed-forward network.
        depth : int
            Dimensionality of embeddings.
        dropout_rate : float
            Dropout probability.
        epsilon : float
            Epsilon for layer normalization.
        num_heads : int
            Number of attention heads.

        """
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
    ) -> torch.Tensor:
        """Apply encoder layer: self-attention followed by feed-forward.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape [B, L, D].
        key_padding_mask : Optional[torch.Tensor], default=None
            Boolean mask of shape [B, L], where True marks padded tokens.

        Returns
        -------
        torch.Tensor
            Output tensor of shape [B, L, D].

        """
        x = self.self_attention(inputs, key_padding_mask=key_padding_mask)
        x = self.ffn(x)
        return x
