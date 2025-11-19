import torch
import torch.nn as nn
import torch.nn.functional as f
from .layers import Embeddings, EncoderLayer
# from torch.utils.data import Dataset, DataLoader
# import os
# import random
# import numpy as np


class Belka(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        dropout_rate: float,
        mode: str,
        num_layers: int,
        vocab_size: int,
        **kwargs: dict
    ):
        super(Belka, self).__init__()

        # Arguments
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.mode = mode

        # Layers
        # Note: Assuming Embeddings and EncoderLayer are implemented elsewhere
        # These would typically be transformer-style embeddings and encoder layers
        self.embeddings = Embeddings(
            max_length=128, depth=32, input_dim=vocab_size
        )  # TODO: don't hard-code, replace once **parameters are implemented

        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    activation=f.gelu,
                    depth=32,
                    dropout_rate=0.1,
                    epsilon=1e-07,
                    num_heads=8,
                )  # TODO: don't hard-code, replace once **parameters are implemented
                for i in range(num_layers)
            ]
        )

        self.mlm_head = nn.Linear(hidden_size, vocab_size)

        if mode == "clf":
            output_units = 1  # Binary classification (single output)
        else:  # fps mode
            output_units = 2048
        
        
        self.fps_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(32, 2048),
            nn.Sigmoid())
        
        self.pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling

        self.clf_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(32, output_units),  # Use depth=32 from encoder
            nn.Sigmoid(),
        )

    def switch_mode(self, mode: str):
        # Switch head for sequential training
        if mode not in {"mlm", "fps", "clf"}:
            raise ValueError(f"Invalid mode: {mode}")
        self.mode = mode

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass of main Belka arch"""
        x, key_padding_mask = self.embeddings(inputs)

        for encoder in self.encoder_layers:
            x = encoder(x, key_padding_mask)

        if self.mode == "mlm":
            x = self.mlm_head(x)
            x = f.softmax(x, dim=-1)
        else:
            # For non-MLM modes: pool over sequence, then classify
            # x shape: (batch, seq_len, depth)
            x = x.transpose(1, 2)  # (batch, depth, seq_len)
            x = self.pool(x)  # (batch, depth, 1)
            x = x.squeeze(-1)  # (batch, depth)
            if self.mode == "fps":
                x = self.fps_head(x)
            elif self.mode == "clf":
                x = self.clf_head(x)
            else:
                raise ValueError(f"Invalid model model {self.mode} provided")

        return x
