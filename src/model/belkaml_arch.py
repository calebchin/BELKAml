import torch.nn as nn
import torch.nn.functional as f
from .layers import Embeddings, EncoderLayer
#from torch.utils.data import Dataset, DataLoader
#import os
#import random
#import numpy as np


class Belka(nn.Module):
    def __init__(self, hidden_size : int, dropout_rate: float, mode: str, num_layers: int, vocab_size: int):
        super(Belka, self).__init__()
        
        # Arguments
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.mode = mode
        
        # Layers
        # Note: Assuming Embeddings and EncoderLayer are implemented elsewhere
        # These would typically be transformer-style embeddings and encoder layers
        self.embeddings = Embeddings(input_dim=vocab_size, name='smiles_emb')  # TODO: Implement Embeddings
        
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(name=f'encoder_{i}')  # TODO: Implement EncoderLayer
            for i in range(num_layers)
        ])
        
        if mode == 'mlm':
            self.head = nn.Linear(hidden_size, vocab_size)
        else:
            if mode == 'clf':
                output_units = 3
            else:  # fps mode
                output_units = 2048
            
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),  # Global average pooling
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size, output_units),
                nn.Sigmoid()
            )
    
    def forward(self, inputs, training=None):
        x = self.embeddings(inputs)
        
        for encoder in self.encoder_layers:
            x = encoder(x, training=training)
        
        if self.mode == 'mlm':
            x = self.head(x)
            x = f.softmax(x, dim=-1)
        else:
            # For non-MLM modes, we need to handle the global pooling differently
            # PyTorch's AdaptiveAvgPool1d expects (batch, channels, length)
            x = x.transpose(1, 2)  # (batch, hidden, seq_len)
            x = self.head(x)
            x = x.squeeze(-1)  # Remove the pooled dimension
        
        return x
