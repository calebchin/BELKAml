"""Finetuning heads for molecular property prediction.

This module provides simple, flexible MLP-based heads that accept pre-extracted
embeddings and project them to task-specific outputs.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Literal
import torch
import torch.nn as nn


def get_activation(name: str) -> nn.Module:
    """Get activation module by name.

    Args:
        name: Activation name ('relu', 'gelu', 'tanh', 'silu', 'sigmoid')

    Returns:
        Activation module

    Raises:
        ValueError: If activation name not recognized
    """
    activations = {
        'relu': nn.ReLU(),
        'gelu': nn.GELU(),
        'tanh': nn.Tanh(),
        'silu': nn.SiLU(),
        'sigmoid': nn.Sigmoid()
    }
    if name.lower() not in activations:
        raise ValueError(f"Unknown activation: {name}")
    return activations[name.lower()]


class FinetuneHead(nn.Module, ABC):
    """Abstract base class for finetuning heads.

    Finetuning heads are simple MLP architectures that accept embeddings
    and project them to task-specific outputs.
    """

    def __init__(
        self,
        embedding_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.1,
        activation: str = 'relu',
        use_normalization: bool = False,
        normalization_type: Literal['layer', 'batch'] = 'layer',
        output_activation: Optional[str] = None
    ):
        """Initialize finetuning head.

        Args:
            embedding_dim: Dimension of input embeddings (e.g., 32 for Belka)
            output_dim: Dimension of output (e.g., 1 for binary classification)
            hidden_dims: List of hidden layer dimensions. If None, direct projection.
            dropout_rate: Dropout probability (applied after each hidden layer)
            activation: Activation function ('relu', 'gelu', 'tanh', 'silu')
            use_normalization: Whether to apply normalization layers
            normalization_type: Type of normalization ('layer' or 'batch')
            output_activation: Output activation ('sigmoid', None)
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.use_normalization = use_normalization
        self.normalization_type = normalization_type
        self.output_activation = output_activation

        # Build MLP layers
        self._build_layers()

    def _build_layers(self):
        """Build the MLP layer architecture."""
        layers = []

        # Build MLP
        if self.hidden_dims is None or len(self.hidden_dims) == 0:
            # Direct projection
            layers.append(nn.Dropout(self.dropout_rate))
            layers.append(nn.Linear(self.embedding_dim, self.output_dim))
        else:
            # MLP with hidden layers
            dims = [self.embedding_dim] + self.hidden_dims + [self.output_dim]

            for i in range(len(dims) - 1):
                # Linear layer
                layers.append(nn.Linear(dims[i], dims[i+1]))

                # Don't add activation/dropout/norm after final layer
                if i < len(dims) - 2:
                    # Normalization (optional)
                    if self.use_normalization:
                        if self.normalization_type == 'layer':
                            layers.append(nn.LayerNorm(dims[i+1]))
                        else:  # batch
                            layers.append(nn.BatchNorm1d(dims[i+1]))

                    # Activation
                    layers.append(get_activation(self.activation))

                    # Dropout
                    layers.append(nn.Dropout(self.dropout_rate))

        self.mlp = nn.Sequential(*layers)

        # Output activation (if specified)
        if self.output_activation:
            self.output_act = get_activation(self.output_activation)
        else:
            self.output_act = None

    @abstractmethod
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass - implemented by subclasses.

        Args:
            embeddings: Pre-extracted embeddings of shape (batch, embedding_dim)

        Returns:
            Task-specific predictions
        """
        pass


class BinaryClassificationHead(FinetuneHead):
    """Finetuning head for binary classification tasks.

    Accepts pre-extracted embeddings and projects to binary output with sigmoid.
    Compatible with BinaryLoss and MaskedAUC.
    """

    def __init__(
        self,
        embedding_dim: int = 32,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.1,
        activation: str = 'relu',
        use_normalization: bool = False,
        normalization_type: Literal['layer', 'batch'] = 'layer'
    ):
        """Initialize binary classification head.

        Args:
            embedding_dim: Dimension of input embeddings (e.g., 32 for Belka)
            hidden_dims: Hidden layer sizes (e.g., [64, 32])
            dropout_rate: Dropout probability
            activation: Hidden layer activation function
            use_normalization: Whether to use normalization
            normalization_type: 'layer' or 'batch' normalization
        """
        super().__init__(
            embedding_dim=embedding_dim,
            output_dim=1,  # Binary classification
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation,
            use_normalization=use_normalization,
            normalization_type=normalization_type,
            output_activation='sigmoid'  # Binary needs sigmoid
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            embeddings: Pre-extracted embeddings of shape (batch, embedding_dim)

        Returns:
            Binary predictions of shape (batch, 1) with sigmoid activation
        """
        # MLP layers
        x = self.mlp(embeddings)

        # Output activation
        if self.output_act is not None:
            x = self.output_act(x)

        return x


class RegressionHead(FinetuneHead):
    """Finetuning head for regression tasks.

    Accepts pre-extracted embeddings and projects to continuous outputs.
    Compatible with MSELoss, MAELoss, etc.
    """

    def __init__(
        self,
        embedding_dim: int = 32,
        output_dim: int = 1,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.1,
        activation: str = 'relu',
        use_normalization: bool = False,
        normalization_type: Literal['layer', 'batch'] = 'layer',
        output_activation: Optional[str] = None
    ):
        """Initialize regression head.

        Args:
            embedding_dim: Dimension of input embeddings (e.g., 32 for Belka)
            output_dim: Number of continuous outputs (default 1)
            hidden_dims: Hidden layer sizes
            dropout_rate: Dropout probability
            activation: Hidden layer activation function
            use_normalization: Whether to use normalization
            normalization_type: 'layer' or 'batch' normalization
            output_activation: Optional output activation (e.g., 'tanh' to bound)
        """
        super().__init__(
            embedding_dim=embedding_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation,
            use_normalization=use_normalization,
            normalization_type=normalization_type,
            output_activation=output_activation
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            embeddings: Pre-extracted embeddings of shape (batch, embedding_dim)

        Returns:
            Continuous predictions of shape (batch, output_dim)
        """
        # MLP layers
        x = self.mlp(embeddings)

        # Output activation
        if self.output_act is not None:
            x = self.output_act(x)

        return x


class ProteinAwareBinaryClassificationHead(nn.Module):
    """Binary classification head with protein target conditioning.

    This head accepts molecule embeddings and protein IDs, learns protein
    embeddings, concatenates them, and passes through an MLP for binary
    classification.

    Compatible with BinaryLoss and MaskedAUC.
    """

    def __init__(
        self,
        molecule_embedding_dim: int,
        num_proteins: int = 3,
        protein_embedding_dim: int = 16,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.1,
        activation: str = 'relu',
        use_normalization: bool = False,
        normalization_type: Literal['layer', 'batch'] = 'layer'
    ):
        """Initialize protein-aware binary classification head.

        Args:
            molecule_embedding_dim: Dimension of molecule embeddings from foundation model
            num_proteins: Number of unique protein targets (default 3)
            protein_embedding_dim: Dimension of learned protein embeddings (default 16)
            hidden_dims: Hidden layer sizes for MLP (e.g., [128, 64])
            dropout_rate: Dropout probability
            activation: Hidden layer activation function
            use_normalization: Whether to use normalization
            normalization_type: 'layer' or 'batch' normalization
        """
        super().__init__()

        self.molecule_embedding_dim = molecule_embedding_dim
        self.num_proteins = num_proteins
        self.protein_embedding_dim = protein_embedding_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.use_normalization = use_normalization
        self.normalization_type = normalization_type

        # Protein embedding layer
        self.protein_embedding = nn.Embedding(
            num_embeddings=num_proteins,
            embedding_dim=protein_embedding_dim
        )

        # Combined embedding dimension
        combined_dim = molecule_embedding_dim + protein_embedding_dim

        # Build MLP for classification
        layers = []

        if hidden_dims is None or len(hidden_dims) == 0:
            # Direct projection
            layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.Linear(combined_dim, 1))
        else:
            # MLP with hidden layers
            dims = [combined_dim] + hidden_dims + [1]

            for i in range(len(dims) - 1):
                # Linear layer
                layers.append(nn.Linear(dims[i], dims[i+1]))

                # Don't add activation/dropout/norm after final layer
                if i < len(dims) - 2:
                    # Normalization (optional)
                    if use_normalization:
                        if normalization_type == 'layer':
                            layers.append(nn.LayerNorm(dims[i+1]))
                        else:  # batch
                            layers.append(nn.BatchNorm1d(dims[i+1]))

                    # Activation
                    layers.append(get_activation(activation))

                    # Dropout
                    layers.append(nn.Dropout(dropout_rate))

        self.mlp = nn.Sequential(*layers)
        self.output_act = nn.Sigmoid()

    def forward(
        self,
        molecule_embeddings: torch.Tensor,
        protein_ids: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            molecule_embeddings: Molecule embeddings of shape (batch, molecule_embedding_dim)
            protein_ids: Protein IDs of shape (batch,) with values in [0, num_proteins)

        Returns:
            Binary predictions of shape (batch, 1) with sigmoid activation
        """
        # Get protein embeddings (batch, protein_embedding_dim)
        protein_emb = self.protein_embedding(protein_ids)

        # Concatenate molecule and protein embeddings
        combined = torch.cat([molecule_embeddings, protein_emb], dim=-1)

        # MLP layers
        x = self.mlp(combined)

        # Sigmoid activation
        x = self.output_act(x)

        return x
