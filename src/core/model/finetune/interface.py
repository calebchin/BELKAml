# Interface for foundation model interactions. Should specify how to query the model through API and through local
# to get necessary embedding that will be passed to finetuning heads. Also, if it exists, should provide way to run complete
# inference pass without finetuning for experimental reasons

from abc import ABC, abstractmethod
from typing import Any

import torch


class FoundationModelInterface(ABC):
    """Abstract interface for foundation model interactions.

    Implementations can use local models, API endpoints, or external libraries.
    """

    @abstractmethod
    def get_embedding(
        self,
        input: Any,
        preprocessed: bool = False,
        **kwargs: Any
    ) -> torch.Tensor:
        """Extract intermediate embedding from foundation model.

        Args:
            input: Raw input (e.g., dict with 'smiles', 'protein') or preprocessed tensors
            preprocessed: Whether input is already tokenized/encoded
            **kwargs: Model-specific parameters (e.g., layer selection, pooling strategy)

        Returns:
            Embedding tensor for use in finetuning. Shape/format is implementation-specific.

        """
        pass

    @abstractmethod
    def inference(
        self,
        input: Any,
        preprocessed: bool = False,
        **kwargs: Any
    ) -> Any:
        """Run complete inference pass without finetuning (experimental).

        Args:
            input: Raw input or preprocessed tensors
            preprocessed: Whether input is already tokenized/encoded
            **kwargs: Model-specific inference parameters

        Returns:
            Model predictions. Type depends on implementation and task.

        """
        pass