from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
import torch
import torch.nn as nn


class ExplainabilityInterface(ABC):
    """Abstract interface for model explainability methods.

    Supports diverse explanation paradigms including counterfactual explanations,
    attribution methods, attention visualization, and feature importance analysis.

    Implementations can work with either a live model or pre-computed predictions.
    """

    @abstractmethod
    def explain(
        self,
        input: Any,
        model: Optional[nn.Module] = None,
        predictions: Optional[torch.Tensor] = None,
        preprocessed: bool = False,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Generate explanations for model predictions.

        Args:
            input: Input data - can be raw (dict with 'smiles', 'protein') or
                   preprocessed tensors. Format is implementation-specific.
            model: Optional PyTorch model for inference/gradient computation.
                   Required for gradient-based and counterfactual methods.
            predictions: Optional pre-computed predictions tensor.
                   Enables post-hoc explanations without model access.
            preprocessed: Whether input is already tokenized/encoded.
            **kwargs: Method-specific parameters such as:
                - target_class: Which output to explain
                - protein_id: Target protein (0=BRD4, 1=HSA, 2=sEH)
                - num_samples: Number of samples for stochastic methods
                - layer_idx: Transformer layer to extract from
                - Implementation-specific options

        Returns:
            Dict with explanation results. Must include:
            {
                'method': str,  # Explainer name (e.g., 'exmol', 'attention')
                'metadata': Dict[str, Any],  # Parameters, timestamp, etc.
                ...  # Method-specific outputs (documented in subclass)
            }

        Raises:
            ValueError: If required inputs (model/predictions) not provided
            NotImplementedError: If functionality unavailable

        """
        raise NotImplementedError("explain method must be implemented in subclass")
