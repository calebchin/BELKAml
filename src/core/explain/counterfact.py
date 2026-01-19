# This implements the explainability interface for counterfactual explanations. We will use the exmol package as the first implementation.

from typing import Any, Dict, Optional, List, Callable
import torch
import torch.nn as nn
from .interface import ExplainabilityInterface


class CounterfactualExplainer(ExplainabilityInterface):
    """Counterfactual explanation using the ExMol package.

    Generates alternative molecular structures (SMILES) that would result
    in different binding predictions, helping understand decision boundaries.

    Reference: https://github.com/ur-whitelab/exmol
    """

    def __init__(self):
        """Initialize counterfactual explainer.

        Note: User will provide prediction function wrapper when calling explain().
        """
        pass  # Minimal initialization

    def _generate_counterfactuals(
        self,
        smiles: str,
        predict_fn: Callable,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Generate counterfactual explanations using exmol.

        Args:
            smiles: Original SMILES string
            predict_fn: Prediction function that takes SMILES string(s) and returns predictions.
                       This function should handle all tokenization/preprocessing internally.
                       Signature: predict_fn(smiles: Union[str, List[str]]) -> Union[float, List[float]]
            **kwargs: ExMol-specific parameters:
                - num_samples: Number of counterfactuals to generate (default: 10)
                - target_class: Desired prediction for counterfactuals (0 or 1)
                - similarity_threshold: Minimum Tanimoto similarity
                - Other exmol parameters

        Returns:
            List of counterfactual dicts with:
            {
                'smiles': str,
                'prediction': float,
                'similarity': float,
                'explanation': str (optional)
            }

        TODO: User implements this method with exmol logic

        """
        # Placeholder implementation
        raise NotImplementedError(
            "User must implement _generate_counterfactuals() with exmol logic. "
            "See docstring for expected signature and return format. "
            "Install exmol: pip install exmol"
        )

    def explain(
        self,
        input: Any,
        model: Optional[nn.Module] = None,
        predictions: Optional[torch.Tensor] = None,
        preprocessed: bool = False,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Generate counterfactual explanations.

        Args:
            input: Dict with 'smiles' key (raw SMILES string)
                   Can optionally include 'protein' for metadata
            model: Ignored - not used (prediction function provided in kwargs instead)
            predictions: Optional - pre-computed original prediction
            preprocessed: Ignored - predict_fn handles preprocessing
            **kwargs: Required and optional parameters:
                - predict_fn: REQUIRED callable that takes SMILES and returns predictions
                - num_samples: Number of counterfactuals (default: 10)
                - target_class: Desired outcome (0 or 1)
                - similarity_threshold: Min similarity to original
                - Other exmol parameters

        Returns:
            Dict with:
            {
                'method': 'counterfactual_exmol',
                'metadata': {
                    'original_smiles': str,
                    'original_prediction': float,
                    'num_samples': int,
                    'target_class': int (optional),
                    ...
                },
                'counterfactuals': [
                    {
                        'smiles': str,
                        'prediction': float,
                        'similarity': float,
                        'explanation': str (optional)
                    },
                    ...
                ]
            }

        Raises:
            ValueError: If predict_fn not provided in kwargs
            NotImplementedError: If user hasn't implemented _generate_counterfactuals()

        """
        # 1. Validate predict_fn is provided
        predict_fn = kwargs.get('predict_fn')
        if predict_fn is None:
            raise ValueError(
                "predict_fn is required in kwargs. "
                "It should be a callable that takes SMILES string(s) and returns predictions."
            )

        # 2. Extract SMILES from input
        if isinstance(input, dict):
            smiles = input.get('smiles')
            if smiles is None:
                raise ValueError("input dict must contain 'smiles' key")
        else:
            raise ValueError("input must be a dict with 'smiles' key")

        # 3. Get original prediction
        if predictions is not None:
            # Use provided prediction
            if isinstance(predictions, torch.Tensor):
                original_prediction = predictions.item()
            else:
                original_prediction = float(predictions)
        else:
            # Compute prediction using predict_fn
            original_prediction = predict_fn(smiles)
            if isinstance(original_prediction, (list, tuple)):
                original_prediction = original_prediction[0]
            original_prediction = float(original_prediction)

        # 4. Extract parameters with defaults
        num_samples = kwargs.get('num_samples', 10)
        target_class = kwargs.get('target_class')

        # 5. Generate counterfactuals (user implements this)
        counterfactuals = self._generate_counterfactuals(
            smiles=smiles,
            predict_fn=predict_fn,
            **kwargs
        )

        # 6. Build result dict
        result = {
            'method': 'counterfactual_exmol',
            'metadata': {
                'original_smiles': smiles,
                'original_prediction': original_prediction,
                'num_samples': num_samples,
            },
            'counterfactuals': counterfactuals
        }

        # Add optional metadata
        if target_class is not None:
            result['metadata']['target_class'] = target_class
        if 'protein' in input:
            result['metadata']['protein'] = input['protein']

        return result
