"""MMELON foundation model wrapper for molecular embeddings.

This module provides a wrapper for the IBM Research MMELON multi-view
foundation model that implements the FoundationModelInterface.

Model: ibm-research/biomed.sm.mv-te-84m
Reference: https://huggingface.co/ibm-research/biomed.sm.mv-te-84m
"""

from typing import Any, Union, List
import warnings
import torch

from .interface import FoundationModelInterface

# Import MMELON dependencies with graceful error handling
try:
    from bmfm_sm.api.smmv_api import SmallMoleculeMultiViewModel
    from bmfm_sm.core.data_modules.namespace import LateFusionStrategy
    MMELON_AVAILABLE = True
except ImportError as e:
    MMELON_AVAILABLE = False
    MMELON_IMPORT_ERROR = e


class MMELONFoundationModel(FoundationModelInterface):
    """Wrapper for IBM Research MMELON multi-view foundation model.

    MMELON is a multi-view model that combines image, graph, and text
    representations of molecules using attention-based fusion.

    The model processes SMILES strings through three encoders:
    - Image encoder: 2D molecular depictions
    - Graph encoder: Atoms and bonds structure
    - Text encoder: SMILES sequence embeddings

    These are combined via an attention-based fusion module to produce
    unified multi-view embeddings suitable for downstream finetuning.

    Example:
        >>> mmelon = MMELONFoundationModel(device="auto")
        >>> embeddings = mmelon.get_embedding("CCO")
        >>> print(embeddings.shape)  # (embedding_dim,)
        >>>
        >>> # Batch processing
        >>> smiles_list = ["CCO", "CC(C)C", "C1=CC=CC=C1"]
        >>> embeddings = mmelon.get_embedding(smiles_list)
        >>> print(embeddings.shape)  # (3, embedding_dim)
    """

    def __init__(
        self,
        model_path: str = "ibm/biomed.sm.mv-te-84m",
        fusion_strategy: str = "ATTENTIONAL",
        device: str = "auto",
        **kwargs: Any
    ):
        """Initialize MMELON foundation model.

        Args:
            model_path: HuggingFace model path (default: ibm/biomed.sm.mv-te-84m)
            fusion_strategy: Fusion strategy for multi-view ("ATTENTIONAL")
            device: Device to run on ("auto", "cpu", "cuda", "mps")
            **kwargs: Additional model initialization parameters

        Raises:
            ImportError: If bmfm_sm package is not installed
        """
        super().__init__()

        # Check if MMELON is available
        if not MMELON_AVAILABLE:
            raise ImportError(
                "bmfm_sm package not installed. Install with:\n"
                "  pip install bmfm-sm\n"
                "See: https://github.com/BiomedSciAI/biomed-multi-view"
            ) from MMELON_IMPORT_ERROR

        self.model_path = model_path
        self.fusion_strategy = fusion_strategy
        self.device_str = self._determine_device(device)
        self.kwargs = kwargs

        # Map fusion strategy string to enum
        fusion_strategy_map = {
            "ATTENTIONAL": LateFusionStrategy.ATTENTIONAL
        }
        if fusion_strategy not in fusion_strategy_map:
            raise ValueError(
                f"Unknown fusion_strategy: {fusion_strategy}. "
                f"Supported: {list(fusion_strategy_map.keys())}"
            )
        self.fusion_enum = fusion_strategy_map[fusion_strategy]

        # Load model from HuggingFace
        try:
            self.model = SmallMoleculeMultiViewModel.from_pretrained(
                self.fusion_enum,
                model_path=self.model_path,
                huggingface=True,
                **self.kwargs
            )
            # Set to evaluation mode
            self.model.eval()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load MMELON model from {self.model_path}. "
                f"Error: {str(e)}"
            ) from e

    def _determine_device(self, device: str) -> str:
        """Determine device to use for model.

        Args:
            device: Requested device ("auto", "cpu", "cuda", "mps")

        Returns:
            Device string ("cpu", "cuda", "mps")
        """
        if device != "auto":
            return device

        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _validate_smiles(self, smiles: str) -> bool:
        """Validate SMILES string using RDKit.

        Args:
            smiles: SMILES string to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except ImportError:
            # If RDKit not available, skip validation
            return True

    def _parse_input(self, input: Union[str, List[str], dict]) -> List[str]:
        """Parse input into list of SMILES strings.

        Args:
            input: SMILES string, list of SMILES, or dict with 'smiles' key

        Returns:
            List of SMILES strings

        Raises:
            ValueError: If input format is invalid
        """
        # Handle dict input
        if isinstance(input, dict):
            if 'smiles' not in input:
                raise ValueError(
                    "Input dict must contain 'smiles' key. "
                    f"Got keys: {list(input.keys())}"
                )
            input = input['smiles']

        # Convert to list
        if isinstance(input, str):
            smiles_list = [input]
        elif isinstance(input, list):
            smiles_list = input
        else:
            raise ValueError(
                f"Input must be str, list, or dict with 'smiles' key. "
                f"Got type: {type(input)}"
            )

        # Validate all SMILES
        for smiles in smiles_list:
            if not isinstance(smiles, str):
                raise ValueError(
                    f"All SMILES must be strings. Got type: {type(smiles)}"
                )
            if not self._validate_smiles(smiles):
                warnings.warn(
                    f"Invalid SMILES: {smiles}. This may cause errors.",
                    UserWarning
                )

        return smiles_list

    def get_embedding(
        self,
        input: Union[str, List[str], dict],
        preprocessed: bool = False,
        **kwargs: Any
    ) -> torch.Tensor:
        """Extract embeddings from MMELON model.

        Args:
            input: SMILES string, list of SMILES, or dict with 'smiles' key
            preprocessed: Ignored (MMELON handles preprocessing internally)
            **kwargs: Additional parameters (ignored)

        Returns:
            Embedding tensor of shape:
            - (embedding_dim,) for single SMILES
            - (batch_size, embedding_dim) for list of SMILES

        Raises:
            ValueError: If input format is invalid
            RuntimeError: If embedding extraction fails

        Example:
            >>> mmelon = MMELONFoundationModel()
            >>> embedding = mmelon.get_embedding("CCO")
            >>> print(embedding.shape)  # (embedding_dim,)
        """
        # Parse input to list of SMILES
        smiles_list = self._parse_input(input)
        is_single = isinstance(input, str)

        # Extract embeddings for each SMILES
        embeddings_list = []
        for smiles in smiles_list:
            try:
                # Get embedding from MMELON
                emb = SmallMoleculeMultiViewModel.get_embeddings(
                    smiles=smiles,
                    model_path=self.model_path,
                    huggingface=True
                )

                # Convert to torch tensor if needed
                if not isinstance(emb, torch.Tensor):
                    emb = torch.tensor(emb)

                embeddings_list.append(emb)

            except Exception as e:
                raise RuntimeError(
                    f"Failed to extract embedding for SMILES: {smiles}. "
                    f"Error: {str(e)}"
                ) from e

        # Stack embeddings
        if len(embeddings_list) == 1:
            result = embeddings_list[0]
        else:
            result = torch.stack(embeddings_list)

        # Move to device
        result = result.to(self.device_str)

        return result

    def inference(
        self,
        input: Any,
        preprocessed: bool = False,
        **kwargs: Any
    ) -> Any:
        """Run inference (not supported for pre-trained MMELON).

        The pre-trained MMELON model only provides embeddings.
        For task-specific predictions, use get_embedding() to extract
        embeddings, then pass them to a finetuning head.

        Args:
            input: SMILES string or list of SMILES
            preprocessed: Ignored
            **kwargs: Additional parameters

        Raises:
            NotImplementedError: MMELON requires finetuning for predictions

        Example:
            >>> # Correct usage
            >>> mmelon = MMELONFoundationModel()
            >>> head = BinaryClassificationHead(embedding_dim=512)
            >>> embeddings = mmelon.get_embedding("CCO")
            >>> predictions = head(embeddings)
        """
        raise NotImplementedError(
            "MMELON pre-trained model does not support direct inference. "
            "Use get_embedding() to extract embeddings, then pass to a "
            "finetuning head for task-specific predictions."
        )
