"""Protein name encoder for BELKA project.

Maps protein names to integer IDs for embedding lookup.
"""

from typing import Dict, Optional
from pathlib import Path


class ProteinEncoder:
    """Encode protein names to integer IDs.

    Maps the 3 BELKA protein targets (BRD4, HSA, sEH) to integer IDs
    for use with PyTorch embedding layers.
    """

    def __init__(self, vocab_path: Optional[str] = None):
        """Initialize protein encoder.

        Args:
            vocab_path: Path to protein vocabulary file (one protein per line).
                       If None, uses default 3-protein mapping.
        """
        if vocab_path and Path(vocab_path).exists():
            # Load from file
            with open(vocab_path, 'r') as f:
                proteins = [line.strip() for line in f if line.strip()]
            self.protein_to_id = {protein: idx for idx, protein in enumerate(proteins)}
        else:
            # Default mapping for 3 BELKA proteins
            self.protein_to_id = {
                "BRD4": 0,
                "HSA": 1,
                "sEH": 2,
            }

        # Reverse mapping
        self.id_to_protein = {idx: protein for protein, idx in self.protein_to_id.items()}

        # Store vocabulary size
        self.vocab_size = len(self.protein_to_id)

    def encode(self, protein_name: str) -> int:
        """Encode protein name to integer ID.

        Args:
            protein_name: Protein name (e.g., "BRD4", "HSA", "sEH")

        Returns:
            Integer ID (0-2 for the 3 proteins)

        Raises:
            KeyError: If protein name is not in vocabulary
        """
        if protein_name not in self.protein_to_id:
            raise KeyError(
                f"Unknown protein: {protein_name}. "
                f"Valid proteins: {list(self.protein_to_id.keys())}"
            )
        return self.protein_to_id[protein_name]

    def decode(self, protein_id: int) -> str:
        """Decode integer ID to protein name.

        Args:
            protein_id: Integer ID (0-2)

        Returns:
            Protein name string

        Raises:
            KeyError: If ID is not in vocabulary
        """
        if protein_id not in self.id_to_protein:
            raise KeyError(
                f"Unknown protein ID: {protein_id}. "
                f"Valid IDs: {list(self.id_to_protein.keys())}"
            )
        return self.id_to_protein[protein_id]

    def encode_batch(self, protein_names: list) -> list:
        """Encode a batch of protein names.

        Args:
            protein_names: List of protein name strings

        Returns:
            List of integer IDs
        """
        return [self.encode(name) for name in protein_names]

    def decode_batch(self, protein_ids: list) -> list:
        """Decode a batch of protein IDs.

        Args:
            protein_ids: List of integer IDs

        Returns:
            List of protein name strings
        """
        return [self.decode(id) for id in protein_ids]

    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size

    def __repr__(self) -> str:
        """String representation."""
        return f"ProteinEncoder(vocab_size={self.vocab_size}, proteins={list(self.protein_to_id.keys())})"
