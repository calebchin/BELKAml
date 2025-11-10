"""PyTorch-based data utilities converted from tf_data_utils.py.

Provides equivalent functionality for parquet file processing using PyTorch.
Maintains API compatibility while leveraging PyTorch's data handling capabilities.
"""

# UTILS >>>
import numpy as np
import pandas as pd
import torch

# import torch.nn as nn
from torch.utils.data import Dataset  # , DataLoader
from sklearn.model_selection import train_test_split
import os
from typing import List, Dict

# import mapply
from skfp.fingerprints import ECFPFingerprint
from rdkit import Chem
import atomInSmiles
import dask.dataframe as dd
import pyarrow as pa
import itertools


def read_parquet(subset: str, root: str) -> pd.DataFrame:
    """Read and process train/test parquet files"""
    df = pd.read_parquet(os.path.join(root, f"{subset}.parquet"))
    df = df.rename(
        columns={
            "buildingblock1_smiles": "block1",
            "buildingblock2_smiles": "block2",
            "buildingblock3_smiles": "block3",
            "molecule_smiles": "smiles",
        }
    )

    cols = ["block1", "block2", "block3", "smiles"]
    values = "binds" if subset == "train" else "id"
    df = df.pivot(index=cols, columns="protein_name", values=values).reset_index()
    return df


def make_parquet(work_dir: str, extra_data: str, seed: int) -> None:
    """Make Dask DataFrame and write to parquet file. Ensure that the validation set includes building blocks that have never been seen."""

    def validation_split(x: pd.Series, validation_blocks: set) -> np.int8:
        blocks = set(x[col] for col in ["block1", "block2", "block3"])
        i = len(blocks.intersection(validation_blocks))
        i = 0 if i == 0 else 1
        return np.int8(i)

    def replace_linker(smiles: str) -> str:
        smiles = smiles.replace("[Dy]", "[H]")
        smiles = Chem.CanonSmiles(smiles)
        return smiles

    dataset = []
    for subset in ["test", "extra", "train"]:
        if subset in ["train", "test"]:
            df = read_parquet(subset=subset, root=work_dir)
        else:
            df = pd.read_csv(
                os.path.join(work_dir, extra_data),
                usecols=["new_structure", "read_count"],  # type: ignore
            )
            df = df.rename(columns={"new_structure": "smiles", "read_count": "binds"})
        cols = ["BRD4", "HSA", "sEH"]
        if subset == "train":
            df["binds"] = np.stack(
                [df[col].to_numpy() for col in cols], axis=-1, dtype=np.int8
            ).tolist()
        elif subset == "test":
            df["binds"] = np.tile(
                np.array([[2, 2, 2]], dtype=np.int8), reps=(df.shape[0], 1)
            ).tolist()
        else:
            df["binds"] = df["binds"].mapply(
                lambda x: [2, 2, np.clip(x, a_min=0, a_max=1)]
            )
        for col in cols:
            df = df.drop(columns=[col]) if col in df.columns else df

        if subset == "train":
            blocks = list(
                set(df["block1"].to_list())
                | set(df["block2"].tolist())
                | set(df["block3"].tolist())
            )
            _, val = train_test_split(blocks, test_size=0.03, random_state=seed)
            df["subset"] = df.mapply(
                lambda x: validation_split(x, validation_blocks=set(val)), axis=1
            )  # type: ignore
        elif subset == "test":
            df["subset"] = 2
        else:
            df["subset"] = 0

        df = df.drop(columns=["block1", "block2", "block3"], errors="ignore")
        df["smiles_no_linker"] = df["smiles"].mapply(replace_linker)
        dataset.append(df)

    df = pd.concat(dataset).sample(frac=1.0, ignore_index=True, random_state=seed)
    ddf = dd.from_pandas(df, npartitions=20)

    schema = {
        "smiles": pa.string(),
        "binds": pa.list_(pa.int8(), 3),
        "subset": pa.int8(),
        "smiles_no_linker": pa.string(),
    }
    ddf.to_parquet(os.path.join(work_dir, "belka.parquet"), schema=schema)


def get_vocab(work_dir: str) -> None:
    """Get vocabularly for SMILES encoding"""
    df = pd.read_parquet(os.path.join(work_dir, "belka.parquet"))
    df["smiles"] = df["smiles"].mapply(
        lambda x: list(set(atomInSmiles.smiles_tokenizer(x)))
    )
    vocab = np.unique(
        list(itertools.chain.from_iterable(df["smiles"].tolist()))
    ).tolist()
    with open(os.path.join(work_dir, "vocab.txt"), "w") as f:
        for token in vocab:
            f.write(f"{token}\n")


class SMILESTokenizer:
    """A simple tokenizer for SMILES data"""

    def __init__(self, vocab_path: str):
        self.vocab = self._load_vocab(vocab_path)
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}

    def _load_vocab(self, vocab_path: str) -> List[str]:
        with open(vocab_path, "r") as f:
            vocab = [line.strip() for line in f]
            return ["[PAD]", "[MASK]", "[UNK]"] + vocab

    def encode(self, smiles_list: List[str], max_length: int) -> torch.Tensor:
        """Encodes list of SMILES strings into torch Tensor format"""
        tokenized_list = [atomInSmiles.smiles_tokenizer(s) for s in smiles_list]
        encoded = []
        for tokens in tokenized_list:
            ids = [
                self.token_to_id.get(token, self.token_to_id.get("[UNK]", 2))
                for token in tokens
            ]
            ids = ids[:max_length]
            padded_ids = ids + [self.token_to_id["[PAD]"]] * (max_length - len(ids))
            encoded.append(padded_ids)
        return torch.tensor(encoded, dtype=torch.long)


class BelkaDataset(Dataset):
    """PyTorch dataset for BELKA parquet data handling.

    Expects parquet with columns:
    - protein_name (string): protein identifier
    - molecule_smiles (string): SMILES representation of molecule
    - binds (integer): binary binding label (0 or 1)
    """

    def __init__(self, parquet_path: str, subset: str = "train", val_split: float = 0.1, seed: int = 42,
                 vocab_path: str = None, max_length: int = 128):
        """Initialize BelkaDataset.

        Args:
            parquet_path: Path to belka.parquet file
            subset: 'train' or 'val'
            val_split: Validation split fraction (default 0.1 = 10%)
            seed: Random seed for reproducibility
            vocab_path: Path to vocabulary file
            max_length: Maximum sequence length for tokenization

        """
        # Read full parquet file
        df = pd.read_parquet(parquet_path)

        # Create train/val split using numpy Generator
        rng = np.random.default_rng(seed)
        total_rows = len(df)
        val_size = int(total_rows * val_split)

        # Shuffle indices
        indices = rng.permutation(total_rows)

        # Split indices
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        # Select subset
        if subset == "train":
            self.data = df.iloc[train_indices].reset_index(drop=True)
        elif subset == "val":
            self.data = df.iloc[val_indices].reset_index(drop=True)
        else:
            raise ValueError(f"subset must be 'train' or 'val', got {subset}")

        # Initialize tokenizer if vocab provided
        self.max_length = max_length
        if vocab_path:
            self.tokenizer = SMILESTokenizer(vocab_path)
        else:
            self.tokenizer = None

        # Initialize ECFP transformer
        self.ecfp_transformer = ECFPFingerprint(fp_size=2048)  # type: ignore

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        row = self.data.iloc[idx]

        smiles = row["molecule_smiles"]
        binds = int(row["binds"])

        # Tokenize SMILES if tokenizer is available
        if self.tokenizer:
            tokens = atomInSmiles.smiles_tokenizer(smiles)
            # Convert tokens to IDs
            token_ids = [self.tokenizer.token_to_id.get(token, self.tokenizer.token_to_id.get("[UNK]", 2))
                        for token in tokens]
            # Truncate or pad
            token_ids = token_ids[:self.max_length]
            token_ids = token_ids + [self.tokenizer.token_to_id["[PAD]"]] * (self.max_length - len(token_ids))
            smiles_tokens = torch.tensor(token_ids, dtype=torch.long)
        else:
            smiles_tokens = torch.zeros(self.max_length, dtype=torch.long)

        # Generate ECFP fingerprint from SMILES
        try:
            ecfp = self.ecfp_transformer.transform([smiles])[0]
        except Exception as e:
            # If SMILES parsing fails, return zero vector
            print(f"Warning: Failed to generate ECFP for SMILES: {smiles}, error: {e}")
            ecfp = np.zeros(2048, dtype=np.float32)

        return {
            "smiles": smiles_tokens,  # Tokenized SMILES
            "binds": torch.tensor([binds], dtype=torch.float32),  # Single binary label
            "ecfp": torch.tensor(ecfp, dtype=torch.float32),
        }


def collate_fn(batch: List[Dict], tokenizer: SMILESTokenizer, max_length: int) -> Dict:
    """Custom collate function to tokenize SMILES strings in a batch."""
    smiles = [item["smiles"] for item in batch]
    binds = torch.stack([item["binds"] for item in batch])
    ecfp = torch.stack([item["ecfp"] for item in batch])

    # Tokenize and pad smiles strings
    encoded_smiles = tokenizer.encode(smiles, max_length=max_length)

    return {"smiles": encoded_smiles, "binds": binds, "ecfp": ecfp}
