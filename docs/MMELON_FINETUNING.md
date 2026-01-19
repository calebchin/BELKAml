# MMELON Foundation Model Finetuning Guide

This guide explains how to finetune the MMELON foundation model for molecular binding prediction with protein targets.

## Overview

The finetuning pipeline consists of:

1. **MMELON Foundation Model** (frozen) - Extracts multi-view molecular embeddings
2. **Protein Embedding Layer** - Learns embeddings for protein targets
3. **Classification Head** - MLP that predicts binding from concatenated embeddings

## Architecture

```
Input: SMILES string + Protein ID
    ↓
MMELON Foundation Model (frozen)
    ↓
Molecule Embedding (batch, embedding_dim)
    ↓
        ┌─────────────────────────┐
        │  Protein Embedding      │
        │  (batch, 16)            │
        └─────────────────────────┘
    ↓
Concatenate → (batch, embedding_dim + 16)
    ↓
MLP Head [128, 64] → Dropout → Linear
    ↓
Sigmoid → Binary Prediction (batch, 1)
```

## Installation

### 1. Install Optional Dependencies

The MMELON model requires the `biomed-multi-view` package:

```bash
pip install -e ".[foundationmodels]"
```

This installs the `biomed-multi-view` package from the GitHub repository.

**Note**: The package requires PyTorch. If you encounter version conflicts, see the troubleshooting section.

### 2. Verify Installation

```python
from core.model.finetune import MMELONFoundationModel

# Initialize model (downloads from HuggingFace on first use)
model = MMELONFoundationModel()

# Test embedding extraction
embedding = model.get_embedding("CCO")
print(f"Embedding shape: {embedding.shape}")
```

## Data Format

Your training data should be a parquet or CSV file with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `smiles` | str | SMILES string representation of molecule |
| `protein_id` | int | Integer protein ID (0, 1, 2 for BRD4, HSA, sEH) |
| `binds` | int | Binary binding label (0 or 1) |

**Example:**

```csv
smiles,protein_id,binds
CCO,0,1
CC(C)C,1,0
C1=CC=CC=C1,2,1
```

**Protein ID Mapping** (for Belka dataset):
- 0: BRD4
- 1: HSA
- 2: sEH

## Configuration

Edit [`configs/mmelon_finetune_config.yaml`](../configs/mmelon_finetune_config.yaml) to customize training:

### Key Parameters

```yaml
# Data
data_path: "data/belka.parquet"
val_ratio: 0.1

# Model
foundation_model_path: "ibm/biomed.sm.mv-te-84m"
freeze_foundation: true  # Only train the head
hidden_dims: [128, 64]   # MLP architecture
protein_embedding_dim: 16

# Training
epochs: 50
batch_size: 32
learning_rate: 0.001
patience: 10  # Early stopping patience

# Loss
focal_gamma: 2.0   # Focal loss for class imbalance
focal_alpha: 0.25
```

## Training

### Basic Usage

```bash
python src/scripts/train_mmelon_finetune.py \
    --config configs/mmelon_finetune_config.yaml
```

### Training Output

The script will:

1. Load and split data (stratified by protein ID)
2. Initialize frozen MMELON foundation model
3. Train protein-aware classification head
4. Save checkpoints to `checkpoints/mmelon_finetune/`
5. Print training/validation metrics

**Example output:**

```
Loading datasets...
Loaded 90000 samples for split='train'
Loaded 10000 samples for split='val'

Loading MMELON foundation model...
Determining embedding dimension...
MMELON embedding dimension: 512

Starting training for 50 epochs...
============================================================
Epoch 1/50
============================================================
Epoch 1 | Batch 10/2813 | Loss: 0.6234
Epoch 1 | Batch 20/2813 | Loss: 0.5891
...
Epoch 1 | Train Loss: 0.4521
Epoch 1 | Val Loss: 0.3892 | Val AUC: 0.7543
Saved checkpoint: checkpoints/mmelon_finetune/mmelon_head_best_epoch001_valloss0.3892.pt
```

## Checkpoints

Checkpoints are saved as:

```
checkpoints/mmelon_finetune/mmelon_head_best_epochXXX_vallossY.YYYY.pt
```

Each checkpoint contains:

- `head_state_dict`: Trained classification head weights
- `optimizer_state_dict`: Optimizer state for resuming training
- `epoch`: Epoch number
- `val_loss`: Validation loss

**Note**: The foundation model weights are NOT saved (they remain frozen and unchanged).

## Inference

### Loading a Trained Model

```python
import torch
from core.model.finetune import (
    MMELONFoundationModel,
    ProteinAwareBinaryClassificationHead
)

# Load foundation model
foundation_model = MMELONFoundationModel()

# Initialize head (must match training config)
head = ProteinAwareBinaryClassificationHead(
    molecule_embedding_dim=512,  # MMELON embedding dim
    num_proteins=3,
    protein_embedding_dim=16,
    hidden_dims=[128, 64]
)

# Load checkpoint
checkpoint = torch.load('checkpoints/mmelon_finetune/mmelon_head_best_epoch010_valloss0.3421.pt')
head.load_state_dict(checkpoint['head_state_dict'])
head.eval()

# Run inference
smiles = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
protein_id = torch.tensor([0])  # BRD4

with torch.no_grad():
    # Extract embedding
    embedding = foundation_model.get_embedding(smiles)
    # Predict binding
    prediction = head(embedding.unsqueeze(0), protein_id)

print(f"Binding probability: {prediction.item():.4f}")
```

### Batch Inference

```python
smiles_list = [
    "CCO",
    "CC(C)C",
    "C1=CC=CC=C1"
]
protein_ids = torch.tensor([0, 1, 2])  # Different proteins

with torch.no_grad():
    embeddings = foundation_model.get_embedding(smiles_list)
    predictions = head(embeddings, protein_ids)

for smiles, protein, pred in zip(smiles_list, protein_ids, predictions):
    print(f"{smiles} + Protein {protein.item()} → {pred.item():.4f}")
```

## Model Components

### 1. MMELONFoundationModel

Wrapper for IBM Research MMELON multi-view foundation model.

**Key methods:**
- `get_embedding(smiles)`: Extract molecule embeddings
- `inference()`: Not supported (raises NotImplementedError)

**Features:**
- Multi-view encoding: Image + Graph + Text
- Attention-based fusion
- Pre-trained on large molecular datasets

### 2. ProteinAwareBinaryClassificationHead

Classification head with protein target conditioning.

**Architecture:**
- Protein embedding layer (3 proteins → 16-dim)
- Concatenation with molecule embeddings
- MLP with configurable hidden layers
- Sigmoid output activation

**Parameters:**
- `molecule_embedding_dim`: From foundation model (512 for MMELON)
- `num_proteins`: Number of unique proteins (3 for Belka)
- `protein_embedding_dim`: Size of protein embeddings (16)
- `hidden_dims`: MLP architecture (e.g., [128, 64])

### 3. Dataset: MoleculeProteinBindingDataset

PyTorch dataset for loading binding data.

**Features:**
- Stratified train/val split (by protein ID)
- Supports parquet and CSV formats
- Custom collation function for MMELON workflow
