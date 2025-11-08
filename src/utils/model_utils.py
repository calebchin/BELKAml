import torch
from typing import Union, Dict, Any, Optional
from model import Belka


def load_model(
    checkpoint_path: str,
    device: Optional[str] = None,
    **parameters
) -> Belka:
    """
    Load a PyTorch model from a checkpoint file.

    Args:
        checkpoint_path: Path to the checkpoint file (.pt or .pth)
        device: Device to load model on ('cuda' or 'cpu'). If None, auto-detects.
        **parameters: Model parameters (used if checkpoint doesn't contain them)

    Returns:
        Loaded Belka model
    """
    # Auto-detect device if not specified
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Check if it's a full checkpoint dict or just state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Full checkpoint with metadata
        state_dict = checkpoint['model_state_dict']
        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('val_loss', None)
        print(f"Loading checkpoint from epoch {epoch}")
        if loss is not None:
            print(f"Checkpoint validation loss: {loss:.4f}")
    else:
        # Just a state_dict
        state_dict = checkpoint

    # Initialize model
    model = Belka(**parameters)

    # Load state dict
    model.load_state_dict(state_dict)

    # Move to device
    model = model.to(device)

    print(f"Model loaded from {checkpoint_path}")
    print(f"Device: {device}")

    return model
