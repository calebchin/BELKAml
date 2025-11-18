"""Entry point script for training BELKA model.

This script loads configuration from YAML, sets up paths, and runs the training pipeline.
"""

import sys
import os
import yaml

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from train import train_model


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Dictionary of configuration parameters

    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function."""
    # Load configuration
    config_path = os.path.join('configs', 'train_config.yaml')
    config = load_config(config_path)

    # Set up paths
    project_root = os.path.dirname(os.path.abspath(__file__))
    working_dir = os.path.join(project_root, config.get('working', '.'))
    checkpoint_dir = os.path.join(project_root, 'checkpoints')
    outputs_dir = os.path.join(project_root, 'outputs')

    # Update config with absolute paths
    config['working'] = working_dir
    config['root'] = working_dir

    # Verify belka.parquet exists
    parquet_path = os.path.join(working_dir, 'belka.parquet')
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(
            f"belka.parquet not found at {parquet_path}\n"
            f"Please place your belka.parquet file in the project root directory."
        )

    print("=" * 80)
    print("BELKA Model Training")
    print("=" * 80)
    print(f"Configuration: {config_path}")
    print(f"Working directory: {working_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Outputs directory: {outputs_dir}")
    print(f"Data file: {parquet_path}")
    print(f"Mode: {config['mode']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Steps per epoch: {config['steps_per_epoch']}")
    print("=" * 80)

    # Extract training parameters
    train_params = {
        'model': config['model'],
        'epochs': config['epochs'],
        'initial_epoch': config['initial_epoch'],
        'mode': config['mode'],
        'model_name': config['model_name'],
        'patience': config['patience'],
        'steps_per_epoch': config['steps_per_epoch'],
        'validation_steps': config['validation_steps'],
        'working': checkpoint_dir,  # Save checkpoints to checkpoints/ dir
    }

    # Pass all config as additional parameters
    train_params.update(config)

    # Override working dir for data loading
    train_params['working'] = working_dir

    # Train model
    try:
        print("\nStarting training...")
        model = train_model(**train_params)
        print("\n" + "=" * 80)
        print("Training completed successfully!")
        print(f"Model checkpoints saved to: {checkpoint_dir}")
        print("=" * 80)
        return model
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"Training failed with error: {e}")
        print("=" * 80)
        raise


if __name__ == "__main__":
    main()
