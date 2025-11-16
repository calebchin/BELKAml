from kfp.dsl import ClassificationMetrics
from kfp.v2.dsl import component, Input, Output, Dataset, Model, Metrics


@component(
    base_image="northamerica-northeast2-docker.pkg.dev/belkaml/belka-repo/belkaml-trainer:latest",
)
def train_model(
    train_data: Input[Dataset],
    val_data: Input[Dataset],
    model: Output[Model],
    train_metrics: Output[Metrics],
    val_metrics: Output[Metrics],
    classification_metrics: Output[ClassificationMetrics],
    config_path: str = "gs://belkamlbucket/configs/vertex_train_config.yaml",
    target_column: str = "binds",
) -> None:
    """Train Belka model on protein-molecule binding data.

    This component loads training configuration from GCS, loads parquet data,
    downloads the vocabulary file, initializes the Belka transformer model,
    and trains it with early stopping and learning rate scheduling.

    Args:
        train_data: Training dataset (parquet format on GCS)
        val_data: Validation dataset (parquet format on GCS)
        model: Output model artifact
        train_metrics: Training metrics output
        val_metrics: Validation metrics output
        classification_metrics: Classification metrics output
        config_path: GCS path to training config YAML file
        target_column: Name of target column in data
    """
    import torch
    import pandas as pd
    import os
    from pathlib import Path
    from google.cloud import storage
    import yaml
    import numpy as np

    # Import BELKAml modules (available in Docker image)
    from model import Belka
    from losses import CategoricalLoss, BinaryLoss, MultiLabelLoss
    from metrics import MaskedAUC
    from utils.torch_data_utils import BelkaDataset
    from torch.utils.data import DataLoader

    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Download and load config from GCS ---
    print(f"Loading training config from {config_path}")
    config_local_path = "/tmp/vertex_train_config.yaml"

    # Parse GCS path
    config_bucket_name = config_path.replace("gs://", "").split("/")[0]
    config_blob_name = "/".join(config_path.replace("gs://", "").split("/")[1:])

    # Download config file
    storage_client = storage.Client()
    config_bucket = storage_client.bucket(config_bucket_name)
    config_blob = config_bucket.blob(config_blob_name)
    config_blob.download_to_filename(config_local_path)
    print(f"Downloaded config to {config_local_path}")

    # Load config
    with open(config_local_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract parameters from config
    seed = config.get('seed', 42)
    batch_size = config.get('batch_size', 32)
    num_workers = config.get('num_workers', 2)
    max_length = config.get('max_length', 128)
    mode = config.get('mode', 'clf')
    epochs = config.get('epochs', 10)
    initial_epoch = config.get('initial_epoch', 0)
    steps_per_epoch = config.get('steps_per_epoch', 100)
    validation_steps = config.get('validation_steps', 20)
    patience = config.get('patience', 3)
    model_name = config.get('model_name', 'belka_clf')
    hidden_size = config.get('hidden_size', 32)
    num_layers = config.get('num_layers', 2)
    vocab_size = config.get('vocab_size', 44)
    dropout_rate = config.get('dropout_rate', 0.1)
    lr = config.get('lr', 0.001)
    epsilon = config.get('epsilon', 1e-7)
    vocab_gcs_path = config.get('vocab_path', 'gs://belkamlbucket/data/raw/vocab.txt')

    print(f"Loaded config: mode={mode}, epochs={epochs}, batch_size={batch_size}")

    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- 2. Download vocab.txt from GCS ---
    print(f"Downloading vocab.txt from {vocab_gcs_path}")
    vocab_local_path = "/tmp/vocab.txt"

    # Parse GCS path
    vocab_bucket_name = vocab_gcs_path.replace("gs://", "").split("/")[0]
    vocab_blob_name = "/".join(vocab_gcs_path.replace("gs://", "").split("/")[1:])

    # Download vocab file
    vocab_bucket = storage_client.bucket(vocab_bucket_name)
    vocab_blob = vocab_bucket.blob(vocab_blob_name)
    vocab_blob.download_to_filename(vocab_local_path)
    print(f"Downloaded vocab.txt to {vocab_local_path}")

    # --- 3. Load data from GCS artifacts ---
    print(f"Loading training data from {train_data.path}")
    print(f"Loading validation data from {val_data.path}")

    # Create PyTorch datasets
    # Note: The data has already been split, so we load each parquet directly
    train_dataset = BelkaDataset(
        parquet_path=train_data.path,
        subset="train",
        val_split=0.0,  # No split needed, already split by previous component
        seed=seed,
        vocab_path=vocab_local_path,
        max_length=max_length
    )

    val_dataset = BelkaDataset(
        parquet_path=val_data.path,
        subset="train",  # Use "train" subset to get all data (no internal split)
        val_split=0.0,
        seed=seed,
        vocab_path=vocab_local_path,
        max_length=max_length
    )

    # Create DataLoaders
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        generator=g
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # --- 4. Initialize model ---
    print(f"Initializing Belka model in {mode} mode...")
    model_params = {
        "hidden_size": hidden_size,
        "dropout_rate": dropout_rate,
        "mode": mode,
        "num_layers": num_layers,
        "vocab_size": vocab_size,
    }

    belka_model = Belka(**model_params).to(device)

    # Sequentially train all modes
    training_order = ["mlm", "fps", "clf"]
    best_val_loss_overall = {}
    best_checkpoint_overall = {}

    # Store the final losses per training mode
    final_train_loss = 0
    final_val_loss = 0
    total_epochs_trained = 0

    # --- 6. Checkpointing function ---
    checkpoint_dir = Path(model.path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(mode, epoch, val_loss):
        """Save model checkpoint to GCS artifact directory"""
        checkpoint_name = f"{model_name}_{mode}_{epoch:03d}_{val_loss:.4f}.pt"
        checkpoint_path = checkpoint_dir / checkpoint_name
        torch.save(belka_model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        return checkpoint_path

    for mode in training_order:

        # --- 7. Training loop ---
        print(f"\nStarting training...")
        print(f"Model architecture:\n{belka_model}")

        print("-" * 20)
        print(f"Training in mode = {mode}")
        print("-" * 20)

        # Switch model mode (assumes Belka.switch_mode handles heads correctly)
        belka_model.switch_mode(mode)
        print(f"Model switched to {mode} mode.")

        # Initialize loss function based on mode
        if mode == "mlm":
            loss_fn = CategoricalLoss(mask=-1, epsilon=epsilon, vocab_size=vocab_size)
            metrics_fn = MaskedAUC(mask=-1, multi_label=False, num_labels=None, mode=mode)
        elif mode == "fps":
            loss_fn = BinaryLoss()
            metrics_fn = MaskedAUC(mask=-1, multi_label=False, num_labels=None, mode=mode)
        else: # clf mode - binary classification
            loss_fn = BinaryLoss()
            metrics_fn = MaskedAUC(mask=-1, multi_label=False, num_labels=None, mode=mode)

        # New optimizer & scheduler for each phase (keeps weights, resets LR state)
        optimizer = torch.optim.Adam(belka_model.parameters(), lr=lr, eps=epsilon)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, mode="min")

        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_checkpoint_path = None

        for epoch in range(initial_epoch, epochs):
            # Training phase
            belka_model.train()
            train_loss = 0

            for step, batch in enumerate(train_loader):
                if step >= steps_per_epoch:
                    break

                # Extract inputs and targets from batch dictionary
                x = batch['smiles'].to(device)
                y = batch['binds'].to(device)

                # Forward pass
                optimizer.zero_grad()
                y_pred = belka_model(x)
                loss = loss_fn(y_pred, y)

                # Backward pass
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= steps_per_epoch

            # Validation phase
            belka_model.eval()
            val_loss = 0

            with torch.no_grad():
                for step, batch in enumerate(val_loader):
                    if validation_steps and step >= validation_steps:
                        break

                    # Extract inputs and targets
                    x = batch['smiles'].to(device)
                    y = batch['binds'].to(device)

                    # Forward pass
                    y_pred = belka_model(x)
                    loss = loss_fn(y_pred, y)
                    val_loss += loss.item()

            val_loss /= (validation_steps if validation_steps else len(val_loader))

            # Update learning rate scheduler
            lr_scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']

            print(f"Epoch {epoch + 1}/{epochs} - "
                f"train_loss: {train_loss:.4f} - "
                f"val_loss: {val_loss:.4f} - "
                f"lr: {current_lr:.6f}")

            # Save checkpoint
            checkpoint_path = save_checkpoint(mode, epoch, val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint_path = checkpoint_path
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
            
            total_epochs_trained += 1
            final_train_loss = train_loss
            final_val_loss = val_loss
        
        print(f"[{mode}] training completed. Best val_loss: {best_val_loss:.4f}")

        # --- 8. Save final model and metrics ---
        print(f"\nTraining completed. Best validation loss: {best_val_loss:.4f}")
        best_val_loss_overall[mode] = best_val_loss
        best_checkpoint_overall[mode] = best_checkpoint_path


        # Load the best checkpoint to use for the next training mode
        if best_checkpoint_path:
            print(f"[{mode}] Loading best checkpoint {best_checkpoint_path} for next phase")
            state_dict = torch.load(best_checkpoint_path, map_location=device)
            belka_model.load_state_dict(state_dict)


    # Copy best checkpoint to main model output path after all modes have been trained
    if best_checkpoint_path:
        final_model_path = checkpoint_dir / "model.pt"
        import shutil
        shutil.copy(best_checkpoint_path, final_model_path)
        print(f"Best model saved to {final_model_path}")

    # --- 8. Log metrics to KFP outputs ---
    # Log final (clf) phase losses
    if final_train_loss is not None:
        train_metrics.log_metric("final_train_loss", final_train_loss)
    if final_val_loss is not None:
        val_metrics.log_metric("final_val_loss", final_val_loss)

    # Log best per mode validation loss
    for m in training_order:
        if m in best_val_loss_overall:
            val_metrics.log_metric(f"best_val_loss_{m}", best_val_loss_overall[m])

    val_metrics.log_metric("total_epochs_trained", total_epochs_trained)

    print("Training component completed successfully!")
