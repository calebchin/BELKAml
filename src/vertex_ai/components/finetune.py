from kfp.dsl import ClassificationMetrics
from kfp.dsl import component, Input, Output, Dataset, Model, Metrics


@component(
    base_image="northamerica-northeast2-docker.pkg.dev/belkaml/belka-repo/belkaml-trainer:latest",
)
def finetune_model(
    train_data: Input[Dataset],
    val_data: Input[Dataset],
    pretrained_model_id: str,
    model: Output[Model],
    train_metrics: Output[Metrics],
    val_metrics: Output[Metrics],
    classification_metrics: Output[ClassificationMetrics],
    aipproject_id: str = "belkaml",
    aipproject_location: str = "northamerica-northeast2",
    config_path: str = "gs://belkamlbucket/configs/vertex_train_config.yaml",
    target_column: str = "binds",
) -> None:
    """Fine-tune Belka model from pretrained weights on new protein-molecule binding data.

    This component loads a pretrained model from Vertex AI Model Registry, extracts the
    shared encoder/embedding weights, and continues training on new data. Task heads
    (MLM, FPS, CLF) are reinitialized to allow full sequential training.

    Args:
        train_data: Training dataset (parquet format on GCS)
        val_data: Validation dataset (parquet format on GCS)
        pretrained_model_id: Model ID from Vertex AI Model Registry (e.g., "projects/.../models/123")
        model: Output model artifact
        train_metrics: Training metrics output
        val_metrics: Validation metrics output
        classification_metrics: Classification metrics output
        aipproject_id: GCP project ID
        aipproject_location: GCP region
        config_path: GCS path to training config YAML file
        target_column: Name of target column in data
    """
    import torch
    import pandas as pd
    import os
    from pathlib import Path
    from google.cloud import storage, aiplatform as aip
    import yaml
    import numpy as np

    # Import BELKAml modules (available in Docker image)
    from model import Belka
    from losses import CategoricalLoss, BinaryLoss, MultiLabelLoss
    from metrics import MaskedAUC
    from utils.torch_data_utils import BelkaIterableDataset
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
    epsilon = float(config.get('epsilon', 1e-7))
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

    # --- 3. Download pretrained model from Model Registry ---
    print(f"\n{'='*60}")
    print(f"LOADING PRETRAINED MODEL")
    print(f"{'='*60}")

    aip.init(project=aipproject_id, location=aipproject_location)
    pretrained_model = aip.Model(pretrained_model_id)
    print(f"Retrieved pretrained model: {pretrained_model.display_name}")
    print(f"  Model URI: {pretrained_model.uri}")

    # Download model artifact from GCS
    pretrained_model_local_dir = "/tmp/pretrained_model"
    Path(pretrained_model_local_dir).mkdir(parents=True, exist_ok=True)

    # Parse GCS URI and download model.pt
    model_uri = pretrained_model.uri
    if model_uri.startswith("gs://"):
        model_bucket_name = model_uri.replace("gs://", "").split("/")[0]
        model_blob_prefix = "/".join(model_uri.replace("gs://", "").split("/")[1:])

        model_bucket = storage_client.bucket(model_bucket_name)
        model_blob = model_bucket.blob(f"{model_blob_prefix}/model.pt")

        pretrained_model_path = Path(pretrained_model_local_dir) / "model.pt"
        model_blob.download_to_filename(str(pretrained_model_path))
        print(f"Downloaded pretrained model to {pretrained_model_path}")
    else:
        raise ValueError(f"Unsupported model URI format: {model_uri}")

    # --- 4. Initialize model and load pretrained weights ---
    print(f"\nInitializing Belka model architecture...")
    model_params = {
        "hidden_size": hidden_size,
        "dropout_rate": dropout_rate,
        "mode": mode,
        "num_layers": num_layers,
        "vocab_size": vocab_size,
    }

    belka_model = Belka(**model_params).to(device)

    # Load pretrained state dict
    print(f"Loading pretrained weights...")
    pretrained_state = torch.load(pretrained_model_path, map_location=device)

    # Filter to only load shared weights (exclude task heads)
    # Shared components: embeddings, encoder_layers
    # Exclude: mlm_head, fps_head, clf_head, pool
    shared_keys = [k for k in pretrained_state.keys()
                   if not k.startswith(('mlm_head', 'fps_head', 'clf_head', 'pool'))]
    shared_state = {k: pretrained_state[k] for k in shared_keys}

    print(f"Loading {len(shared_keys)} pretrained parameters (excluding task heads):")
    for key in shared_keys:
        print(f"  ✓ {key}")

    # Load only shared weights (strict=False allows missing head weights)
    belka_model.load_state_dict(shared_state, strict=False)
    print(f"\n✓ Pretrained encoder/embeddings loaded successfully!")
    print(f"✓ Task heads (MLM, FPS, CLF) initialized from scratch")
    print(f"{'='*60}\n")

    # Sequentially train all modes
    training_order = ["mlm", "fps", "clf"]
    best_val_loss_overall = {}
    best_checkpoint_overall = {}

    # Store the final losses per training mode
    final_train_loss = 0
    final_val_loss = 0
    total_epochs_trained = 0

    # --- 5. Checkpointing function ---
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

        # --- 6. Training loop ---
        print(f"\nStarting fine-tuning...")
        print(f"Model architecture:\n{belka_model}")

        print("-" * 20)
        print(f"Fine-tuning in mode = {mode}")
        print("-" * 20)

        # Create datasets for this mode (using IterableDataset for memory efficiency)
        # Download protein vocab from GCS
        protein_vocab_local_path = "/tmp/protein_vocab.txt"
        protein_vocab_gcs = "gs://belkamlbucket/data/raw/protein_vocab.txt"
        bucket_name = protein_vocab_gcs.split("/")[2]
        blob_path = "/".join(protein_vocab_gcs.split("/")[3:])
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.download_to_filename(protein_vocab_local_path)

        train_dataset = BelkaIterableDataset(
            parquet_path=train_data.path,
            vocab_path=vocab_local_path,
            protein_vocab_path=protein_vocab_local_path,
            max_length=max_length,
            mode=mode  # Mode-specific dataset
        )

        val_dataset = BelkaIterableDataset(
            parquet_path=val_data.path,
            vocab_path=vocab_local_path,
            protein_vocab_path=protein_vocab_local_path,
            max_length=max_length,
            mode=mode  # Mode-specific dataset
        )

        print(f"Train dataset: streaming from {train_data.path}")
        print(f"Validation dataset: streaming from {val_data.path}")

        # Create DataLoaders for this mode
        # Note: IterableDataset doesn't support shuffle=True, but data is already shuffled in split step
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )

        # Switch model mode (assumes Belka.switch_mode handles heads correctly)
        belka_model.switch_mode(mode)
        print(f"Model switched to {mode} mode.")

        # Initialize loss function based on mode
        if mode == "mlm":
            loss_fn = CategoricalLoss(mask=-1, epsilon=epsilon, vocab_size=vocab_size)
            metrics_fn = MaskedAUC(mask=-1, multi_label=False, num_labels=None, mode=mode, vocab_size=vocab_size)
        elif mode == "fps":
            loss_fn = BinaryLoss()
            metrics_fn = MaskedAUC(mask=-1, multi_label=False, num_labels=None, mode=mode, vocab_size=vocab_size)
        else: # clf mode - binary classification
            loss_fn = BinaryLoss()
            metrics_fn = MaskedAUC(mask=-1, multi_label=False, num_labels=None, mode=mode, vocab_size=vocab_size)

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

                # Extract inputs and targets from batch dictionary (mode-specific)
                x_smiles = batch['smiles'].to(device)
                x_protein = batch['protein'].to(device)

                if mode == "mlm":
                    # MLM: binds contains (seq_len, 2) targets
                    y = batch['binds'].to(device)
                elif mode == "fps":
                    # FPS: predict ECFP fingerprints
                    y = batch['ecfp'].to(device)
                else:  # clf
                    # CLF: predict binding labels
                    y = batch['binds'].to(device)

                # Forward pass
                optimizer.zero_grad()
                y_pred = belka_model(x_smiles, x_protein)
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

                    # Extract inputs and targets (mode-specific)
                    x_smiles = batch['smiles'].to(device)
                    x_protein = batch['protein'].to(device)

                    if mode == "mlm":
                        # MLM: binds contains (seq_len, 2) targets
                        y = batch['binds'].to(device)
                    elif mode == "fps":
                        # FPS: predict ECFP fingerprints
                        y = batch['ecfp'].to(device)
                    else:  # clf
                        # CLF: predict binding labels
                        y = batch['binds'].to(device)

                    # Forward pass
                    y_pred = belka_model(x_smiles, x_protein)
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

        print(f"[{mode}] fine-tuning completed. Best val_loss: {best_val_loss:.4f}")

        # --- 7. Save final model and metrics ---
        print(f"\nFine-tuning completed. Best validation loss: {best_val_loss:.4f}")
        best_val_loss_overall[mode] = best_val_loss
        best_checkpoint_overall[mode] = best_checkpoint_path


        # Load the best checkpoint to use for the next training mode
        if best_checkpoint_path:
            print(f"[{mode}] Loading best checkpoint {best_checkpoint_path} for next phase")
            state_dict = torch.load(best_checkpoint_path, map_location=device)
            belka_model.load_state_dict(state_dict)


    # Copy best CLF checkpoint to main model output path after all modes have been trained
    # Use CLF checkpoint since that's the final classification head used for inference
    if "clf" in best_checkpoint_overall and best_checkpoint_overall["clf"]:
        final_model_path = checkpoint_dir / "model.pt"
        import shutil
        shutil.copy(best_checkpoint_overall["clf"], final_model_path)
        print(f"Best CLF model saved to {final_model_path} for inference")

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
    val_metrics.log_metric("finetuned_from_model", pretrained_model_id)

    print("Fine-tuning component completed successfully!")
