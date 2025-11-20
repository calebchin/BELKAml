from kfp.v2.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Model,
    Metrics,
    ClassificationMetrics,
)


@component(
    base_image="northamerica-northeast2-docker.pkg.dev/belkaml/belka-repo/belkaml-trainer:latest",
)
def test_model(
    test_data: Input[Dataset],
    model: Input[Model],
    test_metrics: Output[Metrics],
    classification_metrics: Output[ClassificationMetrics],
    config_path: str = "gs://belkamlbucket/configs/vertex_train_config.yaml",
    batch_size: int = 1024,
    target_column: str = "binds",
) -> None:
    """
    Evaluate a PyTorch binary classification model on held-out test data.

    This component computes standard evaluation metrics (AUROC, AUPRC, F1, precision,
    recall, MCC) and logs them to Vertex AI Pipelines. It also logs the ROC curve and
    confusion matrix for visualization. Supports GPU acceleration and batch inference.

    Parameters
    ----------
    test_data : Input[Dataset]
        Test dataset artifact (CSV or Parquet) with features and target column.
    model : Input[Model]
        Trained PyTorch model artifact (.pt or .pth file).
    test_metrics : Output[Metrics]
        Metrics artifact to log scalar evaluation metrics to Vertex AI.
    classification_metrics : Output[ClassificationMetrics]
        Artifact to log ROC curve and confusion matrix for visualization in Vertex AI.
    config_path : str, optional
        GCS path to training config YAML containing model hyperparameters,
        by default 'gs://belkamlbucket/configs/vertex_train_config.yaml'.
    batch_size : int, optional
        Batch size for GPU inference, by default 1024.
    target_column : str, optional
        Name of the target column in the dataset, by default 'binds'.

    Returns
    -------
    None
        The evaluation metrics on the test set are logged to Vertex AI Pipelines.
    """
    import pandas as pd
    import torch
    import numpy as np
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        matthews_corrcoef,
        roc_curve,
        confusion_matrix,
    )
    from pathlib import Path
    import json
    from google.cloud import storage
    import yaml
    from model import Belka

    # Load training config from GCS to get model hyperparameters
    print(f"Loading model config from {config_path}")
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

    # Extract model hyperparameters
    hidden_size = config.get('hidden_size', 32)
    num_layers = config.get('num_layers', 2)
    vocab_size = config.get('vocab_size', 44)
    dropout_rate = config.get('dropout_rate', 0.1)

    print(f"Loaded config: hidden_size={hidden_size}, num_layers={num_layers}, vocab_size={vocab_size}, dropout_rate={dropout_rate}")

    test_data_path = Path(test_data.path)
    if test_data_path.is_dir():
        # Read all files in directory (from sharded output)
        df = pd.read_parquet(test_data_path)
    elif test_data_path.suffix == ".parquet":
        df = pd.read_parquet(test_data_path)
    elif test_data_path.suffix == ".csv":
        df = pd.read_csv(test_data_path)
    else:
        raise ValueError(f"Unsupported file format: {test_data_path.suffix}")

    # Validate that we have test data
    if len(df) == 0:
        raise ValueError("Test dataset is empty. Ensure test_size > 0 in the split component.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Extract token IDs (model expects token IDs, not concatenated features)
    # token_ids is a list of integers from preprocessing
    token_ids_array = np.stack(df["token_ids"].values)
    X_test = torch.tensor(token_ids_array, dtype=torch.long).to(device)

    y_test = np.asarray(df[target_column].values, dtype=float)

    # Instantiate model architecture using config from training
    # Note: Always use 'clf' mode for testing, regardless of training mode
    loaded_model = Belka(
        hidden_size=hidden_size,
        dropout_rate=dropout_rate,
        mode='clf',  # Always use classification mode for testing
        num_layers=num_layers,
        vocab_size=vocab_size,
    )

    # Load state_dict (model weights) from checkpoint
    state_dict = torch.load(model.path, map_location=device)
    loaded_model.load_state_dict(state_dict)

    # Move to device and set eval mode
    loaded_model.to(device)
    loaded_model.eval()

    # batch inference
    y_test_prob_list = []
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch = X_test[i : i + batch_size]
            # Model already applies sigmoid in clf_head, no need to apply again
            batch_prob = loaded_model(batch).cpu().numpy().ravel()
            y_test_prob_list.append(batch_prob)
    y_test_prob = np.concatenate(y_test_prob_list)
    y_test_pred = (y_test_prob > 0.5).astype(int)

    # 1. Log metrics.
    # ---------------

    # log AUROC, AUPRC (AP), F1 score, precision, recall, MCC
    test_metrics_dict = {
        "__TEST_AUROC": float(roc_auc_score(y_test, y_test_prob)),
        "__TEST_AUPRC": float(average_precision_score(y_test, y_test_prob)),
        "__TEST_F1": float(f1_score(y_test, y_test_pred)),
        "__TEST_precision": precision_score(y_test, y_test_pred),
        "__TEST_recall": float(recall_score(y_test, y_test_pred)),
        "__TEST_MCC": float(matthews_corrcoef(y_test, y_test_pred)),
        "__TEST_sample_size": len(y_test),
        "__TEST_positive_ratio": float(y_test.mean()),
        "__TEST_negative_ratio": float(1 - y_test.mean()),
    }

    # log ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
    classification_metrics.log_roc_curve(
        fpr=fpr.tolist(), tpr=tpr.tolist(), threshold=thresholds.tolist()
    )

    # log confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    classification_metrics.log_confusion_matrix(
        categories=["non-binder", "binder"],
        matrix=[[int(tn), int(fp)], [int(fn), int(tp)]],
    )

    # 2. Log PER protein metrics.
    # ---------------------------

    grouped = df.groupby("protein_name")
    for protein, group in grouped:
        # Use boolean indexing to avoid index mismatch with numpy arrays
        mask = df.index.isin(group.index)
        y_test_p = y_test[mask]
        y_test_prob_p = y_test_prob[mask]
        y_test_pred_p = y_test_pred[mask]

        test_metrics_dict = {
            **test_metrics_dict,
            f"__TEST_AUROC_{protein}": float(roc_auc_score(y_test_p, y_test_prob_p)),
            f"__TEST_AUPRC_{protein}": float(
                average_precision_score(y_test_p, y_test_prob_p)
            ),
            f"__TEST_F1_{protein}": float(f1_score(y_test_p, y_test_pred_p)),
            f"__TEST_precision_{protein}": float(
                precision_score(y_test_p, y_test_pred_p)
            ),
            f"__TEST_recall_{protein}": float(recall_score(y_test_p, y_test_pred_p)),
            f"__TEST_MCC_{protein}": float(matthews_corrcoef(y_test_p, y_test_pred_p)),
            f"__TEST_sample_size_{protein}": len(y_test_p),
            f"__TEST_positive_ratio_{protein}": float(y_test_p.mean()),
            f"__TEST_negative_ratio_{protein}": float(1 - y_test_p.mean()),
        }

    for key, val in test_metrics_dict.items():
        test_metrics.log_metric(key, val)

    with open(test_metrics.path, "w") as f:
        json.dump(test_metrics_dict, f)
