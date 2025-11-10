from kfp.v2.dsl import component, Input, Output, Dataset

# Rule of thumb: Any feature engineering or preprocessing should be done in the pipeline instead of
# while ingesting into BigQuery because for live predictions, you need the same feature engineering
# logic.

# Split components for re-use. For example, preprocessing is used in both the training and batch
# prediction pipelines.
# GCS for intermediate artifact storage as default. Use BigQuery if you need SQL functionality.

# Best solution: package preprocessing logic into model!
# Alternatives are to duplicate the preprocessing logic in KFP and Vertex AI Endpoints.
# (Issue is that the environments are very different, so you can't just reuse the preprocessing
# KFP component in Vertex AI Endpoints.)


@component(base_image="python:3.12", packages_to_install=["pandas"])
def preprocess_gcs(
    raw_data: Input[Dataset],
    data: Output[Dataset],
) -> None:
    """Preprocesses the raw dataset by dropping unnecessary columns.

    This component reads the data exported from BigQuery in Parquet or CSV format, drops any
    unnecessary columns, and writes the dataset to the output artifact location.

    Parameters
    ----------
    raw_data : Input[Dataset]
        Input dataset artifact representing the extracted raw data (from BigQuery → GCS).
    data : Output[Dataset]
        Output dataset artifact representing the preprocessed data.

    Returns
    -------
    None
        The preprocessed dataset is saved to the output artifact's GCS URI.

    """
    import logging
    import pandas as pd
    from pathlib import Path

    raw_data_path = Path(raw_data.path)

    if raw_data_path.suffix == ".parquet":
        df = pd.read_parquet(raw_data_path)
    elif raw_data_path.suffix == ".csv":
        df = pd.read_csv(raw_data_path)
    else:
        raise ValueError(f"Unsupported file format: {raw_data_path.suffix}")

    # Instead of strings in the molecule_smiles and protein_smiles columns, use an string to
    # integer mapping.
    # Keep the mapping file somewhere as a single source of truth and update it dynamically whenever
    # we see a new target protein.
    # Only need to read this mapping in the smiles to fingerprint encoder.
    df = df[["molecule_smiles", "protein", "binds"]]
    df = df.dropna()

    data_path = Path(data.path)
    data_path.parent.mkdir(parents=True, exist_ok=True)

    if raw_data_path.suffix == ".parquet":
        df.to_parquet(data_path, index=False)
    else:
        df.to_csv(data_path, index=False)

    logging.info(f"Dataset saved to GCS at {data_path}")
