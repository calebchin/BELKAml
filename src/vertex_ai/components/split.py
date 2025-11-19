from kfp.v2.dsl import component, Input, Output, Dataset
from typing import Optional


@component(
    base_image="python:3.12",
    packages_to_install=["pandas", "scikit-learn", "pyarrow"],
)
def split_train_val_test_gcs(
    data: Input[Dataset],
    train_data: Output[Dataset],
    val_data: Output[Dataset],
    test_data: Output[Dataset],
    test_size: float = 0.0,  # Changed default to 0.0 (no test split by default)
    val_size: float = 0.1,
    stratify_column: Optional[str] = None,
    random_state: int = 42,
) -> None:
    """Splits a dataset into train, validation, and optionally test sets.

    Parameters
    ----------
    data : Input[Dataset]
        The preprocessed dataset artifact (CSV or Parquet) to split.
    train_data : Output[Dataset]
        Output artifact for the training subset.
    val_data : Output[Dataset]
        Output artifact for the validation subset.
    test_data : Output[Dataset]
        Output artifact for the test subset (only created if test_size > 0).
    test_size : float, optional
        Fraction of data to allocate to the test set (default 0.0 = no test split).
    val_size : float, optional
        Fraction of the remaining data to allocate to the validation set (default 0.1).
    stratify_column : str, optional
        Column name to use for stratified splitting (useful for classification tasks).
    random_state : int, optional
        Random seed for reproducibility (default 42).

    Returns
    -------
    None
        The train, validation, and optionally test sets are saved as separate output artifacts on GCS.

    Notes
    -----
    - When test_size=0.0, only train/val split is performed (test data will be separate)
    - When test_size>0.0, performs train/val/test split

    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from pathlib import Path
    import logging

    data_path = Path(data.path)
    if data_path.is_dir():
        # Read all Parquet files in directory (from sharded BQ export)
        df = pd.read_parquet(data_path)
        output_format = "parquet"
    elif data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
        output_format = "parquet"
    elif data_path.suffix == ".csv":
        df = pd.read_csv(data_path)
        output_format = "csv"
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")

    # If test_size is 0, skip test split and only do train/val
    if test_size > 0.0:
        # Three-way split: train, val, test
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df[stratify_column] if stratify_column else None,
            random_state=random_state,
        )

        val_relative_size = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_relative_size,
            stratify=train_val_df[stratify_column] if stratify_column else None,
            random_state=random_state,
        )

        splits = [train_df, val_df, test_df]
        outputs = [train_data, val_data, test_data]
    else:
        # Two-way split: train, val only (test data is separate)
        train_df, val_df = train_test_split(
            df,
            test_size=val_size,
            stratify=df[stratify_column] if stratify_column else None,
            random_state=random_state,
        )

        # Create empty dataframe for test_data output (required by KFP signature)
        test_df = pd.DataFrame()

        splits = [train_df, val_df, test_df]
        outputs = [train_data, val_data, test_data]

    for split_df, split_data in zip(splits, outputs):
        split_path = Path(split_data.path)
        split_path.parent.mkdir(parents=True, exist_ok=True)

        if output_format == "parquet":
            split_df.to_parquet(split_path, index=False)
        else:
            split_df.to_csv(split_path, index=False)

        logging.info(f"Dataset saved to GCS at {split_path} ({len(split_df)} rows)")
