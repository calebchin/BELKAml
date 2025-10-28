from kfp.v2.dsl import component, Input, Output, Dataset
from typing import Optional


@component(
    base_image="python:3.10",
    packages_to_install=["pandas", "scikit-learn"],
)
def split_train_val_test_gcs(
    data: Input[Dataset],
    train_data: Output[Dataset],
    val_data: Output[Dataset],
    test_data: Output[Dataset],
    test_size: float = 0.2,
    val_size: float = 0.1,
    stratify_column: Optional[str] = None,
    random_state: int = 42,
) -> None:
    """
    Splits a dataset into train, validation, and test sets.

    Parameters
    ----------
    data : Input[Dataset]
        The preprocessed dataset artifact (CSV or Parquet) to split.
    train_data : Output[Dataset]
        Output artifact for the training subset.
    val_data : Output[Dataset]
        Output artifact for the validation subset.
    test_data : Output[Dataset]
        Output artifact for the test subset.
    test_size : float, optional
        Fraction of data to allocate to the test set (default 0.2).
    val_size : float, optional
        Fraction of the remaining data to allocate to the validation set (default 0.1).
    stratify_column : str, optional
        Column name to use for stratified splitting (useful for classification tasks).
    random_state : int, optional
        Random seed for reproducibility (default 42).

    Returns
    -------
    None
        The train, validation, and test sets are saved as separate output artifacts on GCS.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from pathlib import Path
    import logging

    data_path = Path(data.path)
    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    elif data_path.suffix == ".csv":
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")

    # exclude one of the building blocks
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

    for split_df, split_data in zip(
        [train_df, val_df, test_df], [train_data, val_data, test_data]
    ):
        split_path = Path(split_data.path)
        split_path.parent.mkdir(parents=True, exist_ok=True)

        if data_path.suffix == ".parquet":
            split_df.to_parquet(split_path, index=False)
        else:
            split_df.to_csv(split_path, index=False)

        logging.info(f"Dataset saved to GCS at {split_path} ({len(split_df)} rows)")
