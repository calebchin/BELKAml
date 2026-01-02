from kfp.dsl import component, Input, Output, Dataset
from typing import Optional


@component(
    base_image="python:3.12",
    packages_to_install=["pandas", "ray[data]", "pyarrow"],
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
    import ray
    import logging
    from pathlib import Path
    import os

    #os.environ['RAY_DATA_PUSH_BASED_SHUFFLE'] = '1'
    ray.data.DataContext.get_current().execution_options.preserve_order = True
    # Initialize Ray with optimized settings for large datasets
    ray.init(
        ignore_reinit_error=True,
        object_store_memory=int(0.6 * 64 * 1024 * 1024 * 1024)  # 60% of 32GB for object store
    )

    logging.info("Reading and materializing dataset...")
    ds = ray.data.read_parquet(data.path)

    #ds = ds.materialize()

    if stratify_column:
        logging.warning(f"Ignoring stratify_column='{stratify_column}' for large dataset. "
                       f"Using random shuffle instead to avoid memory/disk overflow.")

    logging.info("Shuffling dataset...")
    #ds = ds.random_shuffle(seed=random_state)

    logging.info("Splitting dataset...")
    if test_size > 0:
        #train_size = 1 - test_size - val_size
        train_val_ds, test_ds = ds.streaming_train_test_split(test_size, seed=random_state)
        train_ds, val_ds = train_val_ds.streaming_train_test_split(val_size, seed=random_state)

        #train_ds, val_ds, test_ds = ds.split_proportionately([train_size, val_size])
    else:
        train_ds, val_ds = ds.streaming_train_test_split(val_size, seed=random_state)
        #train_ds, val_ds = ds.split_proportionately([1 - val_size])
        test_ds = None

    logging.info("Writing split data out...")
    train_ds.write_parquet(train_data.path)
    val_ds.write_parquet(val_data.path)
    if test_ds:
        test_ds.write_parquet(test_data.path)
    else:
        Path(test_data.path).mkdir(parents=True, exist_ok=True)
        ray.data.from_items([]).write_parquet(test_data.path)
