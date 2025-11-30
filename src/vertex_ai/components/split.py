@component(
    base_image="python:3.12",
    packages_to_install=["pandas", "ray[data]", "pyarrow"],
)
def split_train_val_test_gcs(
    data: Input[Dataset],
    train_data: Output[Dataset],
    val_data: Output[Dataset],
    test_data: Output[Dataset],
    test_size: float = 0.0,
    val_size: float = 0.1,
    stratify_column: Optional[str] = None,
    random_state: int = 42,
    mem_size_gb: int = 64, # Pass this in from pipeline definition
) -> None:
    import ray
    import logging
    from pathlib import Path
    import os

    # 2. Dynamic Memory Calculation
    # Convert GB to bytes. Ray usually needs a lot of object store memory.
    object_store_memory = int(0.6 * mem_size_gb * 1024 * 1024 * 1024)
    
    ray.init(
        ignore_reinit_error=True,
        object_store_memory=object_store_memory,
        # storage="/tmp/ray" # Explicitly ensure spilling goes to the boot disk
    )

    logging.info("Reading dataset...")
    # Lazy read - does not load data yet
    ds = ray.data.read_parquet(data.path)

    # 3. REMOVED: ds = ds.materialize()
    # This was doubling your storage requirement (Input Copy + Shuffled Copy).
    
    if stratify_column:
        logging.warning(f"Ignoring stratify_column='{stratify_column}'...")

    logging.info("Shuffling dataset...")
    # random_shuffle will trigger the read and spill to disk as needed.
    ds = ds.random_shuffle(seed=random_state)

    logging.info("Splitting dataset...")
    if test_size > 0:
        train_size = 1 - test_size - val_size
        train_ds, val_ds, test_ds = ds.split_proportionately([train_size, val_size])
    else:
        train_ds, val_ds = ds.split_proportionately([1 - val_size])
        test_ds = None

    logging.info("Writing split data out...")
    train_ds.write_parquet(train_data.path)
    val_ds.write_parquet(val_data.path)
    
    if test_ds:
        test_ds.write_parquet(test_data.path)
    else:
        Path(test_data.path).mkdir(parents=True, exist_ok=True)
        ray.data.from_items([]).write_parquet(test_data.path)