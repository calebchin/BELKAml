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


@component(
        base_image="northamerica-northeast2-docker.pkg.dev/belkaml/belka-repo/belkaml-trainer:latest",
        packages_to_install=["ray[data]", "pyarrow"]
)
def preprocess_gcs(
    raw_data: Input[Dataset],
    data: Output[Dataset],
    vocab_gcs_path: str = "gs://belkamlbucket/data/raw/vocab.txt",
    max_length: int = 128,
) -> None:
    """Preprocesses the raw dataset by dropping unnecessary columns and computing features.

    This component reads the data exported from BigQuery in Parquet or CSV format, drops any
    unnecessary columns, tokenizes SMILES strings, computes ECFP fingerprints, and writes
    the dataset to the output artifact location.

    Parameters
    ----------
    raw_data : Input[Dataset]
        Input dataset artifact representing the extracted raw data (from BigQuery → GCS).
    data : Output[Dataset]
        Output dataset artifact representing the preprocessed data.
    vocab_gcs_path : str
        GCS path to vocabulary file for SMILES tokenization (default: gs://belkamlbucket/data/raw/vocab.txt).
    max_length : int
        Maximum sequence length for tokenized SMILES (default: 128).

    Returns
    -------
    None
        The preprocessed dataset is saved to the output artifact's GCS URI.

    """
    import logging
    import ray
    import numpy as np
    from google.cloud import storage

    # import pandas as pd
    # import numpy as np
    # from pathlib import Path
    # from typing import List
    # import atomInSmiles
    # from utils.torch_data_utils import SMILESTokenizer
    # from utils.torch_data_utils import ECFPFingerprint
    # from google.cloud import storage

    ray.init(ignore_reinit_error=True)
    # Download vocab file from GCS if needed
    vocab_path = "/tmp/vocab.txt"
    if vocab_gcs_path.startswith("gs://"):
        bucket_name = vocab_gcs_path.split("/")[2]
        blob_path = "/".join(vocab_gcs_path.split("/")[3:])
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.download_to_filename(vocab_path)
    else:
        vocab_path = vocab_gcs_path
    # class to be loaded once per CPU core
    
    class PreProcessor:
        def __init__(self):
            from utils.torch_data_utils import SMILESTokenizer, ECFPFingerprint
            # Initialize expensive objects once per worker
            self.tokenizer = SMILESTokenizer(vocab_path)
            self.ecfp_transformer = ECFPFingerprint(fp_size=2048)
            self.max_length = max_length
            
            # Cache special tokens
            self.pad_token = self.tokenizer.token_to_id["[PAD]"]
            self.unk_token = self.tokenizer.token_to_id.get("[UNK]", 2)

        def __call__(self, batch: dict) -> dict:
            import atomInSmiles
            
            # Access numpy arrays from the batch
            smiles_list = batch["molecule_smiles"]
            
            token_ids_list = []
            ecfp_list = []

            for smile in smiles_list:
                # --- Tokenization Logic ---
                try:
                    tokens = atomInSmiles.smiles_tokenizer(smile)
                    ids = [
                        self.tokenizer.token_to_id.get(t, self.unk_token)
                        for t in tokens
                    ]
                    # Truncate/Pad
                    ids = ids[:self.max_length]
                    ids = ids + [self.pad_token] * (self.max_length - len(ids))
                    token_ids_list.append(ids)
                except Exception:
                    token_ids_list.append([self.pad_token] * self.max_length)

                # --- ECFP Logic ---
                try:
                    # ECFP transformer expects a list
                    fp = self.ecfp_transformer.transform([smile])[0]
                    ecfp_list.append(fp)
                except Exception:
                    ecfp_list.append(np.zeros(2048, dtype=np.float32))

            return {
                "token_ids": token_ids_list, 
                "ecfp": ecfp_list
            }
    #raw_data_path = Path(raw_data.path)
    logging.info("Starting Ray data pipeline...")
    # Handle both single file and directory of sharded files
    # if raw_data_path.is_dir():
    #     # Read all Parquet files in directory (from sharded BQ export)
    #     df = pd.read_parquet(raw_data_path)
    #     output_format = "parquet"
    # elif raw_data_path.suffix == ".parquet":
    #     df = pd.read_parquet(raw_data_path)
    #     output_format = "parquet"
    # elif raw_data_path.suffix == ".csv":
    #     df = pd.read_csv(raw_data_path)
    #     output_format = "csv"
    # else:
    #     raise ValueError(f"Unsupported file format: {raw_data_path.suffix}")
    ds = ray.data.read_parquet(raw_data.path)
    ds = ds.filter(lambda row: row["molecule_smiles"] is not None)
    
    # Apply transformations in parallel
    # concurrency=None defaults to using all available CPUs
    processed_ds = ds.map_batches(
        PreProcessor, 
        concurrency=None, 
        batch_size=4096  # Adjust based on memory
    )

    processed_ds.write_parquet(data.path)

    # def add_masked_smiles(df: pd.DataFrame) -> pd.DataFrame:
    #     """Adds a 'masked_smiles' column to the dataframe by masking target protein substructures.

    #     Parameters
    #     ----------
    #     df : pd.DataFrame
    #         Input dataframe with a 'molecule_smiles' column.

    #     Returns
    #     -------
    #     pd.DataFrame
    #         Dataframe with an additional 'masked_smiles' column.

    #     """
    #     from utils.smiles_tokenizer import SMILESTokenizer

    #     tokenizer = SMILESTokenizer()

    #     def mask_smiles(smiles: str) -> str:
    #         tokens = tokenizer.tokenize(smiles)
    #         masked_tokens = ['[MASK]' if token in tokenizer.target_tokens else token for token in tokens]
    #         return ''.join(masked_tokens)

    #     df['masked_smiles'] = df['molecule_smiles'].apply(mask_smiles)
    #     return df
    # Instead of strings in the molecule_smiles and protein_smiles columns, use an string to
    # integer mapping.
    # Keep the mapping file somewhere as a single source of truth and update it dynamically whenever
    # we see a new target protein.
    # df = add_masked_smiles(df)
    # df = add_ecfp(df)
    # Only need to read this mapping in the smiles to fingerprint encoder.
    # df = df[["molecule_smiles", "protein_name", "binds"]]
    # df = df.dropna()

    # # Add tokenization
    # logging.info("Tokenizing SMILES strings...")
    # tokenizer = SMILESTokenizer(vocab_path)
    # ecfp_transformer = ECFPFingerprint(fp_size=2048)

    # def tokenize_smiles(smiles: str) -> List[int]:
    #     """Tokenize a SMILES string and convert to token IDs with padding."""
    #     try:
    #         tokens = atomInSmiles.smiles_tokenizer(smiles)
    #         token_ids = [
    #             tokenizer.token_to_id.get(token, tokenizer.token_to_id.get("[UNK]", 2))
    #             for token in tokens
    #         ]
    #         # Truncate or pad to max_length
    #         token_ids = token_ids[:max_length]
    #         token_ids = token_ids + [tokenizer.token_to_id["[PAD]"]] * (
    #             max_length - len(token_ids)
    #         )
    #         return token_ids
    #     except Exception as e:
    #         logging.warning(f"Failed to tokenize SMILES '{smiles}': {e}")
    #         # Return padding tokens on error
    #         return [tokenizer.token_to_id["[PAD]"]] * max_length

    # def compute_ecfp(smiles: str) -> np.ndarray:
    #     """Compute ECFP fingerprint for a SMILES string."""
    #     try:
    #         return ecfp_transformer.transform([smiles])[0]
    #     except Exception as e:
    #         logging.warning(f"Failed to compute ECFP for SMILES '{smiles}': {e}")
    #         return np.zeros(2048, dtype=np.float32)

    # # Process in chunks to avoid memory issues with large datasets
    # data_dir = Path(data.path)
    # data_dir.mkdir(parents=True, exist_ok=True)

    # chunk_size = 1_000_000  # Process 1M rows at a time
    # total_rows = len(df)
    # logging.info(f"Processing {total_rows:,} rows in chunks of {chunk_size:,}")

    # if output_format == "parquet":
    #     # Write each chunk to a separate parquet file to avoid Arrow list overflow
    #     chunk_files = []

    #     for i in range(0, total_rows, chunk_size):
    #         chunk_df = df.iloc[i:i+chunk_size].copy()
    #         chunk_num = i // chunk_size
    #         logging.info(f"Processing chunk {chunk_num + 1}/{(total_rows + chunk_size - 1)//chunk_size} (rows {i:,} to {min(i+chunk_size, total_rows):,})")

    #         # Apply transformations to chunk
    #         chunk_df["token_ids"] = chunk_df["molecule_smiles"].apply(tokenize_smiles)
    #         chunk_df["ecfp"] = chunk_df["molecule_smiles"].apply(compute_ecfp)

    #         # Write chunk to separate parquet file
    #         chunk_file = data_dir / f"data_chunk_{chunk_num}.parquet"
    #         chunk_df.to_parquet(chunk_file, index=False)
    #         chunk_files.append(chunk_file)

    #         del chunk_df  # Free memory

    #     logging.info(f"Wrote {len(chunk_files)} chunk files. Downstream components will read all parquet files from directory.")

    # else:
    #     # CSV can append efficiently
    #     file_path = data_dir / "data.csv"
    #     first_chunk = True

    #     for i in range(0, total_rows, chunk_size):
    #         chunk_df = df.iloc[i:i+chunk_size].copy()
    #         logging.info(f"Processing chunk {i//chunk_size + 1}/{(total_rows + chunk_size - 1)//chunk_size} (rows {i:,} to {min(i+chunk_size, total_rows):,})")

    #         # Apply transformations to chunk
    #         chunk_df["token_ids"] = chunk_df["molecule_smiles"].apply(tokenize_smiles)
    #         chunk_df["ecfp"] = chunk_df["molecule_smiles"].apply(compute_ecfp)

    #         # Write chunk to CSV
    #         chunk_df.to_csv(file_path, mode='w' if first_chunk else 'a', header=first_chunk, index=False)
    #         first_chunk = False

    #         del chunk_df  # Free memory

    logging.info(f"Preprocessing complete. Dataset saved to {data.path}")

