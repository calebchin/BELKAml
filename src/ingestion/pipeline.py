"""Molecule Binding Data Pipeline
Merges parquet molecule data with CSV binding data and loads to BigQuery
"""

import logging
from typing import Dict, Any

import pandas as pd
import yaml
from google.cloud import bigquery
from google.cloud.exceptions import GoogleCloudError


class MoleculeDataPipeline:
    """Pipeline for processing and merging molecule binding data."""

    def __init__(self, config_path: str):
        """Initialize pipeline with configuration."""
        self.config = self._load_config(config_path)
        self._setup_logging()
        self.client = bigquery.Client(project=self.config["bigquery"]["project"])
        self.logger = logging.getLogger(__name__)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=self.config["logging"]["level"],
            format=self.config["logging"]["format"],
        )

    def process_data(self, bucket_name: str, file_name: str) -> pd.DataFrame:
        """Return processed binding data specified in config with file type `file_type`.

        Returns:
            pd.DataFrame: Processed binding data matching config schema

        """
        self.logger.info(f"Reading file '{file_name}' from bucket '{bucket_name}'.")
        gcs_uri = f"gs://{bucket_name}/{file_name}"
        if file_name.endswith(".csv"):
            new_df = pd.read_csv(gcs_uri)
        elif file_name.endswith(".parquet"):
            new_df = pd.read_parquet(gcs_uri)
        else:
            raise ValueError(
                f"Unsupported file type for {file_name}. Must be .csv or .parquet."
            )

        self.logger.info(f"Loaded {len(new_df)} rows from {file_name}")

        # schema cols as a list
        schema_cols = self.config["data_preprocessing"]["schema_cols"]
        # col_name_map should be a dict mapping col names in csv to schema col names
        #col_name_map = self.config["data_preprocessing"]["col_name_map"]
        # experimental batch num
        experimental_batch_num = self.config["data_preprocessing"][
            "experimental_batch_num"
        ]
        # tz info
        tz = self.config["data_preprocessing"]["timezone"]

        # col_orig = col_name_map.keys()
        # col_schema = col_name_map.values()
        # processed_df = new_df.loc[:, col_orig].rename(columns=col_schema)
        # self.logger.info(f"Renamed columns: {col_schema}")
        processed_df = new_df[schema_cols]

        processed_df["experimental_batch"] = experimental_batch_num
        self.logger.info(f"Ingested data has batch number: {experimental_batch_num}")

        stamped_time = pd.Timestamp.now(tz=tz)
        self.logger.info(f"Timestamp for ingestion: {stamped_time}")
        processed_df["timestamp"] = stamped_time

        self.logger.info("Processed new data successfully.")
        return new_df

    def load_to_bigquery(self, df: pd.DataFrame) -> None:
        """Loads and appends DataFrame df to Bigquery Table"""
        table_id = f"{self.config['bigquery']['project']}.{self.config['bigquery']['dataset']}.{self.config['bigquery']['table']}"
        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        )
        try:
            job = self.client.load_table_from_dataframe(
                df, table_id, job_config=job_config
            )
            job.result()
            self.logger.info(f"Loaded {len(df)} new rows to BigQuery table {table_id}.")
        except GoogleCloudError as e:
            self.logger.error(f"Failed to load data to BigQuery: {e}")
            if hasattr(e, "errors"):
                for error in e.errors:
                    self.logger.error(f"Error details: {error}")
            raise
