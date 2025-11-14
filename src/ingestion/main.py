from .pipeline import MoleculeDataPipeline
from google.cloud import storage
from typing import Any
from cloudevents.http import CloudEvent

# Initialize clients globally for efficiency
PIPELINE = MoleculeDataPipeline(config_path="config.yaml")
STORAGE_CLIENT = storage.Client()


def new_data_ingest_entrypoint(cloud_event: CloudEvent) -> None:
    """Cloud Function entry point triggered by a GCS file upload via Eventarc.
    Orchestrates processing, loading, and archiving of the file.

    Args:
        cloud_event (CloudEvent): CloudEvent containing GCS object metadata.

    """
    # Extract GCS event data from CloudEvent
    event_data = cloud_event.get_data()
    bucket_name = event_data["bucket"]
    file_name = event_data["name"]

    target_folder = "uploads/"
    if not file_name.startswith(target_folder):
        PIPELINE.logger.info(
            f"Skipping file '{file_name}' as it's not in the '{target_folder}' folder."
        )
        return

    try:
        archive_bucket_name = PIPELINE.config["gcs"]["archive_bucket_name"]
    except KeyError:
        PIPELINE.logger.error(
            "'archive_bucket_name' not found in config.yaml under 'gcs'."
        )
        raise
    if bucket_name == archive_bucket_name:
        PIPELINE.logger.info(
            f"File '{file_name}' is in the archive bucket. No processing needed."
        )
        return

    try:
        # Process and load data
        processed_df = PIPELINE.process_data(bucket_name, file_name)
        PIPELINE.load_to_bigquery(processed_df)

        # Archive file
        PIPELINE.logger.info(f"Archiving file: {file_name}")
        source_bucket = STORAGE_CLIENT.bucket(bucket_name)
        source_blob = source_bucket.blob(file_name)
        archive_bucket = STORAGE_CLIENT.bucket(archive_bucket_name)

        # Copy the blob to the archive bucket and then delete it from the source
        source_bucket.copy_blob(source_blob, archive_bucket, file_name)
        source_blob.delete()

        PIPELINE.logger.info(
            f"Successfully archived {file_name} to {archive_bucket_name}."
        )

    except Exception as e:
        PIPELINE.logger.error(
            f"Pipeline failed for file {file_name}. Error: {e}", exc_info=True
        )
        raise
