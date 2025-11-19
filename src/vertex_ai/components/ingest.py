from kfp.v2.dsl import component, Dataset, Output


@component(
    base_image="python:3.12",
    packages_to_install=["google-cloud-bigquery==3.38.0"],
)
def extract_bq_to_gcs(
    bq_project_id: str,
    bq_project_location: str,
    bq_dataset_id: str,
    bq_table_id: str,
    raw_data: Output[Dataset],
) -> None:
    """Extracts a BigQuery table and saves it as a Parquet file in Google Cloud Storage (GCS).

    This component uses the BigQuery client to export the specified table to a GCS
    destination automatically provided by the output `raw_data` artifact.

    Parameters
    ----------
    bq_project_id : str
        The Google Cloud project ID containing the BigQuery table. A BigQuery client
        will be created in this project to perform the extraction.
    bq_project_location: str
        The Google Cloud project location, e.g. us-central1, containing the BigQuery table.
    bq_dataset_id : str
        The dataset ID within the project that contains the BigQuery table.
    bq_table_id : str
        The table ID (without project ID or dataset ID prefix) identifying the table to extract.
    raw_data : Output[Dataset]
        The output dataset artifact representing the exported table in GCS.
        The `.uri` property of this artifact specifies the GCS destination path.

    Returns
    -------
    None
        The extracted table is saved to the output artifact's GCS URI.

    Raises
    ------
    GoogleCloudError
        If the BigQuery extract job fails.

    Notes
    -----
    - The extracted file format is Parquet.
    - The extraction is performed in the `US` region by default.
    - The destination GCS URI is automatically set by the Kubeflow Pipelines orchestrator.

    """
    import logging
    from google.cloud import bigquery

    client = bigquery.client.Client(project=bq_project_id, location=bq_project_location)
    table = bigquery.table.Table(
        table_ref=f"{bq_project_id}.{bq_dataset_id}.{bq_table_id}"
    )

    job_config = bigquery.job.ExtractJobConfig(destination_format="PARQUET")
    # Use wildcard to shard output into multiple files for large tables
    destination_uri = raw_data.uri.rstrip('/') + "/data-*.parquet"
    extract_job = client.extract_table(
        table,
        destination_uri,
        job_config=job_config,
    )

    try:
        extract_job.result()
        logging.info(f"BQ table extracted to GCS at {raw_data.uri}")
    except Exception as e:
        logging.error(e)
        logging.error(extract_job.error_result)
        logging.error(extract_job.errors)
        raise e
