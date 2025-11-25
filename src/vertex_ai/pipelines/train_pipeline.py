from kfp.v2.dsl import pipeline

from vertex_ai.components.ingest import extract_bq_to_gcs
from vertex_ai.components.preprocess import preprocess_gcs
from vertex_ai.components.split import split_train_val_test_gcs
from vertex_ai.components.train import train_model
from vertex_ai.components.test import test_model

from typing import Optional

# https://github.com/GoogleCloudPlatform/vertex-pipelines-end-to-end-samples/blob/main/pipelines/src/pipelines/xgboost/training/pipeline.py
#
# Before this pipeline:
#  1. Data as Parquet on GCS.
#  2. ETL (via DataFlow or Cloud Functions) conducts any transformations.
#  3. ETL loads data into BigQuery
#
# During this pipeline:
#  1. Ingestion.
#  2. Preprocessing.
#  3. Splitting.
#  4. Training.
#  5. Testing.

# Model artifacts are saved to GCS automatically by KFP.
# Registration and deployment are done via separate scripts after reviewing metrics.


@pipeline(
    name="BELKAml-train-pipeline", pipeline_root="gs://belkaml_pipeline_artifacts"
)
def train_pipeline(
    bq_project_id: str,
    bq_project_location: str,
    bq_dataset_id: str,
    bq_table_id: str,
    aip_project_id: str,
    aip_project_location: str,
    stratify_column: Optional[str],
    target_column: str,
):
    # Step 1: Ingest
    ingest_task = extract_bq_to_gcs(
        bq_project_id=bq_project_id,
        bq_project_location=bq_project_location,
        bq_dataset_id=bq_dataset_id,
        bq_table_id=bq_table_id,
    )

    # Step 2: Preprocess (includes tokenization and ECFP computation)
    # Uses default vocab_gcs_path and max_length from component
    preprocess_task = preprocess_gcs(raw_data=ingest_task.outputs["raw_data"])
    # Set memory for preprocessing with chunked processing
    preprocess_task.set_memory_limit('32G')
    preprocess_task.set_cpu_limit('16')

    # Step 3: Split (train/val only, test data is separate)
    split_task = split_train_val_test_gcs(
        data=preprocess_task.outputs["data"],
        test_size=0.1,  # No test split (test data is separate)
        val_size=0.1,
        stratify_column=stratify_column,
    )
    # Set memory for splitting large datasets
    split_task.set_memory_limit('32G')
    split_task.set_cpu_limit('4')

    # Step 4: Train
    # Training parameters are loaded from config file in GCS: gs://belkamlbucket/configs/vertex_train_config.yaml
    train_task = train_model(
        train_data=split_task.outputs["train_data"],
        val_data=split_task.outputs["val_data"],
        config_path="gs://belkamlbucket/configs/vertex_train_config.yaml",
        target_column=target_column,
    )
    # Set higher memory for model training
    train_task.set_memory_limit('32G')
    train_task.set_cpu_limit('8')

    # Step 5: Test
    test_task = test_model(
        test_data=split_task.outputs["test_data"],
        model=train_task.outputs["model"],
        batch_size=1024,
        target_column=target_column,
    )

    # Model artifacts are automatically saved to GCS by KFP at:
    # gs://belkaml_pipeline_artifacts/{pipeline_run_id}/train-model_{task_id}/model/model.pt
    # Registration and deployment are done separately after reviewing test metrics
