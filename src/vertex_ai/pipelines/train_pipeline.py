from kfp.v2.dsl import pipeline

from vertex_ai.components.ingest import extract_bq_to_gcs
from vertex_ai.components.preprocess import preprocess_gcs
from vertex_ai.components.split import split_train_val_test_gcs
from vertex_ai.components.train import train_model
from vertex_ai.components.test import test_model
from vertex_ai.components.deploy import deploy_model_to_aip

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
#  6. Deployment.


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

    # Step 2: Preprocess
    preprocess_task = preprocess_gcs(raw_data=ingest_task.outputs["raw_data"])

    # Step 3: Split (train/val only, test data is separate)
    split_task = split_train_val_test_gcs(
        data=preprocess_task.outputs["data"],
        test_size=0.0,  # No test split (test data is separate)
        val_size=0.1,
        stratify_column=stratify_column,
    )

    # Step 4: Train
    # Training parameters are loaded from config file in GCS: gs://belkamlbucket/configs/vertex_train_config.yaml
    train_task = train_model(
        train_data=split_task.outputs["train_data"],
        val_data=split_task.outputs["val_data"],
        config_path="gs://belkamlbucket/configs/vertex_train_config.yaml",
        target_column=target_column,
    )

    # Step 5: Test
    test_task = test_model(
        test_data=split_task.outputs["test_data"],
        model=train_task.outputs["model"],
        batch_size=1024,
        target_column=target_column,
    )

    # Step 6: Deploy
    deploy_task = deploy_model_to_aip(
        aipproject_id=aip_project_id,
        aipproject_location=aip_project_location,
        model=train_task.outputs["model"],
        train_metrics=train_task.outputs["train_metrics"],
        val_metrics=train_task.outputs["val_metrics"],
        test_metrics=test_task.outputs["test_metrics"],
    )
