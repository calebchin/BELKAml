from typing import List, Optional

from kfp.dsl import pipeline

from google_cloud_pipeline_components.v1.custom_job import create_custom_training_job_from_component

from vertex_ai.components.ingest import extract_bq_to_gcs
from vertex_ai.components.preprocess import preprocess_gcs
from vertex_ai.components.split import split_train_val_test_gcs
from vertex_ai.components.test import test_model
from vertex_ai.components.train import train_model

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
    experiment_batch_ids: List[
        int
    ],  # input field in the Vertex AI Console (UI) accepts either 1,2,3 or [1, 2, 3]
):
    # Step 1: Ingest
    ingest_task = extract_bq_to_gcs(
        bq_project_id=bq_project_id,
        bq_project_location=bq_project_location,
        bq_dataset_id=bq_dataset_id,
        bq_table_id=bq_table_id,
        experiment_batch_ids=experiment_batch_ids,
    )

    # Step 2: Preprocess (includes tokenization and ECFP computation)
    # Uses default vocab_gcs_path and max_length from component
    preprocess_task = preprocess_gcs(raw_data=ingest_task.outputs["raw_data"])
    # Set memory for preprocessing with chunked processing
    preprocess_task.set_memory_limit("32G")
    preprocess_task.set_cpu_limit("16")

    # Step 3: Split (train/val only, test data is separate)
    # split_task = split_train_val_test_gcs(
    #     data=preprocess_task.outputs["data"],
    #     test_size=0.1,  # No test split (test data is separate)
    #     val_size=0.1,
    #     stratify_column=stratify_column,
    # )
    # Set memory for splitting large datasets
    # split_task.set_memory_limit("64G")
    # split_task.set_cpu_limit("16")
    # split_task.set_ephemeral_storage_limit("500G")
    split_op = create_custom_training_job_from_component(
        component_spec=split_train_val_test_gcs,
        display_name="split-dataset-large-disk",
        machine_type="e2-standard-16", 
        boot_disk_type="pd-ssd",          
        boot_disk_size_gb=1000             
    )

    # 2. RUN the new op
    # Note: We remove .set_memory_limit/.set_cpu_limit because 
    # the 'machine_type' above already handles that.
    split_task = split_op(
        data=preprocess_task.outputs["data"],
        test_size=0.1,
        val_size=0.1,
        stratify_column=stratify_column
    )


    # Step 4: Train
    # Training parameters are loaded from config file in GCS: gs://belkamlbucket/configs/vertex_train_config.yaml
    train_task = train_model(
        train_data=split_task.outputs["train_data"],
        val_data=split_task.outputs["val_data"],
        config_path="gs://belkamlbucket/configs/vertex_train_config.yaml",
        target_column=target_column,
    )

    train_task.set_memory_limit("256G")
    train_task.set_cpu_limit("8")
    train_task.set_accelerator_type("NVIDIA_L4")
    train_task.set_accelerator_limit(2)

    # Step 5: Test
    test_task = test_model(
        test_data=split_task.outputs["test_data"],
        model=train_task.outputs["model"],
        batch_size=1024,
        target_column=target_column,
    )
    test_task.set_memory_limit("128G")
    test_task.set_cpu_limit("8")
    test_task.set_accelerator_type("NVIDIA_L4")
    test_task.set_accelerator_limit(1)

    # Model artifacts are automatically saved to GCS by KFP at:
    # gs://belkaml_pipeline_artifacts/{pipeline_run_id}/train-model_{task_id}/model/model.pt
    # Registration and deployment are done separately after reviewing test metrics
