from kfp.dsl import pipeline


from vertex_ai.components.ingest import extract_bq_to_gcs
from vertex_ai.components.preprocess import preprocess_gcs
from vertex_ai.components.split import split_train_val_test_gcs
from vertex_ai.components.finetune import finetune_model
from vertex_ai.components.test import test_model

from typing import Optional, List


@pipeline(
    name="BELKAml-finetune-pipeline", pipeline_root="gs://belkaml_pipeline_artifacts"
)
def finetune_pipeline(
    pretrained_model_id: str,
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
    ],
):
    """Fine-tune a pretrained Belka model on new data.

    This pipeline takes a model GCS URI from a previous training run and continues training
    on new data. The pretrained encoder/embeddings are loaded while task heads are
    reinitialized, allowing full sequential training (MLM → FPS → CLF).

    Args:
        pretrained_model_id: GCS URI to pretrained model
            (e.g., "gs://belkaml_pipeline_artifacts/{run_id}/{task_id}/model")
        bq_project_id: BigQuery project ID for data ingestion
        bq_project_location: BigQuery location (e.g., "US")
        bq_dataset_id: BigQuery dataset ID
        bq_table_id: BigQuery table ID
        aip_project_id: Vertex AI project ID
        aip_project_location: Vertex AI location (e.g., "northamerica-northeast2")
        stratify_column: Column to stratify splits (e.g., "protein_name")
        target_column: Target column name (default: "binds")

    Pipeline Steps:
        1. Ingest data from BigQuery
        2. Preprocess (tokenization + ECFP computation)
        3. Split into train/val/test sets
        4. Fine-tune pretrained model on new data
        5. Test fine-tuned model

    Model artifacts are saved to GCS automatically by KFP.
    """
    # Step 1: Ingest
    ingest_task = extract_bq_to_gcs(
        bq_project_id=bq_project_id,
        bq_project_location=bq_project_location,
        bq_dataset_id=bq_dataset_id,
        bq_table_id=bq_table_id,
        experiment_batch_ids=experiment_batch_ids
    )

    # Step 2: Preprocess (includes tokenization and ECFP computation)
    preprocess_task = preprocess_gcs(raw_data=ingest_task.outputs["raw_data"])
    # Set higher memory for preprocessing large datasets
    preprocess_task.set_memory_limit('32G')
    preprocess_task.set_cpu_limit('8')

    # Step 3: Split
    split_op = create_custom_training_job_from_component(
        component_spec=split_train_val_test_gcs,
        display_name="split-dataset-large-disk",
        machine_type="e2-standard-16",    # 16 vCPUs, ~64GB RAM
        boot_disk_type="pd-ssd",          # SSD is much faster for Ray shuffling
        boot_disk_size_gb=500             # <--- THE FIX: Request 500GB
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

    # Step 4: Fine-tune (instead of training from scratch)
    finetune_task = finetune_model(
        train_data=split_task.outputs["train_data"],
        val_data=split_task.outputs["val_data"],
        pretrained_model_id=pretrained_model_id,
        aipproject_id=aip_project_id,
        aipproject_location=aip_project_location,
        config_path="gs://belkamlbucket/configs/vertex_train_config.yaml",
        target_column=target_column,
    )
    # Set higher memory for fine-tuning
    finetune_task.set_memory_limit('32G')
    finetune_task.set_cpu_limit('8')

    # Step 5: Test
    test_task = test_model(
        test_data=split_task.outputs["test_data"],
        model=finetune_task.outputs["model"],
        batch_size=1024,
        target_column=target_column,
    )

    # Model artifacts are automatically saved to GCS by KFP at:
    # gs://belkaml_pipeline_artifacts/{pipeline_run_id}/finetune-model_{task_id}/model/model.pt
