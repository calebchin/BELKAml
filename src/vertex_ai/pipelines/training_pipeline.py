from kfp.v2.dsl import pipeline

from vertex_ai.components.ingest import extract_bq_to_gcs
from vertex_ai.components.preprocess import preprocess_gcs
from vertex_ai.components.split import split_train_val_test_gcs


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


@pipeline(name="BELKAml-train-pipeline", pipeline_root="gs://my-bucket/pipeline_root/")
def train_pipeline(bq_project_id: str, bq_dataset_id: str, bq_table_id: str):
    # Step 1: Ingest
    ingest_task = extract_bq_to_gcs(
        bq_project_id=bq_project_id,
        bq_dataset_id=bq_dataset_id,
        bq_table_id=bq_table_id,
    )

    # Step 2: Preprocess
    preprocess_task = preprocess_gcs(raw_data=ingest_task.outputs["raw_data"])

    # Step 3: Split
    split_task = split_train_val_test_gcs(
        data=preprocess_task.outputs["data"],
        test_size=0.2,
        val_size=0.1,
        stratify_column="protein_smiles",  # TODO: Is this right?
    )

    # Step 4: Train
    # ...

    # Step 5: Test
    # ...

    # Step 6: Deploy
    # ...


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(pipeline_func=train_pipeline, package_path="train_pipeline.yaml")
