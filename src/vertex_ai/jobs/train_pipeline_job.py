from google.cloud import aiplatform as aip
from kfp.v2 import compiler
from vertex_ai.pipelines.train_pipeline import train_pipeline

args = {
    "bq_project_id": "belkaml",
    "bq_project_location": "US",
    "bq_dataset_id": "belka_train_dataset",
    "bq_table_id": "all_data",
    "aip_project_id": "belkaml",
    "aip_project_location": "us-central1-a",
    "stratify_column": "protein_name",
    "target_column": "binds",
}

aip.init(project=args["aip_project_id"], location=args["aip_project_location"])

# 1. Compile pipeline.
compiler.Compiler().compile(
    pipeline_func=train_pipeline, package_path="train_pipeline.yaml"
)

# 2. Run pipeline.
# job = aip.PipelineJob(
#     display_name="train-pipeline-job",
#     template_path="train_pipeline.yaml",
#     pipeline_root="gs://belkaml_pipeline_artifacts",
#     parameter_values=args,
# )
# job.submit()

# Vertex AI:
#  * Compute Engine default service account.
#  * Cloud Storage for storing artifacts.
#  * Vertex ML Metadata to store metadata for pipeline runs.

# Workflow is:
# GA runs this Python file, which compiles the pipeline to a local train_pipeline.yaml file.
# Then, GA uses the gcloud CLI to upload train_pipeline.yaml to GCS.
# Then, we configure Vertex AI to run this yaml file on GCS (see
# https://docs.cloud.google.com/vertex-ai/docs/pipelines/run-pipeline).
