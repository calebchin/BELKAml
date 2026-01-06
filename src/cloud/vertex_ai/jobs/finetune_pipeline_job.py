from google.cloud import aiplatform as aip
from kfp import compiler
from vertex_ai.pipelines.finetune_pipeline import finetune_pipeline

# Pipeline parameters (data source and basic config)
# NOTE: pretrained_model_id should be replaced with actual model ID from Model Registry
# Get model ID from: https://console.cloud.google.com/vertex-ai/models
# Or use: gcloud ai models list --region=northamerica-northeast2
args = {
    "pretrained_model_id": "projects/46518343354/locations/northamerica-northeast2/models/YOUR_MODEL_ID",  # UPDATE THIS
    "bq_project_id": "belkaml",
    "bq_project_location": "US",
    "bq_dataset_id": "belka_train_dataset",
    "bq_table_id": "all_data",
    "aip_project_id": "belkaml",
    "aip_project_location": "northamerica-northeast2",
    "stratify_column": "protein_name",
    "target_column": "binds",
}

aip.init(project=args["aip_project_id"], location=args["aip_project_location"])

# 1. Compile pipeline.
compiler.Compiler().compile(
    pipeline_func=finetune_pipeline, package_path="finetune_pipeline.yaml"
)

# 2. Run pipeline (commented out - trigger manually or via Cloud Build).
# IMPORTANT: Update pretrained_model_id in args before running!
# job = aip.PipelineJob(
#     display_name="finetune-pipeline-job",
#     template_path="finetune_pipeline.yaml",
#     pipeline_root="gs://belkaml_pipeline_artifacts",
#     parameter_values=args,
# )
# job.submit()

# Workflow:
# 1. GitHub Actions runs this Python file to compile pipeline to finetune_pipeline.yaml
# 2. Cloud Build uploads finetune_pipeline.yaml to GCS
# 3. Manually trigger pipeline run in Vertex AI console with actual pretrained_model_id
#    Or use: gcloud ai pipelines run --file=gs://belkaml_pipeline_artifacts/finetune_pipeline.yaml
