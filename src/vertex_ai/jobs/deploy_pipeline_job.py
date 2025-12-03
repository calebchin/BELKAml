from google.cloud import aiplatform as aip
from kfp.v2 import compiler
from vertex_ai.pipelines.deploy_pipeline import deploy_pipeline

# Configuration
# You can load these from args or environment variables if needed
args = {
    "aip_project_id": "belkaml",
    "aip_project_location": "northamerica-northeast2",
}

aip.init(project=args["aip_project_id"], location=args["aip_project_location"])


compiler.Compiler().compile(
    pipeline_func=deploy_pipeline,
    package_path="deploy_pipeline.yaml"
)
