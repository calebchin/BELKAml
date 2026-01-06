from kfp.v2.dsl import pipeline
from vertex_ai.components.deploy import deploy_model_endpoint

@pipeline(
    name="belka-deploy-pipeline",
    pipeline_root="gs://belkaml_pipeline_artifacts"
)
def deploy_pipeline(
    aipproject_id: str,
    aipproject_location: str,
    model_id: str,
    serving_container_image_uri: str,
    endpoint_display_name: str = "belka-protein-binding-endpoint",
    machine_type: str = "n1-standard-4",
):
    """
    Pipeline to deploy a registered model to an endpoint.
    """
    deploy_task = deploy_model_endpoint(
        aipproject_id=aipproject_id,
        aipproject_location=aipproject_location,
        model_id=model_id,
        serving_container_image_uri=serving_container_image_uri,
        endpoint_display_name=endpoint_display_name,
        machine_type=machine_type,
    )
