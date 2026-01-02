from kfp.dsl import component, Input, Output, Artifact


@component(base_image="python:3.12", packages_to_install=["google-cloud-aiplatform"])
def deploy_model_endpoint(
    aipproject_id: str,
    aipproject_location: str,
    model_id: str,
    serving_container_image_uri: str,
    endpoint_display_name: str = "belka-protein-binding-endpoint",
    machine_type: str = "n1-standard-4",
    min_replica_count: int = 1,
    max_replica_count: int = 1,
):
    """Deploy a registered model to a Vertex AI endpoint for online predictions.

    This component takes a model that was previously registered in the Model Registry
    and deploys it to a serving endpoint with your custom serving container.

    Prerequisites:
    - Model must be registered in Model Registry (use register.py component first)
    - Custom serving container must be built and pushed to GCR/Artifact Registry

    Args:
        aipproject_id: GCP project ID
        aipproject_location: GCP region (e.g., 'northamerica-northeast2')
        model_id: The model resource ID from Model Registry
        serving_container_image_uri: URI of your custom serving container
        endpoint_display_name: Display name for the endpoint
        machine_type: Machine type for serving (e.g., 'n1-standard-4')
        min_replica_count: Minimum number of replicas
        max_replica_count: Maximum number of replicas for autoscaling
    """
    from google.cloud import aiplatform as aip

    aip.init(project=aipproject_id, location=aipproject_location)

    # Get the registered model
    model = aip.Model(model_id)
    print(f"Retrieved model: {model.display_name}")
    print(f"  Model URI: {model.uri}")

    # Create or get endpoint
    endpoints = aip.Endpoint.list(
        filter=f'display_name="{endpoint_display_name}"',
        order_by="create_time desc",
    )

    if len(endpoints) > 0:
        endpoint = endpoints[0]
        print(f"Using existing endpoint: {endpoint.display_name}")
    else:
        endpoint = aip.Endpoint.create(display_name=endpoint_display_name)
        print(f"Created new endpoint: {endpoint.display_name}")

    # Deploy model to endpoint
    print(f"Deploying model to endpoint...")
    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=f"{model.display_name}-v{model.version_id}",
        machine_type=machine_type,
        min_replica_count=min_replica_count,
        max_replica_count=max_replica_count,
        container_image_uri=serving_container_image_uri,
        traffic_percentage=100,
    )

    print(f"\n✓ Model deployed successfully!")
    print(f"  Endpoint resource name: {endpoint.resource_name}")
    print(f"  Endpoint URL: {endpoint.gca_resource.deployed_models[0].display_name}")
    print(f"\nYou can now send prediction requests to this endpoint.")
