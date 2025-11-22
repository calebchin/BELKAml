from kfp.v2.dsl import component, Input, Model, Metrics


@component(base_image="python:3.12", packages_to_install=["google-cloud-aiplatform"])
def register_model_to_aip(
    aipproject_id: str,
    aipproject_location: str,
    model: Input[Model],
    train_metrics: Input[Metrics],
    val_metrics: Input[Metrics],
    test_metrics: Input[Metrics],
):
    """Register model in Vertex AI Model Registry and upload evaluation metrics.

    This component registers the trained model for tracking and versioning, but does NOT
    deploy it to an endpoint. For endpoint deployment, use the deploy.py component separately.
    """
    from google.cloud import aiplatform as aip
    import json

    aip.init(project=aipproject_id, location=aipproject_location)

    # Register model in Vertex AI Model Registry (without endpoint deployment)
    # Omit serving_container_image_uri to avoid requiring TorchServe format
    aipmodel = aip.Model.upload(
        display_name="belka-protein-binding-model",
        artifact_uri=model.uri,
    )

    print(f"✓ Model registered in Model Registry")
    print(f"  Resource name: {aipmodel.resource_name}")
    print(f"  Model URI: {aipmodel.uri}")

    # Upload evaluation metrics for each split
    for display_name, metrics in zip(
        ["train_metrics", "val_metrics", "test_metrics"],
        [train_metrics, val_metrics, test_metrics],
    ):
        with open(metrics.path, "r") as f:
            parsed_metrics = json.load(f)

            aipmodel.upload_evaluation(
                display_name=display_name,
                metrics=parsed_metrics,
                metrics_schema_uri=aip.schema.dataset.metadata.metric.classification,
            )

    print(f"\n✓ Uploaded metrics: train, val, test")
    print(f"\nModel ID: {aipmodel.name}")
    print(f"Use this model ID to deploy to an endpoint via deploy.py component")
