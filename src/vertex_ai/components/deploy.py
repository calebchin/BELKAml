from kfp.v2.dsl import component, Input, Model, Metrics


@component(base_image="python:3.10", packages_to_install=["google-cloud-aiplatform"])
def deploy_model_to_aiplatform(
    model: Input[Model],
    train_metrics: Input[Metrics],
    val_metrics: Input[Metrics],
    test_metrics: Input[Metrics],
):
    from google.cloud import aiplatform
    import json

    with open(metrics.path, "r") as f:
        parsed_metrics = json.load(f)

    # idk about this uploading part
    aiplatform.init(project=project, location=location)
    aiplatform_model = aiplatform.Model.upload(
        display_name="protein-binding-model",
        artifact_uri=model_path,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.2-1:latest",
    )

    for display_name, metrics in zip(
        ["train_metrics", "val_metrics", "test_metrics"],
        [train_metrics, val_metrics, test_metrics],
    ):
        aiplatform_model.upload_evaluation(
            display_name=display_name,
            metrics=parsed_metrics,
            metrics_schema_uri=aiplatform.schema.dataset.metadata.metric.classification,
        )
