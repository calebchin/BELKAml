from kfp.v2.dsl import component, Input, Model, Metrics


@component(base_image="python:3.12", packages_to_install=["google-cloud-aiplatform"])
def deploy_model_to_aip(
    aipproject_id: str,
    aipproject_location: str,
    model: Input[Model],
    train_metrics: Input[Metrics],
    val_metrics: Input[Metrics],
    test_metrics: Input[Metrics],
):
    from google.cloud import aiplatform as aip
    import json

    aip.init(project=aipproject_id, location=aipproject_location)

    # Upload model to Vertex AI Model Registry
    aipmodel = aip.Model.upload(
        display_name="protein-binding-model",
        artifact_uri=model.uri,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.2-1:latest",
    )
    aipmodel.wait()

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
