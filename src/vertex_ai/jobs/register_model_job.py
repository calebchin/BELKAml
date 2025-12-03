from google.cloud import aiplatform as aip

# Configuration
PROJECT_ID = "belkaml"
LOCATION = "northamerica-northeast2"
# UPDATE THIS: Path to the FOLDER containing your model.pt in GCS
MODEL_ARTIFACT_URI = "gs://belkaml_model_output/46518343354/belkaml-train-pipeline-20251120035032/train-model_545364364446662656/model"
# The container we defined above
SERVING_IMAGE_URI = f"northamerica-northeast2-docker.pkg.dev/{PROJECT_ID}/belka-repo/belka-serving:latest"


def register():
    aip.init(project=PROJECT_ID, location=LOCATION)

    print(f"Registering model from {MODEL_ARTIFACT_URI}...")

    model = aip.Model.upload(
        display_name="belka-custom-container-model",
        artifact_uri=MODEL_ARTIFACT_URI,
        serving_container_image_uri=SERVING_IMAGE_URI,
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
        serving_container_ports=[8080],
    )

    print(f"✓ Model Registered. ID: {model.name}")
    print(f"Resource Name: {model.resource_name}")


if __name__ == "__main__":
    register()
