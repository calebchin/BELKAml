from kfp.dsl import ClassificationMetrics
from kfp.v2.dsl import component, Input, Output, Dataset, Model, Metrics


@component(
    base_image="python:3.10", packages_to_install=["pandas", "scikit-learn", "joblib"]
)
def train_model(
    train_data: Input[Dataset],
    val_data: Input[Dataset],
    model: Output[Model],
    train_metrics: Output[Metrics],
    val_metrics: Output[Metrics],
    classification_metrics: Output[ClassificationMetrics],
    batch_size: int = 1024,
    target_column: str = "binds",
) -> None:
    pass
