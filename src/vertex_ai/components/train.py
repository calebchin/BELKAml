from kfp.v2.dsl import component, Input, Output, Dataset, Model


@component(
    base_image="python:3.10", packages_to_install=["pandas", "scikit-learn", "joblib"]
)
def train_model(
    train_data: Input[Dataset],
    val_data: Input[Dataset],
    model: Output[Model],
    random_state: int = 42,
) -> None:
    pass
