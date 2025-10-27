from kfp.v2.dsl import component, Input, Dataset, Model


@component(base_image="python:3.10", packages_to_install=["pandas", "scikit-learn"])
def test_model(
    test_data: Input[Dataset], model: Input[Model], random_state: int = 42
) -> None:
    pass
