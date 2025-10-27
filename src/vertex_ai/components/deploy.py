from kfp.v2.dsl import component


@component(base_image="python:3.10", packages_to_install=["google-cloud-aiplatform"])
def deploy_to_aiplatform():
    pass
