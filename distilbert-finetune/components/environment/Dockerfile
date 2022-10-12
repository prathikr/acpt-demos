#FROM ptebic.azurecr.io/internal/azureml/aifx/nightly-ubuntu2004-cu113-py38-torch1130dev:latest
FROM ptebic.azurecr.io/public/azureml/aifx/stable-ubuntu2004-cu116-py38-torch1120:latest

RUN python -m pip install azureml-core \
                          accelerate \
                          datasets \
                          optimum \
                          transformers
