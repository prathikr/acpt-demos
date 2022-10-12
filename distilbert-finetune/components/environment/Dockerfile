#FROM  mcr.microsoft.com/azureml/curated/acpt-pytorch-1.11-py38-cuda11.3-gpu:1
FROM ptebic.azurecr.io/internal/azureml/aifx/nightly-ubuntu2004-cu113-py38-torch1130dev:latest

RUN python -m pip install accelerate \
                          datasets \
                          optimum \
                          transformers
