FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

LABEL maintainer="Gregory Bryant"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        software-properties-common \
        wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 5000 vscode \
    && useradd --home-dir /home/tfuser --create-home --uid 5000 \
    --gid 5000 --shell /bin/sh --skel /dev/null vscode

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda

# Conda Install for Channel: pytorch
RUN /opt/conda/bin/conda install \
    pytorch \
    torchvision \
    torchaudio \
    cudatoolkit=11.3 \
    -c pytorch \
    && /opt/conda/bin/conda clean --all

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

