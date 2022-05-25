FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

LABEL maintainer="Gregory Bryant"

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# This Dockerfile adds a non-root user with sudo access. Use the "remoteUser"
# property in devcontainer.json to use it. On Linux, the container user's GID/UIDs
# will be updated to match your local UID/GID (when using the dockerFile property).
# See https://aka.ms/vscode-remote/containers/non-root-user for details.

ARG CACHEBUST=1
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

LABEL maintainer="Gregory Bryant"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        apt-utils \
        git \
        build-essential \
        software-properties-common \
        wget \
        sudo \
    && groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME\
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Rust Install
RUN curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain stable -y
ENV PATH=$HOME/.cargo/bin:$PATH
RUN rustup component add rls rust-analysis rust-src rustfmt clippy
RUN cargo install fd-find ripgrep

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
    && /opt/conda/bin/conda clean --all \
    #
    # Clean up
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=dialog
ENV PATH=$CONDA_DIR/bin:$PATH