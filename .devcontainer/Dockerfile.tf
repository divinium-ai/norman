FROM tensorflow/tensorflow:latest-gpu-jupyter

RUN groupadd --gid 5000 vscode \
    && useradd --home-dir /home/tfuser --create-home --uid 5000 \
    --gid 5000 --shell /bin/sh --skel /dev/null vscode

ENV SHELL /bin/bash