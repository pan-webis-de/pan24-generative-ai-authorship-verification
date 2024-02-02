FROM nvcr.io/nvidia/cuda:12.3.1-devel-ubuntu22.04

RUN set -x \
    && apt update \
    && apt install -y python3 python3-pip \
    && python3 -m pip install --no-cache poetry \
    && rm -rf /var/lib/apt/lists/*

COPY . /opt/pan24-generative-authorship-detection
WORKDIR /opt/pan24-generative-authorship-detection

RUN set -x \
    && POETRY_INSTALLER_MAX_WORKERS=10 poetry --no-cache --no-interaction install
