FROM nvcr.io/nvidia/cuda:12.3.1-devel-ubuntu22.04

RUN set -x \
    && apt update \
    && apt install -y python3 python3-pip \
    && python3 -m pip install --no-cache poetry poetry-plugin-export \
    && rm -rf /var/lib/apt/lists/*

RUN set -x && mkdir /opt/pan24-generative-authorship-detection
WORKDIR /opt/pan24-generative-authorship-detection

# Install dependencies before copying actual source files
COPY pyproject.toml /opt/pan24-generative-authorship-detection
COPY poetry.lock /opt/pan24-generative-authorship-detection
RUN set -x \
      && poetry export > requirements.txt \
      && pip --no-cache install -r requirements.txt \
      && rm requirements.txt

COPY . /opt/pan24-generative-authorship-detection

RUN set -x && pip --no-cache install --no-deps .
