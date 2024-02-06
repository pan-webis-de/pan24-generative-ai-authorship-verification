#!/usr/bin/env bash

set -e
docker build -t registry.webis.de/code-research/authorship/pan24-generative-authorship-detection .
docker push registry.webis.de/code-research/authorship/pan24-generative-authorship-detection
