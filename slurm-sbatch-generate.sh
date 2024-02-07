#!/bin/bash
#SBATCH --job-name=pan24-llm-generate
#SBATCH --gres=gpu:ampere
#SBATCH --mem=48g
#SBATCH --cpus-per-task=10
#SBATCH --output=pan-%j.log
#SBATCH --container-image=registry.webis.de/code-research/authorship/pan24-generative-authorship-detection

if [ -z "$1" ]; then
    echo "Usage: sbatch slurm-sbatch.sh INDIR MODEL [ ARGS... ]" >&2
    exit 1
fi

llm-generate huggingface-chat "$@" --flash-attn
