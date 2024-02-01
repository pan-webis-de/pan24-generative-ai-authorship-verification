import logging

import click
import torch
from transformers import set_seed

logger = logging.getLogger(__name__)

GPU_DEVICE = -1

set_seed(42)


@click.group()
def main():
    global GPU_DEVICE
    GPU_DEVICE = torch.cuda.current_device() if torch.cuda.is_available() else -1


@main.command(help='Generate articles using the OpenAI API')
@click.argument('input_dir', type=click.Path(file_okay=False, exists=True))
@click.option('-o', '--output-dir', type=click.Path(file_okay=False), help='Output directory')
@click.option('-k', '--api_key', type=click.Path(dir_okay=False, exists=True),
              help='File containing OpenAI API key (if not given, OPENAI_API_KEY env var must be set)')
@click.option('-n', '--assistant-name', default='news-article-synthesizer', show_default=True)
@click.option('-m', '--model-name', default='gpt-4-turbo-preview', show_default=True)
@click.option('-p', '--parallelism', default=10, show_default=True)
def openai(input_dir, output_dir, api_key, assistant_name, model_name, parallelism):
    pass


@main.command(help='Generate texts with GPT-2-XL')
@click.argument('input_dir', type=click.Path(file_okay=False, exists=True))
@click.option('-o', '--output-dir', type=click.Path(file_okay=False), help='Output directory')
def gpt2_xl(input_dir, output_dir):
    # generator = pipeline('text-generation', model='gpt2-xl', device=GPU_DEVICE)
    # generator = pipeline('text-generation', model='openai-community/gpt2-xl', device=GPU_DEVICE)
    pass


if __name__ == "__main__":
    main()
