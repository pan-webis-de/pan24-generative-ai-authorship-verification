import glob
from functools import partial
import logging
from multiprocessing import pool
import os
import time

import click
from openai import OpenAI
from openai.types.beta import Assistant
import torch
from transformers import pipeline, set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

logger = logging.getLogger(__name__)

GPU_DEVICE = -1

set_seed(42)


@click.group()
def main():
    global GPU_DEVICE
    GPU_DEVICE = torch.cuda.current_device() if torch.cuda.is_available() else -1


SUMMARIZER_INSTRUCTIONS = '''
You are a news article and press release summarizer. Given an article, you summarize the key points in 10 bullet points.
You also classify the article type ("breaking news",  "press release", "government agency statement", "financial news",
"opinion piece", "fact check", "celebrity news", "general reporting", "speech transcript").
Extract the dateline from the beginning of the article if one exists (e.g. "WASHINGTON " or "May 28 (Reuters)").
If spokespersons are cited verbatim, list their names, functions, and titles (if any).
Determine the article's target audience ("general public", "professionals", "children").
Classify whether the article's stance is "left-leaning", "right-leaning", or "neutral".

Answer in structured JSON format (without Markdown formatting) like so:
{
    "key_points": ["key point 1", "key point 2", ...],
    "spokespersons": ["person1 (title, function)", ...],
    "article_type": "article type",
    "dateline": "dateline",
    "audience": "audience",
    "stance": "stance"
}
'''


def _summarize_article(article: str, client: OpenAI, assistant: Assistant):
    thread = client.beta.threads.create()
    client.beta.threads.messages.create(thread_id=thread.id, role='user', content=article)

    run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant.id)
    while run.status in ('queued', 'in_progress'):
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        time.sleep(0.5)
    if run.status == 'failed':
        logger.error('Run %s failed: %s', run.id, run.last_error.message)
        return

    response = client.beta.threads.messages.list(thread_id=thread.id).data[0]
    response = '\n'.join(r.text.value for r in response.content)
    if response.startswith('```json'):
        response = response.strip('`')[len('json'):]

    client.beta.threads.delete(thread_id=thread.id)
    return response.strip()


def _map_from_to_file(fnames, *args, fn, skip_existing=True, max_chars=None, **kwargs):
    """
    Call ``fn`` on tuples of input and output filenames, reading input from one file and writing to another.
    """
    file_in, file_out = fnames

    if skip_existing and os.path.exists(file_out):
        return
    os.makedirs(os.path.dirname(file_out), exist_ok=True)

    result = fn(open(file_in, 'r').read()[:max_chars], *args, **kwargs)
    if not result:
        return
    open(file_out, 'w').write(result)


@main.command(help='Generate news article summaries using OpenAI API')
@click.argument('input_dir', type=click.Path(file_okay=False, exists=True))
@click.option('-o', '--output-dir', type=click.Path(file_okay=False), help='Output directory', default='data')
@click.option('-k', '--api_key', type=click.Path(dir_okay=False, exists=True),
              help='File containing OpenAI API key (if not given, OPENAI_API_KEY env var must be set)')
@click.option('-n', '--assistant-name', default='news-article-summarizer', show_default=True)
@click.option('-m', '--model-name', default='gpt-4-turbo-preview', show_default=True)
@click.option('-p', '--parallelism', default=10, show_default=True)
@click.option('-c', '--max-chars', help='Maximum article length to send to OpenAI API in characters',
              default=8192, show_default=True)
def summarize_news(input_dir, output_dir, api_key, assistant_name, model_name, parallelism, max_chars):
    if not api_key and not os.environ.get('OPENAI_API_KEY'):
        raise click.UsageError('Need one of --api-key or OPENAI_API_KEY!')

    client = OpenAI(api_key=open(api_key).read().strip() if api_key else os.environ.get('OPENAI_API_KEY'))

    # Create or update assistant
    assistant = next((a for a in client.beta.assistants.list() if a.name == assistant_name), None)
    if not assistant:
        assistant = client.beta.assistants.create(
            name=assistant_name,
            instructions=SUMMARIZER_INSTRUCTIONS,
            model=model_name)
    elif assistant.model != model_name or assistant.instructions != SUMMARIZER_INSTRUCTIONS:
        assistant = client.beta.assistants.update(
            assistant_id=assistant.id,
            instructions=SUMMARIZER_INSTRUCTIONS,
            model=model_name)

    # List input files and map to (in-dir/art-*.txt, out-dir/art-*.json) tuples
    in_files = glob.glob(os.path.join(input_dir, '*', 'art-*.txt'))
    in_out_files = ((f, os.path.join(output_dir, 'article-summaries', os.path.basename(os.path.dirname(f)),
                                     os.path.splitext(os.path.basename(f))[0] + '.json')) for f in in_files)
    fn = partial(_map_from_to_file, fn=_summarize_article, max_chars=max_chars, client=client, assistant=assistant)

    with pool.ThreadPool(processes=parallelism) as p:
        with click.progressbar(p.imap(fn, in_out_files), label='Generating summaries', length=len(in_files)) as bar:
            list(bar)


@main.command(help='Generate texts with GPT-2-XL')
@click.argument('input_dir', type=click.Path(file_okay=False, exists=True))
@click.option('-o', '--output-dir', type=click.Path(file_okay=False), help='Output directory')
def gpt2_xl(input_dir, output_dir):
    # generator = pipeline('text-generation', model='gpt2-xl', device=GPU_DEVICE)
    # generator = pipeline('text-generation', model='openai-community/gpt2-xl', device=GPU_DEVICE)
    pass


if __name__ == "__main__":
    main()
