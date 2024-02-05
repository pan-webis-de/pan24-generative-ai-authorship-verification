import re
from functools import partial
import glob
import json
import logging
from multiprocessing import pool
import os

import backoff
import click
import markdown
from openai import OpenAI, OpenAIError
from resiliparse.extract import html2text
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

logger = logging.getLogger(__name__)

GPU_DEVICE = -1

set_seed(42)


@click.group()
def main():
    global GPU_DEVICE
    GPU_DEVICE = 'cuda' if torch.cuda.is_available() else -1


def _generate_instruction_prompt(article_data):
    """
    Generate an instruction prompt for generating an article from the given source article data.
    """
    publisher = article_data.get('gnews_meta', {}).get('publisher', {}).get('title', '').replace('The ', '')
    summary = article_data['summary']
    if summary['article_type'] in ['press release', 'government agency statement']:
        personality = f'You are a {publisher} spokesperson writing a {summary["article_type"]}.'
        base_instruction = f'Write a press release covering the following key points:'
    elif summary['article_type'] == 'speech transcript':
        personality = 'You write a speech for a public figure.'
        base_instruction = 'The following key points must be addressed in the speech. ' + \
                           'Write the speech from the perspective of the person mentioned in the key points.'
    else:
        art_type = summary['article_type']
        if art_type == 'general reporting':
            art_type = 'a news article'
        elif art_type == 'opinion piece':
            art_type = f'an {art_type}'
        else:
            art_type = f'a {art_type} article'
        personality = f'You are a {publisher} journalist writing {art_type}. '
        base_instruction = 'In your article, cover the following key points:'

    key_points = '\n- ' + '\n- '.join(summary['key_points']) + '\n'
    prompt = '\n'.join((personality, base_instruction, key_points))

    if summary["stance"] != 'neutral':
        prompt += f'\nWrite the text from a {summary["stance"]} perspective.'

    if summary['article_type'] != 'speech transcript':
        if summary['spokespersons']:
            prompt += f'\nWithin the text, quote the following persons directly:\n- ' + \
                      '\n- '.join(summary['spokespersons']) + '\n'
        if summary['audience'] in ['professionals', 'children']:
            prompt += f'\nYour target audience are {summary["audience"]}.'

    prompt += '\nThe first line of your text is the headline.'
    if summary['article_type'] != 'speech transcript' and summary['dateline']:
        prompt += f'\nStart the article body with the dateline "{summary["dateline"]} – ".'

    n_paragraphs = article_data['text'].count('\n\n')
    n_words = round(int(len(re.split(r'\s+', article_data['text']))) + 9, -1)
    prompt += f'\nYour article should be about {n_paragraphs} paragraphs long (at least {n_words} words).'

    return prompt


@backoff.on_exception(backoff.expo, OpenAIError, max_tries=5)
def _openai_gen_article(article_data, client: OpenAI, model_name: str):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {'role': 'system', 'content': _generate_instruction_prompt(article_data)}
        ]
    )
    return html2text.extract_plain_text(markdown.markdown(response.choices[0].message.content))


def _iter_jsonl_files(in_files):
    for f in in_files:
        for l in open(f, 'r'):
            yield f, json.loads(l)


def _map_records_to_files(fname_and_record, *args, fn, out_dir, skip_existing=True, **kwargs):
    """
    Take a tuple of ``(input file name, parsed JSON record)``, apply ``fn`` on the JSON and write its output to
    individual text files based on the record's ID under ``out_dir``.
    """

    file_in, record = fname_and_record

    out_dir = os.path.join(out_dir, os.path.splitext(os.path.basename(file_in))[0])
    os.makedirs(out_dir, exist_ok=True)

    out_file = os.path.join(out_dir, record['id'] + '.txt')
    if skip_existing and os.path.isfile(out_file):
        return

    result = fn(record, *args, **kwargs)
    if not result:
        return
    try:
        print('OUT', out_file)
        open(out_file, 'w').write(result)
    except Exception as e:
        print(e)


def _generate_articles(input_dir, gen_fn, parallelism):
    jsonl_it = _iter_jsonl_files(glob.glob(os.path.join(input_dir, '*.jsonl')))
    with pool.ThreadPool(processes=parallelism) as p:
        with click.progressbar(p.imap(gen_fn, jsonl_it), label='Generating articles') as bar:
            list(bar)


@main.command(help='Generate articles using the OpenAI API')
@click.argument('input_dir', type=click.Path(file_okay=False, exists=True))
@click.option('-o', '--output-dir', type=click.Path(file_okay=False), help='Output directory',
              default=os.path.join('data', 'articles-llm'), show_default=True)
@click.option('-k', '--api_key', type=click.Path(dir_okay=False, exists=True),
              help='File containing OpenAI API key (if not given, OPENAI_API_KEY env var must be set)')
@click.option('-m', '--model-name', default='gpt-4-turbo-preview', show_default=True)
@click.option('-p', '--parallelism', default=10, show_default=True)
def openai(input_dir, output_dir, api_key, model_name, parallelism):
    if not api_key and not os.environ.get('OPENAI_API_KEY'):
        raise click.UsageError('Need one of --api-key or OPENAI_API_KEY!')

    output_dir = os.path.join(output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)

    client = OpenAI(api_key=open(api_key).read().strip() if api_key else os.environ.get('OPENAI_API_KEY'))

    fn = partial(
        _map_records_to_files,
        fn=_openai_gen_article,
        client=client,
        model_name=model_name)
    _generate_articles(input_dir, fn, parallelism)


def _huggingface_chat_gen_article(article_data, model, tokenizer, max_new_tokens):
    messages = [
        {'role': 'user', 'content': _generate_instruction_prompt(article_data)},
    ]
    model_inputs = tokenizer.apply_chat_template(messages, return_tensors='pt',).to(GPU_DEVICE)
    generated_ids = model.generate(model_inputs, max_new_tokens=max_new_tokens, do_sample=True)
    response = tokenizer.batch_decode(generated_ids[:, len(model_inputs[0]):], skip_special_tokens=True)[0]
    response = re.sub(r'^(?:Title|Headline):\s', '', response, flags=re.I, count=1)
    response = re.sub(r'^[\[(]?Paragraph(?: \d+)?[)\]]?:?\n', '', response, flags=re.M | re.I)
    return html2text.extract_plain_text(markdown.markdown(response))


@main.command(help='Generate texts using a Huggingface chat model')
@click.argument('input_dir', type=click.Path(file_okay=False, exists=True))
@click.argument('model_name')
@click.option('-o', '--output-dir', type=click.Path(file_okay=False), help='Output directory',
              default=os.path.join('data', 'articles-llm'))
@click.option('-m', '--max-new-tokens', type=int, default=1000, show_default=True, help='Maximum new tokens')
@click.option('-b', '--better-transformer', is_flag=True, help='Use BetterTransformer')
@click.option('-q', '--quantization', type=click.Choice(['4', '8']))
@click.option('-p', '--parallelism', default=1, show_default=True)
def huggingface_chat(input_dir, model_name, output_dir, max_new_tokens, better_transformer, quantization, parallelism):
    try:
        if quantization == '4':
            model = AutoModelForCausalLM.from_pretrained(
                model_name, load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            model_name_out = model_name + '-4bit'
        elif quantization == '8':
            model = AutoModelForCausalLM.from_pretrained(
                model_name, load_in_8bit=True, bnb_8bit_compute_dtype=torch.float16)
            model_name_out = model_name + '-4bit'
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name)
            model.to(GPU_DEVICE)
            model_name_out = model_name

        if better_transformer:
            model = model.to_bettertransformer()

    except Exception as e:
        raise click.UsageError('Failed to load model: ' + str(e))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    output_dir = os.path.join(output_dir, model_name_out.lower().replace('/', '-'))
    fn = partial(
        _map_records_to_files,
        fn=_huggingface_chat_gen_article,
        out_dir=output_dir,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens)
    _generate_articles(input_dir, fn, parallelism)


if __name__ == "__main__":
    main()
