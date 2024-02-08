import re
import string
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
from tqdm import tqdm
import torch
try:
    from optimum.nvidia import AutoModelForCausalLM
except ModuleNotFoundError:
    from transformers import AutoModelForCausalLM
import transformers
from transformers import AutoTokenizer, set_seed

logger = logging.getLogger(__name__)

set_seed(42)


@click.group()
def main():
    pass


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

    prompt += '\nStart with a short and fitting headline for your article.'
    if summary['article_type'] != 'speech transcript' and summary['dateline']:
        prompt += f'\nBelow the headline, start the article body with the dateline "{summary["dateline"]} – ".'

    n_paragraphs = article_data['text'].count('\n\n')
    n_words = round(int(len(re.split(r'\s+', article_data['text']))) + 9, -1)
    prompt += f'\nYour article should be about {n_paragraphs} paragraphs long (at least {n_words} words).'
    prompt += ' Do not comment on what you do. Do not speak to or address the user.'

    return prompt


@backoff.on_exception(backoff.expo, OpenAIError, max_tries=5)
def _openai_gen_article(article_data, client: OpenAI, model_name: str):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {'role': 'system', 'content': _generate_instruction_prompt(article_data)}
        ]
    )
    return html2text.extract_plain_text(markdown.markdown(response.choices[0].message.content)).strip()


def _iter_jsonl_files(in_files):
    for f in in_files:
        for l in open(f, 'r'):
            yield f, json.loads(l)


def _map_records_to_files(topic_and_record, *args, fn, out_dir, skip_existing=True, out_file_suffix='.txt', **kwargs):
    """
    Take a tuple of ``(topic name, parsed JSON record)``, apply ``fn`` on the JSON and write its output to
    individual text files based on the record's topic and ID under ``out_dir``.
    """

    topic, record = topic_and_record
    out_dir = os.path.join(out_dir, topic)
    os.makedirs(out_dir, exist_ok=True)

    out_file = os.path.join(out_dir, record['id'] + out_file_suffix)

    if skip_existing and os.path.isfile(out_file):
        return

    try:
        result = fn(record, *args, **kwargs)
    except Exception as e:
        logger.error('Failed to generate article: %s', str(e))
        logger.exception(e)
        return

    if not result:
        return

    open(out_file, 'w').write(result)


# noinspection PyStatementEffect
def _generate_articles(input_dir, gen_fn, parallelism=1):
    it = _iter_jsonl_files(glob.glob(os.path.join(input_dir, '*.jsonl')))
    it = ((os.path.splitext(os.path.basename(f))[0], a) for f, a in it)

    if parallelism == 1:
        [_ for _ in tqdm(map(gen_fn, it), desc='Generating articles', unit=' articles')]
        return

    with pool.ThreadPool(processes=parallelism) as p:
        [_ for _ in tqdm(p.imap(gen_fn, it), desc='Generating articles', unit=' articles')]


# noinspection PyStatementEffect
def _generate_missing_article_headlines(input_dir, gen_fn):
    article_it = glob.iglob(os.path.join(input_dir, '*', 'art-*.txt'))

    for f in tqdm(article_it, desc='Checking and generating headlines', unit=' articles'):
        article = open(f, 'r').read()
        first_line = article.split('\n', 1)[0]
        if len(first_line) < 25 or len(first_line) > 160 or first_line[-1] == '.':
            gen_fn((os.path.basename(os.path.dirname(f)), {
                'id': os.path.splitext(os.path.basename(f))[0],
                'text': article
            }))


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
        out_dir=output_dir,
        client=client,
        model_name=model_name)
    _generate_articles(input_dir, fn, parallelism)


def _huggingface_chat_gen_article(article_data, model, tokenizer, headline_only=False, **kwargs):
    if headline_only:
        prompt = ('The following text is a news article or press release. ' +
                  'Write a short and fitting headline that summarizes the main point.\n\n' +
                  'Article: ' + article_data['text'])
    else:
        prompt = _generate_instruction_prompt(article_data)

    role = 'user'
    if model.config.model_type in ['llama']:
        role = 'system'
    messages = [{'role': role, 'content': prompt}]
    if role == 'system':
        messages.append({'role': 'user', 'content': ''})

    model_inputs = tokenizer.apply_chat_template(
        messages, return_tensors='pt', add_generation_prompt=True).to(model.device)

    for _ in range(3):
        generated_ids = model.generate(
            model_inputs,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            **kwargs)

        response = tokenizer.decode(generated_ids[0][len(model_inputs[0]):], skip_special_tokens=True)

        # Strip markdown
        response = html2text.extract_plain_text(markdown.markdown(response)).strip()

        # Remove certain generation quirks
        response = re.sub(r'^[IVX0-9]+\.\s+', '', response, flags=re.M)
        response = re.sub(
            r'^(?:Title|Headline|Paragraph|Introduction|Article|Dateline)(?: \d+)?(?::\s+|\n+)', '',
            response, flags=re.M | re.I)
        response = re.sub(r'^[\[(]?(?:Paragraph|Headline)(?: \d+)[])]?:?\s+', '', response, flags=re.M | re.I)
        response = re.sub(r'^FOR IMMEDIATE RELEASE:?\n\n', '', response)
        response = re.sub(r'\n{3,}', '\n\n', response).strip()

        if article_data.get('dateline'):
            response = response.replace('\n' + article_data['dateline'] + '\n\n', '\n' + article_data['dateline'] + ' ')

        # Retry if response empty
        if not response:
            continue

        if headline_only:
            response = response.split('\n', 1)[0]       # Take only first line
        elif response[-1] in string.ascii_letters:
            response = response[:response.rfind('\n\n')]            # Some models tend to stop mid-sentence

        # Strip quotes around headlines
        response = response.split('\n', 1)
        if len(response) == 2:
            response[0] = re.sub(r'^"(.+)"$', r'\1', response[0], flags=re.M)
            response = '\n'.join(response)

        return response.rstrip()

    return ''


@main.command(help='Generate texts using a Huggingface chat model')
@click.argument('input_dir', type=click.Path(file_okay=False, exists=True))
@click.argument('model_name')
@click.option('-o', '--output-dir', type=click.Path(file_okay=False),
              default=os.path.join('data', 'articles-llm'), show_default=True, help='Output directory')
@click.option('-d', '--device', type=click.Choice(['auto', 'cuda', 'cpu']), default='auto',
              help='Select device to run model on')
@click.option('-m', '--min-length', type=click.IntRange(1), default=400,
              show_default=True, help='Minimum length in tokens')
@click.option('-x', '--max-new-tokens', type=click.IntRange(1), default=3600,
              show_default=True, help='Maximum new tokens')
@click.option('-s', '--decay-start', type=click.IntRange(1), default=600,
              show_default=True, help='Length decay penalty start')
@click.option('--decay-factor', type=click.FloatRange(1), default=1.001,
              show_default=True, help='Length decay penalty factor')
@click.option('-b', '--num-beams', type=click.IntRange(1), default=5,
              show_default=True, help='Number of search beams')
@click.option('-k', '--top-k', type=click.IntRange(0), default=100,
              show_default=True, help='Top-k sampling (0 to disable)')
@click.option('-t', '--temperature', type=click.FloatRange(0), default=2,
              show_default=True, help='Model temperature')
@click.option('-f', '--flash-attn', is_flag=True,
              help='Use flash-attn 2 (must be installed separately)')
@click.option('-b', '--better-transformer', is_flag=True, help='Use BetterTransformer')
@click.option('-q', '--quantization', type=click.Choice(['4', '8']))
@click.option('-h', '--headlines-only', is_flag=True, help='Run on previous output and generate missing headlines')
@click.option('--trust-remote-code', is_flag=True, help='Trust remote code')
def huggingface_chat(input_dir, model_name, output_dir, device, quantization, top_k,
                     decay_start, decay_factor, better_transformer, flash_attn, headlines_only,
                     trust_remote_code, **kwargs):

    model_name_out = model_name
    model_args = {
        'torch_dtype': torch.bfloat16
    }
    if flash_attn:
        model_args.update({'attn_implementation': 'flash_attention_2'})
    if quantization:
        model_args.update({
            f'load_in_{quantization}bit': True,
            f'bnb_{quantization}bit_compute_dtype': torch.bfloat16
        })
        model_name_out = model_name + f'-{quantization}bit'

    model_name_out = model_name_out.replace('\\', '/').rstrip('/')
    if '/' in model_name_out:
        model_name_out = '-'.join(model_name_out.split('/')[-2:])
    output_dir = os.path.join(output_dir, model_name_out.lower())

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=device, trust_remote_code=trust_remote_code, **model_args)
        if better_transformer:
            model = model.to_bettertransformer()
    except Exception as e:
        raise click.UsageError('Failed to load model: ' + str(e))

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_cache=False, padding_side='left', trust_remote_code=trust_remote_code)

    kwargs.update(dict(
        model=model,
        tokenizer=tokenizer,
        top_k=top_k if top_k > 0 else None,
        exponential_decay_length_penalty=(decay_start, decay_factor)
    ))

    if headlines_only:
        del kwargs['min_length']
        del kwargs['exponential_decay_length_penalty']
        kwargs['max_new_tokens'] = 60

    fn = partial(_map_records_to_files, fn=_huggingface_chat_gen_article, out_dir=output_dir, **kwargs)

    if headlines_only:
        click.echo('Trying to detect and generate missing headlines...', err=True)
        _generate_missing_article_headlines(input_dir, partial(fn, headline_only=True, out_file_suffix='-headline.txt'))
        return

    _generate_articles(input_dir, fn)


if __name__ == "__main__":
    main()
