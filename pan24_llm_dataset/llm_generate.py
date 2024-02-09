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
import jinja2
import markdown
from openai import OpenAI, OpenAIError
from resiliparse.extract import html2text
from tqdm import tqdm
import torch
try:
    from optimum.nvidia import AutoModelForCausalLM
except ModuleNotFoundError:
    from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, set_seed

logger = logging.getLogger(__name__)

set_seed(42)


def _generate_instruction_prompt(article_data, template_name):
    """
    Generate an instruction prompt for generating an article from the given source article data.
    """

    target_paragraphs = article_data['text'].count('\n\n')
    target_words = round(int(len(re.split(r'\s+', article_data['text']))) + 9, -1)

    env = jinja2.Environment(
        loader=jinja2.PackageLoader('pan24_llm_dataset', 'prompt_templates')
    )
    template = env.get_template(template_name)
    return template.render(article_data=article_data, target_paragraphs=target_paragraphs, target_words=target_words)


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


def _clean_text_quirks(text, article_data):
    """Clean up some common LLM text quirks."""

    # Remove certain generation quirks
    text = re.sub(r'^[a-z-]+>\s*', '', text)   # Cut-off special tokens at the beginning
    text = re.sub(r'^[IVX0-9]+\.\s+', '', text, flags=re.M)
    text = re.sub(
        r'^(?:(?:Sub)?Title|(?:Sub)?Headline|Paragraph|Introduction|Article(?: Title)?|Dateline)(?: \d+)?(?::\s+|\n+)',
        '',
        text, flags=re.M | re.I)
    text = re.sub(r'^[\[(]?(?:Paragraph|Headline)(?: \d+)[])]?:?\s+', '', text, flags=re.M | re.I)
    text = re.sub(r'^FOR IMMEDIATE RELEASE:?\n\n', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    if article_data.get('dateline'):
        text = text.replace('\n' + article_data['dateline'] + ' –\n\n', '\n' + article_data['dateline'] + ' – ')

    return text.strip()


@backoff.on_exception(backoff.expo, OpenAIError, max_tries=5)
def _openai_gen_article(article_data, client: OpenAI, model_name: str, prompt_template: str):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {'role': 'system', 'content': _generate_instruction_prompt(article_data, prompt_template)}
        ]
    )
    response = html2text.extract_plain_text(markdown.markdown(response.choices[0].message.content)).strip()
    return _clean_text_quirks(response, article_data)


def _huggingface_chat_gen_article(article_data, model, tokenizer, prompt_template, headline_only=False, **kwargs):
    role = 'user'
    if model.config.model_type in ['llama']:
        role = 'system'
    messages = [{'role': role, 'content': _generate_instruction_prompt(article_data, prompt_template)}]
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
        response = _clean_text_quirks(response, article_data)

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


@click.group()
def main():
    pass


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
        prompt_template='news_article_chat.jinja2',
        out_dir=output_dir,
        client=client,
        model_name=model_name)
    _generate_articles(input_dir, fn, parallelism)


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

    prompt_template = 'news_article_chat.jinja2'

    if headlines_only:
        del kwargs['min_length']
        del kwargs['exponential_decay_length_penalty']
        kwargs['max_new_tokens'] = 60
        prompt_template = 'headline_chat.jinja2'

    fn = partial(_map_records_to_files, fn=_huggingface_chat_gen_article,
                 prompt_template=prompt_template, out_dir=output_dir, **kwargs)

    if headlines_only:
        click.echo('Trying to detect and generate missing headlines...', err=True)
        _generate_missing_article_headlines(input_dir, partial(fn, headline_only=True, out_file_suffix='-headline.txt'))
        return

    _generate_articles(input_dir, fn)


if __name__ == "__main__":
    main()
