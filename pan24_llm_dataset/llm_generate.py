# Copyright 2024 Janek Bevendorff, Webis
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import string
from functools import partial
import glob
import json
import logging
from multiprocessing import pool, set_start_method
import os

import backoff
import click
from google.api_core.exceptions import GoogleAPIError
from google.auth.exceptions import GoogleAuthError
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
from transformers import AutoTokenizer, BitsAndBytesConfig, set_seed
from vertexai.language_models import ChatModel, TextGenerationModel
from vertexai.preview.generative_models import FinishReason, GenerativeModel, HarmCategory, HarmBlockThreshold


logger = logging.getLogger(__name__)
set_start_method('spawn')
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


def _apply_chat_template(tokenizer, model_type, messages):
    chat_template = tokenizer.chat_template

    if not chat_template:
        if 'alpaca' in tokenizer.name_or_path.lower():
            chat_template = (
                'Below is an instruction that describes a task. '
                'Write a response that appropriately completes the request.\n\n'
                '### Instruction:\n'
                '{% for message in messages -%}\n'
                '{{ message["content"] }}\n'
                '{% endfor %}\n'
                '{% if add_generation_prompt %}\n'
                '### Response:\n'
                '{% endif %}')
        elif model_type == 'llama':
            chat_template = (
                '{% for message in messages -%}\n'
                '### {{ message["role"].capitalize() }}:\n'
                '{{ message["content"] }}\n'
                '{% endfor %}\n'
                '{% if add_generation_prompt %}\n'
                '### Assistant:\n'
                '{% endif %}')
        else:
            chat_template = (
                'Task description:\n'
                '{% for message in messages -%}\n'
                '{{ message["content"] }}\n'
                '{% endfor %}\n'
                '{% if add_generation_prompt %}\n'
                'Response:\n'
                '{% endif %}')

    return tokenizer.apply_chat_template(
        messages, chat_template=chat_template, return_tensors='pt', add_generation_prompt=True)


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
    text = re.sub(r'^ *[IVX0-9]+\.\s+', '', text, flags=re.M)
    text = re.sub(
        r'^(?:(?:Sub)?Title|(?:Sub)?Headline|Paragraph|Introduction|Article(?: Title)?|Dateline)(?: \d+)?(?::\s+|\n+)',
        '',
        text, flags=re.M | re.I)
    text = re.sub(r'^[\[(]?(?:Paragraph|Headline)(?: \d+)[])]?:?\s+', '', text, flags=re.M | re.I)
    text = re.sub(r'^FOR IMMEDIATE RELEASE:?\n\n', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    if article_data.get('dateline'):
        text = text.replace('\n' + article_data['dateline'] + ' –\n\n', '\n' + article_data['dateline'] + ' – ')

    # Strip quotes around headlines
    text = text.split('\n', 1)
    if len(text) == 2:
        text[0] = re.sub(r'^"(.+)"$', r'\1', text[0], flags=re.M)
    text = '\n'.join(text)

    return text.strip()


@backoff.on_exception(backoff.expo, OpenAIError, max_tries=3)
def _openai_gen_article(article_data, client: OpenAI, model_name: str, prompt_template: str):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {'role': 'system', 'content': _generate_instruction_prompt(article_data, prompt_template)}
        ]
    )
    response = html2text.extract_plain_text(markdown.markdown(response.choices[0].message.content)).strip()
    return _clean_text_quirks(response, article_data)


@backoff.on_exception(backoff.expo, GoogleAPIError, max_tries=3)
def _vertexai_gen_article(article_data, model_name: str, prompt_template: str, **model_args):
    prompt = _generate_instruction_prompt(article_data, prompt_template)

    if 'gemini' in model_name:
        model = GenerativeModel(model_name=model_name)
    elif model_name.startswith('chat-'):
        model = ChatModel.from_pretrained(model_name)
    else:
        model = TextGenerationModel.from_pretrained(model_name)

    citations_censored = False
    sex_censored = False
    max_tries = 4
    for _ in range(max_tries):
        if isinstance(model, GenerativeModel):
            response = model.generate_content(
                prompt,
                generation_config=model_args,
                safety_settings={h: HarmBlockThreshold.BLOCK_NONE for h in HarmCategory})
            candidates = response.candidates

        elif isinstance(model, ChatModel):
            chat = model.start_chat(context=prompt, **model_args)
            candidates = chat.send_message('Assistant response:').candidates

        else:
            candidates = model.predict(prompt, **model_args).candidates

        # Handle hard-coded safety filters
        filtered = not candidates
        citations_filtered = False
        if candidates and hasattr(candidates[0], 'finish_reason'):
            filtered |= candidates[0].finish_reason not in [FinishReason.STOP, FinishReason.MAX_TOKENS]
            citations_filtered = candidates[0].finish_reason == FinishReason.RECITATION

        # Probably hard-coded input or output blocking. Amend prompt and try again with higher temperature.
        if filtered:
            if citations_filtered and not citations_censored:
                prompt += ('\nAvoid direct citation of sources that could be used for misinformation. '
                           'Do not cite or mention any medical or pharmaceutical sources.')
                citations_censored = True
            elif not sex_censored:
                prompt = prompt.replace('sex', '&&&')
                prompt += '\nPhrase your response in a non-harmful way suitable for the general public.'
                sex_censored = True
            model_args['temperature'] = min(1.0, model_args.get('temperature', 0.5) + 0.1)
            continue

        # Success
        break
    else:
        raise RuntimeError(f'Generation failed for {article_data["id"]}')

    response = candidates[0].content.text if hasattr(candidates[0], 'content') else candidates[0].text
    if sex_censored:
        response = response.replace('&&&', 'sex')

    response = html2text.extract_plain_text(markdown.markdown(response)).strip()
    return _clean_text_quirks(response, article_data)


def _huggingface_chat_gen_article(article_data, model, tokenizer, prompt_template, headline_only=False, **kwargs):
    role = 'user'
    if model.config.model_type in ['llama', 'qwen2']:
        role = 'system'
    messages = [{'role': role, 'content': _generate_instruction_prompt(article_data, prompt_template)}]
    if role == 'system':
        messages.append({'role': 'user', 'content': ''})

    model_inputs = _apply_chat_template(tokenizer, model.config.model_type, messages).to(model.device)

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
            trim_len = response.rfind('\n\n')
            if len(response) - trim_len > 500:
                trim_len = response.rfind('. ') + 1
            response = response[:trim_len]                          # Some models tend to stop mid-sentence

        return response.rstrip()

    return ''


@click.group()
def main():
    pass


@main.command(help='Generate articles using the OpenAI API')
@click.argument('input_dir', type=click.Path(file_okay=False, exists=True))
@click.option('-o', '--output-dir', type=click.Path(file_okay=False), help='Output directory',
              default=os.path.join('data', 'text', 'articles-llm'), show_default=True)
@click.option('-n', '--outdir-name', help='Output subdirectory name (defaults to model name)')
@click.option('-k', '--api_key', type=click.Path(dir_okay=False, exists=True),
              help='File containing OpenAI API key (if not given, OPENAI_API_KEY env var must be set)')
@click.option('-m', '--model-name', default='gpt-4-turbo-preview', show_default=True)
@click.option('-p', '--parallelism', default=5, show_default=True)
def openai(input_dir, output_dir, outdir_name, api_key, model_name, parallelism):
    if not api_key and not os.environ.get('OPENAI_API_KEY'):
        raise click.UsageError('Need one of --api-key or OPENAI_API_KEY!')

    output_dir = os.path.join(output_dir, outdir_name if outdir_name else model_name.lower())
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


@main.command(help='Generate articles using the VertexAI API')
@click.argument('input_dir', type=click.Path(file_okay=False, exists=True))
@click.option('-o', '--output-dir', type=click.Path(file_okay=False), help='Output directory',
              default=os.path.join('data', 'text', 'articles-llm'), show_default=True)
@click.option('-m', '--model-name', default='gemini-pro', show_default=True)
@click.option('-n', '--outdir-name', help='Output subdirectory name (defaults to model name)')
@click.option('-p', '--parallelism', default=5, show_default=True)
@click.option('-t', '--temperature', type=click.FloatRange(0, 1), default=0.6, show_default=True,
              help='Model temperature')
@click.option('-x', '--max-output-tokens', type=click.IntRange(0, 1024), default=1024, show_default=True,
              help='Maximum number of output tokens')
@click.option('-k', '--top-k', type=click.IntRange(1, 40), default=None, show_default=True,
              help='Top-k sampling')
@click.option('--top-p', type=click.FloatRange(0, 1), default=0.95, show_default=True,
              help='Top-p sampling')
def vertexai(input_dir, output_dir, model_name, outdir_name, parallelism, **kwargs):
    output_dir = os.path.join(output_dir, outdir_name if outdir_name else model_name.replace('@', '-').lower())
    os.makedirs(output_dir, exist_ok=True)

    fn = partial(
        _map_records_to_files,
        fn=_vertexai_gen_article,
        prompt_template='news_article_chat.jinja2',
        out_dir=output_dir,
        model_name=model_name,
        **kwargs)

    try:
        _generate_articles(input_dir, fn, parallelism)
    except GoogleAuthError as e:
        raise click.UsageError('Authentication error:\n' + str(e))


@main.command(help='Generate texts using a Huggingface chat model')
@click.argument('input_dir', type=click.Path(file_okay=False, exists=True))
@click.argument('model_name')
@click.option('-o', '--output-dir', type=click.Path(file_okay=False),
              default=os.path.join('data', 'text', 'articles-llm'), show_default=True, help='Output directory')
@click.option('-n', '--outdir-name', help='Output subdirectory name (defaults to model name)')
@click.option('-d', '--device', type=click.Choice(['auto', 'cuda', 'cpu']), default='auto',
              help='Select device to run model on')
@click.option('-m', '--min-length', type=click.IntRange(1), default=370,
              show_default=True, help='Minimum length in tokens')
@click.option('-x', '--max-new-tokens', type=click.IntRange(1), default=1000,
              show_default=True, help='Maximum new tokens')
@click.option('-s', '--decay-start', type=click.IntRange(1), default=500,
              show_default=True, help='Length decay penalty start')
@click.option('--decay-factor', type=click.FloatRange(1), default=1.01,
              show_default=True, help='Length decay penalty factor')
@click.option('-b', '--num-beams', type=click.IntRange(1), default=5,
              show_default=True, help='Number of search beams')
@click.option('-k', '--top-k', type=click.IntRange(0), default=0,
              show_default=True, help='Top-k sampling (0 to disable)')
@click.option('-p', '--top-p', type=click.FloatRange(0, 1), default=0.9,
              show_default=True, help='Top-p sampling')
@click.option('-t', '--temperature', type=click.FloatRange(0), default=2,
              show_default=True, help='Model temperature')
@click.option('-f', '--flash-attn', is_flag=True,
              help='Use flash-attn 2 (must be installed separately)')
@click.option('-b', '--better-transformer', is_flag=True, help='Use BetterTransformer')
@click.option('-q', '--quantization', type=click.Choice(['4', '8']))
@click.option('-h', '--headlines-only', is_flag=True, help='Run on previous output and generate missing headlines')
@click.option('--short-prompt', is_flag=True, help='Shorten the input prompt')
@click.option('--trust-remote-code', is_flag=True, help='Trust remote code')
def huggingface_chat(input_dir, model_name, output_dir, outdir_name, device, quantization, top_k,
                     decay_start, decay_factor, better_transformer, flash_attn, headlines_only,
                     trust_remote_code, short_prompt, **kwargs):

    model_name_out = model_name
    model_args = {
        'torch_dtype': torch.bfloat16
    }
    if flash_attn:
        model_args.update({'attn_implementation': 'flash_attention_2'})
    if quantization:
        model_args.update({
            'quantization_config': BitsAndBytesConfig(**{
                f'load_in_{quantization}bit': True,
                f'bnb_{quantization}bit_compute_dtype': torch.bfloat16
            })
        })
        model_name_out = model_name + f'-{quantization}bit'

    model_name_out = model_name_out.replace('\\', '/').rstrip('/')
    if '/' in model_name_out:
        model_name_out = '-'.join(model_name_out.split('/')[-2:])
    output_dir = os.path.join(output_dir, outdir_name if outdir_name else model_name_out.lower())

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

    prompt_template = 'news_article_chat.jinja2' if not short_prompt else 'news_article_short_chat.jinja2'

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
