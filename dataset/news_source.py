from base64 import urlsafe_b64decode
from functools import partial
import glob
import json
import logging
import lzma
from multiprocessing import pool
import os
import re
import sys
import time

import click
import jsonschema
import gnews
from gnews_url import GNewsURL
from matplotlib import pyplot as plt
import newspaper
import numpy as np
from openai import NotFoundError, OpenAI
from openai.types.beta import Assistant
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

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

SUMMARY_JSON_SCHEMA = {
    'type': 'object',
    'properties': {
        'key_points': {'type': 'array', 'items': {'type': 'string'}},
        'spokespersons': {'type': 'array', 'items': {'type': 'string'}},
        'article_type': {
            'type': 'string',
            'enum': ['breaking news', 'press release', 'government agency statement', 'financial news',
                     'opinion piece', 'fact check', 'celebrity news', 'general reporting', 'speech transcript']},
        'dateline': {'type': 'string'},
        'audience': {'type': 'string', 'enum': ['general public', 'professionals', 'children']},
        'stance': {'type': 'string', 'enum': ['left-leaning', 'right-leaning', 'neutral']},
    },
    'required': ['key_points', 'spokespersons', 'article_type', 'dateline', 'audience', 'stance']
}


@click.group()
def main():
    pass


@main.command(help='Download Google News listings for given topic list')
@click.argument('start_date', metavar='START_DATE', type=click.DateTime(formats=['%Y-%m-%d']))
@click.argument('end_date', metavar='END_DATE', type=click.DateTime(formats=['%Y-%m-%d']))
@click.argument('topic_file', type=click.File('r'))
@click.option('-o', '--output-dir', type=click.Path(file_okay=False), help='Output directory', default='data')
@click.option('-l', '--language', help='News language', default='en')
@click.option('-c', '--country', help='News country', default='US')
@click.option('-n', '--num-results', type=int, help='Maximum number of results to download', default=200)
@click.option('--sleep-time', type=int, default=5, help='Sleep time between requests')
def list_news(start_date, end_date, topic_file, output_dir, language, country, num_results, sleep_time):
    output_dir = os.path.join(output_dir, 'article-lists')
    os.makedirs(output_dir, exist_ok=True)

    with click.progressbar(topic_file.readlines(), label='Downloading news for topics') as progress:
        for topic in progress:
            topic = topic.strip()
            if not topic:
                continue

            news = gnews.GNews(language=language, country=country,
                               start_date=start_date, end_date=end_date, max_results=num_results)
            news = news.get_news(topic)

            d1_s = start_date.strftime('%Y-%m-%d')
            d2_s = end_date.strftime('%Y-%m-%d')
            topic = ''.join(filter(str.isalnum, topic))
            with open(os.path.join(output_dir, f'news-{d1_s}-{d2_s}-{topic}.jsonl'), 'w') as f:
                for n in news:
                    # Decode Google News URLs
                    n['url'] = n['url'][len('https://news.google.com/rss/articles/'):].split('?', 1)[0]
                    n['url'] = urlsafe_b64decode(n['url'] + '==')
                    n['url'] = GNewsURL().parse(n['url']).url

                    json.dump(n, f, ensure_ascii=False)
                    f.write('\n')

            time.sleep(sleep_time)


@main.command(help='Download news articles from article lists')
@click.argument('input_dir', type=click.Path(file_okay=False, exists=True))
@click.option('-o', '--output-dir', type=click.Path(file_okay=False), help='Output directory', default='data')
def scrape_articles(input_dir, output_dir):
    output_dir = os.path.join(output_dir, 'articles')
    os.makedirs(output_dir, exist_ok=True)

    newspaper_cfg = newspaper.Config()
    newspaper_cfg_browser = newspaper.Config()
    newspaper_cfg_browser.headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Language': 'en-US,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate',
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
    }
    browser_ua_re = re.compile(r'(?:reuters|washingtonpost|forbes|thehill|newsweek)\.com|abc\.net')
    urls_scraped = set()

    with click.progressbar(glob.glob(os.path.join(input_dir, '*.jsonl')), label='Downloading news articles') as progress:
        for news_list in progress:
            d = os.path.join(output_dir, os.path.basename(news_list[:-6]))
            os.makedirs(d, exist_ok=True)

            for i, news_item in enumerate(open(news_list, 'r')):
                art_id = f'art-{i:03d}'
                out_name_html = os.path.join(d, f'{art_id}.html.xz')
                out_name_txt = os.path.join(d, f'{art_id}.txt.xz')
                if os.path.exists(out_name_html) and os.path.exists(out_name_txt):
                    continue

                news_item = json.loads(news_item)

                # Shouldn't happen, but filter duplicate URLs, just in case
                if news_item['url'] in urls_scraped:
                    logger.debug('Skipped duplicate URL: %s', news_item['url'])
                    continue
                urls_scraped.add(news_item['url'])

                try:
                    cfg = newspaper_cfg_browser if browser_ua_re.search(news_item['url']) else newspaper_cfg
                    article = newspaper.Article(url=news_item['url'], config=cfg, language='en')
                    article.download()
                    article.parse()
                except newspaper.article.ArticleException:
                    logger.error('Failed to download %s/%s (URL: %s)', os.path.basename(d),  art_id, news_item['url'])
                    continue

                text = '\n\n'.join((article.title, article.text))
                lzma.open(out_name_html, 'wt').write(article.html)
                lzma.open(out_name_txt, 'wt').write(text)


@main.command(help='Filter downloaded articles')
@click.argument('input_dir', type=click.Path(file_okay=False, exists=True))
@click.option('-o', '--output-dir', type=click.Path(file_okay=False), help='Output directory', default='data')
@click.option('-n', '--min-length', type=int, default=2000, help='Minimum post length in characters', show_default=True)
def filter_articles(input_dir, output_dir, min_length):
    output_dir = os.path.join(output_dir, 'articles-filtered')
    os.makedirs(output_dir, exist_ok=True)

    with click.progressbar(os.listdir(input_dir), label='Filtering articles') as bar:
        for d in bar:
            if not os.path.isdir(os.path.join(input_dir, d)) or not d.startswith('news-'):
                continue

            out = os.path.join(output_dir, d)
            os.makedirs(out, exist_ok=True)

            for f in glob.glob(os.path.join(input_dir, d, 'art-*.txt.xz')):
                lines = lzma.open(f, 'rt').readlines()
                while len(lines) > 2 and lines[0] == lines[2] and lines[1] == '\n':
                    # Delete duplicate title lines at the beginning
                    lines = lines[2:]

                text = ''.join(lines).strip()
                if len(text) < min_length:
                    continue
                open(os.path.join(out, os.path.basename(f)[:-3]), 'wt').write(text)


@main.command(help='Plot text length distribution')
@click.argument('input_dir', type=click.Path(file_okay=False, exists=True))
def plot_length_dist(input_dir):
    ws_re = re.compile(r'\s+')
    tokens = []

    with click.progressbar(glob.glob(os.path.join(input_dir, '*', 'art-*.txt')), label='Counting tokens') as bar:
        for f in bar:
            tokens.append(len(ws_re.sub(open(f, 'r').read().strip(), ' ')))

    tokens = pd.DataFrame(tokens, columns=['Characters'])

    tokens_log = np.log(tokens)
    print(f'μ = {tokens_log.mean().iloc[0]:.2f}, σ = {tokens_log.std().iloc[0]:.2f} (log-normal)')

    sns.histplot(data=tokens, x='Characters', kde=True, log_scale=True)
    plt.show()


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

    try:
        client.beta.threads.delete(thread_id=thread.id)
    except NotFoundError:
        pass
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


@main.command(help='Validate LLM-generated JSON files')
@click.argument('input_dir', type=click.Path(file_okay=False, exists=True))
def validate_llm_json(input_dir):
    with click.progressbar(glob.glob(os.path.join(input_dir, '*', 'art-*.json')), label='Validating JSON files') as bar:
        syntax_errors = []
        validation_errors = []
        for fname in bar:
            try:
                jsonschema.validate(instance=json.load(open(fname, 'r')), schema=SUMMARY_JSON_SCHEMA)
            except json.JSONDecodeError as e:
                syntax_errors.append((e.msg, fname))
            except jsonschema.ValidationError as e:
                validation_errors.append((e.message, fname))

    if not syntax_errors and not validation_errors:
        click.echo('No errors.', err=True)
        sys.exit(0)

    if syntax_errors:
        click.echo('Not well-formed:', err=True)
        for e, f in sorted(syntax_errors, key=lambda x: x[1]):
            click.echo(f'  {f}: {e}', err=True)

    if validation_errors:
        click.echo('Validation errors:', err=True)
        for e, f in sorted(validation_errors, key=lambda x: x[1]):
            click.echo(f'  {f}: {e}', err=True)

    sys.exit(1)


@main.command(name='Combine source article lists and texts with LLM summaries to JSONL files')
@click.argument('article_list_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('article_text_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('summary_dir', type=click.Path(exists=True, file_okay=False))
@click.option('-o', '--output-dir', type=click.Path(file_okay=False), help='Output directory')
def combine_source_data(article_list_dir, article_text_dir, summary_dir, output_dir):
    pass


if __name__ == "__main__":
    main()
