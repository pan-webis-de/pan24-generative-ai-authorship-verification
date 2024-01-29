import glob
import json
import logging
import lzma
import os
import re
import time

import click
import gnews
import newspaper
import requests


logger = logging.getLogger(__name__)


@click.group()
def main():
    pass


@main.command(help='Download Google News listings for given topic list')
@click.argument('start_date', metavar='START_DATE', type=click.DateTime(formats=['%Y-%m-%d']))
@click.argument('end_date', metavar='END_DATE', type=click.DateTime(formats=['%Y-%m-%d']))
@click.argument('topic_file', type=click.File('r'))
@click.option('-o', '--output', type=click.Path(file_okay=False), help='Output directory', default='data')
@click.option('-l', '--language', help='News language', default='en')
@click.option('-c', '--country', help='News country', default='US')
@click.option('-n', '--num-results', type=int, help='Maximum number of results to download', default=200)
@click.option('--sleep-time', type=int, default=5, help='Sleep time between requests')
def list_news(start_date, end_date, topic_file, output, language, country, num_results, sleep_time):
    output = os.path.join(output, 'article-lists')
    os.makedirs(output, exist_ok=True)

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
            with open(os.path.join(output, f'news-{d1_s}-{d2_s}-{topic}.jsonl'), 'w') as f:
                for n in news:
                    # Resolve Google News proxy URLs
                    n['url'] = requests.head(n['url']).headers.get('Location') or n['url']

                    json.dump(n, f, ensure_ascii=False)
                    f.write('\n')

            time.sleep(sleep_time)


@main.command(help='Download news articles from article lists')
@click.argument('input_dir', type=click.Path(file_okay=False, exists=True))
@click.option('-o', '--output', type=click.Path(file_okay=False), help='Output directory', default='data')
def scrape_articles(input_dir, output):
    output = os.path.join(output, 'articles')
    os.makedirs(output, exist_ok=True)

    newspaper_cfg = newspaper.Config()
    newspaper_cfg_browser = newspaper.Config()
    newspaper_cfg_browser.headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Language': 'en-US,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate',
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
    }
    browser_ua_re = re.compile(r'(?:(?:reuters|washingtonpost|forbes|thehill|newsweek)\.com|abc\.net)')
    urls_scraped = set()

    with click.progressbar(glob.glob(os.path.join(input_dir, '*.jsonl')), label='Downloading news articles') as progress:
        for news_list in progress:
            d = os.path.join(output, os.path.basename(news_list[:-6]))
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
@click.option('-o', '--output', type=click.Path(file_okay=False), help='Output directory', default='data')
@click.option('-n', '--min-length', type=int, default=2000, help='Minimum post length in characters', show_default=True)
def filter_articles(input_dir, output, min_length):
    output = os.path.join(output, 'articles-filtered')
    os.makedirs(output, exist_ok=True)

    with click.progressbar(os.listdir(input_dir), label='Filtering articles') as bar:
        for d in bar:
            if not os.path.isdir(os.path.join(input_dir, d)) or not d.startswith('news-'):
                continue

            out = os.path.join(output, d)
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


if __name__ == "__main__":
    main()
