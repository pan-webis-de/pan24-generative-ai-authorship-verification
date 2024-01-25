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
@click.option('--sleep-time', type=int, default=10, help='Sleep time between requests')
def list_news(start_date, end_date, topic_file, output, language, country, num_results, sleep_time):
    output = os.path.join(output, 'lists')
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
                    json.dump(n, f, ensure_ascii=False)
                    f.write('\n')

            time.sleep(sleep_time)


@main.command(help='Download posts from news lists')
@click.argument('input_dir', type=click.Path(file_okay=False, exists=True))
@click.option('-o', '--output', type=click.Path(file_okay=False), help='Output directory', default='data')
def scrape_posts(input_dir, output):
    output = os.path.join(output, 'posts')
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

    with click.progressbar(glob.glob(os.path.join(input_dir, '*.jsonl')), label='Downloading news posts') as progress:
        for news_list in progress:
            d = os.path.join(output, os.path.basename(news_list[:-6]))
            os.makedirs(d, exist_ok=True)

            for i, news_item in enumerate(open(news_list, 'r')):
                art_id = f'art-{i:03d}'
                out_name_html = os.path.join(d, f'{art_id}.html.xz')
                out_name_txt = os.path.join(d, f'{art_id}.txt')
                if os.path.exists(out_name_html) and os.path.exists(out_name_txt):
                    continue

                news_item = json.loads(news_item)

                # First, retrieve target URL with a non-browser User-Agent.
                article_url = requests.head(news_item['url']).headers.get('Location')

                # Second, scrape the article itself with a browser User-Agent.
                # We cannot do both in one go, since Google displays cookie banners for browsers, but
                # some news pages block scrapers / non-browsers.
                try:
                    cfg = newspaper_cfg_browser if browser_ua_re.search(article_url) else newspaper_cfg
                    article = newspaper.Article(url=article_url or news_item['url'], config=cfg, language='en')
                    article.download()
                    article.parse()
                except newspaper.article.ArticleException:
                    logger.error('Failed to download %s/%s (URL: %s)', os.path.basename(d),  art_id, article_url)
                    continue

                text = '\n\n'.join((article.title, article.text))
                lzma.open(out_name_html, 'w').write(article.html)
                open(out_name_txt, 'w').write(text)


if __name__ == "__main__":
    main()
