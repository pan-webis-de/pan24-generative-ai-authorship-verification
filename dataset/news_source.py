import json
import os
import time

import click
import gnews


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

    with click.progressbar(topic_file, label='Downloading news for topics') as progress:
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


if __name__ == "__main__":
    main()
