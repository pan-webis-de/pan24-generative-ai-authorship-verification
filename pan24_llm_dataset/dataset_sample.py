import glob
import json
import os
import random

import click
from tqdm import tqdm


@click.group()
def main():
    pass


@main.command(help='Convert text files to JSONL')
@click.argument('article_text_dir', type=click.Path(exists=True, file_okay=False), nargs=-1)
@click.option('-o', '--output-dir', type=click.Path(file_okay=False), help='Output directory',
              default=os.path.join('data', 'articles-jsonl'), show_default=True)
def text2jsonl(article_text_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for at in article_text_dir:
        topics = sorted([d for d in os.listdir(at) if os.path.isdir(os.path.join(at, d))])
        if not topics:
            continue

        with open(os.path.join(output_dir, os.path.basename(at) + '.jsonl'), 'w') as out:
            for topic in tqdm(topics, desc='Reading source files', unit='files', leave=False):
                for text_file in sorted(glob.glob(os.path.join(at, topic, 'art-*.txt'))):
                    art_id = '/'.join((topic, os.path.splitext(os.path.basename(text_file))[0]))
                    article_data = {
                        'id': art_id,
                        'text': open(text_file, 'r').read().strip(),
                    }
                    json.dump(article_data, out, ensure_ascii=False)
                    out.write('\n')


@main.command(help='Generate train/test splits and save IDs as text files')
@click.argument('jsonl_in', type=click.File('r'))
@click.option('-o', '--output-dir', type=click.Path(file_okay=False), help='Output directory',
              default=os.path.join('data'), show_default=True)
@click.option('-t', '--train-size', type=click.FloatRange(0, 1), default=0.8,
              help='Training set size')
@click.option('-s', '--seed', type=int, default=42, help='Random seed')
def gen_splits(jsonl_in, output_dir, train_size, seed):
    random.seed(seed)
    ids = [json.loads(l)["id"] for l in jsonl_in]
    random.shuffle(ids)
    split_idx = int(len(ids) * train_size)
    train, test = ids[:split_idx], ids[split_idx:]

    open(os.path.join(output_dir, 'ids-train.txt'), 'w').write('\n'.join(train))
    open(os.path.join(output_dir, 'ids-test.txt'), 'w').write('\n'.join(test))


@main.command(help='Assemble the final dataset from a given split')
@click.argument('train_ids', type=click.File('r'))
@click.argument('test_ids', type=click.File('r'))
@click.argument('human_jsonl', type=click.File('r'))
@click.argument('machine_jsonl',  type=click.File('r'), nargs=-1)
@click.option('-o', '--output-dir', type=click.Path(file_okay=False), help='Output directory',
              default=os.path.join('data', 'dataset-final'), show_default=True)
@click.option('-s', '--seed', type=int, default=42, help='Random seed')
def assemble_dataset(train_ids, test_ids, human_jsonl, machine_jsonl, output_dir, seed):
    random.seed(seed)

    machine_jsonl = [m for m in machine_jsonl if m.name != human_jsonl.name]
    if not machine_jsonl:
        raise click.UsageError('At least one machine folder must be specified.')

    train_ids = [l.strip().split('/') for l in train_ids.readlines()]
    test_ids = [l.strip().split('/') for l in test_ids.readlines()]

    os.makedirs(output_dir, exist_ok=True)
    for lhuman in human_jsonl:
        pass
        # TODO


if __name__ == '__main__':
    main()
