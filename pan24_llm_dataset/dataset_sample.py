import glob
import json
from hashlib import sha256
from logging import getLogger
import os
import random
import re
import unicodedata

import click
from tqdm import tqdm


logger = getLogger(__name__)


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


_SINGLE_QUOTE_RE = re.compile(r'[‘’‚‛]')
_DOUBLE_QUOTE_RE = re.compile(r'[“”„‟]')


def _normalize_text(text):
    text = unicodedata.normalize('NFKC', text)
    text = _SINGLE_QUOTE_RE.sub('\'', text)
    text = _DOUBLE_QUOTE_RE.sub('"', text)
    return text.strip()


def _read_texts(base_dir, ids):
    for i in ids:
        topic, article = i.split('/')
        file_name = os.path.join(base_dir, topic, article + '.txt')
        if not os.path.isfile(file_name):
            logger.warning('File not found: %s', file_name)
            yield i, ''
            continue
        yield i, _normalize_text(open(file_name, 'r').read())


def _write_jsonl(it, ids, suffix, output_dir):
    for i, in_dir in enumerate(it):
        name = os.path.basename(in_dir) if i > 0 else 'human'
        out_name = os.path.join(output_dir, name + f'-{suffix}.jsonl')
        with open(out_name, 'w') as out:
            for tid, text in _read_texts(in_dir, ids):
                json.dump({'id': tid, 'text': text}, out, ensure_ascii=False)
                out.write('\n')


@main.command(help='Assemble dataset from sorted (!) human and machine JSONL files')
@click.argument('train_ids', type=click.File('r'))
@click.argument('test_ids', type=click.File('r'))
@click.argument('human_txt', type=click.Path(exists=True, file_okay=False))
@click.argument('machine_txt',  type=click.Path(exists=True, file_okay=False), nargs=-1)
@click.option('-o', '--output-dir', type=click.Path(file_okay=False), help='Output directory',
              default=os.path.join('data', 'dataset-final'), show_default=True)
@click.option('-s', '--seed', type=int, default=42, help='Random seed')
@click.option('-p', '--test-pairwise', is_flag=True, help='Output test set as pairs')
@click.option('-h', '--hash-test-ids', is_flag=True, help='Hash test case IDs')
def assemble_dataset(train_ids, test_ids, human_txt, machine_txt, output_dir, seed, test_pairwise, hash_test_ids):
    random.seed(seed)

    machine_txt = [m for m in set(machine_txt) if m != human_txt]
    if not machine_txt:
        raise click.UsageError('At least one machine folder must be specified.')

    train_ids = [l.strip() for l in train_ids.readlines() if l.strip()]
    test_ids = [l.strip() for l in test_ids.readlines() if l.strip()]

    os.makedirs(output_dir, exist_ok=True)

    train_it = tqdm([human_txt] + machine_txt, desc='Assembling train split', unit=' inputs')
    _write_jsonl(train_it, train_ids, 'train', output_dir)

    if not test_pairwise:
        test_it = tqdm([human_txt] + machine_txt, desc='Assembling test split', unit=' inputs')
        _write_jsonl(test_it, test_ids, 'test', output_dir)
        return

    # Pairwise output with randomised (human, machine) or (machine, human) pairs
    for machine in tqdm(machine_txt, desc='Assembling pairwise test split', unit=' inputs'):
        machine_name = os.path.basename(machine)
        h_it = _read_texts(human_txt, test_ids)
        m_it = _read_texts(machine, test_ids)
        out_name = os.path.join(output_dir, machine_name + '-test.jsonl')
        with open(out_name, 'w') as out:
            for (i, t1), (_, t2) in zip(h_it, m_it):
                l1, l2 = True, False
                if random.randint(0, 1):
                    t1, t2 = t2, t1
                    l1, l2 = l2, l1
                case_id = f'{machine_name}/{i}'
                json.dump({
                    'id': case_id if not hash_test_ids else sha256(case_id.encode()).hexdigest(),
                    'text1': t1,
                    'text2': t2,
                    'is_human': [l1, l2]
                }, out, ensure_ascii=False)
                out.write('\n')


if __name__ == '__main__':
    main()
