from base64 import urlsafe_b64encode
import glob
import json
from logging import getLogger
import os
import random
import re
import string
import unicodedata
import uuid

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import lognorm, norm
import seaborn as sns

import click
from tqdm import tqdm


logger = getLogger(__name__)


@click.group()
def main():
    pass


@main.command(help='Convert directories with text files to JSONL files')
@click.argument('article_text_dir', type=click.Path(exists=True), nargs=-1)
@click.option('-o', '--output-dir', type=click.Path(file_okay=False), help='Output directory',
              default=os.path.join('data', 'articles-jsonl'), show_default=True)
@click.option('-c', '--combine-topics', is_flag=True, help='Combine all topics into one file')
def text2jsonl(article_text_dir, output_dir, combine_topics):
    os.makedirs(output_dir, exist_ok=True)

    article_text_dir = [d for d in article_text_dir if os.path.isdir(d)]
    if not article_text_dir:
        raise click.UsageError('No valid inputs provided.')

    def _write_jsonl(outfile, input_files, topic):
        for text_file in sorted(input_files):
            art_id = os.path.splitext(os.path.basename(text_file))[0]
            if combine_topics:
                art_id = '/'.join((topic, art_id))
            article_data = {
                'id': art_id,
                'text': open(text_file, 'r').read().strip(),
            }
            json.dump(article_data, outfile, ensure_ascii=False)
            outfile.write('\n')

    for at in tqdm(article_text_dir, desc='Reading input dirs', unit='dirs', leave=False):
        topics = sorted([d for d in os.listdir(at) if os.path.isdir(os.path.join(at, d))])
        if not topics:
            continue

        if combine_topics:
            outname = os.path.join(output_dir, os.path.basename(at) + '.jsonl')
            with open(outname, 'w') as out:
                for topic in topics:
                    _write_jsonl(out, glob.glob(os.path.join(at, topic, 'art-*.txt')), topic)
        else:
            for topic in topics:
                outname = os.path.join(output_dir, os.path.basename(at), topic + '.jsonl')
                os.makedirs(os.path.dirname(outname), exist_ok=True)
                with open(outname, 'w') as out:
                    _write_jsonl(out, glob.glob(os.path.join(at, topic, 'art-*.txt')), topic)


@main.command(help='Convert directories with JSONL files to directories with text files')
@click.argument('article_jsonl_dir', type=click.Path(exists=True, file_okay=False), nargs=-1)
@click.option('-o', '--output-dir', type=click.Path(file_okay=False), help='Output directory',
              default=os.path.join('data', 'articles-text'), show_default=True)
def jsonl2text(article_jsonl_dir, output_dir):
    for aj in article_jsonl_dir:
        aj = aj.rstrip(os.path.sep)
        out_dir_base = os.path.join(output_dir, os.path.basename(aj))

        infiles = sorted(glob.glob(os.path.join(aj, '*.jsonl')))
        if not infiles:
            continue

        for infile in tqdm(infiles, desc='Reading source files', unit='files', leave=False):
            topic = os.path.splitext(os.path.basename(infile))[0]
            out_dir = os.path.join(out_dir_base, topic)
            os.makedirs(out_dir, exist_ok=True)

            for l in open(infile, 'r'):
                j = json.loads(l)
                open(os.path.join(out_dir, j['id'] + '.txt'), 'w').write(j['text'])


@main.command(help='Truncate text character lengths according to a specific log-normal distribution')
@click.argument('input_dir', type=click.Path(exists=True, file_okay=False))
@click.option('-o', '--output-dir', type=click.Path(file_okay=False), help='Output directory',
              default=os.path.join('data', 'articles-truncated'), show_default=True)
@click.option('-m', '--scale', type=float, default=3300.0, show_default=True, help='Distribution scale')
@click.option('-l', '--loc', type=float, default=0.0, show_default=True, help='Distribution left location')
@click.option('-s', '--sigma', type=float, default=.28, show_default=True, help='Distribution standard deviation')
@click.option('-x', '--hard-max', type=int, default=8000, show_default=True, help='Hard maximum number of characters')
def truncate(input_dir, output_dir, scale, loc, sigma, hard_max):
    for f in tqdm(glob.glob(os.path.join(input_dir, '*', 'art-*.txt')), desc='Resampling text lengths', unit='texts'):
        out = os.path.join(output_dir, os.path.basename(os.path.dirname(f)))
        os.makedirs(out, exist_ok=True)
        out = os.path.join(out, os.path.basename(f))

        t = open(f, 'r').read()
        r = lognorm.rvs(loc=loc, s=sigma, scale=scale)
        while len(t) > hard_max or len(t) > r + 200:
            t = t[:t.rfind('\n\n')]

        # Strip one more line if last character isn't punctuation
        if t[-1] not in string.punctuation and len(t[t.rfind('\n'):]) < 70:
            t = t[:t.rfind('\n\n')]

        open(out, 'w').write(t)


@main.command(help='Plot text length distribution')
@click.argument('input_dir', type=click.Path(exists=True), nargs=-1)
@click.option('-l', '--no-log', is_flag=True, help='Fit a normal distribution instead of log-normal')
@click.option('--prune-outliers', type=click.FloatRange(0, 0.9), default=0.01, show_default=True,
              help='Prune percentage of outliers')
@click.option('-b', '--num-bins', type=int, default=50, show_default=True, help="Number of bins")
def plot_length_dist(input_dir, no_log, prune_outliers, num_bins):
    ws_re = re.compile(r'\s+')
    input_dir = [d for d in input_dir if os.path.isdir(d)]
    tokens = pd.DataFrame(columns=['dataset', 'art_id', 'characters'])

    ds_col = 'dataset'
    art_col = 'art_id'
    val_col = 'characters'

    for indir in tqdm(input_dir, desc='Calculating statistics', leave=False):
        token_list = []
        ds = os.path.basename(indir.rstrip(os.path.sep))
        for f in glob.glob(os.path.join(indir, '*', 'art-*.txt')):
            art_id = os.path.splitext(os.path.basename(f))[0]
            l = len(ws_re.sub(open(f, 'r').read().strip(), ' '))
            token_list.append((ds, art_id, l))

        pd_tmp = pd.DataFrame(token_list, columns=[ds_col, art_col, val_col])
        if prune_outliers > 0:
            p_lo, p_hi = pd_tmp[val_col].quantile(q=[prune_outliers / 2, 1.0 - prune_outliers / 2])
            pd_tmp = pd_tmp[(pd_tmp[val_col] > p_lo) & (pd_tmp[val_col] < p_hi)]
        tokens = pd.concat((tokens, pd_tmp))
        del pd_tmp

    tokens.reset_index(inplace=True)

    n_ds = tokens[ds_col].nunique()
    if n_ds == 0:
        raise click.UsageError('No valid input data provided.')

    first_col_w = tokens[ds_col].map(len).max()
    col_wrap = min(n_ds, max(3, int(np.sqrt(n_ds))))
    x_lim = (tokens[val_col].min(), tokens[val_col].max())

    def _plot_hist(*, data=None, x=None, **kwargs):
        ax = sns.histplot(data=data, x=x, **kwargs)
        ds_name = data[ds_col].iloc[0]

        # Overlay (log-)normal distribution
        if no_log:
            mean, std = norm.fit(data[x].astype(int))
            x_pdf = np.linspace(*x_lim, 100)
            y_pdf = norm.pdf(x_pdf, loc=mean, scale=std)
            y_pdf *= max(ax.lines[0].get_ydata()) / np.max(y_pdf)
            ax.plot(x_pdf, y_pdf, 'r', label='Normal dist.')
            print(f'{ds_name:<{first_col_w + 1}} μ = {mean:.2f}, σ = {std:.2f}')
        else:
            s, loc, scale = lognorm.fit(data[x].astype(int))
            x_pdf = np.logspace(*np.log10(np.clip(x_lim, 1, None)), 100, base=10)
            y_pdf = lognorm.pdf(x_pdf, s=s, loc=loc, scale=scale)
            y_pdf *= x_pdf / (scale * np.exp((s ** 2) / 2))                   # Correct for x bin shift
            y_pdf *= max(ax.lines[0].get_ydata()) / np.max(y_pdf)  # Scale up height to match histogram
            ax.plot(x_pdf, y_pdf, 'r', label='Log-normal dist.')
            print(f'{ds_name:<{first_col_w + 1}} loc = {loc:.2f}, scale = {scale:.2f}, σ = {s:.2f} (log-normal)')

    if no_log:
        bins = np.linspace(*x_lim, num_bins, dtype=int)
    else:
        bins = np.log10(np.logspace(*np.log10(x_lim), num_bins, dtype=int, base=10))
    g = sns.FacetGrid(tokens, col=ds_col, col_wrap=col_wrap, height=3, sharex=True,
                      sharey=True, aspect=1.5, legend_out=False)
    g.map_dataframe(_plot_hist, x=val_col, kde=True, log_scale=not no_log,
                    line_kws={'label': 'Kernel density'}, bins=bins)
    g.add_legend()
    plt.show()


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
@click.argument('machine_txt',  type=click.Path(exists=True), nargs=-1)
@click.option('-o', '--output-dir', type=click.Path(file_okay=False), help='Output directory',
              default=os.path.join('data', 'dataset-final'), show_default=True)
@click.option('-s', '--seed', type=int, default=42, help='Random seed')
@click.option('-p', '--test-pairwise', is_flag=True, help='Output test set as pairs')
@click.option('-r', '--scramble-test-ids', is_flag=True, help='Scramble test case IDs')
def assemble_dataset(train_ids, test_ids, human_txt, machine_txt, output_dir, seed, test_pairwise, scramble_test_ids):
    random.seed(seed)

    # Secret for scrambling test case IDs (depends on the set seed!).
    # This is not cryptographically secure, but should suffice for us!
    uuid_secret = random.randbytes(24)

    machine_txt = list({m for m in set(machine_txt) if os.path.isdir(m) and m != human_txt})
    if not machine_txt:
        raise click.UsageError('At least one machine folder must be specified.')

    train_ids = {l.strip() for l in train_ids.readlines() if l.strip()}
    test_ids = {l.strip() for l in test_ids.readlines() if l.strip()}
    if train_ids == test_ids:
        raise click.UsageError('Train and test input are the same.')
    for t in test_ids:
        if t in train_ids:
            raise click.UsageError(f'Test ID "{t}" found in training set.')

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
                case_id = '/'.join((machine_name, i))
                if scramble_test_ids:
                    case_id = '/'.join((uuid_secret.hex(), case_id))
                    case_id = urlsafe_b64encode(uuid.uuid5(uuid.NAMESPACE_OID, case_id).bytes).decode().rstrip('=')
                json.dump({
                    'id': case_id,
                    'text1': t1,
                    'text2': t2,
                    'is_human': [l1, l2]
                }, out, ensure_ascii=False)
                out.write('\n')


if __name__ == '__main__':
    main()
