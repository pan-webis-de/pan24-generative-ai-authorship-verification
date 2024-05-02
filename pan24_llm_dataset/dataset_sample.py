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
import scipy.stats as st
import seaborn as sns

import click
from tqdm import tqdm


logger = getLogger(__name__)

_SINGLE_QUOTE_RE = re.compile(r'[‘’‚‛]')
_DOUBLE_QUOTE_RE = re.compile(r'[“”„‟]')


def _normalize_text(text):
    text = unicodedata.normalize('NFKC', text)
    text = _SINGLE_QUOTE_RE.sub('\'', text)
    text = _DOUBLE_QUOTE_RE.sub('"', text)
    return text.strip()


def _read_texts(base_dir, topic_name, article_ids=None, normalize=False, skip_empty=True):
    if not article_ids:
        article_ids = (os.path.basename(a)[:-4] for a in glob.glob(os.path.join(base_dir, topic_name, 'art-*.txt')))

    for art_id in article_ids:
        text = ''
        file_name = os.path.join(base_dir, topic_name, art_id + '.txt')
        if os.path.isfile(file_name):
            text = open(file_name, 'r').read().strip()
        elif skip_empty:
            logger.warning('File not found: %s', file_name)
            continue

        if normalize:
            text = _normalize_text(text)

        if skip_empty and not text:
            logger.warning('Skipped empty training text: %s/%s', topic_name, art_id)
            continue

        yield art_id, text


def _write_jsonl(outfile, input_text_it, id_prefix=None,):
    for tid, text in input_text_it:
        if id_prefix:
            tid = '/'.join((id_prefix, tid))

        json.dump({
            'id': tid,
            'text': text,
        }, outfile, ensure_ascii=False)
        outfile.write('\n')


@click.group()
def main():
    pass


@main.command(help='Convert directories with text files to JSONL files')
@click.argument('article_text_dir', type=click.Path(exists=True), nargs=-1)
@click.option('-o', '--output-dir', type=click.Path(file_okay=False), help='Output directory',
              default=os.path.join('data', 'jsonl', 'articles-converted-jsonl'), show_default=True)
@click.option('-c', '--combine-topics', is_flag=True, help='Combine all topics into one file')
def text2jsonl(article_text_dir, output_dir, combine_topics):
    os.makedirs(output_dir, exist_ok=True)

    article_text_dir = [d for d in article_text_dir if os.path.isdir(d)]
    if not article_text_dir:
        raise click.UsageError('No valid inputs provided.')

    for at in tqdm(article_text_dir, desc='Reading input dirs', unit='dirs', leave=False):
        topics = sorted([d for d in os.listdir(at) if os.path.isdir(os.path.join(at, d))])
        if not topics:
            continue

        if combine_topics:
            outname = os.path.join(output_dir, os.path.basename(at) + '.jsonl')
            with open(outname, 'w') as out:
                for topic in topics:
                    _write_jsonl(out, _read_texts(at, topic), topic)
        else:
            for topic in topics:
                outname = os.path.join(output_dir, os.path.basename(at), topic + '.jsonl')
                os.makedirs(os.path.dirname(outname), exist_ok=True)
                with open(outname, 'w') as out:
                    _write_jsonl(out, _read_texts(at, topic))


@main.command(help='Convert directories with JSONL files to directories with text files')
@click.argument('article_jsonl_dir', type=click.Path(exists=True, file_okay=False), nargs=-1)
@click.option('-o', '--output-dir', type=click.Path(file_okay=False), help='Output directory',
              default=os.path.join('data', 'text', 'articles-converted-text'), show_default=True)
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
              default=os.path.join('data', 'text', 'articles-truncated'), show_default=True)
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
        r = st.lognorm.rvs(loc=loc, s=sigma, scale=scale)
        while len(t) > hard_max or len(t) > r + 200:
            t = t[:t.rfind('\n\n')]

        # Strip one more line if last character isn't punctuation
        if t[-1] not in string.punctuation and len(t[t.rfind('\n'):]) < 70:
            t = t[:t.rfind('\n\n')]

        open(out, 'w').write(t)


@main.command(help='Plot text length distribution')
@click.argument('input_dir', type=click.Path(exists=True), nargs=-1)
@click.option('-l', '--no-log', is_flag=True, help='Fit a normal distribution instead of log-normal')
@click.option('-p', '--prune-outliers', type=click.FloatRange(0, 0.9), default=0.01, show_default=True,
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
            a, mean, std = st.skewnorm.fit(data[x].astype(int))
            x_pdf = np.linspace(*x_lim, 100)
            y_pdf = st.skewnorm.pdf(x_pdf, a=a, loc=mean, scale=std)
            y_pdf *= max(ax.lines[0].get_ydata()) / np.max(y_pdf)
            ax.plot(x_pdf, y_pdf, 'r', label='Skew-normal dist.')
            print(f'{ds_name:<{first_col_w + 1}} α = {a:.2f}, μ = {mean:.2f}, σ = {std:.2f}')
        else:
            s, loc, scale = st.lognorm.fit(data[x].astype(int))
            x_pdf = np.logspace(*np.log10(np.clip(x_lim, 1, None)), 100, base=10)
            y_pdf = st.lognorm.pdf(x_pdf, s=s, loc=loc, scale=scale)
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


@main.command(help='Assemble dataset from human and machine text files')
@click.argument('human_txt', type=click.Path(exists=True, file_okay=False))
@click.argument('machine_txt',  type=click.Path(exists=True), nargs=-1)
@click.option('-a', '--train-ids', type=click.File('r'), help='File with train IDs')
@click.option('-b', '--test-ids', type=click.File('r'), help='File with test IDs')
@click.option('-o', '--output-dir', type=click.Path(file_okay=False), help='Output directory',
              default=os.path.join('data', 'dataset-final'), show_default=True)
@click.option('-s', '--seed', type=int, default=42, help='Seed for randomizing test IDs')
@click.option('-n', '--no-source', is_flag=True, help='Do not include source IDs')
@click.option('-m', '--min-length', type=int, default=120, show_default=True,
              help='Minimum text length for test cases in words')
def assemble_dataset(human_txt, machine_txt, train_ids, test_ids, output_dir, seed, no_source, min_length):
    random.seed(seed)

    if (train_ids and not test_ids) or (test_ids and not train_ids):
        raise click.UsageError('Need to specify either train and test IDs or neither.')

    human_txt = human_txt.rstrip(os.path.sep)
    human_name = os.path.basename(human_txt)

    machine_txt = sorted({m.rstrip(os.path.sep) for m in set(machine_txt) if os.path.isdir(m) and m != human_txt})
    if not machine_txt:
        raise click.UsageError('At least one machine folder must be specified.')

    if train_ids:
        train_ids = sorted({l.strip() for l in train_ids.readlines() if l.strip()})
        test_ids = sorted(sorted({l.strip() for l in test_ids.readlines() if l.strip()}), key=lambda _: random.random())
        if not train_ids or not test_ids:
            raise click.UsageError('Train or test set empty.')
        if train_ids == test_ids:
            raise click.UsageError('Train and test input are the same.')
        for t in test_ids:
            if t in train_ids:
                raise click.UsageError(f'Test ID "{t}" found in training set.')
    else:
        logger.info('No train / test split specified, using all inputs as test.')
        test_ids = sorted({os.path.splitext(p)[0][len(human_txt) + 1:]
                           for p in glob.glob(os.path.join(human_txt, '**', '*.txt'), recursive=True)})
        test_ids = sorted(test_ids, key=lambda _: random.random())

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'machines'), exist_ok=True)

    if train_ids:
        # Output train set
        train_it = tqdm([human_txt] + machine_txt, desc='Assembling train split', unit=' inputs')
        for i, in_dir in enumerate(train_it):
            name = os.path.basename(in_dir)
            if in_dir != human_txt:
                out_name = os.path.join(output_dir, 'machines', name + '.jsonl')
            else:
                out_name = os.path.join(output_dir, 'human.jsonl')
            _write_jsonl(open(out_name, 'w'),
                         _read_texts(os.path.dirname(in_dir), name, train_ids, skip_empty=False, normalize=True), name)

    # Pairwise output with randomised (human, machine) or (machine, human) pairs
    for machine in tqdm(machine_txt, desc='Assembling pairwise test split', unit=' inputs'):
        machine_name = os.path.basename(machine)
        h_it = _read_texts(os.path.dirname(human_txt), human_name, test_ids, normalize=True)
        m_it = _read_texts(os.path.dirname(machine), machine_name, test_ids, normalize=True)
        out_name = os.path.join(output_dir, 'machines', machine_name + '-test.jsonl')
        out_name_truth = os.path.join(output_dir, 'machines', machine_name + '-test-truth.jsonl')

        with open(out_name, 'w') as out, open(out_name_truth, 'w') as out_truth:
            for (i, t1), (_, t2) in zip(h_it, m_it):
                l1, l2 = True, False
                if random.randint(0, 1):
                    t1, t2 = t2, t1
                    l1, l2 = l2, l1
                case_id = '/'.join((machine_name, i))
                random_case_id = urlsafe_b64encode(
                    uuid.UUID(int=random.getrandbits(128), version=4).bytes).decode().rstrip('=')

                min_len_case = min(len(t1), len(t2))
                if min_len_case < min_length:
                    logger.warning('Skipped test case %s due to short/empty text (%d words).',
                                   case_id, min_len_case)
                    continue

                # Cut texts to same length in white-space-separated words + random margin
                t1, t2 = t1.split(' '), t2.split(' ')
                t_tmp = t1 if len(t1) > len(t2) else t2
                margin = random.randint(30, 50)
                while t_tmp and (len(t_tmp) > min_len_case + margin or
                                 not t_tmp[-1].strip() or t_tmp[-1][-1] not in string.punctuation):
                    t_tmp.pop()
                if len(t_tmp) < min_length:
                    logger.warning('Skipped test case %s due to short/empty text after truncation (%d words).',
                                   case_id, len(t_tmp))
                    continue

                t1, t2 = ' '.join(t1), ' '.join(t2)

                json.dump({
                    'id': random_case_id,
                    'text1': t1,
                    'text2': t2,
                }, out, ensure_ascii=False)
                out.write('\n')

                json.dump({
                    'id': random_case_id,
                    **({'source_id': case_id} if not no_source else {}),
                    'is_human': [l1, l2]
                }, out_truth, ensure_ascii=False)
                out_truth.write('\n')


if __name__ == '__main__':
    main()
