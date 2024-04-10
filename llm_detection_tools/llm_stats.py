from collections import defaultdict
from pathlib import Path
import re
import unicodedata

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm


def normalize_text(string):
    """
    Normalize a text string.

    :param string: input string
    :return: normalized string
    """
    string = string.replace("\xef\xbb\xbf", "")                           # remove UTF-8 BOM
    string = string.replace("\ufeff", "")                                 # remove UTF-16 BOM
    string = unicodedata.normalize("NFKD", string)                 # convert to NFKD normal form
    string = re.compile(r"[0-9]").sub("0", string)                   # map all numbers to "0"
    string = re.compile(r"''|``|[\"„“”‘’«»]").sub("'", string)       # normalize quotes
    string = re.compile(r"[‒–—―]+|-{2,}").sub("--", string)          # normalize dashes
    string = re.compile(r"\s+").sub(" ", string)                     # collapse whitespace characters

    return string.strip()


def char_ngram_freq(text, n=3):
    """
    Product character n-gram frequency dict.

    :param text: input text
    :param n: n-gram order
    :return: frequency distribution
    """

    freq_dist = defaultdict(int)
    for i in range(len(text) - n + 1):
        freq_dist[text[i:i + n]] += 1
    return freq_dist


@click.group()
def main():
    pass


def log_range(start, stop, step, base=2):
    i = start
    stop = base ** stop
    while (e := base ** i) <= stop:
        yield e
        i += step


@main.command()
@click.argument('input_dir', type=click.Path(exists=True), nargs=-1)
def plot_entropy(input_dir):
    """
    Plot the entropy w.r.t text length of human or LLM texts.
    """
    entropies = []

    for i in tqdm(input_dir, desc='Loading input directories', unit=' dirs'):
        dir_path = Path(i)
        if not dir_path.is_dir():
            continue

        for p in tqdm(dir_path.rglob('*.txt'), desc='Reading text files', leave=False, unit=' files'):
            t = normalize_text(open(p, errors='ignore').read())

            chunk_len = 100
            for chunk_idx in log_range(np.log2(chunk_len), np.log2(len(t)), 0.1):
                freq = char_ngram_freq(t[:int(chunk_idx)])
                freq = np.fromiter(freq.values(), dtype=np.float32, count=len(freq))
                freq /= np.sum(freq)
                entropy = -np.sum(freq * np.log2(freq))
                entropies.append((dir_path.name, chunk_idx + chunk_len, entropy))

    entropies = pd.DataFrame(entropies, columns=['Model', 'Text Length', 'Entropy (bits)'])

    plt.figure(figsize=(8.5, 6))
    ax = sns.lineplot(data=entropies, x='Text Length', y='Entropy (bits)', hue='Model',
                      style='Model', errorbar=['ci', 95])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xscale('log', base=2)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
