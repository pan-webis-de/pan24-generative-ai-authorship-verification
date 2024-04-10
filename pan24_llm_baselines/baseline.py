import json
import os

import click
import numpy as np
from tqdm import tqdm


@click.group()
def main():
    """
    PAN'24 Generative Authorship Detection baselines.
    """
    pass


def comparative_score(score1, score2, epsilon=1e-3):
    """
    Return a single score in [0, 1] based on the comparison of two input scores.

    :param score1: first score
    :param score2: second score
    :param epsilon: non-answer (output score = 0.5) epsilon threshold
    :return: [0, 0.5) if score1 > score2; (0.5, 1] if score2 > score1; 0.5 otherwise
    """
    if abs(score1 - score2) < epsilon:
        return 0.5
    if score1 > score2:
        return max(min(1.0 - score1, 0.49), 0.0)
    return min(max(0.51, score2), 1.0)


@main.command()
@click.argument('input_file', type=click.File('r'))
@click.argument('output_directory', type=click.Path(file_okay=False, exists=True))
@click.option('-o', '--outfile-name', help='Output file name', default='binoculars.jsonl', show_default=True)
@click.option('-q', '--quantize', type=click.Choice(['4', '8']))
@click.option('-f', '--flash-attn', is_flag=True, help='Use flash-attn 2 (must be installed separately)')
@click.option('--observer', help='Observer model', default='tiiuae/falcon-7b', show_default=True)
@click.option('--performer', help='Performer model', default='tiiuae/falcon-7b-instruct', show_default=True)
@click.option('--device1', help='Observer model device', default='auto', show_default=True)
@click.option('--device2', help='Performer model device', default='auto', show_default=True)
def binoculars(input_file, output_directory, outfile_name, quantize, flash_attn,
               observer, performer, device1, device2):
    """
    PAN'24 baseline: Binoculars.

    References:
    ===========
        Hans, A., Schwarzschild, A., Cherepanova, V., Kazemi, H., Saha, A.,
        Goldblum, M., ... & Goldstein, T. (2024). Spotting LLMs With
        Binoculars: Zero-Shot Detection of Machine-Generated Text.
        arXiv preprint arXiv:2401.12070.
    """
    from pan24_llm_baselines.thirdparty_binoculars import Binoculars

    bino = Binoculars(
        quantization_bits=quantize,
        use_flash_attn=flash_attn,
        observer_name_or_path=observer,
        performer_name_or_path=performer,
        device1=device1,
        device2=device2)

    with open(os.path.join(output_directory, outfile_name), 'w') as out:
        for l in tqdm(input_file, desc='Predicting cases'):
            j = json.loads(l)
            score1 = bino.compute_score(j['text1'])
            score2 = bino.compute_score(j['text2'])

            json.dump({'id': j['id'], 'is_human': comparative_score(score1, score2)}, out)
            out.write('\n')


@main.command()
@click.argument('input_file', type=click.File('r'))
@click.argument('output_directory', type=click.Path(file_okay=False, exists=True))
@click.option('-o', '--outfile-name', help='Output file name', default='ppmd.jsonl', show_default=True)
def ppmd(input_file, output_directory, outfile_name):
    """
    PAN'24 baseline: Compression-based cosine.

    References:
    ===========
        Sculley, D., & Brodley, C. E. (2006, March). Compression and machine learning:
        A new perspective on feature space vectors. In Data Compression Conference (DCC'06)
        (pp. 332-341). IEEE.

        Halvani, O., Winter, C., & Graner, L. (2017, August). On the usefulness of
        compression models for authorship verification. In Proceedings of the 12th
        international conference on availability, reliability and security (pp. 1-10).
    """
    import pyppmd

    def _cbc(t1, t2):
        cx = len(pyppmd.compress(t1))
        cy = len(pyppmd.compress(t2))
        cxy = len(pyppmd.compress(t1 + t2))
        return 1.0 - (cx + cy - cxy) / np.sqrt(cx * cy)

    with open(os.path.join(output_directory, outfile_name), 'w') as out:
        for l in tqdm(input_file, desc='Predicting cases'):
            j = json.loads(l)

            # Trim texts to the same length
            t1 = j['text1'][:min(len(j['text1']), len(j['text2']))]
            t2 = j['text2'][:min(len(j['text1']), len(j['text2']))]

            # Cut texts into halves and compare them
            cbc1 = _cbc(t1[:len(t1) // 2], t1[len(t1) // 2:])
            cbc2 = _cbc(t2[:len(t2) // 2], t2[len(t2) // 2:])

            json.dump({'id': j['id'], 'is_human': comparative_score(cbc1, cbc2)}, out)
            out.write('\n')


@main.command()
@click.argument('input_file', type=click.File('r'))
@click.argument('output_directory', type=click.Path(file_okay=False, exists=True))
@click.option('-o', '--outfile-name', help='Output file name', default='length.jsonl', show_default=True)
def length(input_file, output_directory, outfile_name):
    """
    PAN'24 baseline: Text length.
    """

    with open(os.path.join(output_directory, outfile_name), 'w') as out:
        for l in tqdm(input_file, desc='Predicting cases'):
            j = json.loads(l)
            l1 = len(j['text1'])
            l2 = len(j['text2'])
            json.dump({'id': j['id'], 'is_human': float(l1 < l2)}, out)
            out.write('\n')


if __name__ == '__main__':
    main()
