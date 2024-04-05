import json
import os

import click
from tqdm import tqdm


@click.group()
def main():
    pass


@main.command()
@click.argument('input_file', type=click.File('r'))
@click.argument('output_directory', type=click.Path(file_okay=False, exists=True))
@click.option('-n', '--out-name', help='Output file name', default='binoculars.jsonl')
@click.option('-q', '--quantization', type=click.Choice(['4', '8']))
@click.option('-f', '--flash-attn', is_flag=True,
              help='Use flash-attn 2 (must be installed separately)')
def binoculars(input_file, output_directory, out_name, quantization, flash_attn):
    from pan24_llm_baselines.thirdparty_binoculars import Binoculars

    bino = Binoculars(quantization_bits=quantization, use_flash_attn=flash_attn)

    with open(os.path.join(output_directory, out_name), 'w') as out:
        for l in tqdm(input_file, desc='Predicting cases'):
            j = json.loads(l)
            score1 = bino.compute_score(j['text1'])
            score2 = bino.compute_score(j['text2'])

            if abs(score1 - score2) < 1e-3:
                score = 0.5
            elif score1 > score2:
                score = max(min(1.0 - score1, 0.49), 0.0)
            else:
                score = min(max(0.51, score2), 1.0)

            json.dump({'id': j['id'], 'is_human': score}, out)
            out.write('\n')


if __name__ == '__main__':
    main()
