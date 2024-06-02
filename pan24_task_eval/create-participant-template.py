#!/usr/bin/env python3
import json
from tqdm import tqdm
from pathlib import Path
from shutil import copytree, rmtree
from subprocess import check_output
from statistics import median, mean
from numpy import quantile

TEAM_TO_SUBMISSIONS = json.load(open('team-to-submissions.json'))
EVALUATION_SCORES = json.load(open('evaluation-scores.json'))

scores_per_dataset = {}

for t, submissions in EVALUATION_SCORES.items():
    for submission, evaluations in submissions.items():
        for d, e in evaluations.items():
            if d not in scores_per_dataset:
                scores_per_dataset[d] = {}
            for k,v in e.items():
                if k not in scores_per_dataset[d]:
                    scores_per_dataset[d][k] = []
                scores_per_dataset[d][k] += [v]

def load_evaluation(team, software, dataset='pan24-generative-authorship-test-20240502-test'):
    return EVALUATION_SCORES[team][software][dataset]

def table_line(team, software, display_name=None):
    ret = load_evaluation(team, software)
    return (display_name if display_name else software.replace('_', '\\_')) + ' & ' + (' & '.join([str(ret[i]) for i in ['roc-auc', 'brier', 'c@1', 'f1', 'f05u', 'mean']]))

def f(i):
    return "{:.3f}".format(i)

def table_line_aggregated(display_name, aggregation):
    ret = scores_per_dataset['pan24-generative-authorship-test-20240502-test']
    return (display_name if display_name else software.replace('_', '\\_')) + ' & ' + (' & '.join([f(aggregation(ret[i])) for i in ['roc-auc', 'brier', 'c@1', 'f1', 'f05u', 'mean']]))

def report_table(team):
    lines = []
    
    for software in TEAM_TO_SUBMISSIONS[team]:
        lines += [table_line(team, software)]
    
    lines = '\\\\\n'.join(lines)

    return """\\begin{table}[t]
\\centering
\\caption{Overview of the accuracy in detecting if a text is written by an human in task~4 on PAN~2024 (Voight-Kampff Generative AI Authorship Verification). We report ROC-AUC, Brier, C@1, F$_{1}$, F$_{0.5u}$ and their mean.}
\\label{table-obfuscation-results}
\\renewcommand{\\tabcolsep}{4.5pt}
\\begin{tabular}{@{}lcccccc@{}}
\\toprule
  \\textbf{Approach}       & \\textbf{ROC-AUC} & \\textbf{Brier} & \\textbf{C@1} & \\textbf{F$_{1}$} & \\textbf{F$_{0.5u}$} & \\textbf{Mean} \\\\
\\midrule
  """ + lines + """\\\\
\\midrule
  """ + table_line('baseline', 'baseline-binoculars', 'Baseline Binoculars') + """\\\\
  """ + table_line('baseline', 'baseline-fastdetectgpt-mistral', 'Baseline Fast-DetectGPT (Mistral)') + """\\\\
  """ + table_line('baseline', 'baseline-ppmd', 'Baseline PPMd') + """\\\\
  """ + table_line('baseline', 'baseline-unmasking', 'Baseline Unmasking') + """\\\\
  """ + table_line('baseline', 'baseline-fastdetectgpt', 'Baseline Fast-DetectGPT') + """\\\\
\\midrule
  """ + table_line_aggregated('95-th quantile', lambda i: quantile(i, 0.95)) + """\\\\
  """ + table_line_aggregated('75-th quantile', lambda i: quantile(i, 0.75)) + """\\\\
  """ + table_line_aggregated('Median', median) + """\\\\
  """ + table_line_aggregated('25-th quantile', lambda i: quantile(i, 0.25)) + """\\\\
  """ + table_line_aggregated('Min', min) + """\\\\
\\bottomrule
\\end{tabular}
\\end{table}
"""


def table_obfuscation_line(team, software, display_name=None):
    scores = []
    for d, e in EVALUATION_SCORES[team][software].items():
        scores += [e['mean']]
    
    display_name = (display_name if display_name else software.replace('_', '\\_')) + ' & '
    
    return display_name + (' & '.join([f(min(scores)), f(quantile(scores, 0.25)), f(median(scores)), f(quantile(scores, 0.75)), f(max(scores))]))

def table_obfuscation_line_aggregated(display_name, aggregation):
    scores = []
    for d, e in scores_per_dataset.items():
        scores += [aggregation(e['mean'])]

    return display_name + '& ' + (' & '.join([f(min(scores)), f(quantile(scores, 0.25)), f(median(scores)), f(quantile(scores, 0.75)), f(max(scores))]))


def report_obfuscation_table(team):
    lines = []
    
    for software in TEAM_TO_SUBMISSIONS[team]:
        lines += [table_obfuscation_line(team, software)]
    
    lines = '\\\\\n'.join(lines)

    return """\\begin{table}[t]
\\centering
\\caption{Overview of the mean accuracy over 9~variants of the test set. We report the minumum, median, the maximum, the 25-th, and the 75-th quantile, of the mean per the 9~datasets.}
\\label{table-evaluation-results}
\\renewcommand{\\tabcolsep}{4.5pt}
\\begin{tabular}{@{}lccccc@{}}
\\toprule
  \\textbf{Approach}       & \\textbf{Minimum} & \\textbf{25-th Quantile} & \\textbf{Median} & \\textbf{75-th Quantile} & \\textbf{Max} \\\\
\\midrule
  """ + lines + """\\\\
\\midrule
  """ + table_obfuscation_line('baseline', 'baseline-binoculars', 'Baseline Binoculars') + """\\\\
  """ + table_obfuscation_line('baseline', 'baseline-detectgpt-mistral', 'Baseline DetectGPT (Mistral)') + """\\\\
  """ + table_obfuscation_line('baseline', 'baseline-detectgpt-falcon', 'Baseline DetectGPT (Falcon)') + """\\\\
  """ + table_obfuscation_line('baseline', 'baseline-detectllm-npr-mistral', 'Baseline DetectLLM NPR (Mistral)') + """\\\\
  """ + table_obfuscation_line('baseline', 'baseline-detectllm-npr-falcon', 'Baseline DetectLLM NPR (Falcon)') + """\\\\
  """ + table_obfuscation_line('baseline', 'baseline-detectllm-lrr-mistral', 'Baseline DetectLLM NPR (Mistral)') + """\\\\
  """ + table_obfuscation_line('baseline', 'baseline-detectllm-lrr', 'Baseline DetectLLM NPR (Falcon)') + """\\\\
  """ + table_obfuscation_line('baseline', 'baseline-fastdetectgpt-mistral', 'Baseline Fast-DetectGPT (Mistral)') + """\\\\
  """ + table_obfuscation_line('baseline', 'baseline-fastdetectgpt', 'Baseline Fast-DetectGPT (Falcon)') + """\\\\
  """ + table_obfuscation_line('baseline', 'baseline-ppmd', 'Baseline PPMd') + """\\\\
  """ + table_obfuscation_line('baseline', 'baseline-unmasking', 'Baseline Unmasking') + """\\\\

\\midrule
  """ + table_obfuscation_line_aggregated('95-th quantile', lambda i: quantile(i, 0.95)) + """\\\\
  """ + table_obfuscation_line_aggregated('75-th quantile', lambda i: quantile(i, 0.75)) + """\\\\
  """ + table_obfuscation_line_aggregated('Median', median) + """\\\\
  """ + table_obfuscation_line_aggregated('25-th quantile', lambda i: quantile(i, 0.25)) + """\\\\
  """ + table_obfuscation_line_aggregated('Min', min) + """\\\\
\\bottomrule
\\end{tabular}
\\end{table}
"""


def process_team(team):
    results_table = report_table(team)
    obfuscation_results_table = report_obfuscation_table(team)
    output_dir = Path('output')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = output_dir / f'notebook-{team}'

    copytree('template-pan-2024', output_dir)

    open(output_dir / 'table-evaluation-results.tex', 'w').write(results_table)
    open(output_dir / 'table-obfuscation-results.tex', 'w').write(obfuscation_results_table)
    paper = open(output_dir / 'pan24-paper.tex', 'r').read()
    paper = paper.replace('<team name>', f'Team {team}')
    open(output_dir / 'pan24-paper.tex', 'w').write(paper)

    check_output(['bash', '-c', f'cd {output_dir.absolute()} && pdflatex pan24-paper.tex && pdflatex pan24-paper.tex'])
    check_output(['bash', '-c', f'cd {Path("output").absolute()} && zip -r notebook-{team}.zip notebook-{team}/*'])
    rmtree(output_dir)

for team in tqdm(TEAM_TO_SUBMISSIONS):
    if len(TEAM_TO_SUBMISSIONS[team]) <= 0:
        print(f'TODO: Finalize team {team}')
        continue
    process_team(team)

