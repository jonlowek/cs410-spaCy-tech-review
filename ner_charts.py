#!/usr/bin/env python3
"""
Generate NER metric charts from ner_eval.csv

Input:
  ner_eval.csv with columns:
    model, sample_id, gold_count, pred_count, precision, recall, f1

Outputs:
  results/plots/ner_macro_prf_per_model.png  (bar chart: macro Precision/Recall/F1 per model)
  results/plots/ner_f1_per_model.png         (bar chart: macro F1 per model only)

Usage:
  python ner_charts.py --ner_csv results/ner_eval.csv --outdir results/plots
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def macro_average(df: pd.DataFrame) -> pd.DataFrame:
    # group by model and average precision/recall/f1
    grouped = df.groupby('model', as_index=False)[['precision','recall','f1']].mean()
    return grouped

def plot_prf(grouped: pd.DataFrame, out_path: str) -> None:
    # Transform to long format for grouped bars
    long = grouped.melt(id_vars='model', value_vars=['precision','recall','f1'], var_name='metric', value_name='score')
    # Pivot so each metric is a series
    pivot = long.pivot_table(index='model', columns='metric', values='score', aggfunc='mean')
    ax = pivot.plot(kind='bar')
    ax.set_title('NER Macro-Averaged Precision/Recall/F1 by Model')
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.legend(title='Metric')
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_f1(grouped: pd.DataFrame, out_path: str) -> None:
    plt.figure()
    plt.bar(grouped['model'], grouped['f1'])
    plt.title('NER Macro-Averaged F1 by Model')
    plt.xlabel('Model')
    plt.ylabel('F1')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ner_csv', type=str, default='results/ner_eval.csv')
    parser.add_argument('--outdir', type=str, default='results/plots')
    args = parser.parse_args()

    ensure_outdir(args.outdir)
    df = pd.read_csv(args.ner_csv)

    grouped = macro_average(df)

    prf_path = os.path.join(args.outdir, 'ner_macro_prf_per_model.png')
    plot_prf(grouped, prf_path)

    f1_path = os.path.join(args.outdir, 'ner_f1_per_model.png')
    plot_f1(grouped, f1_path)

    print(f"[OK] Wrote charts to:\n  {prf_path}\n  {f1_path}")

if __name__ == '__main__':
    main()
