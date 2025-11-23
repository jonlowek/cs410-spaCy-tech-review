#!/usr/bin/env python3
"""
Generate charts for SpaCy Tech Review timing results.

Reads: results/timing_results.csv (columns: model, text_id, len_chars, len_tokens, elapsed_sec_avg)
Outputs:
  - results/plots/mean_elapsed_per_model.png
  - results/plots/elapsed_per_textid_per_model.png

Usage:
  python make_charts.py --timing_csv results/timing_results.csv --outdir results/plots
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def mean_elapsed_per_model(df: pd.DataFrame, out_path: str) -> None:
    means = df.groupby('model', as_index=False)['elapsed_sec_avg'].mean()
    plt.figure()
    plt.bar(means['model'], means['elapsed_sec_avg'])
    plt.title('Mean Elapsed Seconds per Model')
    plt.xlabel('Model')
    plt.ylabel('Mean Elapsed Seconds')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def elapsed_per_textid_per_model(df: pd.DataFrame, out_path: str) -> None:
    # Pivot to have models as columns, text_id as index
    pivot = df.pivot_table(index='text_id', columns='model', values='elapsed_sec_avg', aggfunc='mean')
    # Bar chart with grouped bars per text_id
    ax = pivot.plot(kind='bar')
    ax.set_title('Elapsed Seconds per Text (Grouped by Model)')
    ax.set_xlabel('Text ID')
    ax.set_ylabel('Elapsed Seconds')
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timing_csv', type=str, default='results/timing_results.csv')
    parser.add_argument('--outdir', type=str, default='results/plots')
    args = parser.parse_args()

    ensure_outdir(args.outdir)
    df = pd.read_csv(args.timing_csv)

    mean_path = os.path.join(args.outdir, 'mean_elapsed_per_model.png')
    mean_elapsed_per_model(df, mean_path)

    grouped_path = os.path.join(args.outdir, 'elapsed_per_textid_per_model.png')
    elapsed_per_textid_per_model(df, grouped_path)

    print(f"[OK] Wrote charts to:\n  {mean_path}\n  {grouped_path}")

if __name__ == '__main__':
    main()
