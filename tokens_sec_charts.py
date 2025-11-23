#!/usr/bin/env python3
"""
Generate tokens/sec chart from timing_results.csv

Input:
  timing_results.csv with columns:
    model, text_id, len_chars, len_tokens, elapsed_sec_avg

Outputs:
  results/plots/mean_tokens_per_sec_per_model.png

Usage:
  python tokens_sec_charts.py --timing_csv results/timing_results.csv --outdir results/plots
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timing_csv', type=str, default='results/timing_results.csv')
    parser.add_argument('--outdir', type=str, default='results/plots')
    args = parser.parse_args()

    ensure_outdir(args.outdir)
    df = pd.read_csv(args.timing_csv)

    # compute tokens/sec per row
    df = df.copy()
    df['tokens_per_sec'] = df['len_tokens'] / df['elapsed_sec_avg']

    # mean per model
    means = df.groupby('model', as_index=False)['tokens_per_sec'].mean()

    # save augmented CSV (optional helper)
    out_aug = os.path.join(os.path.dirname(args.timing_csv), 'timing_with_tokens_per_sec.csv')
    df.to_csv(out_aug, index=False)

    # plot
    plt.figure()
    plt.bar(means['model'], means['tokens_per_sec'])
    plt.title('Mean Tokens per Second by Model')
    plt.xlabel('Model')
    plt.ylabel('Tokens per Second')
    plt.tight_layout()
    out_png = os.path.join(args.outdir, 'mean_tokens_per_sec_per_model.png')
    plt.savefig(out_png)
    plt.close()

    print(f"[OK] Wrote chart to:\n  {out_png}\n[OK] Wrote augmented CSV to:\n  {out_aug}")

if __name__ == '__main__':
    main()
