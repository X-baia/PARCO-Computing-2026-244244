#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


INPUT_CSV = "final_results_best.csv"
OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


df = pd.read_csv(INPUT_CSV, sep=';')
df['approach'] = df['approach'].str.strip()
df['matrix'] = df['matrix'].str.strip()

matrices = df['matrix'].unique()
approaches = ['sequential', 'parallel(omp+csr)', 'blocked']


def plot_grouped_bar(df, field, ylabel, fname):
    n_matrices = len(matrices)
    n_approaches = len(approaches)
    bar_width = 0.2
    x = np.arange(n_matrices) 

    values = []
    for a in approaches:
        vals = []
        for m in matrices:
            row = df[(df['matrix'] == m) & (df['approach'] == a)]
            if len(row) == 1:
                vals.append(float(row[field].iloc[0]))
            else:
                vals.append(np.nan)
        values.append(vals)

    plt.figure(figsize=(max(10, n_matrices*1.5), 6))

    for i, vals in enumerate(values):
        plt.bar(x + i*bar_width, vals, width=bar_width, label=approaches[i])

    plt.xticks(x + bar_width, matrices, rotation=45, ha='right')
    plt.ylabel(ylabel)
    plt.title(field)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, fname))
    plt.close()
    print(f"Saved {fname}")


plot_grouped_bar(df, 'p90_ms', 'Execution Time [ms]', 'all_matrices_p90.png')
plot_grouped_bar(df, 'gflops', 'GFLOPs', 'all_matrices_gflops.png')
plot_grouped_bar(df, 'bandwidth', 'Bandwidth [GB/s]', 'all_matrices_bandwidth.png')

print("All grouped charts created in 'plots/'")
