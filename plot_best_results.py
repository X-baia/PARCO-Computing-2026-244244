#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


INPUT_CSV = "final_results_best.csv"  
OUTPUT_DIR = "plots"
MATRICES = None  


import os
os.makedirs(OUTPUT_DIR, exist_ok=True)


df = pd.read_csv(INPUT_CSV, sep=';')


if MATRICES is None:
    MATRICES = df['matrix'].unique()


def plot_bar(df, field, ylabel, title_suffix):
    for mat in MATRICES:
        sub = df[df['matrix'] == mat]
        approaches = ['sequential', 'parallel(omp+csr)', 'blocked']
        values = []
        for a in approaches:
            row = sub[sub['approach'] == a]
            if len(row) == 1:
                values.append(float(row[field]))
            else:
                values.append(np.nan)  
        
        plt.figure(figsize=(6,4))
        bars = plt.bar(approaches, values, color=['skyblue', 'orange', 'green'])
        plt.ylabel(ylabel)
        plt.title(f"{mat} {title_suffix}")
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                plt.text(bar.get_x() + bar.get_width()/2, val*1.01, f"{val:.3f}", ha='center', fontsize=9)
        plt.tight_layout()
        fname = os.path.join(OUTPUT_DIR, f"{mat}_{field}.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")


plot_bar(df, 'p90_ms', 'Execution time [ms]', 'p90')
plot_bar(df, 'gflops', 'GFLOPs', 'GFLOPs')
plot_bar(df, 'bandwidth', 'Bandwidth [GB/s]', 'Bandwidth')

print("All plots generated in folder:", OUTPUT_DIR)
