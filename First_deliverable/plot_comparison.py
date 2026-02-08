import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


CSV_FILE = "final_results_all.csv"   
PLOT_DIR = "plots"


os.makedirs(PLOT_DIR, exist_ok=True)

df = pd.read_csv(CSV_FILE, sep=';')

df['threads'] = df['threads'].replace(0, 1)

serial_df = df[df['approach'] == "sequential"]

openmp_df = df[
    (df['approach'] == "parallel(omp+csr)") &
    (df['variant'] == "loop")
]

block_df = df[
    (df['approach'].str.contains("blocked", case=False)) |
    (df['block_rows'] > 0)
]


def best_per_thread(df_in, matrix_name):
    df_m = df_in[df_in['matrix'] == matrix_name]
    if df_m.empty:
        return None
    idx = df_m.groupby('threads')['p90_ms'].idxmin()
    return df_m.loc[idx].sort_values('threads')


def plot_runtime_vs_threads(matrix):
    serial = serial_df[serial_df['matrix'] == matrix]
    omp = best_per_thread(openmp_df, matrix)
    blk = best_per_thread(block_df, matrix)

    threads = sorted(df['threads'].unique())

    plt.figure(figsize=(7,4))

    if not serial.empty:
        serial_time = serial['p90_ms'].values[0]
        plt.plot(threads, [serial_time]*len(threads),
                 marker='o', label="Serial", linewidth=2)

    if omp is not None and not omp.empty:
        plt.plot(omp['threads'], omp['p90_ms'], marker='o',
                 label="OpenMP (best per thread)", linewidth=2)

    if blk is not None and not blk.empty:
        plt.plot(blk['threads'], blk['p90_ms'], marker='o',
                 label="Cache-blocked (best per thread)", linewidth=2)

    plt.xlabel("Threads")
    plt.ylabel("Time (ms)")
    plt.title(f"Runtime vs Threads — {matrix}")
    plt.xticks(threads)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{matrix}_runtime_vs_threads.png")
    plt.close()

def plot_speedup(matrix):
    serial = serial_df[serial_df['matrix'] == matrix]
    omp = best_per_thread(openmp_df, matrix)
    blk = best_per_thread(block_df, matrix)

    if serial.empty: 
        return

    serial_time = serial['p90_ms'].values[0]

    threads = sorted(df['threads'].unique())
    plt.figure(figsize=(7,4))

    if omp is not None and not omp.empty:
        plt.plot(omp['threads'], serial_time / omp['p90_ms'],
                 marker='o', label="OpenMP best", linewidth=2)

    if blk is not None and not blk.empty:
        plt.plot(blk['threads'], serial_time / blk['p90_ms'],
                 marker='o', label="Blocking best", linewidth=2)

    plt.xlabel("Threads")
    plt.ylabel("Speedup (relative to serial)")
    plt.title(f"Speedup vs Threads — {matrix}")
    plt.xticks(threads)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{matrix}_speedup.png")
    plt.close()


def plot_schedule_heatmap(matrix, threads_target=8):
    df_m = openmp_df[(openmp_df['matrix']==matrix) & 
                     (openmp_df['threads']==threads_target)]

    if df_m.empty: 
        return

    pivot = df_m.pivot_table(
        values="p90_ms",
        index="schedule",
        columns="chunk",
        aggfunc='mean'
    )

    plt.figure(figsize=(8,5))
    plt.imshow(pivot, cmap='viridis', aspect='auto')
    plt.colorbar(label="Time (ms)")
    plt.title(f"Schedule × Chunk Heatmap ({threads_target} threads) — {matrix}")
    plt.xlabel("Chunk size")
    plt.ylabel("Schedule")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{matrix}_schedule_heatmap_{threads_target}threads.png")
    plt.close()


def plot_blocksize_sweep(matrix, threads_target=8):
    df_m = block_df[(block_df['matrix']==matrix) &
                    (block_df['threads']==threads_target)]

    if df_m.empty: 
        return

    plt.figure(figsize=(7,4))
    plt.plot(df_m['block_rows'], df_m['p90_ms'], marker='o')
    plt.xlabel("Block rows")
    plt.ylabel("Time (ms)")
    plt.title(f"Block size sweep ({threads_target} threads) — {matrix}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{matrix}_blocksize_sweep_{threads_target}threads.png")
    plt.close()


matrices = df['matrix'].unique()

for m in matrices:
    print(f"Generating plots for {m}...")
    plot_runtime_vs_threads(m)
    plot_speedup(m)
    plot_schedule_heatmap(m, threads_target=8)
    plot_blocksize_sweep(m, threads_target=8)

print("\nAll plots saved in:", PLOT_DIR)
