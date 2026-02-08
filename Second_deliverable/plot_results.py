#!/usr/bin/env python3
import sys
import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

FIXED_OMP = 4
XTICKS = [1, 2, 4, 8, 16, 32, 64, 128]

USAGE = (
    "Usage:\n"
    "  python3 plot_results.py "
    "1d_spmv_results_mpi.csv 1d_spmv_results_hybrid.csv "
    "2d_spmv_results_mpi.csv 2d_spmv_results_hybrid.csv\n"
)

if len(sys.argv) != 5:
    print(USAGE)
    sys.exit(1)

CSV_1D_MPI = sys.argv[1]
CSV_1D_HYB = sys.argv[2]
CSV_2D_MPI = sys.argv[3]
CSV_2D_HYB = sys.argv[4]

TIME_COL   = "spmv_total_p90_ms"
COMM_COL   = "spmv_comm_p90_ms"
COMP_COL   = "spmv_comp_p90_ms"
GFLOPS_COL = "gflops"
DIST_COL   = "dist_sum_mb"
ITER_COL   = "iter_sum_mb"

RM_ROOT  = "root"
RM_PAR   = "parallel"
RM_MPIIO = "mpiio"


def load_csv(path, tag):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["run_tag"] = tag
    return df


df = pd.concat([
    load_csv(CSV_1D_MPI, "1D_MPI"),
    load_csv(CSV_1D_HYB, "1D_HYB"),
    load_csv(CSV_2D_MPI, "2D_MPI"),
    load_csv(CSV_2D_HYB, "2D_HYB"),
], ignore_index=True)

# keep only OK
if "status" in df.columns:
    df = df[df["status"] == "OK"].copy()

# normalize
df["read_mode"] = df.get("read_mode", RM_MPIIO).astype(str).str.lower().str.strip()
df["matrix_name"] = df["matrix"].astype(str).apply(os.path.basename)

num_cols = ["mpi_ranks", "omp_threads", TIME_COL, COMM_COL, COMP_COL,
            GFLOPS_COL, DIST_COL, ITER_COL]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["mpi_ranks", "omp_threads", TIME_COL]).copy()
df["mpi_ranks"] = df["mpi_ranks"].astype(int)
df["omp_threads"] = df["omp_threads"].astype(int)

df_strong = df[df["scaling_type"] == "strong"].copy()
df_weak   = df[df["scaling_type"] == "weak"].copy()
strong_mats = sorted(df_strong["matrix_name"].unique())

BASE = "plots"
DIRS = [
    "report_time",
    "report_speedup_eff",
    "report_weak",
    "report_comm_comp",
    "report_comm_volume",
    "report_best",
    "report_1d_readmode",
    "extra_all_modes",
    "extra_parallel_vs_mpiio",
]
for d in DIRS:
    os.makedirs(os.path.join(BASE, d), exist_ok=True)


def setup_rank_axis(ax, logy=False):
    # Robust log2 x-axis across Matplotlib versions.
    try:
        ax.set_xscale("log", base=2)
    except TypeError:
        ax.set_xscale("log", basex=2)

    ax.set_xticks(XTICKS)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(style="plain", axis="x")
    ax.set_xlim(0.9, 140)

    if logy:
        ax.set_yscale("log")

    ax.grid(True, which="both", alpha=0.35)


def plot_line(ax, x, y, label, marker):
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    if d.empty:
        return False
    d = d.sort_values("x")
    ax.plot(d["x"].values, d["y"].values, linewidth=2, label=label)
    ax.scatter(d["x"].values, d["y"].values, s=55, marker=marker,
               edgecolors="black", linewidths=0.6, zorder=5)
    return True


def best_per_rank(d):
    return (
        d.sort_values(["mpi_ranks", TIME_COL])
         .groupby("mpi_ranks", as_index=False)
         .first()
         .sort_values("mpi_ranks")
    )


def get_series(df_in, matrix, tag, rm=None, omp=None, choose_best=True):
    d = df_in[(df_in["matrix_name"] == matrix) & (df_in["run_tag"] == tag)]
    if rm is not None:
        d = d[d["read_mode"] == rm]
    if omp is not None:
        d = d[d["omp_threads"] == omp]
    if d.empty:
        return d
    d = d.sort_values("mpi_ranks")
    return best_per_rank(d) if choose_best else d


def choose_distread(df_in, matrix, tag, omp):
    # Prefer mpiio, else parallel
    for rm in [RM_MPIIO, RM_PAR]:
        d = get_series(df_in, matrix, tag, rm=rm, omp=omp, choose_best=True)
        if not d.empty:
            return d, rm
    return pd.DataFrame(), None


# ==============================================================
# A) EXTRA: all modes (strong)
# ==============================================================
for matrix in strong_mats:
    fig, ax = plt.subplots()
    ok = False

    for rm, mk in [(RM_ROOT, "^"), (RM_PAR, "o"), (RM_MPIIO, "o")]:
        d = get_series(df_strong, matrix, "1D_MPI", rm=rm, omp=1, choose_best=True)
        ok |= plot_line(ax, d.mpi_ranks, d[TIME_COL], f"1D MPI ({rm})", mk)

    for rm, mk in [(RM_ROOT, "D"), (RM_PAR, "s"), (RM_MPIIO, "s")]:
        d = get_series(df_strong, matrix, "1D_HYB", rm=rm, omp=FIXED_OMP, choose_best=True)
        ok |= plot_line(ax, d.mpi_ranks, d[TIME_COL], f"1D HYB ({rm}, omp={FIXED_OMP})", mk)

    for rm, mk in [(RM_PAR, "v"), (RM_MPIIO, "v")]:
        d = get_series(df_strong, matrix, "2D_MPI", rm=rm, omp=1, choose_best=True)
        ok |= plot_line(ax, d.mpi_ranks, d[TIME_COL], f"2D MPI ({rm})", mk)

    for rm, mk in [(RM_PAR, "P"), (RM_MPIIO, "P")]:
        d = get_series(df_strong, matrix, "2D_HYB", rm=rm, omp=FIXED_OMP, choose_best=True)
        ok |= plot_line(ax, d.mpi_ranks, d[TIME_COL], f"2D HYB ({rm}, omp={FIXED_OMP})", mk)

    if ok:
        ax.set_xlabel("MPI ranks")
        ax.set_ylabel("SpMV total time [ms]")
        ax.set_title(f"{matrix} – Strong scaling (ALL modes)")
        setup_rank_axis(ax, logy=True)
        ax.legend(fontsize=8)
        fig.savefig(f"{BASE}/extra_all_modes/{matrix}_strong_allmodes.png",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)


# ==============================================================
# B) REPORT: strong scaling time (clean selection)
# ==============================================================
for matrix in strong_mats:
    fig, ax = plt.subplots()
    ok = False

    for tag, rm, lab, mk, omp in [
        ("1D_MPI", RM_ROOT,  "1D MPI (root)",  "^", 1),
        ("1D_MPI", RM_MPIIO, "1D MPI (mpiio)", "o", 1),
        ("1D_HYB", RM_ROOT,  f"1D HYB (root, omp={FIXED_OMP})", "D", FIXED_OMP),
        ("1D_HYB", RM_MPIIO, f"1D HYB (mpiio, omp={FIXED_OMP})", "s", FIXED_OMP),
    ]:
        d = get_series(df_strong, matrix, tag, rm=rm, omp=omp, choose_best=True)
        ok |= plot_line(ax, d.mpi_ranks, d[TIME_COL], lab, mk)

    d, rm = choose_distread(df_strong, matrix, "2D_MPI", 1)
    ok |= plot_line(ax, d.mpi_ranks, d[TIME_COL], f"2D MPI ({rm})", "v")

    d, rm = choose_distread(df_strong, matrix, "2D_HYB", FIXED_OMP)
    ok |= plot_line(ax, d.mpi_ranks, d[TIME_COL], f"2D HYB ({rm}, omp={FIXED_OMP})", "P")

    if ok:
        ax.set_xlabel("MPI ranks")
        ax.set_ylabel("SpMV total time [ms]")
        ax.set_title(f"{matrix} – Strong scaling (report)")
        setup_rank_axis(ax, logy=True)
        ax.legend(fontsize=8)
        fig.savefig(f"{BASE}/report_time/{matrix}_strong_report.png",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)


# ==============================================================
# C) REPORT: speedup + efficiency
# ==============================================================
def speedup_eff(d):
    d = d.sort_values("mpi_ranks").copy()
    if d.empty:
        return d
    if 1 in set(d["mpi_ranks"]):
        base = float(d.loc[d["mpi_ranks"] == 1, TIME_COL].iloc[0])
        base_p = 1
    else:
        base = float(d.iloc[0][TIME_COL])
        base_p = int(d.iloc[0]["mpi_ranks"])
    d["speedup"] = base / d[TIME_COL]
    d["efficiency"] = d["speedup"] / (d["mpi_ranks"] / base_p)
    return d

for matrix in strong_mats:
    series = []

    d = get_series(df_strong, matrix, "1D_MPI", rm=RM_ROOT, omp=1, choose_best=True)
    if not d.empty: series.append(("1D MPI (root)", d, "^"))

    d = get_series(df_strong, matrix, "1D_MPI", rm=RM_MPIIO, omp=1, choose_best=True)
    if not d.empty: series.append(("1D MPI (mpiio)", d, "o"))

    d = get_series(df_strong, matrix, "1D_HYB", rm=RM_MPIIO, omp=FIXED_OMP, choose_best=True)
    if not d.empty: series.append((f"1D HYB (mpiio, omp={FIXED_OMP})", d, "s"))

    d, rm = choose_distread(df_strong, matrix, "2D_MPI", 1)
    if not d.empty: series.append((f"2D MPI ({rm})", d, "v"))

    d, rm = choose_distread(df_strong, matrix, "2D_HYB", FIXED_OMP)
    if not d.empty: series.append((f"2D HYB ({rm}, omp={FIXED_OMP})", d, "P"))

    # speedup
    fig, ax = plt.subplots()
    ok = False
    for lab, d, mk in series:
        dd = speedup_eff(d)
        ok |= plot_line(ax, dd.mpi_ranks, dd["speedup"], lab, mk)

    if ok:
        ax.set_xlabel("MPI ranks")
        ax.set_ylabel("Speedup")
        ax.set_title(f"{matrix} – Strong scaling speedup")
        setup_rank_axis(ax, logy=False)
        ax.legend(fontsize=8)
        fig.savefig(f"{BASE}/report_speedup_eff/{matrix}_speedup.png",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)

    # efficiency
    fig, ax = plt.subplots()
    ok = False
    for lab, d, mk in series:
        dd = speedup_eff(d)
        ok |= plot_line(ax, dd.mpi_ranks, dd["efficiency"], lab, mk)

    if ok:
        ax.set_xlabel("MPI ranks")
        ax.set_ylabel("Parallel efficiency")
        ax.set_title(f"{matrix} – Strong scaling efficiency")
        setup_rank_axis(ax, logy=False)
        ax.set_ylim(bottom=0.0)
        ax.legend(fontsize=8)
        fig.savefig(f"{BASE}/report_speedup_eff/{matrix}_efficiency.png",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)


# ==============================================================
# D) REPORT: weak scaling (time)
# ==============================================================
fig, ax = plt.subplots()
ok = False

for tag, omp, mk, lab in [
    ("1D_MPI", 1, "o", "1D MPI"),
    ("2D_MPI", 1, "s", "2D MPI"),
    ("1D_HYB", FIXED_OMP, "^", f"1D HYB (omp={FIXED_OMP})"),
    ("2D_HYB", FIXED_OMP, "D", f"2D HYB (omp={FIXED_OMP})"),
]:
    d = df_weak[(df_weak["run_tag"] == tag) & (df_weak["omp_threads"] == omp)]
    if d.empty:
        continue
    d = best_per_rank(d)
    ok |= plot_line(ax, d.mpi_ranks, d[TIME_COL], lab, mk)

if ok:
    ax.set_xlabel("MPI ranks (problem size ∝ ranks)")
    ax.set_ylabel("SpMV total time [ms]")
    ax.set_title("Weak scaling – total time")
    setup_rank_axis(ax, logy=True)
    ax.legend(fontsize=9)
    fig.savefig(f"{BASE}/report_weak/weak_scaling_total_time.png",
                dpi=200, bbox_inches="tight")
plt.close(fig)


# ==============================================================
# E) REPORT: comm vs comp
#    - For 1D: show COMP + COMM + OVERHEAD so they sum to TOTAL
#    - For 2D: keep COMP + COMM + OVERHEAD as well (it’s also useful)
# ==============================================================
def comm_comp_with_overhead(matrix, tag, rm, omp, title, outpath):
    d = get_series(df_strong, matrix, tag, rm=rm, omp=omp, choose_best=True)
    if d.empty:
        return False
    if (COMM_COL not in d.columns) or (COMP_COL not in d.columns):
        return False

    d = d.sort_values("mpi_ranks").copy()

    x = d["mpi_ranks"].to_numpy(dtype=float)
    comp = np.nan_to_num(d[COMP_COL].to_numpy(dtype=float), nan=0.0)
    comm = np.nan_to_num(d[COMM_COL].to_numpy(dtype=float), nan=0.0)
    total = np.nan_to_num(d[TIME_COL].to_numpy(dtype=float), nan=0.0)

    overhead = total - (comp + comm)

    # Make overhead non-negative for stacking.
    # If it becomes negative (due to measurement noise / different timer scopes),
    # clamp to 0 so stacked areas never exceed total.
    overhead = np.maximum(overhead, 0.0)

    upper_comp = comp
    upper_comm = comp + comm
    upper_over = comp + comm + overhead  # should match total after clamp

    fig, ax = plt.subplots()

    ax.fill_between(x, 0.0, upper_comp, alpha=0.35, label="COMP (p90) [ms]")
    ax.fill_between(x, upper_comp, upper_comm, alpha=0.35, label="COMM (p90) [ms]")
    ax.fill_between(x, upper_comm, upper_over, alpha=0.35, label="OVERHEAD = TOTAL-(COMP+COMM) [ms]")

    ax.plot(x, total, linewidth=2.2, label="TOTAL (p90) [ms]")
    ax.scatter(x, total, s=55, edgecolors="black", linewidths=0.6, zorder=5)

    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("Time [ms]")
    ax.set_title(title)

    # X log2, Y linear (composition readable)
    setup_rank_axis(ax, logy=False)

    ax.legend(fontsize=8)
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


for matrix in strong_mats:
    # 1D MPI root + mpiio
    comm_comp_with_overhead(
        matrix, "1D_MPI", RM_ROOT, 1,
        "1D MPI (root) – Total time composition (COMP+COMM+OVERHEAD)",
        f"{BASE}/report_comm_comp/{matrix}_1d_mpi_root_commcomp_overhead.png"
    )
    comm_comp_with_overhead(
        matrix, "1D_MPI", RM_MPIIO, 1,
        "1D MPI (mpiio) – Total time composition (COMP+COMM+OVERHEAD)",
        f"{BASE}/report_comm_comp/{matrix}_1d_mpi_mpiio_commcomp_overhead.png"
    )

    # 1D HYB mpiio
    comm_comp_with_overhead(
        matrix, "1D_HYB", RM_MPIIO, FIXED_OMP,
        f"1D HYB (mpiio, omp={FIXED_OMP}) – Total time composition (COMP+COMM+OVERHEAD)",
        f"{BASE}/report_comm_comp/{matrix}_1d_hyb_mpiio_commcomp_overhead.png"
    )

    # 2D choose mpiio else parallel
    d2, rm2 = choose_distread(df_strong, matrix, "2D_MPI", 1)
    if rm2 is not None:
        comm_comp_with_overhead(
            matrix, "2D_MPI", rm2, 1,
            f"2D MPI ({rm2}) – Total time composition (COMP+COMM+OVERHEAD)",
            f"{BASE}/report_comm_comp/{matrix}_2d_mpi_{rm2}_commcomp_overhead.png"
        )

    d2, rm2 = choose_distread(df_strong, matrix, "2D_HYB", FIXED_OMP)
    if rm2 is not None:
        comm_comp_with_overhead(
            matrix, "2D_HYB", rm2, FIXED_OMP,
            f"2D HYB ({rm2}, omp={FIXED_OMP}) – Total time composition (COMP+COMM+OVERHEAD)",
            f"{BASE}/report_comm_comp/{matrix}_2d_hyb_{rm2}_commcomp_overhead.png"
        )


# ==============================================================
# F) REPORT: communication volume
# ==============================================================
for matrix in strong_mats:
    fig, ax = plt.subplots()
    ok = False

    selections = [
        ("1D MPI (root)", get_series(df_strong, matrix, "1D_MPI", rm=RM_ROOT, omp=1, choose_best=True), "o"),
        ("1D MPI (mpiio)", get_series(df_strong, matrix, "1D_MPI", rm=RM_MPIIO, omp=1, choose_best=True), "s"),
    ]
    d2, rm2 = choose_distread(df_strong, matrix, "2D_MPI", 1)
    if not d2.empty:
        selections.append((f"2D MPI ({rm2})", d2, "^"))

    for lab, d, mk in selections:
        if d.empty or (DIST_COL not in d.columns) or (ITER_COL not in d.columns):
            continue
        ok |= plot_line(ax, d.mpi_ranks, d[DIST_COL], f"{lab} – dist_sum_mb", mk)
        ok |= plot_line(ax, d.mpi_ranks, d[ITER_COL], f"{lab} – iter_sum_mb", mk)

    if ok:
        ax.set_xlabel("MPI ranks")
        ax.set_ylabel("Communication volume [MB]")
        ax.set_title(f"{matrix} – Communication volume")
        setup_rank_axis(ax, logy=True)
        ax.legend(fontsize=7)
        fig.savefig(f"{BASE}/report_comm_volume/{matrix}_comm_volume.png",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)


# ==============================================================
# G) REPORT: best-of-all comparison (global best per rank)
# ==============================================================
for matrix in strong_mats:
    dmat = df_strong[df_strong["matrix_name"] == matrix].copy()
    if dmat.empty:
        continue

    best = best_per_rank(dmat)

    fig, ax = plt.subplots()
    ok = plot_line(ax, best.mpi_ranks, best[TIME_COL], "Best configuration (min total time)", "o")
    if ok:
        ax.set_xlabel("MPI ranks")
        ax.set_ylabel("SpMV total time [ms]")
        ax.set_title(f"{matrix} – Best per MPI rank (global min)")
        setup_rank_axis(ax, logy=True)
        fig.savefig(f"{BASE}/report_best/{matrix}_best_per_rank.png",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)


# ==============================================================
# H) REPORT: 1D read-mode comparison
# ==============================================================
for matrix in strong_mats:
    fig, ax = plt.subplots()
    ok = False

    for rm, mk in [(RM_ROOT, "^"), (RM_PAR, "o"), (RM_MPIIO, "s")]:
        d = get_series(df_strong, matrix, "1D_MPI", rm=rm, omp=1, choose_best=True)
        ok |= plot_line(ax, d.mpi_ranks, d[TIME_COL], f"1D MPI ({rm})", mk)

    if ok:
        ax.set_xlabel("MPI ranks")
        ax.set_ylabel("SpMV total time [ms]")
        ax.set_title(f"{matrix} – 1D MPI read-mode comparison")
        setup_rank_axis(ax, logy=True)
        ax.legend(fontsize=8)
        fig.savefig(f"{BASE}/report_1d_readmode/{matrix}_1d_readmodes.png",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)


# ==============================================================
# I) EXTRA: parallel vs mpiio
# ==============================================================
for matrix in strong_mats:
    fig, ax = plt.subplots()
    ok = False

    for tag, omp, mk in [
        ("1D_MPI", 1, "o"),
        ("1D_HYB", FIXED_OMP, "s"),
        ("2D_MPI", 1, "^"),
        ("2D_HYB", FIXED_OMP, "D"),
    ]:
        for rm in [RM_PAR, RM_MPIIO]:
            d = get_series(df_strong, matrix, tag, rm=rm, omp=omp, choose_best=True)
            if d.empty:
                continue
            ok |= plot_line(ax, d.mpi_ranks, d[TIME_COL], f"{tag} ({rm}, omp={omp})", mk)

    if ok:
        ax.set_xlabel("MPI ranks")
        ax.set_ylabel("SpMV total time [ms]")
        ax.set_title(f"{matrix} – parallel vs mpiio (extra)")
        setup_rank_axis(ax, logy=True)
        ax.legend(fontsize=7)
        fig.savefig(f"{BASE}/extra_parallel_vs_mpiio/{matrix}_parallel_vs_mpiio.png",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)

print("All plots generated successfully.")
