# cliffs_delta.py — effect size for SPEC 4.4 (TODO B4)
#
# Reads results/sweep_neurons.tsv and results/master_hit_counts.tsv, splits
# neurons by whether the corpus reached them, and reports Cliff's delta with
# a bootstrap CI. No GPU, no model, a couple of seconds.
#
#     python cliffs_delta.py --results results
#
# delta is P(reached louder) - P(unreached louder), in [-1, 1]. 0 means the
# two groups are interchangeable. Thresholds (Romano et al.): 0.147 small,
# 0.33 medium, 0.474 large.

import argparse, csv, math, os
import numpy as np

COL = {"imp_r": "imp_max_all", "imp_i": "imp_max_last",
       "exp_r": "exp_max",     "exp_i": "exp_max"}
VOID = {"exp_r", "exp_i"}   # E1: single explicit token is a degenerate regime
M = 3072


def read_tsv(path):
    with open(path, encoding="utf-8") as fh:
        lines = [l for l in fh if not l.startswith("#")]
    return list(csv.DictReader(lines, delimiter="\t"))


def delta(a, b):
    """Cliff's delta via sorted searchsorted. O(n log n), exact, ties handled."""
    a = np.sort(np.asarray(a, float))
    b = np.asarray(b, float)
    gt = len(a) * len(b) - np.searchsorted(a, b, side="right").sum()
    lt = np.searchsorted(a, b, side="left").sum()
    return (gt - lt) / (len(a) * len(b))


def boot_ci(a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    ds = np.array([delta(rng.choice(a, len(a), True), rng.choice(b, len(b), True))
                   for _ in range(n)])
    return np.percentile(ds, [2.5, 97.5])


def magnitude(d):
    a = abs(d)
    return ("negligible" if a < 0.147 else "small" if a < 0.33
            else "medium" if a < 0.474 else "large")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()

    rows = read_tsv(os.path.join(a.results, "sweep_neurons.tsv"))
    if len(rows) != M:
        print(f"warning: expected {M} neurons, got {len(rows)}")

    hits = {}
    for r in read_tsv(os.path.join(a.results, "master_hit_counts.tsv")):
        hits.setdefault(r["quadrant"], {})[int(r["neuron"])] = int(r["corpus_hits"])

    print(f"{'quad':6} {'n_R':>5} {'n_U':>5} {'med_R':>8} {'med_U':>8} "
          f"{'delta':>7} {'95% CI':>18} {'P(R>U)':>7}  magnitude")
    for q in ("imp_r", "imp_i", "exp_r", "exp_i"):
        vals = np.array([float(r[COL[q]]) for r in rows])
        h = np.array([hits[q][j] for j in range(len(rows))])
        R, U = vals[h > 0], vals[h == 0]
        d = delta(R, U)
        lo, hi = boot_ci(R, U, a.boot)
        mark = "  <- void, see E1" if q in VOID else ""
        print(f"{q:6} {len(R):5d} {len(U):5d} {np.median(R):8.4f} {np.median(U):8.4f} "
              f"{d:+7.3f} [{lo:+7.3f},{hi:+6.3f}] {(d+1)/2:7.3f}  "
              f"{magnitude(d)}{mark}")

    print()
    print("Overlap detail (implicit quadrants only):")
    for q in ("imp_r", "imp_i"):
        vals = np.array([float(r[COL[q]]) for r in rows])
        h = np.array([hits[q][j] for j in range(len(rows))])
        hi_q, lo_q = np.quantile(vals, 0.75), np.quantile(vals, 0.25)
        print(f"  {q}: loudest quartile yet never reached = "
              f"{int(((vals >= hi_q) & (h == 0)).sum())}; "
              f"quietest quartile yet reached = "
              f"{int(((vals < lo_q) & (h > 0)).sum())}")


if __name__ == "__main__":
    main()
