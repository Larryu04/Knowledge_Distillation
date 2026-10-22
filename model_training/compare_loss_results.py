"""
compare_loss_results.py — merge N evaluate_models.py loss_results.csv files (same benchmark,
different adapters) into one side-by-side comparison CSV.

Default: v6 vs v7 vs v8 on hebrew_cs_pilot_dataset_92.json, all scored with the same
23-course PREAMBLE ->
    eval_results_v8/v6_vs_v7_vs_v8_comparison.csv

Domains are read from the input CSVs (nothing hardcoded); every run must cover the same
domain set with the same N per domain, and the frozen base-model loss must match across runs.

Columns per domain:
    Base_Loss, <run>_Loss, <run>_Delta_vs_Base (> 0 => adapter beats base), <run>_PPL,
    Gain_<new>_over_<old> = old_loss - new_loss (> 0 => newer adapter is better), for each
    consecutive pair plus last-over-first, and Best_Run (lowest loss).

Summary rows: optional group means split by --prior-train-data (domains the previous
training set already contained vs. domains new in this version), ALL_DOMAINS_MEAN
(unweighted over domains), and the evaluator's OVERALL row (mean over all items).

Usage:
    python compare_loss_results.py
    python compare_loss_results.py --run v7=eval_results_v7_fair/loss_results.csv \\
        --run v8=eval_results_v8/loss_results.csv --out eval_results_v8/v7_vs_v8.csv
"""

import argparse
import csv
import json
import os
import sys

BASE_TOLERANCE = 1e-3   # every run scores the same frozen base model -> base loss must match
SUMMARY_ROW = "OVERALL"
DEFAULT_RUNS = [
    "v6=eval_results_v6_fair/loss_results.csv",
    "v7=eval_results_v7_fair/loss_results.csv",
    "v8=eval_results_v8/loss_results.csv",
]


def read_loss_csv(path):
    if not os.path.exists(path):
        sys.exit(f"[error] missing loss CSV: {path} (run evaluate_models.py for that adapter first)")
    with open(path, encoding="utf-8-sig") as f:
        return {r["Domain"]: r for r in csv.DictReader(f)}


def parse_runs(specs):
    runs = []
    for spec in specs:
        name, sep, path = spec.partition("=")
        if not sep or not name or not path:
            sys.exit(f"[error] --run must look like NAME=PATH, got {spec!r}")
        runs.append((name, read_loss_csv(path)))
    if len(runs) < 2:
        sys.exit("[error] need at least two runs to compare")
    return runs


def load_prior_domains(path):
    if not path:
        return None
    if not os.path.exists(path):
        print(f"[warn] --prior-train-data {path} not found -> no prior/new group split")
        return None
    with open(path, encoding="utf-8") as f:
        return {r["domain"] for r in json.load(f)}


def mean_row(label, group, members, keys):
    return {"Domain": label, "Group": group, "N": sum(r["N"] for r in members),
            **{k: sum(r[k] for r in members) / len(members) for k in keys}, "Best_Run": ""}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", metavar="NAME=PATH",
                    help="a run to compare, oldest first; repeat for each run "
                         f"(default: {' '.join(DEFAULT_RUNS)})")
    ap.add_argument("--prior-train-data", default="merged_12_courses.json",
                    help="training set of the previous version; its domains are grouped as "
                         "'prior', all others as 'new' (empty string disables the split)")
    ap.add_argument("--out", default="eval_results_v8/v6_vs_v7_vs_v8_comparison.csv")
    args = ap.parse_args()

    runs = parse_runs(args.run or DEFAULT_RUNS)
    names = [n for n, _ in runs]
    ref_name, ref = runs[0]

    domains = [d for d in ref if d != SUMMARY_ROW]
    for name, data in runs[1:]:
        diff = set(data) ^ set(ref)
        if diff:
            sys.exit(f"[error] domain sets differ between {ref_name} and {name}: {sorted(diff)}")
    prior = load_prior_domains(args.prior_train_data)

    pairs = list(zip(names, names[1:]))
    if len(names) > 2:
        pairs.append((names[0], names[-1]))

    rows = []
    for dom in domains + [SUMMARY_ROW]:
        recs = {name: data[dom] for name, data in runs}
        base = float(recs[ref_name]["Base_Avg_Loss"])
        n = int(recs[ref_name]["N"])
        for name, r in recs.items():
            if abs(float(r["Base_Avg_Loss"]) - base) >= BASE_TOLERANCE:
                sys.exit(f"[error] {dom}: base loss differs between {ref_name} ({base}) and "
                         f"{name} ({r['Base_Avg_Loss']}) -- were they run on the same eval set?")
            if int(r["N"]) != n:
                sys.exit(f"[error] {dom}: N differs between {ref_name} ({n}) and {name} ({r['N']})")

        loss = {name: float(r["FineTuned_Avg_Loss"]) for name, r in recs.items()}
        if dom == SUMMARY_ROW:
            group = "all"
        elif prior is None:
            group = ""
        else:
            group = "prior" if dom in prior else "new"
        row = {"Domain": dom, "Group": group, "N": n, "Base_Loss": base}
        row.update({f"{name}_Loss": loss[name] for name in names})
        row.update({f"{name}_Delta_vs_Base": base - loss[name] for name in names})
        row.update({f"{name}_PPL": float(recs[name]["FineTuned_Perplexity"]) for name in names})
        row.update({f"Gain_{new}_over_{old}": loss[old] - loss[new] for old, new in pairs})
        row["Best_Run"] = min(names, key=loss.get)
        rows.append(row)

    # Summary rows, inserted before OVERALL. Unweighted over domains.
    numeric = [k for k in rows[0] if k not in ("Domain", "Group", "N", "Best_Run")]
    domain_rows = rows[:-1]
    summaries = []
    if prior is not None:
        for label, group in (("PRIOR_DOMAINS_MEAN", "prior"), ("NEW_DOMAINS_MEAN", "new")):
            members = [r for r in domain_rows if r["Group"] == group]
            if members:
                summaries.append(mean_row(label, group, members, numeric))
    summaries.append(mean_row("ALL_DOMAINS_MEAN", "all", domain_rows, numeric))
    rows[-1:-1] = summaries

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})

    gain_keys = [f"Gain_{new}_over_{old}" for old, new in pairs]
    header = f"{'Domain':36s} {'Base':>7s} " + " ".join(f"{n:>7s}" for n in names) + " " + \
             " ".join(f"{k.replace('Gain_', ''):>11s}" for k in gain_keys) + "  best"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['Domain']:36s} {r['Base_Loss']:7.4f} "
              + " ".join(f"{r[f'{n}_Loss']:7.4f}" for n in names) + " "
              + " ".join(f"{r[k]:+11.4f}" for k in gain_keys) + f"  {r['Best_Run']}")
    print(f"\n[csv] wrote {args.out}")


if __name__ == "__main__":
    main()
