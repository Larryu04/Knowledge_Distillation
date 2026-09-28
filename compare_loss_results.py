"""
compare_loss_results.py — merge two evaluate_models.py loss_results.csv files (same benchmark,
different adapters) into one side-by-side comparison CSV.

Default: v6 vs v7 on hebrew_cs_pilot_dataset_48.json ->
eval_results_v7/v6_vs_v7_loss_comparison.csv

Columns per domain: base loss, each adapter's loss / delta vs base / perplexity, and
Adapter_Gain = old_FT_loss - new_FT_loss (> 0 => the new adapter is better). Adds
ORIGINAL_6_MEAN / NEW_6_MEAN group rows and keeps the evaluator's OVERALL row.
"""

import argparse
import csv

ORIGINAL_DOMAINS = ["Computer Networks", "Cybersecurity", "Operating Systems", "C++", "Java", "Python"]
NEW_DOMAINS = ["Data Structures", "Algorithms", "Computer Architecture", "Databases",
               "Automata Theory", "Software Engineering"]
BASE_TOLERANCE = 1e-3   # both runs score the same frozen base model -> base loss must match


def read_loss_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return {r["Domain"]: r for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="eval_results_v6_on48/loss_results.csv")
    ap.add_argument("--new", default="eval_results_v7/loss_results.csv")
    ap.add_argument("--old-name", default="v6")
    ap.add_argument("--new-name", default="v7")
    ap.add_argument("--out", default="eval_results_v7/v6_vs_v7_loss_comparison.csv")
    args = ap.parse_args()

    old, new = read_loss_csv(args.old), read_loss_csv(args.new)
    assert set(old) == set(new), f"domain sets differ: {set(old) ^ set(new)}"
    o, n = args.old_name, args.new_name

    rows = []
    for dom in ORIGINAL_DOMAINS + NEW_DOMAINS + ["OVERALL"]:
        a, b = old[dom], new[dom]
        base_a, base_b = float(a["Base_Avg_Loss"]), float(b["Base_Avg_Loss"])
        assert abs(base_a - base_b) < BASE_TOLERANCE, f"{dom}: base loss differs ({base_a} vs {base_b})"
        ft_o, ft_n = float(a["FineTuned_Avg_Loss"]), float(b["FineTuned_Avg_Loss"])
        rows.append({
            "Domain": dom,
            "Group": "original" if dom in ORIGINAL_DOMAINS else "new" if dom in NEW_DOMAINS else "all",
            "N": int(b["N"]),
            "Base_Loss": base_b,
            f"{o}_Loss": ft_o,
            f"{n}_Loss": ft_n,
            f"{o}_Delta_vs_Base": base_b - ft_o,
            f"{n}_Delta_vs_Base": base_b - ft_n,
            f"Gain_{n}_over_{o}": ft_o - ft_n,
            f"{o}_PPL": float(a["FineTuned_Perplexity"]),
            f"{n}_PPL": float(b["FineTuned_Perplexity"]),
        })

    # Group means (unweighted over domains; every domain has the same N).
    for label, group in (("ORIGINAL_6_MEAN", "original"), ("NEW_6_MEAN", "new")):
        members = [r for r in rows if r["Group"] == group]
        mean = {k: sum(r[k] for r in members) / len(members)
                for k in members[0] if k not in ("Domain", "Group", "N")}
        rows.insert(-1, {"Domain": label, "Group": group, "N": sum(r["N"] for r in members), **mean})

    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})

    print(f"{'Domain':24s} {'Base':>7s} {o:>7s} {n:>7s} {'Δ'+o:>7s} {'Δ'+n:>7s} {'gain':>7s}")
    for r in rows:
        print(f"{r['Domain']:24s} {r['Base_Loss']:7.4f} {r[f'{o}_Loss']:7.4f} {r[f'{n}_Loss']:7.4f} "
              f"{r[f'{o}_Delta_vs_Base']:+7.4f} {r[f'{n}_Delta_vs_Base']:+7.4f} {r[f'Gain_{n}_over_{o}']:+7.4f}")
    print(f"[csv] wrote {args.out}")


if __name__ == "__main__":
    main()
