"""
merge_datasets.py — merge the 23 per-course JSON files in raw_courses_data/ into a single
flat training file, merged_all_courses.json.

Each source file is a JSON list of objects with exactly 5 keys: course_name, domain,
instruction, rationale, output (see project_memory.md §2 for the full schema contract).
This script does NOT re-validate content (generate_pilot_full-dataset.py already did that
per-item at generation time) — it only checks structural integrity (right keys, non-empty
strings, expected domain) before concatenating, so a corrupted/truncated source file fails
loudly instead of silently producing a bad merged file.

Usage:
    python merge_datasets.py
    python merge_datasets.py --output merged_all_courses.json --min-per-domain 1200
"""

import argparse
import json
import sys
from collections import Counter

REQUIRED_KEYS = {"course_name", "domain", "instruction", "rationale", "output"}

# (source file, expected domain) — the original 6 courses, the 6 added 2026-09-27, and the
# 11 added 2026-09-30 (v8).
SOURCES = [
    ("raw_courses_data/networks_data.json", "Computer Networks"),
    ("raw_courses_data/os_data.json", "Operating Systems"),
    ("raw_courses_data/cyber_data.json", "Cybersecurity"),
    ("raw_courses_data/cpp_data.json", "C++"),
    ("raw_courses_data/java_data.json", "Java"),
    ("raw_courses_data/python_data.json", "Python"),
    ("raw_courses_data/data_structures_data.json", "Data Structures"),
    ("raw_courses_data/algorithms_data.json", "Algorithms"),
    ("raw_courses_data/computer_architecture_data.json", "Computer Architecture"),
    ("raw_courses_data/databases_data.json", "Databases"),
    ("raw_courses_data/automata_theory_data.json", "Automata Theory"),
    ("raw_courses_data/software_engineering_data.json", "Software Engineering"),
    ("raw_courses_data/discrete_math_data.json", "Discrete Mathematics"),
    ("raw_courses_data/logic_set_theory_data.json", "Logic and Set Theory"),
    ("raw_courses_data/graph_theory_data.json", "Graph Theory"),
    ("raw_courses_data/computability_data.json", "Computability and Complexity"),
    ("raw_courses_data/cryptography_data.json", "Introduction to Cryptography"),
    ("raw_courses_data/intro_ai_data.json", "Introduction to AI"),
    ("raw_courses_data/intro_ml_data.json", "Introduction to Machine Learning"),
    ("raw_courses_data/deep_learning_data.json", "Deep Learning"),
    ("raw_courses_data/ai_software_data.json", "AI-Based Software Development"),
    ("raw_courses_data/info_systems_data.json", "Information Systems Foundations"),
    ("raw_courses_data/comp_biology_data.json", "Computational Biology Applications"),
]


def load_and_check(path, expected_domain):
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        sys.exit(f"[error] {path}: expected a JSON list, got {type(rows).__name__}")
    for i, r in enumerate(rows):
        if set(r.keys()) != REQUIRED_KEYS:
            sys.exit(f"[error] {path} row {i}: keys {sorted(r.keys())} != {sorted(REQUIRED_KEYS)}")
        for k in REQUIRED_KEYS:
            if not isinstance(r[k], str) or not r[k].strip():
                sys.exit(f"[error] {path} row {i}: field {k!r} is empty or not a string")
        if r["domain"] != expected_domain:
            sys.exit(f"[error] {path} row {i}: domain {r['domain']!r} != expected {expected_domain!r}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default="merged_all_courses.json")
    ap.add_argument("--min-per-domain", type=int, default=None,
                     help="If set, fail if any domain has fewer rows than this "
                          "(e.g. 1200, to catch a generation run that fell short).")
    args = ap.parse_args()

    merged = []
    counts = Counter()
    for path, domain in SOURCES:
        try:
            rows = load_and_check(path, domain)
        except FileNotFoundError:
            sys.exit(f"[error] missing source file: {path}")
        counts[domain] = len(rows)
        merged.extend(rows)
        print(f"[load] {path}: {len(rows)} rows ({domain})")

    # Duplicate-instruction check across the WHOLE merged set (per-source dedup already
    # happened at generation time via difflib similarity, but that only guards within a
    # single course's run, not across courses).
    seen = {}
    for i, r in enumerate(merged):
        seen.setdefault(r["instruction"], []).append(i)
    dupes = {ins: idxs for ins, idxs in seen.items() if len(idxs) > 1}
    if dupes:
        print(f"[warn] {len(dupes)} exact-duplicate instruction(s) across courses "
              f"(kept as-is — inspect before training if this count is large).")

    print(f"\n[summary] total rows: {len(merged)}")
    for domain, n in counts.items():
        flag = ""
        if args.min_per_domain is not None and n < args.min_per_domain:
            flag = f"  <-- below --min-per-domain={args.min_per_domain}"
        print(f"    {domain}: {n}{flag}")

    if args.min_per_domain is not None:
        short = {d: n for d, n in counts.items() if n < args.min_per_domain}
        if short:
            sys.exit(f"[error] {len(short)} domain(s) fell short of --min-per-domain="
                      f"{args.min_per_domain}: {short}. Re-run the generator(s) or lower "
                      f"--min-per-domain before merging.")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"\n[done] wrote {len(merged)} rows to {args.output}")


if __name__ == "__main__":
    main()
