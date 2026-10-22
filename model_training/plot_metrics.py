"""
Presentation charts for the Hebrew CS tutor (Base -> v6 -> v7 -> v8).

Reads eval_results_v8/v6_vs_v7_vs_v8_comparison.csv (written by compare_loss_results.py)
and writes PNG (slides) + SVG (lossless) figures to figures/.

  .venv\\Scripts\\python.exe plot_metrics.py                 # light theme (white slides)
  .venv\\Scripts\\python.exe plot_metrics.py --theme dark    # dark slides
  .venv\\Scripts\\python.exe plot_metrics.py --judge-demo    # also render the judge chart with FAKE data

Charts:
  1. loss_progression     - overall gold-set loss (92 items), Base -> v6 -> v7 -> v8
  2. prior_vs_new_domains - mean loss on the 12 prior vs 11 new domains, per version
  3. judge_results        - v8 win/tie/loss vs larger models (needs judge_summary.csv; see
                            plot_judge_results for the expected schema)
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

CSV_PATH = "eval_results_v8/v6_vs_v7_vs_v8_comparison.csv"
JUDGE_CSV_PATH = "eval_results_judge/judge_summary.csv"
OUT_DIR = "figures"
VERSIONS = ["Base", "v6", "v7", "v8"]
SUFFIX = ""  # "_dark" for the dark theme, so both sets can coexist

# Base = neutral gray; v6 -> v8 = one-hue ordinal blue ramp, v8 strongest against the surface.
# Ramps validated (ordinal: monotone lightness, visible steps, light end clears surface).
THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7",
        "series": {"Base": "#c3c2b7", "v6": "#86b6ef", "v7": "#3987e5", "v8": "#1c5cab"},
        "win": "#2a78d6", "tie": "#c3c2b7", "loss": "#e34948",
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835",
        "series": {"Base": "#5f5e5a", "v6": "#256abf", "v7": "#3987e5", "v8": "#86b6ef"},
        "win": "#3987e5", "tie": "#5f5e5a", "loss": "#e66767",
    },
}


# ---------------------------------------------------------------- data

def load_comparison(path=CSV_PATH):
    """Return {row_label: {Base, v6, v7, v8}} of mean losses, keyed by Domain column."""
    rows = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            rows[r["Domain"]] = {
                "Base": float(r["Base_Loss"]),
                "v6": float(r["v6_Loss"]),
                "v7": float(r["v7_Loss"]),
                "v8": float(r["v8_Loss"]),
                "N": int(r["N"]),
            }
    for key in ("ALL_DOMAINS_MEAN", "PRIOR_DOMAINS_MEAN", "NEW_DOMAINS_MEAN"):
        if key not in rows:
            raise KeyError(f"{path} has no {key} row -- rerun compare_loss_results.py")
    return rows


# ---------------------------------------------------------------- styling

def apply_style(theme):
    t = THEMES[theme]
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 12,
        "figure.facecolor": t["surface"],
        "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"],
        "axes.edgecolor": t["axis"],
        "axes.labelcolor": t["ink2"],
        "axes.titlecolor": t["ink"],
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 30,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": t["grid"],
        "grid.linewidth": 0.8,
        "xtick.color": t["muted"],
        "ytick.color": t["muted"],
        "xtick.labelcolor": t["ink2"],
        "ytick.labelcolor": t["muted"],
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "legend.labelcolor": t["ink2"],
    })
    return t


def subtitle(ax, text, t):
    ax.text(0, 1.02, text, transform=ax.transAxes, fontsize=11, color=t["ink2"], va="bottom")


def footnote(fig, text, t):
    fig.text(0.01, -0.02, text, fontsize=9, color=t["muted"], ha="left", va="top")


def save(fig, name):
    name += SUFFIX
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("png", "svg"):
        path = os.path.join(OUT_DIR, f"{name}.{ext}")
        fig.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight", pad_inches=0.25)
        print(f"wrote {path}")
    plt.close(fig)


def ink_on(fill):
    """Black or white text, whichever reads better on the given fill."""
    r, g, b = (int(fill[i:i + 2], 16) / 255 for i in (1, 3, 5))
    lum = 0.2126 * r ** 2.2 + 0.7152 * g ** 2.2 + 0.0722 * b ** 2.2
    return "#0b0b0b" if lum > 0.28 else "#ffffff"


def bar_labels(ax, bars, values, t, fmt="{:.3f}", bold_last=False):
    for i, (b, v) in enumerate(zip(bars, values)):
        bold = bold_last and i == len(bars) - 1
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012, fmt.format(v),
                ha="center", va="bottom", fontsize=11,
                color=t["ink"] if bold else t["ink2"],
                fontweight="bold" if bold else "normal")


# ---------------------------------------------------------------- chart 1

def plot_loss_progression(data, t):
    overall = data["ALL_DOMAINS_MEAN"]
    vals = [overall[v] for v in VERSIONS]
    n = overall["N"]

    fig, ax = plt.subplots(figsize=(10, 5.6))
    bars = ax.bar(VERSIONS, vals, width=0.56,
                  color=[t["series"][v] for v in VERSIONS],
                  edgecolor=t["surface"], linewidth=2)
    bar_labels(ax, bars, vals, t, bold_last=True)

    # % reduction vs base, under each fine-tuned bar's value
    for ver, b, v in zip(VERSIONS[1:], bars[1:], vals[1:]):
        pct = (vals[0] - v) / vals[0] * 100
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() / 2, f"−{pct:.0f}%\nvs Base",
                ha="center", va="center", fontsize=10, color=ink_on(t["series"][ver]))

    ax.set_ylim(0, max(vals) * 1.15)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_ylabel("Mean token loss (lower is better)")
    ax.set_title("Gold-set loss falls with every training round")
    subtitle(ax, f"Masked teacher-forcing loss on the {n}-item, 23-course Hebrew benchmark", t)
    footnote(fig, "v6: 6 courses · v7: 12 courses · v8: 23 courses. "
                  "Same DictaLM-2.0-instruct 7B base, QLoRA adapters.", t)
    save(fig, "loss_progression")


# ---------------------------------------------------------------- chart 2

def plot_prior_vs_new(data, t):
    groups = [
        ("PRIOR_DOMAINS_MEAN", "12 prior domains"),
        ("NEW_DOMAINS_MEAN", "11 new domains"),
    ]
    width = 0.19
    gap = 0.02  # surface gap between adjacent bars
    fig, ax = plt.subplots(figsize=(11, 5.8))

    for i, ver in enumerate(VERSIONS):
        offset = (i - (len(VERSIONS) - 1) / 2) * (width + gap)
        xs = [g + offset for g in range(len(groups))]
        vals = [data[key][ver] for key, _ in groups]
        bars = ax.bar(xs, vals, width=width, color=t["series"][ver], label=ver,
                      edgecolor=t["surface"], linewidth=1.5)
        for b, v in zip(bars, vals):
            is_v8 = ver == "v8"
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center",
                    va="bottom", fontsize=10,
                    color=t["ink"] if is_v8 else t["ink2"],
                    fontweight="bold" if is_v8 else "normal")

    labels = [f"{name}\n(n={data[key]['N']} items)" for key, name in groups]
    ax.set_xticks(range(len(groups)), labels)
    ax.tick_params(axis="x", labelsize=12, pad=8)
    ymax = max(data[k][v] for k, _ in groups for v in VERSIONS)
    ax.set_ylim(0, ymax * 1.25)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_ylabel("Mean token loss (lower is better)")
    ax.set_title("v8 learns 11 new courses at a ~2% cost on the old 12")
    prior, new = data["PRIOR_DOMAINS_MEAN"], data["NEW_DOMAINS_MEAN"]
    subtitle(ax, f"New domains: v8 {new['v8']:.3f} vs v7 {new['v7']:.3f}  ·  "
                 f"Prior domains: v8 {prior['v8']:.3f} vs v7 {prior['v7']:.3f} "
                 f"({(prior['v8'] - prior['v7']) / prior['v7'] * 100:+.1f}%)", t)
    ax.legend(ncols=4, loc="upper right", handlelength=1, handleheight=1)
    footnote(fig, "Prior = the 12 courses v7 was trained on; new = 11 courses added in v8. "
                  "v6 saw only 6 of the prior 12.", t)
    save(fig, "prior_vs_new_domains")


# ---------------------------------------------------------------- chart 3 (placeholder)

DEMO_JUDGE_ROWS = [
    # PLACEHOLDER numbers -- only used with --judge-demo, never for the real deck.
    {"Opponent": "Base DictaLM-2.0 7B", "Params": "7B", "Wins": 70, "Ties": 12, "Losses": 10},
    {"Opponent": "Open-weight 24-32B", "Params": "~30B", "Wins": 41, "Ties": 20, "Losses": 31},
    {"Opponent": "Frontier API model", "Params": "proprietary", "Wins": 22, "Ties": 18, "Losses": 52},
]


def load_judge_summary(path=JUDGE_CSV_PATH):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return [{**r, "Wins": int(r["Wins"]), "Ties": int(r["Ties"]), "Losses": int(r["Losses"])}
                for r in csv.DictReader(f)]


def plot_judge_results(rows, t, demo=False):
    """
    v8's head-to-head record vs each opponent, as a 100% stacked horizontal bar.

    Expected CSV schema (eval_results_judge/judge_summary.csv), one row per opponent:
        Opponent,Params,Wins,Ties,Losses
    Counts are from v8's point of view, after position-swap reconciliation
    (a verdict that flips when A/B are swapped counts as a tie).
    """
    t_win, t_tie, t_loss = t["win"], t["tie"], t["loss"]
    fig, ax = plt.subplots(figsize=(11, 1.2 + 0.9 * len(rows)))
    ax.grid(False)
    ax.spines["bottom"].set_visible(False)

    names = [f"vs {r['Opponent']}  ({r['Params']})" for r in rows][::-1]
    for y, r in enumerate(rows[::-1]):
        total = r["Wins"] + r["Ties"] + r["Losses"]
        left = 0.0
        for key, color, label in (("Wins", t_win, "v8 wins"), ("Ties", t_tie, "Tie"),
                                  ("Losses", t_loss, "Opponent wins")):
            share = r[key] / total * 100
            ax.barh(y, share, left=left, height=0.56, color=color, edgecolor=t["surface"],
                    linewidth=2, label=label if y == 0 else None)
            if share >= 7:
                ax.text(left + share / 2, y, f"{share:.0f}%", ha="center", va="center",
                        fontsize=10.5, color=ink_on(color))
            left += share

    ax.set_yticks(range(len(rows)), names)
    ax.tick_params(axis="y", labelsize=11.5, labelcolor=t["ink2"])
    ax.set_xlim(0, 100)
    ax.set_xticks([])
    ax.set_title("Head-to-head: v8 (7B) vs its base model and larger models" + ("  [DEMO DATA]" if demo else ""))
    n = rows[0]["Wins"] + rows[0]["Ties"] + rows[0]["Losses"]
    subtitle(ax, f"Pairwise LLM-as-judge on {n} gold questions, reference-guided, A/B order swapped", t)
    ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, 0), handlelength=1, handleheight=1)
    if demo:
        fig.text(0.5, 0.5, "PLACEHOLDER", fontsize=60, color=t["muted"], alpha=0.18,
                 ha="center", va="center", rotation=12, fontweight="bold")
    save(fig, "judge_results_DEMO" if demo else "judge_results")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=CSV_PATH)
    ap.add_argument("--theme", choices=THEMES, default="light")
    ap.add_argument("--judge-csv", default=JUDGE_CSV_PATH)
    ap.add_argument("--judge-demo", action="store_true",
                    help="render the judge chart with placeholder data (watermarked)")
    args = ap.parse_args()

    global SUFFIX
    SUFFIX = "" if args.theme == "light" else f"_{args.theme}"
    t = apply_style(args.theme)
    data = load_comparison(args.csv)
    plot_loss_progression(data, t)
    plot_prior_vs_new(data, t)

    if os.path.exists(args.judge_csv):
        plot_judge_results(load_judge_summary(args.judge_csv), t)
    elif args.judge_demo:
        plot_judge_results(DEMO_JUDGE_ROWS, t, demo=True)
    else:
        print(f"skip judge chart: {args.judge_csv} not found (use --judge-demo for a preview)")


if __name__ == "__main__":
    main()
