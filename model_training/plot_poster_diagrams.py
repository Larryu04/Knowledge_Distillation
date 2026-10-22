"""
Poster process-flow diagrams for CoreCS / CoreCS Global, in the same visual language as
plot_metrics.py (same palette, fonts, title/subtitle/footnote layout, PNG 300 dpi + SVG).

  .venv\\Scripts\\python.exe plot_poster_diagrams.py                 # light theme
  .venv\\Scripts\\python.exe plot_poster_diagrams.py --theme dark    # dark, writes *_dark.*

Diagrams:
  1. pipeline_architecture - syllabi -> synthetic Q&A -> QLoRA -> GGUF -> desktop app / HF
  2. judge_methodology     - double-blind, position-swapped, reference-guided LLM-as-judge
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

import plot_metrics as pm
from plot_metrics import THEMES, apply_style, ink_on

# Box fills that plot_metrics doesn't need (it has no panels). Neutral panel + one blue tint.
PANELS = {
    "light": {"panel": "#f3f2ee", "tint": "#e6eefa", "edge": "#c3c2b7"},
    "dark":  {"panel": "#242423", "tint": "#1c2a3d", "edge": "#383835"},
}


# ---------------------------------------------------------------- primitives

def canvas(w, h, t):
    """Axes in inch units: (0,0) bottom-left, (w,h) top-right, so font sizes stay predictable."""
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def header(ax, x, y, title, sub, t):
    ax.text(x, y, title, fontsize=17, fontweight="bold", color=t["ink"], va="top")
    ax.text(x, y - 0.42, sub, fontsize=11.5, color=t["ink2"], va="top")


def box(ax, x, y, w, h, title, lines, t, style="panel", step=None, title_size=13):
    """
    Rounded stage box with (x, y) = bottom-left. style: panel (neutral), tint (light blue),
    strong (solid v8 blue, the stage the poster is about).
    """
    fill = {"panel": t["panel"], "tint": t["tint"], "strong": t["series"]["v8"]}[style]
    edge = t["series"]["v8"] if style == "strong" else (t["series"]["v7"] if style == "tint" else t["edge"])
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.14",
                                facecolor=fill, edgecolor=edge, linewidth=1.3, zorder=2))
    title_ink = ink_on(fill) if style == "strong" else t["ink"]
    body_ink = ink_on(fill) if style == "strong" else t["ink2"]

    top = y + h - 0.26
    tx = x + 0.22
    if step is not None:
        r = 0.17
        badge = t["surface"] if style == "strong" else t["series"]["v8"]
        ax.add_patch(Circle((tx + r, top - 0.02), r, facecolor=badge, edgecolor="none", zorder=3))
        ax.text(tx + r, top - 0.03, str(step), ha="center", va="center", fontsize=10.5,
                fontweight="bold", color=ink_on(badge), zorder=4)
        tx += 2 * r + 0.14
    ax.text(tx, top, title, fontsize=title_size, fontweight="bold", color=title_ink,
            va="center", zorder=4)
    for i, line in enumerate(lines):
        ax.text(x + 0.22, top - 0.5 - i * 0.33, line, fontsize=10.5, color=body_ink,
                va="center", zorder=4)
    return (x, y, w, h)


def arrow(ax, p0, p1, t, rad=0.0, color=None, lw=1.8):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=16,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0,
                                 color=color or t["muted"], linewidth=lw, zorder=1))


def right(b, dy=0.0):
    x, y, w, h = b
    return (x + w, y + h / 2 + dy)


def left(b, dy=0.0):
    x, y, w, h = b
    return (x, y + h / 2 + dy)


def phase(ax, x0, x1, y, label, t):
    """Thin bracket line under a group of boxes with a small-caps phase label."""
    ax.plot([x0, x1], [y, y], color=t["axis"], linewidth=1.2, solid_capstyle="butt")
    for xx in (x0, x1):
        ax.plot([xx, xx], [y, y + 0.1], color=t["axis"], linewidth=1.2)
    ax.text((x0 + x1) / 2, y - 0.12, label.upper(), ha="center", va="top", fontsize=9.5,
            color=t["muted"], fontweight="bold")


def chip(ax, x, y, w, h, text, fill, t):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.1",
                                facecolor=fill, edgecolor="none", zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10.5,
            fontweight="bold", color=ink_on(fill), zorder=4)


# ---------------------------------------------------------------- diagram 1

def plot_pipeline(t):
    W, H = 17.6, 4.55
    fig, ax = canvas(W, H, t)
    header(ax, 0.2, H - 0.15, "From course syllabi to an offline Hebrew CS tutor",
           "CoreCS & CoreCS Global: one distillation pipeline, from teacher API to a 7B model "
           "that runs on a student laptop", t)

    bw, bh, gap, y = 2.95, 2.25, 0.62, 1.15
    xs = [0.2 + i * (bw + gap) for i in range(5)]

    b1 = box(ax, xs[0], y, bw, bh, "Syllabi Upload",
             ["Course syllabi & topic lists", "23 CS courses (CoreCS)", "Any lecturer's course",
              "(CoreCS Global)"], t, step=1)
    b2 = box(ax, xs[1], y, bw, bh, "Synthetic Q&A",
             ["Teacher: GPT-4o-mini", "27K Hebrew Q&A items", "covering 23 courses",
              "explain-then-conclude style"], t, style="tint", step=2)
    b3 = box(ax, xs[2], y, bw, bh, "QLoRA Fine-Tuning",
             ["DictaLM-2.0 Instruct 7B", "4-bit NF4 base + LoRA (r=16)", "single RTX 4070 Super",
              "(12 GB VRAM)"], t, style="strong", step=3)
    b4 = box(ax, xs[3], y, bw, bh, "GGUF Quantization",
             ["Merge adapter into base", "Export GGUF, Q4_K_M", "llama.cpp-compatible",
              "CPU / consumer-GPU ready"], t, style="tint", step=4)

    # Stage 5 forks into two delivery targets.
    sh = (bh - 0.25) / 2
    b5a = box(ax, xs[4], y + sh + 0.25, bw, sh, "Offline Desktop App",
              ["CoreCS (Tauri), fully on-device"], t, step=5, title_size=12)
    b5b = box(ax, xs[4], y, bw, sh, "HuggingFace Release",
              ["Public GGUF + LoRA adapter"], t, step=5, title_size=12)

    for a, b in ((b1, b2), (b2, b3), (b3, b4)):
        arrow(ax, right(a), left(b), t)
    arrow(ax, right(b4, 0.12), left(b5a), t, rad=0.12)
    arrow(ax, right(b4, -0.12), left(b5b), t, rad=-0.12)

    py = y - 0.28
    phase(ax, xs[0], xs[1] + bw, py, "Data", t)
    phase(ax, xs[2], xs[3] + bw, py, "Model", t)
    phase(ax, xs[4], xs[4] + bw, py, "Delivery", t)

    ax.text(0.2, 0.18, "No student data leaves the device: inference runs locally from the "
                       "quantized GGUF. The OpenAI teacher is used only offline, at dataset-build time.",
            fontsize=9, color=t["muted"], va="bottom")
    pm.save(fig, "pipeline_architecture")


# ---------------------------------------------------------------- diagram 2

def plot_judge(t):
    W, H = 17.6, 5.35
    fig, ax = canvas(W, H, t)
    header(ax, 0.2, H - 0.15, "Double-blind, position-swapped LLM-as-a-judge",
           "Every v8-vs-opponent pair is judged twice with A/B swapped; "
           "a verdict that flips with position counts as a tie", t)

    bh, y = 2.7, 1.35
    ymid = y + bh / 2

    b1 = box(ax, 0.2, y, 2.7, bh, "Gold Benchmark",
             ["92 Hebrew exam questions", "23 courses", "staff-written gold answers"], t, step=1)
    b2 = box(ax, 3.5, y, 3.05, bh, "Candidate vs Opponent",
             ["Candidate: v8 (7B, ours)", "Opponents:", "  DictaLM-2.0 7B (base)",
              "  DictaLM-3.0 12B / 24B · GPT-4o"], t, style="tint", step=2)

    # Two orders, stacked: the anonymisation + swap step.
    ow, oh, ox = 2.55, 1.05, 7.15
    o1 = box(ax, ox, ymid + 0.12, ow, oh, "Order A / B",
             ["A = v8,  B = opponent"], t, title_size=12)
    o2 = box(ax, ox, ymid - 0.12 - oh, ow, oh, "Order B / A",
             ["A = opponent,  B = v8"], t, title_size=12)
    ax.text(ox + ow / 2, y + bh + 0.32, "Step 3: anonymised as “A” / “B”",
            ha="center", fontsize=10, color=t["muted"])

    b4 = box(ax, 10.3, y, 2.75, bh, "Reference-Guided Judge",
             ["gpt-4o-mini", "sees question + gold answer", "ranks: correctness first,",
              "then clarity for a student"], t, style="strong", step=4, title_size=12)
    b5 = box(ax, 13.65, y + 0.95, 3.75, bh - 0.95, "Position-Bias Filter",
             ["score v8 +1 / 0 / −1 per order", "sum > 0 win · = 0 tie · < 0 loss",
              "order-dependent verdict → tie"], t, style="tint", step=5)

    arrow(ax, right(b1), left(b2), t)
    arrow(ax, right(b2, 0.12), left(o1), t, rad=0.12)
    arrow(ax, right(b2, -0.12), left(o2), t, rad=-0.12)
    arrow(ax, right(o1), left(b4, 0.12), t, rad=-0.12)
    arrow(ax, right(o2), left(b4, -0.12), t, rad=0.12)
    arrow(ax, right(b4, 0.475), left(b5), t)

    # Final verdict chips under the filter box.
    cw, chh, cy = 1.15, 0.42, y
    cx0 = 13.65
    for i, (label, key) in enumerate((("v8 win", "win"), ("Tie", "tie"), ("v8 loss", "loss"))):
        chip(ax, cx0 + i * (cw + 0.15), cy, cw, chh, label, t[key], t)
    ax_x = b5[0] + b5[2] / 2
    arrow(ax, (ax_x, b5[1]), (ax_x, cy + chh + 0.04), t, lw=1.4)
    ax.text(ax_x + 0.15, (b5[1] + cy + chh) / 2, "Step 6: final verdict per item",
            fontsize=10, color=t["muted"], va="center")

    phase(ax, 0.2, 6.55, y - 0.28, "Inputs", t)
    phase(ax, 7.15, 13.05, y - 0.28, "Blind pairwise judging (2 calls / pair)", t)
    phase(ax, 13.65, 17.4, y - 0.28, "Aggregation", t)

    ax.text(0.2, 0.18, "92 questions × 4 opponents × 2 orders = 736 judge calls. "
                       "Empty answers (e.g. a thinking model out of token budget) lose without a call. "
                       "Judge model is fixed to gpt-4o-mini.",
            fontsize=9, color=t["muted"], va="bottom")
    pm.save(fig, "judge_methodology")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--theme", choices=THEMES, default="light")
    args = ap.parse_args()

    pm.SUFFIX = "" if args.theme == "light" else f"_{args.theme}"
    t = {**apply_style(args.theme), **PANELS[args.theme]}
    plot_pipeline(t)
    plot_judge(t)


if __name__ == "__main__":
    main()
