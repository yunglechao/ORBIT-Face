#!/usr/bin/env python3
"""Replot archived aggregate metrics; this script does not run experiments."""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "assets" / "figures"
DATA = ROOT / "assets" / "data" / "experiment_results.json"
COLORS = {
    "old_old": "#94a3b8",
    "direct": "#e89b28",
    "procrustes": "#008b80",
    "procrustes_mlp": "#6d5bd0",
    "lfw": "#3f82b7",
    "ifrt": "#008b80",
    "new_new": "#6d5bd0",
}


def configure_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "text.color": "#152a40",
        "axes.labelcolor": "#334155",
        "xtick.color": "#475569",
        "ytick.color": "#152a40",
        "axes.edgecolor": "#cbd5e1",
        "svg.fonttype": "none",
        "svg.hashsalt": "orbit-face-results",
        "savefig.facecolor": "white",
    })


def style_axis(ax, metric):
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_title(metric, loc="left", fontsize=13, weight="bold", pad=14)
    ax.grid(axis="x", color="#e2e8f0", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, pad=7)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)


def save_figure(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    svg = FIGURES / f"{name}.svg"
    fig.savefig(svg, metadata={"Date": None})
    # Matplotlib emits trailing spaces inside multiline SVG path attributes.
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    fig.savefig(FIGURES / f"{name}.png", dpi=180)
    plt.close(fig)


def plot_identification(data):
    rows = [data["rows"][i] for i in data["plot_indices"]]
    methods = [
        ("old_old", "Old-old"),
        ("direct", "Direct"),
        ("procrustes", "Procrustes"),
        ("procrustes_mlp", "Procrustes + residual MLP"),
    ]
    labels = [
        "Seed change\nR100: seed 1 → 2",
        "Loss change\nR100: CosFace → ArcFace",
        "Depth change\nR50 → R100 (CosFace)",
        "Architecture change\nR100 → ViT-B (CosFace)",
    ]
    positions = np.arange(len(rows)) * 1.65
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    fig.subplots_adjust(left=0.235, right=0.975, top=0.80, bottom=0.14, wspace=0.16)
    fig.text(0.035, 0.965, "A fixed gallery across model upgrades", fontsize=21, weight="bold")
    p = data["protocol"]
    fig.text(0.035, 0.925,
             f"Original derived IFRT-MR split  |  {p['n_gallery']:,} gallery images  |  "
             f"{p['n_probes']:,} probes  |  K = {p['calibration_k']:,}", fontsize=11)
    fig.legend(handles=[Patch(color=COLORS[m], label=label) for m, label in methods],
               loc="upper center", bbox_to_anchor=(0.60, 0.89), ncol=4,
               frameon=False, fontsize=10.5, handlelength=1.2, columnspacing=1.5)
    for ax, metric, title in zip(axes, ["rank_1", "rank_5"], ["Rank-1 (%)", "Rank-5 (%)"]):
        style_axis(ax, title)
        for i, y in enumerate(positions):
            if i % 2:
                ax.axhspan(y - 0.75, y + 0.75, color="#f4f7fa", zorder=0)
        for j, (method, _) in enumerate(methods):
            values = [100 * row["metrics"][method][metric] for row in rows]
            yy = positions + (j - 1.5) * 0.31
            ax.barh(yy, values, height=0.255, color=COLORS[method], zorder=3)
            for y, value in zip(yy, values):
                ax.text(value + 1.1, y, f"{value:.2f}", va="center", fontsize=9.3)
        ax.set_yticks(positions)
        ax.set_ylim(positions[-1] + 0.8, -0.8)
    axes[0].set_yticklabels(labels, fontsize=10.8, linespacing=1.5)
    axes[1].set_yticklabels([])
    fig.text(0.035, 0.072,
             "Selected upgrades; all seven pairs are reported in the README. Old-old is the legacy reference, not full backfill.",
             fontsize=10, color="#475569")
    fig.text(0.035, 0.037,
             "WebFace12M models. Archived paper results, re-plotted without rerunning experiments. Higher is better.",
             fontsize=10, color="#475569")
    save_figure(fig, "ifrt_identification")


def plot_calibration(data):
    rows = data["rows"]
    labels = ["Old-old", "Direct new-old", "Procrustes\nExternal LFW calibration",
              "Procrustes\nDisjoint IFRT calibration", "New-new\nFull backfill"]
    positions = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.subplots_adjust(left=0.225, right=0.975, top=0.76, bottom=0.20, wspace=0.16)
    fig.text(0.035, 0.95, "Representative calibration matters", fontsize=21, weight="bold")
    fig.text(0.035, 0.89,
             "MS1MV2-R100-ArcFace → WebFace12M-R100-ArcFace (seed 1)", fontsize=12)
    p = data["protocol"]
    fig.text(0.035, 0.835,
             f"Controlled derived IFRT-MR split  |  {p['n_gallery']:,} gallery images  |  "
             f"{p['n_probes']:,} probes  |  K = {p['calibration_k']:,}", fontsize=10.5)
    for ax, metric, title in zip(axes, ["rank_1", "rank_5"], ["Rank-1 (%)", "Rank-5 (%)"]):
        style_axis(ax, title)
        values = [100 * row["metrics"][metric] for row in rows]
        ax.barh(positions, values, height=0.56,
                color=[COLORS[row["key"]] for row in rows], zorder=3)
        for y, value in zip(positions, values):
            ax.text(value + 1.1, y, f"{value:.2f}", va="center", fontsize=10.5)
        ax.set_yticks(positions)
        ax.set_ylim(len(rows) - 0.4, -0.6)
    axes[0].set_yticklabels(labels, fontsize=11, linespacing=1.4)
    axes[1].set_yticklabels([])
    by_key = {row["key"]: row["metrics"] for row in rows}
    r1, r5 = [100 * (by_key["ifrt"][m] - by_key["lfw"][m]) for m in ["rank_1", "rank_5"]]
    fig.text(0.035, 0.105,
             f"Changing calibration source: +{r1:.2f} / +{r5:.2f} percentage points (Rank-1 / Rank-5).",
             fontsize=11, weight="bold", color=COLORS["ifrt"])
    fig.text(0.035, 0.049,
             "Same probes and gallery in both runs; single-view features. All reserved IFRT calibration identities are excluded.",
             fontsize=10, color="#475569")
    save_figure(fig, "calibration_transfer")


def result_tables(data):
    lines = [
        "| Upgrade | Old-old | Direct | Procrustes | Procrustes + MLP |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in data["original_identification"]["rows"]:
        cells = [f"{100 * row['metrics'][m]['rank_1']:.2f} / {100 * row['metrics'][m]['rank_5']:.2f}"
                 for m in ["old_old", "direct", "procrustes", "procrustes_mlp"]]
        lines.append("| " + " | ".join([row["label"], *cells]) + " |")
    original = "\n".join(lines)
    lines = ["| Configuration | Rank-1 (%) | Rank-5 (%) |", "|---|---:|---:|"]
    for row in data["calibration_transfer"]["rows"]:
        lines.append(f"| {row['label']} | {100 * row['metrics']['rank_1']:.2f} | "
                     f"{100 * row['metrics']['rank_5']:.2f} |")
    return {"IFRT_TABLE": original, "CALIBRATION_TABLE": "\n".join(lines)}


def update_readme(tables):
    path = ROOT / "README.md"
    text = path.read_text()
    for key, table in tables.items():
        begin, end = f"<!-- BEGIN {key} -->", f"<!-- END {key} -->"
        if text.count(begin) != 1 or text.count(end) != 1:
            raise ValueError(f"Expected exactly one README marker pair for {key}")
        before, tail = text.split(begin)
        _, after = tail.split(end)
        text = before + begin + "\n\n" + table + "\n\n" + end + after
    path.write_text(text)


def main():
    data = json.loads(DATA.read_text())
    for row in data["original_identification"]["rows"]:
        for metric in row["metrics"].values():
            assert 0 <= metric["rank_1"] <= metric["rank_5"] <= 1
    for row in data["calibration_transfer"]["rows"]:
        assert 0 <= row["metrics"]["rank_1"] <= row["metrics"]["rank_5"] <= 1
    configure_style()
    plot_identification(data["original_identification"])
    plot_calibration(data["calibration_transfer"])
    update_readme(result_tables(data))
    print("Updated two figures (SVG and PNG) and both README tables from archived metrics.")


if __name__ == "__main__":
    main()
