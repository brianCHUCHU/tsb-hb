from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_shrinkage_scatter(x, y, xlabel: str, ylabel: str, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x, y, color="darkblue", alpha=0.3)
    lim_min = min(float(min(x)), float(min(y))) if len(x) and len(y) else 0.0
    lim_max = max(float(max(x)), float(max(y))) if len(x) and len(y) else 1.0
    lim_min = max(lim_min, 0.0)
    lim_max = lim_max * 1.05
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", label="y=x (No Shrinkage)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pit_hist(pit_values, out_path: Path, title: str = "PIT Histogram") -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(pit_values, bins=10, range=(0, 1), edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel("PIT value")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

