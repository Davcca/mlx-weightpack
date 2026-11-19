"""Redraw the README figures from checked-in measurements. Requires matplotlib."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / "docs" / "assets"
PAPER, INK, MUTED = "#F7F5EF", "#18352E", "#475A52"
GREEN, PALE, GRID = "#08745C", "#B9C8BF", "#D9DFD6"


def report(name):
    return json.loads((HERE / "results" / (name + ".json")).read_text())


def text(ax, x, y, value, size, **style):
    ax.text(x, y, value, fontsize=size, color=style.pop("color", INK), va="center", **style)


def memory():
    stock, packed = [
        report("resident_" + mode)["active_mlx_bytes_above_start"] / 1e9
        for mode in ("stock", "packed")
    ]
    fig = plt.figure(figsize=(10.2, 2.35), facecolor=PAPER)
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 100), ylim=(0, 100))
    ax.set_axis_off()
    text(
        ax,
        4.2,
        81,
        "QWEN 3.8 27B   /   M1 MAX",
        8.5,
        family="monospace",
        weight="bold",
        color=MUTED,
    )
    text(ax, 3.8, 48, f"{stock - packed:.2f} GB", 48, family="serif", color=INK)
    text(ax, 4.2, 17, "less memory for scales and biases", 10.8)
    ax.plot([45, 45], [16, 85], color=GRID, linewidth=0.9)
    left, width = 49, 46.8
    for label, value, y, fill in [
        ("Stock MLX", stock, 71, PALE),
        ("Weightpack", packed, 34, GREEN),
    ]:
        text(ax, left, y + 6, label, 10.5)
        text(
            ax,
            left + width,
            y + 6,
            f"{value:.3f} GB",
            9.5,
            family="monospace",
            ha="right",
            color=MUTED,
        )
        ax.add_patch(Rectangle((left, y - 12), width, 9, facecolor="#E8ECE4", edgecolor="none"))
        ax.add_patch(
            Rectangle((left, y - 12), width * value / stock, 9, facecolor=fill, edgecolor="none")
        )
        # Fine divisions give the two bars a shared physical scale.
        for i in range(1, 20):
            pos = left + width * i / 20
            ax.plot([pos, pos], [y - 12, y - 9], color=PAPER, lw=0.6)
    fig.savefig(ASSETS / "metadata.png", dpi=180, facecolor=PAPER)
    plt.close(fig)


def latency():
    data = report("cached_blocks")["sum_of_stage_medians_ms"]
    fig = plt.figure(figsize=(8.7, 4.1), facecolor=PAPER)
    ax = fig.add_axes([0.13, 0.20, 0.81, 0.63], facecolor=PAPER)
    fig.text(0.062, 0.89, "Time to score a cached block", fontsize=17, family="serif", color=INK)
    widths = list(data)
    for i, width in enumerate(widths):
        center = len(widths) - 1 - i
        pair = data[width]
        for side, offset, color in [("stock", 0.15, PALE), ("packed", -0.15, GREEN)]:
            value = pair[side]
            ax.barh(center + offset, value, height=0.23, color=color, zorder=2)
            ax.text(value + 5, center + offset, f"{value:.1f}", fontsize=9, va="center", color=INK)
    ax.set_yticks(
        list(reversed(range(len(widths)))),
        [f"{w} token" if w == "1" else f"{w} tokens" for w in widths],
        fontsize=9,
    )
    ax.set_xlim(0, 1.12 * max(max(pair.values()) for pair in data.values()))
    ax.set_ylim(-0.48, len(widths) - 0.52)
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.set_xlabel("milliseconds", color=MUTED, size=9)
    ax.tick_params(axis="both", length=0, labelcolor=MUTED, pad=5)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.spines[:].set_visible(False)
    from matplotlib.patches import Patch

    fig.legend(
        handles=[Patch(color=PALE, label="Stock MLX"), Patch(color=GREEN, label="Weightpack")],
        loc="upper right",
        bbox_to_anchor=(0.96, 0.95),
        ncol=2,
        frameon=False,
        handlelength=0.9,
        columnspacing=1.3,
        labelcolor=INK,
        fontsize=9,
    )
    fig.savefig(ASSETS / "cached-blocks.png", dpi=180, facecolor=PAPER)
    plt.close(fig)


if __name__ == "__main__":
    ASSETS.mkdir(parents=True, exist_ok=True)
    memory()
    latency()
