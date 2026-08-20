"""Draw the head-to-head chart the README and the docs embed.

The numbers are read from the published runs' own ``summary.json`` rather than
typed in, because the chart and the tables beside it have drifted before: a
rerun updates the table and leaves the picture claiming last week's result.
Point ``--results`` at a different directory to plot other runs.

The chart is deliberately English-only, and the font is pinned to DejaVu Sans
(matplotlib ships it) so the same file comes out of a laptop and out of CI.

    python evaluation/code_benchmark/plot_pk.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

AGENTICA = "#2563eb"
BASELINE = "#94a3b8"
INK = "#0f172a"
MUTED = "#64748b"
RULE = "#e2e8f0"

HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE / "results"
DEFAULT_OUT = HERE.parents[1] / "docs" / "assets" / "benchmark-agentica-vs-codex.png"

# The published runs. Keep in sync with docs/guides/benchmark.md, which cites
# the same ids — that table is how a reader gets from the picture to the raw
# predictions.
SUITES = [
    (
        "Coding — Aider Polyglot, 34 tasks",
        "20260819-195326-polyglot",
        "20260817-215956-polyglot",
    ),
    (
        "Data analysis — InfiAgent-DABench, 257 tasks",
        "20260820-153724-dabench",
        "20260820-134628-dabench",
    ),
]


@dataclass(frozen=True)
class Metric:
    """One column: what to read, how to label it, and which way is better."""

    header: str
    field: str
    fmt: str  # "pct" | "secs" | "millions"


METRICS = [
    Metric("Accuracy · higher is better", "accuracy", "pct"),
    Metric("Wall-clock · lower is better", "sum_wall_clock_s", "secs"),
    Metric("Input tokens · lower is better", "sum_input_tokens", "millions"),
]


def load(results: Path, run_id: str) -> dict:
    path = results / run_id / "summary.json"
    if not path.exists():
        raise SystemExit(f"missing {path} — pass --results if the runs live elsewhere")
    return json.loads(path.read_text())


def value(summary: dict, metric: Metric) -> float:
    """`accuracy` sits at the top level, the `sum_*` totals under `metrics`."""
    for scope in (summary.get("metrics", {}), summary):
        if metric.field in scope:
            return float(scope[metric.field])
    raise SystemExit(f"{metric.field} is in neither the summary nor its metrics block")


def label(v: float, fmt: str) -> str:
    if fmt == "pct":
        # 100 rather than 100.0: a suite where every task passed should read
        # like a clean sweep, not like a measurement.
        return f"{v:.0f}%" if abs(v - round(v)) < 0.05 else f"{v:.1f}%"
    if fmt == "secs":
        return f"{v:,.0f} s"
    return f"{v / 1e6:.2f}M"


def draw(results: Path, out: Path) -> Path:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.unicode_minus": False,
    })

    rows = []
    for title, ours, theirs in SUITES:
        rows.append((title, load(results, ours), load(results, theirs)))

    fig = plt.figure(figsize=(12.4, 5.3), dpi=180)
    fig.patch.set_facecolor("white")
    grid = fig.add_gridspec(
        len(rows), len(METRICS),
        left=0.135, right=0.975, top=0.755, bottom=0.185,
        hspace=0.7, wspace=0.34,
    )

    for r, (title, ours, theirs) in enumerate(rows):
        axes = []
        for c, metric in enumerate(METRICS):
            ax = fig.add_subplot(grid[r, c])
            axes.append(ax)
            vals = [value(ours, metric), value(theirs, metric)]
            # y = 1 on top: the reader's eye lands on Agentica first, and the
            # bars stay in the same order as the legend.
            ax.barh([1, 0], vals, height=0.62, color=[AGENTICA, BASELINE], zorder=3)
            for y, v in zip((1, 0), vals):
                ax.text(
                    v + max(vals) * 0.035, y, label(v, metric.fmt),
                    va="center", ha="left", fontsize=10.5, color=INK, zorder=4,
                )
            ax.set_xlim(0, max(vals) * 1.32)
            ax.set_ylim(-0.55, 1.55)
            ax.set_yticks([1, 0])
            # Names on the left column only. Repeating them six times is the
            # same information three more times; the legend closes the gap.
            ax.set_yticklabels(["Agentica", "OpenAI Codex"] if c == 0 else ["", ""],
                               fontsize=10.5, color=INK)
            ax.set_xticks([])
            ax.tick_params(axis="y", length=0)
            for side in ("top", "right", "bottom", "left"):
                ax.spines[side].set_visible(False)

        # Row heading, left-aligned over the whole row.
        top = max(ax.get_position().y1 for ax in axes)
        fig.text(0.018, top + 0.052, title, fontsize=12, fontweight="bold", color=INK)
        fig.add_artist(plt.Line2D(
            [0.018, 0.975], [top + 0.036, top + 0.036],
            color=RULE, linewidth=1, transform=fig.transFigure,
        ))

        if r == 0:
            for ax, metric in zip(axes, METRICS):
                box = ax.get_position()
                fig.text(
                    (box.x0 + box.x1) / 2, 0.895, metric.header,
                    fontsize=10.5, color=MUTED, ha="center",
                )

    fig.text(0.018, 0.955, "Agentica vs OpenAI Codex", fontsize=15,
             fontweight="bold", color=INK)
    # A real legend rather than hand-placed swatches: the longer label ran off
    # the right edge when its width was guessed.
    fig.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=AGENTICA, label="Agentica"),
            plt.Rectangle((0, 0), 1, 1, color=BASELINE, label="OpenAI Codex"),
        ],
        loc="upper right", bbox_to_anchor=(0.978, 1.005), frameon=False, ncol=2,
        fontsize=10.5, labelcolor=MUTED, handlelength=1.2, handleheight=1.1,
        columnspacing=1.5, handletextpad=0.6,
    )

    fig.text(
        0.018, 0.085,
        "Both harnesses run the same model (deepseek-v4-flash-official) over the same "
        "OpenAI Responses endpoint at reasoning effort=high,\non the full public suites. "
        "Accuracy is the suite total; wall-clock and input tokens are suite sums.",
        fontsize=9, color=MUTED, va="top", linespacing=1.5,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    path = draw(args.results, args.out)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
