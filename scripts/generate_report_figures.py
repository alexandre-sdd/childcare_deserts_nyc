#!/usr/bin/env python3
"""Reference-only helper.

Canonical report figures are now generated from
`notebooks/07_results_visualization_comparison.ipynb`.
This script is kept only as a backup copy of the plotting logic.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "report" / "figures"


def read_metric_map(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    return dict(zip(df["metric"], df["value"]))


def read_component_map(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    return dict(zip(df["component"], df["value"]))


def annotate_bars(ax, bars, fmt="{:.1f}", suffix=""):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            fmt.format(height) + suffix,
            ha="center",
            va="bottom",
            fontsize=9,
        )


def make_ideal_cost_breakdown(ideal_costs: dict[str, float]) -> None:
    labels = ["Expansion", "New build", "Under-5 equipment"]
    values = np.array([
        ideal_costs["expansion_cost"],
        ideal_costs["new_build_cost"],
        ideal_costs["under5_equipment_cost"],
    ]) / 1_000_000
    colors = ["#4C78A8", "#F28E2B", "#59A14F"]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bars = ax.bar(labels, values, color=colors)
    annotate_bars(ax, bars, fmt="{:.1f}", suffix="M")
    ax.set_ylabel("Cost (million $)")
    ax.set_title("Part 1 Cost Breakdown")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ideal_cost_breakdown.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_part_comparison_overview(
    ideal_costs: dict[str, float],
    real_costs: dict[str, float],
    ideal_summary: dict[str, float],
    real_summary: dict[str, float],
) -> None:
    parts = ["Part 1", "Part 2"]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    cost_labels = ["Expansion", "New build", "Under-5"]
    cost_colors = ["#4C78A8", "#F28E2B", "#59A14F"]
    cost_data = np.array([
        [
            ideal_costs["expansion_cost"],
            ideal_costs["new_build_cost"],
            ideal_costs["under5_equipment_cost"],
        ],
        [
            real_costs["expansion_cost"],
            real_costs["new_build_cost"],
            real_costs["under5_equipment_cost"],
        ],
    ]) / 1_000_000

    bottom = np.zeros(len(parts))
    for idx, label in enumerate(cost_labels):
        axes[0].bar(parts, cost_data[:, idx], bottom=bottom, color=cost_colors[idx], label=label)
        bottom += cost_data[:, idx]
    axes[0].set_title("Cost Composition")
    axes[0].set_ylabel("Cost (million $)")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=9)

    slot_labels = ["Expansion slots", "New-build slots"]
    slot_colors = ["#4C78A8", "#E15759"]
    slot_data = np.array([
        [
            ideal_summary["total_added_expansion_slots"],
            ideal_summary["total_added_newbuild_slots"],
        ],
        [
            real_summary["total_expansion_slots"],
            real_summary["total_new_capacity"],
        ],
    ]) / 1_000

    bottom = np.zeros(len(parts))
    for idx, label in enumerate(slot_labels):
        axes[1].bar(parts, slot_data[:, idx], bottom=bottom, color=slot_colors[idx], label=label)
        bottom += slot_data[:, idx]
    axes[1].set_title("Added Capacity by Source")
    axes[1].set_ylabel("Slots (thousands)")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_axisbelow(True)
    axes[1].legend(frameon=False, fontsize=9)

    fig.suptitle("Benchmark vs. Realistic Intervention Shift", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "part_comparison_overview.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_realistic_mix_and_zipcodes(
    ideal_build_counts: pd.Series,
    real_build_counts: pd.Series,
    realistic_zip: pd.DataFrame,
) -> None:
    size_order = ["small", "medium", "large"]
    ideal_vals = [float(ideal_build_counts.get(size, 0.0)) for size in size_order]
    real_vals = [float(real_build_counts.get(size, 0.0)) for size in size_order]

    zip_df = realistic_zip.copy()
    zip_df["total_added_capacity"] = zip_df["expanded_total"] + zip_df["new_total_capacity"]
    top_zip = zip_df.sort_values("total_added_capacity", ascending=False).head(10)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))

    x = np.arange(len(size_order))
    width = 0.36
    bars1 = axes[0].bar(x - width / 2, ideal_vals, width, label="Part 1", color="#4C78A8")
    bars2 = axes[0].bar(x + width / 2, real_vals, width, label="Part 2", color="#F28E2B")
    annotate_bars(axes[0], bars1, fmt="{:.0f}")
    annotate_bars(axes[0], bars2, fmt="{:.0f}")
    axes[0].set_xticks(x, [s.title() for s in size_order])
    axes[0].set_ylabel("Number of new facilities")
    axes[0].set_title("New-Facility Size Mix")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].set_axisbelow(True)

    bars = axes[1].barh(
        top_zip["zipcode"].astype(str),
        top_zip["total_added_capacity"] / 1_000,
        color="#59A14F",
    )
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Added capacity (thousands of slots)")
    axes[1].set_title("Top Realistic-Model ZIP Codes by Added Capacity")
    axes[1].grid(axis="x", alpha=0.25)
    axes[1].set_axisbelow(True)
    for bar in bars:
        width_val = bar.get_width()
        axes[1].text(
            width_val,
            bar.get_y() + bar.get_height() / 2,
            f" {width_val:.1f}",
            va="center",
            ha="left",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(FIG_DIR / "realistic_mix_and_zipcodes.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    ideal_costs = read_component_map(ROOT / "results" / "solutions" / "objective_breakdown_ideal.csv")
    real_costs = read_component_map(ROOT / "results" / "realistic" / "solutions" / "objective_breakdown_realistic.csv")

    ideal_summary = read_metric_map(ROOT / "results" / "solutions" / "summary_stats_ideal.csv")
    real_summary = read_metric_map(ROOT / "results" / "realistic" / "solutions" / "summary_stats_realistic.csv")

    ideal_new = pd.read_csv(ROOT / "results" / "solutions" / "new_build_solution_ideal.csv")
    ideal_build_counts = ideal_new.groupby("size")["num_new_facilities"].sum()

    real_build = pd.read_csv(ROOT / "results" / "realistic" / "solutions" / "build_size_summary_realistic.csv")
    real_build_counts = real_build.set_index("size")["n_new_facilities"]

    realistic_zip = pd.read_csv(ROOT / "results" / "realistic" / "solutions" / "zipcode_solution_realistic.csv")

    make_ideal_cost_breakdown(ideal_costs)
    make_part_comparison_overview(ideal_costs, real_costs, ideal_summary, real_summary)
    make_realistic_mix_and_zipcodes(ideal_build_counts, real_build_counts, realistic_zip)

    for name in [
        "ideal_cost_breakdown.png",
        "part_comparison_overview.png",
        "realistic_mix_and_zipcodes.png",
    ]:
        print("Saved", FIG_DIR / name)


if __name__ == "__main__":
    main()
