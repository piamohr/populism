# ============================================================
# APA 7 FIGURES: H1a/H2a, RQ1/RQ2 AND TIME SERIES
#
# Figures created:
# 1. figure_h1a_interaction.png
#    Left: distance values | Right: absolute values
# 2. figure_h2a_interaction.png
#    Left: distance values | Right: absolute values
# 3. figure_ff1_pairwise_matrix.png
#    Left: distance values | Right: absolute values
# 4. figure_ff2_pairwise_matrix.png
#    Left: distance values | Right: absolute values
# 5. figure_gal_tan_scores_and_afd_poll_over_time.png
# 6. figure_populism_scores_and_afd_poll_over_time.png
#
# Output exclusively as PNG at 300 dpi.
# ============================================================

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. FILE PATHS
# ============================================================

RESULTS_DIR = Path("results")
FIGURE_DIR = RESULTS_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

PARTY_INTERVAL_FILE = Path(
    "data/processed/final_analysis/03_party_polling_interval.csv"
)

# ------------------------------------------------------------
# H1a/H2a: prediction files
# ------------------------------------------------------------

H1A_DISTANCE_PREDICTION_FILE = (
    RESULTS_DIR / "hypothesis1a" / "h1a_distance_predicted_values.csv"
)
H1A_ABSOLUTE_PREDICTION_FILE = (
    RESULTS_DIR / "hypothesis1a" / "h1a_absolute_gal_tan_predicted_values.csv"
)
H2A_DISTANCE_PREDICTION_FILE = (
    RESULTS_DIR / "hypothesis2a" / "h2a_distance_predicted_values.csv"
)
H2A_ABSOLUTE_PREDICTION_FILE = (
    RESULTS_DIR / "hypothesis2a" / "h2a_absolute_populism_predicted_values.csv"
)

# ------------------------------------------------------------
# RQ1/RQ2: pairwise comparisons
# ------------------------------------------------------------

FF1_DISTANCE_PAIRWISE_FILE = (
    RESULTS_DIR / "research_question1" / "rq1_distance_pairwise_dimension_tests.csv"
)
FF1_ABSOLUTE_PAIRWISE_FILE = (
    RESULTS_DIR / "research_question1" / "rq1_absolute_pairwise_dimension_tests.csv"
)
FF2_DISTANCE_PAIRWISE_FILE = (
    RESULTS_DIR / "research_question2" / "rq2_distance_pairwise_feature_tests.csv"
)
FF2_ABSOLUTE_PAIRWISE_FILE = (
    RESULTS_DIR / "research_question2" / "rq2_absolute_pairwise_feature_tests.csv"
)


# ============================================================
# 2. APA-LIKE BASE SETTINGS
# ============================================================

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.unicode_minus": True,
    }
)

PARTY_ORDER = ["cdu/csu", "spd", "fdp", "grüne", "linke", "afd"]

PARTY_LABELS = {
    "cdu/csu": "CDU/CSU",
    "spd": "SPD",
    "fdp": "FDP",
    "grüne": "Bündnis 90/Die Grünen",
    "linke": "Die Linke",
    "afd": "AfD",
}

PARTY_COLORS = {
    "cdu/csu": "#000000",
    "spd": "#E3000F",
    "fdp": "#D4B900",
    "grüne": "#64A12D",
    "linke": "#BE3075",
    "afd": "#009EE0",
}


# ============================================================
# 3. GENERAL HELPER FUNCTIONS
# ============================================================

def check_file(file_path: Path) -> None:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found:\n{file_path.resolve()}")


def check_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    file_description: str,
) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"The following columns are missing in {file_description}:\n"
            + "\n".join(f"- {column}" for column in missing)
        )


def remove_chart_junk(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.set_facecolor("white")


def save_figure(fig: plt.Figure, filename_stem: str) -> None:
    """Saves exclusively a PNG file at 300 dpi."""
    png_file = FIGURE_DIR / f"{filename_stem}.png"
    fig.savefig(
        png_file,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    print(f"Saved: {png_file}")


# ============================================================
# 4. H1a/H2a: TWO-PANEL INTERACTION PLOTS
# ============================================================

def prepare_interaction_data(
    file_path: Path,
    prediction_column: str,
) -> tuple[pd.DataFrame, str, str, str]:
    """Reads and cleans a prediction file for an interaction plot."""
    check_file(file_path)
    df = pd.read_csv(file_path)

    x_column = "afd_support_t"
    group_column = "afd_proximity_ordinal"
    check_columns(
        df,
        [x_column, group_column, prediction_column],
        f"the prediction file {file_path}",
    )

    numeric_columns = [x_column, group_column, prediction_column]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=numeric_columns).copy()
    if df.empty:
        raise ValueError(
            f"No valid predicted values in:\n{file_path}\n\n"
            f"Columns used:\n"
            f"- AfD support: {x_column}\n"
            f"- Moderator: {group_column}\n"
            f"- Prediction: {prediction_column}"
        )

    print("\nInteraction data loaded:")
    print(f"File: {file_path}")
    print(f"x column: {x_column}")
    print(f"Group column: {group_column}")
    print(f"Prediction column: {prediction_column}")
    print(f"Valid rows: {len(df)}")

    return df, x_column, group_column, prediction_column


def draw_interaction_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_column: str,
    group_column: str,
    prediction_column: str,
    panel_title: str,
    y_label: str,
) -> None:
    party_labels = {
        0: "Die Linke",
        1: "Bündnis 90/Die Grünen",
        2: "SPD",
        3: "FDP",
        4: "CDU/CSU",
    }
    line_styles = {
        0: "-",
        1: "--",
        2: "-.",
        3: ":",
        4: (0, (5, 2)),
    }
    markers = {0: "o", 1: "s", 2: "^", 3: "D", 4: "v"}

    for value in sorted(df[group_column].unique()):
        group = df.loc[df[group_column] == value].sort_values(x_column)
        value_int = int(value)
        marker_positions = np.unique(
            np.linspace(0, len(group) - 1, min(6, len(group)), dtype=int)
        )
        ax.plot(
            group[x_column],
            group[prediction_column],
            label=party_labels.get(value_int, f"Proximity = {value_int}"),
            linestyle=line_styles.get(value_int, "-"),
            marker=markers.get(value_int, "o"),
            markevery=marker_positions,
            linewidth=1.3,
            markersize=4,
            color="black",
            markerfacecolor="white",
            markeredgecolor="black",
        )

    ax.set_title(panel_title, fontsize=12, pad=8)
    ax.set_xlabel("AfD support in percentage points")
    ax.set_ylabel(y_label, fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(axis="y", linewidth=0.5, color="0.88")
    remove_chart_junk(ax)


def create_paired_interaction_plot(
    distance_file: Path,
    absolute_file: Path,
    distance_prediction_column: str,
    absolute_prediction_column: str,
    distance_y_label: str,
    absolute_y_label: str,
    filename_stem: str,
) -> None:
    distance = prepare_interaction_data(
        distance_file,
        prediction_column=distance_prediction_column,
    )
    absolute = prepare_interaction_data(
        absolute_file,
        prediction_column=absolute_prediction_column,
    )

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(11.2, 4.8),
        sharex=True,
    )

    draw_interaction_panel(
        axes[0], *distance, "Distance values", distance_y_label
    )
    draw_interaction_panel(
        axes[1], *absolute, "Absolute values", absolute_y_label
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=min(5, len(labels)),
        handlelength=2.8,
        columnspacing=1.3,
        fontsize=11,
    )

    fig.tight_layout(rect=(0, 0.12, 1, 1), w_pad=2.5)
    save_figure(fig, filename_stem)
    plt.close(fig)


def create_h1a_interaction_plot() -> None:
    create_paired_interaction_plot(
        H1A_DISTANCE_PREDICTION_FILE,
        H1A_ABSOLUTE_PREDICTION_FILE,
        distance_prediction_column="predicted_distance",
        absolute_prediction_column="predicted_gal_tan",
        distance_y_label="Predicted GAL–TAN distance to the AfD",
        absolute_y_label="Predicted absolute GAL–TAN overall score",
        filename_stem="figure_h1a_interaction",
    )


def create_h2a_interaction_plot() -> None:
    create_paired_interaction_plot(
        H2A_DISTANCE_PREDICTION_FILE,
        H2A_ABSOLUTE_PREDICTION_FILE,
        distance_prediction_column="predicted_distance",
        absolute_prediction_column="predicted_populism",
        distance_y_label="Predicted populism distance to the AfD",
        absolute_y_label="Predicted absolute populism overall score",
        filename_stem="figure_h2a_interaction",
    )


# ============================================================
# 5. RQ1/RQ2: TWO-PANEL PAIRWISE MATRICES
# ============================================================

def load_pairwise_matrix_data(
    file_path: Path,
    label_a_column: str,
    label_b_column: str,
    preferred_order: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Reads the pairwise comparisons and creates a symmetric
    matrix of coefficient differences.

    For RQ1, the columns dimension_a_label and
    dimension_b_label are used. For RQ2, feature_a_label and
    feature_b_label are used.
    """
    check_file(file_path)
    df = pd.read_csv(file_path)

    difference_column = "slope_difference"
    p_column = "p_value_holm"
    wcr31_column = "wcr31_p_value_holm"
    check_columns(
        df,
        [label_a_column, label_b_column, difference_column, p_column, wcr31_column],
        f"the pairwise comparison file {file_path}",
    )

    numeric_columns = [
        difference_column,
        p_column,
        wcr31_column,
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            label_a_column,
            label_b_column,
            difference_column,
        ]
    ).copy()

    labels_in_data = list(
        dict.fromkeys(
            df[label_a_column].astype(str).tolist()
            + df[label_b_column].astype(str).tolist()
        )
    )

    if preferred_order:
        order = [
            label
            for label in preferred_order
            if label in labels_in_data
        ]
        order += [
            label
            for label in labels_in_data
            if label not in order
        ]
    else:
        order = labels_in_data

    if not order:
        raise ValueError(
            f"No valid feature labels in:\n{file_path}"
        )

    n = len(order)
    lookup = {
        label: index
        for index, label in enumerate(order)
    }

    values = np.full((n, n), np.nan)
    labels = np.full((n, n), "", dtype=object)

    for index in range(n):
        values[index, index] = 0.0
        labels[index, index] = "—"

    for _, row in df.iterrows():
        label_a = str(row[label_a_column])
        label_b = str(row[label_b_column])

        if label_a not in lookup or label_b not in lookup:
            continue

        difference = float(row[difference_column])
        i = lookup[label_a]
        j = lookup[label_b]

        significance_marker = ""

        if (
            pd.notna(row[p_column])
            and float(row[p_column]) < 0.05
        ):
            significance_marker += "*"

        if (
            pd.notna(row[wcr31_column])
            and float(row[wcr31_column]) < 0.05
        ):
            significance_marker += "†"

        values[i, j] = difference
        values[j, i] = -difference

        labels[i, j] = (
            f"{difference:+.3f}{significance_marker}"
        )
        labels[j, i] = (
            f"{-difference:+.3f}{significance_marker}"
        )

    print("\nPairwise comparison data loaded:")
    print(f"File: {file_path}")
    print(f"Feature A: {label_a_column}")
    print(f"Feature B: {label_b_column}")
    print(f"Difference: {difference_column}")
    print(f"Order: {order}")

    return values, labels, order


def draw_pairwise_matrix_panel(
    ax: plt.Axes,
    values: np.ndarray,
    labels: np.ndarray,
    dimensions: list[str],
    panel_title: str,
    vmax: float,
    x_axis_label: str,
    y_axis_label: str,
) -> mpl.image.AxesImage:
    shading = np.abs(values)

    image = ax.imshow(
        shading,
        cmap="Greys",
        vmin=0,
        vmax=vmax,
        aspect="equal",
    )

    n = len(dimensions)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(
        dimensions,
        rotation=40,
        ha="right",
        fontsize=11,
    )
    ax.set_yticklabels(dimensions, fontsize=11)
    ax.set_xlabel(x_axis_label, fontsize=12)
    ax.set_ylabel(y_axis_label, fontsize=12)
    ax.set_title(panel_title, fontsize=12, pad=8)

    for i in range(n):
        for j in range(n):
            text = labels[i, j]

            if text == "":
                continue

            cell_value = shading[i, j]
            text_color = (
                "white"
                if np.isfinite(cell_value)
                and cell_value > vmax * 0.55
                else "black"
            )

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=10.2,
                color=text_color,
            )

    ax.set_xticks(
        np.arange(-0.5, n, 1),
        minor=True,
    )
    ax.set_yticks(
        np.arange(-0.5, n, 1),
        minor=True,
    )
    ax.grid(
        which="minor",
        linewidth=0.7,
        color="white",
    )
    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    return image


def create_paired_pairwise_matrix(
    distance_file: Path,
    absolute_file: Path,
    label_a_column: str,
    label_b_column: str,
    preferred_order: list[str] | None,
    x_axis_label: str,
    y_axis_label: str,
    filename_stem: str,
) -> None:

    (
        distance_values,
        distance_labels,
        distance_order,
    ) = load_pairwise_matrix_data(
        file_path=distance_file,
        label_a_column=label_a_column,
        label_b_column=label_b_column,
        preferred_order=preferred_order,
    )

    (
        absolute_values,
        absolute_labels,
        absolute_order,
    ) = load_pairwise_matrix_data(
        file_path=absolute_file,
        label_a_column=label_a_column,
        label_b_column=label_b_column,
        preferred_order=preferred_order,
    )

    if distance_order != absolute_order:
        raise ValueError(
            "The features or their order differ "
            "between the distance file and the absolute-value file.\n"
            f"Distance: {distance_order}\n"
            f"Absolute: {absolute_order}"
        )

    maxima = [
        np.nanmax(np.abs(distance_values)),
        np.nanmax(np.abs(absolute_values)),
    ]
    vmax = max(maxima)

    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    n_dimensions = len(distance_order)
    width = max(12.6, 2.0 * n_dimensions + 3.4)

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(width, 5.8),
    )

    image = draw_pairwise_matrix_panel(
        ax=axes[0],
        values=distance_values,
        labels=distance_labels,
        dimensions=distance_order,
        panel_title="Distance values",
        vmax=vmax,
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
    )

    draw_pairwise_matrix_panel(
        ax=axes[1],
        values=absolute_values,
        labels=absolute_labels,
        dimensions=absolute_order,
        panel_title="Absolute values",
        vmax=vmax,
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
    )

    # Reserve more space on the right so that the color-scale legend
    # does not overlap the right panel.
    fig.subplots_adjust(
        left=0.08,
        right=0.84,
        bottom=0.19,
        top=0.90,
        wspace=0.34,
    )

    cbar_ax = fig.add_axes([0.87, 0.24, 0.018, 0.54])
    colorbar = fig.colorbar(
        image,
        cax=cbar_ax,
    )
    colorbar.set_label(
        "Absolute difference in\nregression coefficients",
        fontsize=11,
        labelpad=10,
    )
    colorbar.ax.tick_params(labelsize=10)

    save_figure(fig, filename_stem)
    plt.close(fig)


def create_ff1_pairwise_matrix() -> None:
    """RQ1: Comparison of the five GAL–TAN subdimensions."""
    create_paired_pairwise_matrix(
        distance_file=FF1_DISTANCE_PAIRWISE_FILE,
        absolute_file=FF1_ABSOLUTE_PAIRWISE_FILE,
        label_a_column="dimension_a_label",
        label_b_column="dimension_b_label",
        preferred_order=[
            "Environment",
            "Migration",
            "Social policy",
            "Civil rights",
            "Cosmopolitanism",
        ],
        x_axis_label="Subdimension B",
        y_axis_label="Subdimension A",
        filename_stem="figure_ff1_pairwise_matrix",
    )


def create_ff2_pairwise_matrix() -> None:
    """RQ2: Comparison of the three features of populist communication."""
    create_paired_pairwise_matrix(
        distance_file=FF2_DISTANCE_PAIRWISE_FILE,
        absolute_file=FF2_ABSOLUTE_PAIRWISE_FILE,
        label_a_column="feature_a_label",
        label_b_column="feature_b_label",
        preferred_order=[
            "People-centrism",
            "Anti-elitism",
            "Exclusion of outgroups",
        ],
        x_axis_label="Feature B",
        y_axis_label="Feature A",
        filename_stem="figure_ff2_pairwise_matrix",
    )


# ============================================================
# 6. TIME SERIES: ABSOLUTE SCORES + AfD POLL VALUES
# ============================================================

def create_score_and_afd_poll_plot(
    outcome: str,
    y_label: str,
    filename_stem: str,
) -> None:
    check_file(PARTY_INTERVAL_FILE)
    df = pd.read_csv(PARTY_INTERVAL_FILE)

    afd_poll_column = "afd_support_t"

    required_columns = ["poll_date", "party_clean", outcome, afd_poll_column]
    check_columns(df, required_columns, "the party–polling-interval dataset")

    plot_df = df[required_columns].copy()
    plot_df["poll_date"] = pd.to_datetime(plot_df["poll_date"], errors="coerce")
    plot_df[outcome] = pd.to_numeric(plot_df[outcome], errors="coerce")
    plot_df[afd_poll_column] = pd.to_numeric(plot_df[afd_poll_column], errors="coerce")
    plot_df["party_clean"] = (
        plot_df["party_clean"].astype(str).str.strip().str.lower()
    )
    plot_df = plot_df.loc[plot_df["party_clean"].isin(PARTY_ORDER)].copy()
    plot_df = plot_df.dropna(subset=["poll_date", "party_clean", outcome])

    available_parties = set(plot_df["party_clean"].unique())
    missing_parties = [party for party in PARTY_ORDER if party not in available_parties]
    if missing_parties:
        raise ValueError(
            f"The following parties are missing for {outcome}:\n"
            + "\n".join(f"- {PARTY_LABELS[party]}" for party in missing_parties)
        )

    score_df = (
        plot_df.groupby(["poll_date", "party_clean"], as_index=False, observed=True)[outcome]
        .mean()
        .sort_values(["party_clean", "poll_date"])
    )

    afd_poll_df = (
        plot_df.loc[plot_df[afd_poll_column].notna(), ["poll_date", afd_poll_column]]
        .groupby("poll_date", as_index=False)[afd_poll_column]
        .mean()
        .sort_values("poll_date")
    )
    if afd_poll_df.empty:
        raise ValueError("No valid AfD poll values available.")

    fig, score_ax = plt.subplots(figsize=(8.4, 5.1))

    for party in PARTY_ORDER:
        party_df = score_df.loc[score_df["party_clean"] == party].sort_values("poll_date")
        score_ax.plot(
            party_df["poll_date"],
            party_df[outcome],
            label=PARTY_LABELS[party],
            color=PARTY_COLORS[party],
            linewidth=1.35,
            linestyle="-",
            zorder=2,
        )

    score_ax.set_xlabel("Poll date", fontsize=12)
    score_ax.set_ylabel(y_label, fontsize=12)
    score_ax.tick_params(axis="both", labelsize=11)
    score_ax.grid(axis="y", linewidth=0.5, color="0.88", zorder=0)
    score_ax.grid(axis="x", visible=False)
    remove_chart_junk(score_ax)
    score_ax.xaxis.set_major_locator(mdates.YearLocator())
    score_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    poll_ax = score_ax.twinx()
    poll_ax.plot(
    afd_poll_df["poll_date"],
    afd_poll_df[afd_poll_column],
    label="AfD poll value",
    color=PARTY_COLORS["afd"],
    linestyle="--",
    linewidth=2.0,
    zorder=4,
)
    poll_ax.set_ylabel("AfD poll value in percentage points", fontsize=12)
    poll_ax.tick_params(axis="y", labelsize=11)
    poll_ax.spines["top"].set_visible(False)
    poll_ax.spines["right"].set_linewidth(0.8)

    score_handles, score_labels = score_ax.get_legend_handles_labels()
    poll_handles, poll_labels = poll_ax.get_legend_handles_labels()
    score_ax.legend(
        score_handles + poll_handles,
        score_labels + poll_labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=4,
        handlelength=2.8,
        columnspacing=1.3,
        fontsize=11
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.27, right=0.88)
    save_figure(fig, filename_stem)
    plt.close(fig)


def create_gal_tan_and_afd_poll_plot() -> None:
    create_score_and_afd_poll_plot(
        outcome="mean_gal_tan_score",
        y_label="Mean overall GAL–TAN score",
        filename_stem="figure_gal_tan_scores_and_afd_poll_over_time",
    )


def create_populism_and_afd_poll_plot() -> None:
    create_score_and_afd_poll_plot(
        outcome="mean_populism_score",
        y_label="Mean overall populism score",
        filename_stem="figure_populism_scores_and_afd_poll_over_time",
    )


# ============================================================
# 7. CREATE ALL DESIRED FIGURES
# ============================================================

def main() -> None:
    print("=" * 72)
    print("Creating the selected APA 7 figures")
    print("=" * 72)

    # H1a/H2a: distance values on the left, absolute values on the right
    create_h1a_interaction_plot()
    create_h2a_interaction_plot()

    # RQ1/RQ2: distance values on the left, absolute values on the right
    create_ff1_pairwise_matrix()
    create_ff2_pairwise_matrix()

    # Absolute overall scores with AfD poll values
    create_gal_tan_and_afd_poll_plot()
    create_populism_and_afd_poll_plot()

    print("\nAll desired figures have been created.")
    print(f"Output folder: {FIGURE_DIR.resolve()}")


if __name__ == "__main__":
    main()