from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# 1. SETTINGS
# ============================================================

INPUT_SPEECH = Path(
    "data/processed/final_analysis/"
    "02_speech_level_with_polling.csv"
)

INPUT_PARTY_INTERVAL = Path(
    "data/processed/final_analysis/"
    "05_party_polling_interval_distances.csv"
)

INPUT_MAINSTREAM_INTERVAL = Path(
    "data/processed/final_analysis/"
    "06_mainstream_polling_interval_distances.csv"
)

OUTPUT_DIR = Path(
    "data/processed/descriptives"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DECIMALS = 3

PARTY_ORDER = [
    "linke",
    "grüne",
    "spd",
    "fdp",
    "cdu/csu",
    "afd",
]

MAINSTREAM_PARTIES = [
    "linke",
    "grüne",
    "spd",
    "fdp",
    "cdu/csu",
]


# ============================================================
# 2. VARIABLE DEFINITIONS
# ============================================================

POPULISM_SCORE = "populism_score"

POPULISM_SUBDIMENSIONS = [
    "P1_people",
    "P2_anti_elite",
    "P3_outgroup",
]

GAL_TAN_SCORE = "gal_tan_score"

GAL_TAN_SUBDIMENSIONS = [
    "G1_environment",
    "G2_migration",
    "G3_society",
    "G4_civil_rights",
    "G5_cosmopolitanism",
]

ALL_SCORE_VARIABLES = (
    [POPULISM_SCORE]
    + POPULISM_SUBDIMENSIONS
    + [GAL_TAN_SCORE]
    + GAL_TAN_SUBDIMENSIONS
)

VARIABLE_LABELS = {
    "populism_score": "Populism overall score",
    "P1_people": "People-centrism",
    "P2_anti_elite": "Anti-elitism",
    "P3_outgroup": "Exclusion of out-groups",

    "gal_tan_score": "GAL-TAN overall score",
    "G1_environment": "Environment",
    "G2_migration": "Migration",
    "G3_society": "Social policy",
    "G4_civil_rights": "Civil rights",
    "G5_cosmopolitanism": "Cosmopolitanism",
}

VARIABLE_GROUPS = {
    "populism_score": "Populism overall score",
    "P1_people": "Populism subdimension",
    "P2_anti_elite": "Populism subdimension",
    "P3_outgroup": "Populism subdimension",

    "gal_tan_score": "GAL-TAN overall score",
    "G1_environment": "GAL-TAN subdimension",
    "G2_migration": "GAL-TAN subdimension",
    "G3_society": "GAL-TAN subdimension",
    "G4_civil_rights": "GAL-TAN subdimension",
    "G5_cosmopolitanism": "GAL-TAN subdimension",
}

VARIABLE_CONSTRUCTS = {
    "populism_score": "Populism",
    "P1_people": "Populism",
    "P2_anti_elite": "Populism",
    "P3_outgroup": "Populism",

    "gal_tan_score": "GAL-TAN",
    "G1_environment": "GAL-TAN",
    "G2_migration": "GAL-TAN",
    "G3_society": "GAL-TAN",
    "G4_civil_rights": "GAL-TAN",
    "G5_cosmopolitanism": "GAL-TAN",
}


# ============================================================
# 3. GAL-TAN TOPIC INDICATORS AT SPEECH LEVEL
# ============================================================

GAL_TAN_TOPIC_INDICATORS = {
    "G1_environment": "has_environment",
    "G2_migration": "has_migration",
    "G3_society": "has_society",
    "G4_civil_rights": "has_civil_rights",
    "G5_cosmopolitanism": "has_cosmopolitanism",
}


# ============================================================
# 4. VARIABLES AT INTERVAL LEVEL
# ============================================================

INTERVAL_OVERALL_VARIABLES = {
    "Populism": {
        "mainstream_absolute": "mean_populism_score",
        "afd_absolute": "afd_mean_populism_score",
        "distance_to_afd": "populism_distance_to_afd",
    },
    "GAL-TAN": {
        "mainstream_absolute": "mean_gal_tan_score",
        "afd_absolute": "afd_mean_gal_tan_score",
        "distance_to_afd": "gal_tan_distance_to_afd",
    },
}

INTERVAL_SUBDIMENSION_VARIABLES = {
    "P1_people": {
        "mainstream_absolute": "mean_P1_people",
        "afd_absolute": "afd_mean_P1_people",
        "distance_to_afd": "P1_people_distance_to_afd",
    },
    "P2_anti_elite": {
        "mainstream_absolute": "mean_P2_anti_elite",
        "afd_absolute": "afd_mean_P2_anti_elite",
        "distance_to_afd": "P2_anti_elite_distance_to_afd",
    },
    "P3_outgroup": {
        "mainstream_absolute": "mean_P3_outgroup",
        "afd_absolute": "afd_mean_P3_outgroup",
        "distance_to_afd": "P3_outgroup_distance_to_afd",
    },

    "G1_environment": {
        "mainstream_absolute": "mean_G1_environment",
        "afd_absolute": "afd_mean_G1_environment",
        "distance_to_afd": "G1_environment_distance_to_afd",
    },
    "G2_migration": {
        "mainstream_absolute": "mean_G2_migration",
        "afd_absolute": "afd_mean_G2_migration",
        "distance_to_afd": "G2_migration_distance_to_afd",
    },
    "G3_society": {
        "mainstream_absolute": "mean_G3_society",
        "afd_absolute": "afd_mean_G3_society",
        "distance_to_afd": "G3_society_distance_to_afd",
    },
    "G4_civil_rights": {
        "mainstream_absolute": "mean_G4_civil_rights",
        "afd_absolute": "afd_mean_G4_civil_rights",
        "distance_to_afd": "G4_civil_rights_distance_to_afd",
    },
    "G5_cosmopolitanism": {
        "mainstream_absolute": "mean_G5_cosmopolitanism",
        "afd_absolute": "afd_mean_G5_cosmopolitanism",
        "distance_to_afd": "G5_cosmopolitanism_distance_to_afd",
    },
}


# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def check_file_exists(filepath):
    """
    Check whether an input file exists.
    """
    if not filepath.exists():
        raise FileNotFoundError(
            f"File not found:\n{filepath.resolve()}"
        )


def check_required_columns(
    data,
    required_columns,
    dataset_name,
):
    """
    Check whether all required variables are present.
    """
    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        missing_text = "\n".join(
            f"- {column}"
            for column in missing_columns
        )

        raise KeyError(
            f"The following variables are missing in "
            f"dataset '{dataset_name}':\n{missing_text}"
        )


def standardize_party(series):
    """
    Standardize party names.
    """
    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )


def descriptive_stats_for_series(series):
    """
    Compute descriptive statistics for one variable.
    """
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    return pd.Series({
        "n_total": int(numeric.size),
        "n_valid": int(numeric.count()),
        "n_missing": int(numeric.isna().sum()),
        "share_missing": numeric.isna().mean(),
        "percent_missing": numeric.isna().mean() * 100,
        "mean": numeric.mean(),
        "sd": numeric.std(ddof=1),
        "min": numeric.min(),
        "p25": numeric.quantile(0.25),
        "median": numeric.median(),
        "p75": numeric.quantile(0.75),
        "max": numeric.max(),
    })


def descriptive_stats_overall(
    data,
    variable,
    group_label="overall",
):
    """
    Compute overall descriptive statistics.
    """
    stats = descriptive_stats_for_series(
        data[variable]
    ).to_frame().T

    stats.insert(
        0,
        "group",
        group_label,
    )

    stats.insert(
        1,
        "construct",
        VARIABLE_CONSTRUCTS[variable],
    )

    stats.insert(
        2,
        "variable",
        variable,
    )

    stats.insert(
        3,
        "variable_label",
        VARIABLE_LABELS[variable],
    )

    stats.insert(
        4,
        "variable_group",
        VARIABLE_GROUPS[variable],
    )

    return stats


def descriptive_stats_by_party(
    data,
    variable,
):
    """
    Compute descriptive statistics per party.
    """
    output = (
        data
        .groupby(
            "party_clean",
            observed=False,
        )[variable]
        .apply(
            descriptive_stats_for_series
        )
        .unstack()
        .reset_index()
    )

    output.insert(
        1,
        "construct",
        VARIABLE_CONSTRUCTS[variable],
    )

    output.insert(
        2,
        "variable",
        variable,
    )

    output.insert(
        3,
        "variable_label",
        VARIABLE_LABELS[variable],
    )

    output.insert(
        4,
        "variable_group",
        VARIABLE_GROUPS[variable],
    )

    return output


def create_party_rankings(
    data,
    variables,
):
    """
    Create means, standard deviations and rankings of the parties
    for several variables.
    """
    tables = []

    for variable in variables:
        grouped = (
            data
            .groupby(
                "party_clean",
                observed=False,
            )[variable]
            .agg(
                n_valid="count",
                mean="mean",
                sd="std",
                median="median",
                min="min",
                max="max",
            )
            .reset_index()
        )

        grouped.insert(
            1,
            "construct",
            VARIABLE_CONSTRUCTS[variable],
        )

        grouped.insert(
            2,
            "variable",
            variable,
        )

        grouped.insert(
            3,
            "variable_label",
            VARIABLE_LABELS[variable],
        )

        grouped["rank_highest_value"] = (
            grouped["mean"]
            .rank(
                method="min",
                ascending=False,
            )
            .astype("Int64")
        )

        grouped["rank_lowest_value"] = (
            grouped["mean"]
            .rank(
                method="min",
                ascending=True,
            )
            .astype("Int64")
        )

        tables.append(grouped)

    output = pd.concat(
        tables,
        ignore_index=True,
    )

    output["party_clean"] = pd.Categorical(
        output["party_clean"],
        categories=PARTY_ORDER,
        ordered=True,
    )

    return (
        output
        .sort_values(
            [
                "construct",
                "variable",
                "rank_highest_value",
                "party_clean",
            ]
        )
        .reset_index(drop=True)
    )


def create_mainstream_speech_stats(
    data,
    variables,
):
    """
    Compute speech-weighted values of the mainstream parties.
    """
    mainstream_data = data.loc[
        data["party_clean"].isin(
            MAINSTREAM_PARTIES
        )
    ].copy()

    rows = []

    for variable in variables:
        stats = descriptive_stats_for_series(
            mainstream_data[variable]
        ).to_dict()

        rows.append({
            "group": "Mainstream parties",
            "weighting": "speech-weighted",
            "construct": VARIABLE_CONSTRUCTS[variable],
            "variable": variable,
            "variable_label": VARIABLE_LABELS[variable],
            **stats,
        })

    return pd.DataFrame(rows)


def create_party_balanced_mainstream_stats(
    data,
    variables,
):
    """
    Compute party-weighted mainstream values.

    Each mainstream party enters with the same weight.
    """
    mainstream_data = data.loc[
        data["party_clean"].isin(
            MAINSTREAM_PARTIES
        )
    ].copy()

    rows = []

    for variable in variables:
        party_means = (
            mainstream_data
            .groupby(
                "party_clean",
                observed=True,
            )[variable]
            .mean()
            .dropna()
        )

        rows.append({
            "group": "Mainstream parties",
            "weighting": "party-weighted",
            "construct": VARIABLE_CONSTRUCTS[variable],
            "variable": variable,
            "variable_label": VARIABLE_LABELS[variable],
            "n_parties": int(party_means.count()),
            "mean": party_means.mean(),
            "sd_between_parties": party_means.std(ddof=1),
            "min_party_mean": party_means.min(),
            "median_party_mean": party_means.median(),
            "max_party_mean": party_means.max(),
        })

    return pd.DataFrame(rows)


def create_interval_overall_stats(
    data,
    variable_mapping,
    analysis_level,
):
    """
    Compute absolute mainstream values, absolute AfD values
    and distances to the AfD.
    """
    rows = []

    for construct, mapping in variable_mapping.items():
        for measure_type, variable in mapping.items():
            stats = descriptive_stats_for_series(
                data[variable]
            ).to_dict()

            rows.append({
                "analysis_level": analysis_level,
                "construct": construct,
                "measure_type": measure_type,
                "variable": variable,
                **stats,
            })

    return pd.DataFrame(rows)


def create_interval_subdimension_stats(
    data,
    variable_mapping,
    analysis_level,
):
    """
    Compute parallel statistics for all populism and
    GAL-TAN subdimensions.
    """
    rows = []

    for subdimension, mapping in variable_mapping.items():
        construct = VARIABLE_CONSTRUCTS[subdimension]

        for measure_type, variable in mapping.items():
            stats = descriptive_stats_for_series(
                data[variable]
            ).to_dict()

            rows.append({
                "analysis_level": analysis_level,
                "construct": construct,
                "subdimension": subdimension,
                "subdimension_label": (
                    VARIABLE_LABELS[subdimension]
                ),
                "measure_type": measure_type,
                "variable": variable,
                **stats,
            })

    return pd.DataFrame(rows)


def create_interval_overall_by_party(
    data,
    variable_mapping,
):
    """
    Compute absolute values and distances at party-interval level,
    separately by party.

    AfD values are not reported per mainstream party, because
    within an interval they are identical for all mainstream parties.
    """
    rows = []

    for construct, mapping in variable_mapping.items():
        for measure_type in [
            "mainstream_absolute",
            "distance_to_afd",
        ]:
            variable = mapping[measure_type]

            grouped = (
                data
                .groupby(
                    "party_clean",
                    observed=False,
                )[variable]
                .agg(
                    n_valid="count",
                    mean="mean",
                    sd="std",
                    median="median",
                    min="min",
                    max="max",
                )
                .reset_index()
            )

            grouped.insert(
                1,
                "construct",
                construct,
            )

            grouped.insert(
                2,
                "measure_type",
                measure_type,
            )

            grouped.insert(
                3,
                "variable",
                variable,
            )

            grouped["rank_highest_value"] = (
                grouped["mean"]
                .rank(
                    method="min",
                    ascending=False,
                )
                .astype("Int64")
            )

            grouped["rank_lowest_value"] = (
                grouped["mean"]
                .rank(
                    method="min",
                    ascending=True,
                )
                .astype("Int64")
            )

            rows.append(grouped)

    return pd.concat(
        rows,
        ignore_index=True,
    )


def create_interval_subdimensions_by_party(
    data,
    variable_mapping,
):
    """
    Compute absolute subdimension values and distances
    separately by mainstream party.
    """
    rows = []

    for subdimension, mapping in variable_mapping.items():
        construct = VARIABLE_CONSTRUCTS[subdimension]

        for measure_type in [
            "mainstream_absolute",
            "distance_to_afd",
        ]:
            variable = mapping[measure_type]

            grouped = (
                data
                .groupby(
                    "party_clean",
                    observed=False,
                )[variable]
                .agg(
                    n_valid="count",
                    mean="mean",
                    sd="std",
                    median="median",
                    min="min",
                    max="max",
                )
                .reset_index()
            )

            grouped.insert(
                1,
                "construct",
                construct,
            )

            grouped.insert(
                2,
                "subdimension",
                subdimension,
            )

            grouped.insert(
                3,
                "subdimension_label",
                VARIABLE_LABELS[subdimension],
            )

            grouped.insert(
                4,
                "measure_type",
                measure_type,
            )

            grouped.insert(
                5,
                "variable",
                variable,
            )

            grouped["rank_highest_value"] = (
                grouped["mean"]
                .rank(
                    method="min",
                    ascending=False,
                )
                .astype("Int64")
            )

            grouped["rank_lowest_value"] = (
                grouped["mean"]
                .rank(
                    method="min",
                    ascending=True,
                )
                .astype("Int64")
            )

            rows.append(grouped)

    return pd.concat(
        rows,
        ignore_index=True,
    )


def create_support_change_stats(data):
    """
    Compute the change in support.

    party_support_change =
    current poll value minus previous poll value.

    Negative values are losses.
    Positive values are gains.
    """
    values = pd.to_numeric(
        data["party_support_change"],
        errors="coerce",
    )

    descriptives = (
        descriptive_stats_for_series(values)
        .to_frame()
        .T
    )

    descriptives.insert(
        0,
        "variable",
        "party_support_change",
    )

    valid = pd.DataFrame({
        "party_support_change": values
    }).dropna()

    tolerance = 1e-10

    valid["change_category"] = np.select(
        [
            valid["party_support_change"] < -tolerance,
            valid["party_support_change"].abs() <= tolerance,
            valid["party_support_change"] > tolerance,
        ],
        [
            "Loss",
            "Unchanged",
            "Gain",
        ],
        default="Unclear",
    )

    categories = (
        valid
        .groupby(
            "change_category",
            observed=True,
        )
        .size()
        .reset_index(name="n")
    )

    categories["share"] = (
        categories["n"]
        / categories["n"].sum()
    )

    categories["percent"] = (
        categories["share"] * 100
    )

    category_order = [
        "Loss",
        "Unchanged",
        "Gain",
        "Unclear",
    ]

    categories["change_category"] = pd.Categorical(
        categories["change_category"],
        categories=category_order,
        ordered=True,
    )

    categories = (
        categories
        .sort_values("change_category")
        .reset_index(drop=True)
    )

    return descriptives, categories


def safe_round(
    dataframe,
    decimals=DECIMALS,
):
    """
    Round all numeric columns.
    """
    output = dataframe.copy()

    numeric_columns = output.select_dtypes(
        include=[np.number]
    ).columns

    output[numeric_columns] = (
        output[numeric_columns]
        .round(decimals)
    )

    return output


def write_csv(
    dataframe,
    filename,
):
    """
    Save a DataFrame as a CSV file.

    - semicolon as column separator
    - period as decimal separator
    - floats always with three decimal places
    """
    if dataframe is None or dataframe.empty:
        return

    output_path = OUTPUT_DIR / f"{filename}.csv"

    dataframe.to_csv(
        output_path,
        index=False,
        sep=";",
        decimal=".",
        encoding="utf-8-sig",
        float_format="%.3f",
    )


# ============================================================
# 6. CHECK AND LOAD FILES
# ============================================================

check_file_exists(INPUT_SPEECH)
check_file_exists(INPUT_PARTY_INTERVAL)
check_file_exists(INPUT_MAINSTREAM_INTERVAL)

speech_df = pd.read_csv(
    INPUT_SPEECH,
    low_memory=False,
)

party_interval_df = pd.read_csv(
    INPUT_PARTY_INTERVAL,
    low_memory=False,
)

mainstream_interval_df = pd.read_csv(
    INPUT_MAINSTREAM_INTERVAL,
    low_memory=False,
)

print("=" * 78)
print("DESCRIPTIVE STATISTICS")
print("=" * 78)

print(
    f"\nSpeech level: {len(speech_df)} rows, "
    f"{len(speech_df.columns)} columns"
)

print(
    f"Party-interval level: "
    f"{len(party_interval_df)} rows, "
    f"{len(party_interval_df.columns)} columns"
)

print(
    f"Aggregated mainstream level: "
    f"{len(mainstream_interval_df)} rows, "
    f"{len(mainstream_interval_df.columns)} columns"
)


# ============================================================
# 7. CHECK REQUIRED VARIABLES
# ============================================================

required_speech_columns = [
    "rede_id",
    "party_clean",
    "wahlperiode",
    "word_count",

    "populism_score",
    "P1_people",
    "P2_anti_elite",
    "P3_outgroup",

    "gal_tan_score",
    "G1_environment",
    "G2_migration",
    "G3_society",
    "G4_civil_rights",
    "G5_cosmopolitanism",

    "has_environment",
    "has_migration",
    "has_society",
    "has_civil_rights",
    "has_cosmopolitanism",
]

check_required_columns(
    data=speech_df,
    required_columns=required_speech_columns,
    dataset_name="02_speech_level_with_polling.csv",
)

required_party_interval_columns = [
    "poll_interval_id",
    "party_clean",
    "is_mainstream_party",
    "party_support_change",
    "party_support_change_abs",

    "mean_populism_score",
    "afd_mean_populism_score",
    "populism_distance_to_afd",

    "mean_gal_tan_score",
    "afd_mean_gal_tan_score",
    "gal_tan_distance_to_afd",
]

for mapping in INTERVAL_SUBDIMENSION_VARIABLES.values():
    required_party_interval_columns.extend(
        mapping.values()
    )

required_party_interval_columns = list(
    dict.fromkeys(
        required_party_interval_columns
    )
)

check_required_columns(
    data=party_interval_df,
    required_columns=required_party_interval_columns,
    dataset_name=(
        "05_party_polling_interval_distances.csv"
    ),
)

required_mainstream_interval_columns = [
    "poll_interval_id",

    "mean_populism_score",
    "afd_mean_populism_score",
    "populism_distance_to_afd",

    "mean_gal_tan_score",
    "afd_mean_gal_tan_score",
    "gal_tan_distance_to_afd",
]

for mapping in INTERVAL_SUBDIMENSION_VARIABLES.values():
    required_mainstream_interval_columns.extend(
        mapping.values()
    )

required_mainstream_interval_columns = list(
    dict.fromkeys(
        required_mainstream_interval_columns
    )
)

check_required_columns(
    data=mainstream_interval_df,
    required_columns=required_mainstream_interval_columns,
    dataset_name=(
        "06_mainstream_polling_interval_distances.csv"
    ),
)


# ============================================================
# 8. CLEAN SPEECH DATASET
# ============================================================

speech_df["party_clean"] = standardize_party(
    speech_df["party_clean"]
)

speech_df = speech_df.loc[
    speech_df["party_clean"].isin(
        PARTY_ORDER
    )
].copy()

speech_numeric_columns = (
    ALL_SCORE_VARIABLES
    + [
        "word_count",
        "wahlperiode",
        "has_environment",
        "has_migration",
        "has_society",
        "has_civil_rights",
        "has_cosmopolitanism",
    ]
)

for column in speech_numeric_columns:
    speech_df[column] = pd.to_numeric(
        speech_df[column],
        errors="coerce",
    )

speech_df["party_clean"] = pd.Categorical(
    speech_df["party_clean"],
    categories=PARTY_ORDER,
    ordered=True,
)


# ============================================================
# 9. CLEAN PARTY-INTERVAL DATASET
# ============================================================

party_interval_df["party_clean"] = standardize_party(
    party_interval_df["party_clean"]
)

party_interval_df["is_mainstream_party"] = (
    pd.to_numeric(
        party_interval_df["is_mainstream_party"],
        errors="coerce",
    )
)

party_interval_df = party_interval_df.loc[
    (
        party_interval_df["is_mainstream_party"] == 1
    )
    & (
        party_interval_df["party_clean"].isin(
            MAINSTREAM_PARTIES
        )
    )
].copy()

party_interval_df["party_clean"] = pd.Categorical(
    party_interval_df["party_clean"],
    categories=MAINSTREAM_PARTIES,
    ordered=True,
)

party_interval_numeric_columns = [
    "party_support_change",
    "party_support_change_abs",

    "mean_populism_score",
    "afd_mean_populism_score",
    "populism_distance_to_afd",

    "mean_gal_tan_score",
    "afd_mean_gal_tan_score",
    "gal_tan_distance_to_afd",
]

for mapping in INTERVAL_SUBDIMENSION_VARIABLES.values():
    party_interval_numeric_columns.extend(
        mapping.values()
    )

party_interval_numeric_columns = list(
    dict.fromkeys(
        party_interval_numeric_columns
    )
)

for column in party_interval_numeric_columns:
    party_interval_df[column] = pd.to_numeric(
        party_interval_df[column],
        errors="coerce",
    )


# ============================================================
# 10. CLEAN AGGREGATED MAINSTREAM DATASET
# ============================================================

mainstream_interval_numeric_columns = [
    "mean_populism_score",
    "afd_mean_populism_score",
    "populism_distance_to_afd",

    "mean_gal_tan_score",
    "afd_mean_gal_tan_score",
    "gal_tan_distance_to_afd",
]

for mapping in INTERVAL_SUBDIMENSION_VARIABLES.values():
    mainstream_interval_numeric_columns.extend(
        mapping.values()
    )

mainstream_interval_numeric_columns = list(
    dict.fromkeys(
        mainstream_interval_numeric_columns
    )
)

for column in mainstream_interval_numeric_columns:
    mainstream_interval_df[column] = pd.to_numeric(
        mainstream_interval_df[column],
        errors="coerce",
    )


# ============================================================
# 11. SPEECH LEVEL:
#     OVERALL DESCRIPTIVE STATISTICS
# ============================================================

overall_tables = []

for variable in ALL_SCORE_VARIABLES:
    overall_tables.append(
        descriptive_stats_overall(
            data=speech_df,
            variable=variable,
        )
    )

speech_scores_overall = pd.concat(
    overall_tables,
    ignore_index=True,
)

speech_scores_overall = (
    speech_scores_overall
    .sort_values(
        [
            "construct",
            "variable_group",
            "variable",
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# 12. SPEECH LEVEL:
#     DESCRIPTIVE STATISTICS BY PARTY
# ============================================================

party_tables = []

for variable in ALL_SCORE_VARIABLES:
    party_tables.append(
        descriptive_stats_by_party(
            data=speech_df,
            variable=variable,
        )
    )

speech_scores_by_party = pd.concat(
    party_tables,
    ignore_index=True,
)

speech_scores_by_party = (
    speech_scores_by_party
    .sort_values(
        [
            "construct",
            "variable_group",
            "variable",
            "party_clean",
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# 13. POPULISM PREVALENCE AT SPEECH LEVEL
#
# Share of valid codings with score 1 or 2.
# ============================================================

populism_prevalence_rows = []

for variable in POPULISM_SUBDIMENSIONS:
    values = pd.to_numeric(
        speech_df[variable],
        errors="coerce",
    )

    valid = values.dropna()

    score_1_or_2 = valid.isin(
        [1, 2]
    )

    populism_prevalence_rows.append({
        "construct": "Populism",
        "variable": variable,
        "variable_label": VARIABLE_LABELS[variable],
        "n_speeches_total": int(len(speech_df)),
        "n_valid": int(valid.count()),
        "n_missing": int(values.isna().sum()),
        "n_score_1_or_2": int(score_1_or_2.sum()),
        "share_score_1_or_2": score_1_or_2.mean(),
        "percent_score_1_or_2": (
            score_1_or_2.mean() * 100
        ),
        "mean": valid.mean(),
        "sd": valid.std(ddof=1),
        "min": valid.min(),
        "median": valid.median(),
        "max": valid.max(),
    })

populism_prevalence = pd.DataFrame(
    populism_prevalence_rows
)

populism_prevalence = (
    populism_prevalence
    .sort_values(
        "mean",
        ascending=False,
    )
    .reset_index(drop=True)
)


# ============================================================
# 14. GAL-TAN TOPIC FREQUENCY AT SPEECH LEVEL
#
# Uses the existing has_* variables.
# ============================================================

gal_tan_topic_rows = []

for variable, indicator_variable in (
    GAL_TAN_TOPIC_INDICATORS.items()
):
    indicator = pd.to_numeric(
        speech_df[indicator_variable],
        errors="coerce",
    )

    score = pd.to_numeric(
        speech_df[variable],
        errors="coerce",
    )

    mentioned = indicator.eq(1)

    valid_scores = score.loc[
        mentioned & score.notna()
    ]

    gal_tan_topic_rows.append({
        "construct": "GAL-TAN",
        "variable": variable,
        "variable_label": VARIABLE_LABELS[variable],
        "indicator_variable": indicator_variable,
        "n_speeches_total": int(len(speech_df)),
        "n_topic_mentioned": int(mentioned.sum()),
        "share_topic_mentioned": mentioned.mean(),
        "percent_topic_mentioned": (
            mentioned.mean() * 100
        ),
        "n_valid_scores": int(valid_scores.count()),
        "mean": valid_scores.mean(),
        "sd": valid_scores.std(ddof=1),
        "min": valid_scores.min(),
        "median": valid_scores.median(),
        "max": valid_scores.max(),
    })

gal_tan_topic_prevalence = pd.DataFrame(
    gal_tan_topic_rows
)

gal_tan_topic_prevalence = (
    gal_tan_topic_prevalence
    .sort_values(
        "percent_topic_mentioned",
        ascending=False,
    )
    .reset_index(drop=True)
)


# ============================================================
# 15. PARTY RANKINGS AT SPEECH LEVEL
# ============================================================

speech_party_rankings = create_party_rankings(
    data=speech_df,
    variables=ALL_SCORE_VARIABLES,
)

populism_party_rankings = (
    speech_party_rankings.loc[
        speech_party_rankings["construct"]
        == "Populism"
    ]
    .copy()
    .reset_index(drop=True)
)

gal_tan_party_rankings = (
    speech_party_rankings.loc[
        speech_party_rankings["construct"]
        == "GAL-TAN"
    ]
    .copy()
    .reset_index(drop=True)
)


# ============================================================
# 16. MAINSTREAM VALUES AT SPEECH LEVEL
# ============================================================

mainstream_speech_stats = (
    create_mainstream_speech_stats(
        data=speech_df,
        variables=ALL_SCORE_VARIABLES,
    )
)

mainstream_party_balanced_stats = (
    create_party_balanced_mainstream_stats(
        data=speech_df,
        variables=ALL_SCORE_VARIABLES,
    )
)


# ============================================================
# 17. WORD COUNT AND NUMBER OF SPEECHES
# ============================================================

word_count_overall = (
    descriptive_stats_for_series(
        speech_df["word_count"]
    )
    .to_frame()
    .T
)

word_count_overall.insert(
    0,
    "variable",
    "word_count",
)

word_count_overall.insert(
    1,
    "variable_label",
    "Word count",
)

speech_counts_by_party = (
    speech_df
    .groupby(
        "party_clean",
        observed=False,
    )
    .size()
    .reset_index(
        name="n_speeches"
    )
)

speech_counts_by_party["share_speeches"] = (
    speech_counts_by_party["n_speeches"]
    / speech_counts_by_party["n_speeches"].sum()
)

speech_counts_by_party["percent_speeches"] = (
    speech_counts_by_party["share_speeches"]
    * 100
)

speech_counts_by_party = (
    speech_counts_by_party
    .sort_values("party_clean")
    .reset_index(drop=True)
)

speech_counts_by_wahlperiode = (
    speech_df
    .groupby(
        "wahlperiode",
        dropna=False,
    )
    .size()
    .reset_index(
        name="n_speeches"
    )
)

speech_counts_by_wahlperiode["share_speeches"] = (
    speech_counts_by_wahlperiode["n_speeches"]
    / speech_counts_by_wahlperiode["n_speeches"].sum()
)

speech_counts_by_wahlperiode["percent_speeches"] = (
    speech_counts_by_wahlperiode["share_speeches"]
    * 100
)

speech_counts_by_wahlperiode = (
    speech_counts_by_wahlperiode
    .sort_values("wahlperiode")
    .reset_index(drop=True)
)

speech_counts_by_party_wahlperiode = (
    speech_df
    .groupby(
        [
            "wahlperiode",
            "party_clean",
        ],
        observed=False,
        dropna=False,
    )
    .size()
    .reset_index(
        name="n_speeches"
    )
)

speech_counts_by_party_wahlperiode[
    "n_total_wahlperiode"
] = (
    speech_counts_by_party_wahlperiode
    .groupby("wahlperiode")["n_speeches"]
    .transform("sum")
)

speech_counts_by_party_wahlperiode[
    "share_within_wahlperiode"
] = np.where(
    speech_counts_by_party_wahlperiode[
        "n_total_wahlperiode"
    ] > 0,
    (
        speech_counts_by_party_wahlperiode[
            "n_speeches"
        ]
        / speech_counts_by_party_wahlperiode[
            "n_total_wahlperiode"
        ]
    ),
    np.nan,
)

speech_counts_by_party_wahlperiode[
    "percent_within_wahlperiode"
] = (
    speech_counts_by_party_wahlperiode[
        "share_within_wahlperiode"
    ]
    * 100
)

speech_counts_by_party_wahlperiode = (
    speech_counts_by_party_wahlperiode
    .sort_values(
        [
            "wahlperiode",
            "party_clean",
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# 18. PARTY-INTERVAL LEVEL:
#     OVERALL SCORES AND DISTANCES
# ============================================================

party_interval_overall_stats = (
    create_interval_overall_stats(
        data=party_interval_df,
        variable_mapping=INTERVAL_OVERALL_VARIABLES,
        analysis_level="party_interval",
    )
)

party_interval_overall_by_party = (
    create_interval_overall_by_party(
        data=party_interval_df,
        variable_mapping=INTERVAL_OVERALL_VARIABLES,
    )
)


# ============================================================
# 19. PARTY-INTERVAL LEVEL:
#     SUBDIMENSIONS AND DISTANCES
# ============================================================

party_interval_subdimension_stats = (
    create_interval_subdimension_stats(
        data=party_interval_df,
        variable_mapping=(
            INTERVAL_SUBDIMENSION_VARIABLES
        ),
        analysis_level="party_interval",
    )
)

party_interval_subdimensions_by_party = (
    create_interval_subdimensions_by_party(
        data=party_interval_df,
        variable_mapping=(
            INTERVAL_SUBDIMENSION_VARIABLES
        ),
    )
)


# ============================================================
# 20. CHANGE IN SUPPORT
# ============================================================

(
    support_change_descriptives,
    support_change_categories,
) = create_support_change_stats(
    data=party_interval_df
)


# ============================================================
# 21. AGGREGATED MAINSTREAM LEVEL:
#     OVERALL SCORES AND DISTANCES
# ============================================================

mainstream_interval_overall_stats = (
    create_interval_overall_stats(
        data=mainstream_interval_df,
        variable_mapping=INTERVAL_OVERALL_VARIABLES,
        analysis_level=(
            "aggregated_mainstream_interval"
        ),
    )
)


# ============================================================
# 22. AGGREGATED MAINSTREAM LEVEL:
#     SUBDIMENSIONS AND DISTANCES
# ============================================================

mainstream_interval_subdimension_stats = (
    create_interval_subdimension_stats(
        data=mainstream_interval_df,
        variable_mapping=(
            INTERVAL_SUBDIMENSION_VARIABLES
        ),
        analysis_level=(
            "aggregated_mainstream_interval"
        ),
    )
)


# ============================================================
# 23. CONSTRUCT-SPECIFIC TABLES
# ============================================================

party_interval_populism = (
    party_interval_overall_stats.loc[
        party_interval_overall_stats["construct"]
        == "Populism"
    ]
    .copy()
    .reset_index(drop=True)
)

party_interval_gal_tan = (
    party_interval_overall_stats.loc[
        party_interval_overall_stats["construct"]
        == "GAL-TAN"
    ]
    .copy()
    .reset_index(drop=True)
)

party_interval_pop_subdimensions = (
    party_interval_subdimension_stats.loc[
        party_interval_subdimension_stats["construct"]
        == "Populism"
    ]
    .copy()
    .reset_index(drop=True)
)

party_interval_gal_subdimensions = (
    party_interval_subdimension_stats.loc[
        party_interval_subdimension_stats["construct"]
        == "GAL-TAN"
    ]
    .copy()
    .reset_index(drop=True)
)

mainstream_interval_populism = (
    mainstream_interval_overall_stats.loc[
        mainstream_interval_overall_stats["construct"]
        == "Populism"
    ]
    .copy()
    .reset_index(drop=True)
)

mainstream_interval_gal_tan = (
    mainstream_interval_overall_stats.loc[
        mainstream_interval_overall_stats["construct"]
        == "GAL-TAN"
    ]
    .copy()
    .reset_index(drop=True)
)

mainstream_interval_pop_subdimensions = (
    mainstream_interval_subdimension_stats.loc[
        mainstream_interval_subdimension_stats[
            "construct"
        ]
        == "Populism"
    ]
    .copy()
    .reset_index(drop=True)
)

mainstream_interval_gal_subdimensions = (
    mainstream_interval_subdimension_stats.loc[
        mainstream_interval_subdimension_stats[
            "construct"
        ]
        == "GAL-TAN"
    ]
    .copy()
    .reset_index(drop=True)
)


# ============================================================
# 24. SAVE CSV FILES
# ============================================================

# ------------------------------------------------------------
# Speech level: general statistics
# ------------------------------------------------------------

write_csv(
    speech_scores_overall,
    "01_scores_overall",
)

write_csv(
    speech_scores_by_party,
    "02_scores_by_party",
)


# ------------------------------------------------------------
# Speech level: populism prevalence and GAL-TAN topics
# ------------------------------------------------------------

write_csv(
    populism_prevalence,
    "03_populism_prevalence",
)

write_csv(
    gal_tan_topic_prevalence,
    "04_gal_tan_topics",
)


# ------------------------------------------------------------
# Speech level: party rankings
# ------------------------------------------------------------

write_csv(
    populism_party_rankings,
    "05_populism_party_rankings",
)

write_csv(
    gal_tan_party_rankings,
    "06_gal_tan_party_rankings",
)

write_csv(
    speech_party_rankings,
    "07_all_party_rankings",
)


# ------------------------------------------------------------
# Speech level: mainstream parties
# ------------------------------------------------------------

write_csv(
    mainstream_speech_stats,
    "08_mainstream_speech_weighted",
)

write_csv(
    mainstream_party_balanced_stats,
    "09_mainstream_party_weighted",
)


# ------------------------------------------------------------
# Word count and number of speeches
# ------------------------------------------------------------

write_csv(
    word_count_overall,
    "10_word_count_overall",
)

write_csv(
    speech_counts_by_party,
    "11_speech_counts_by_party",
)

write_csv(
    speech_counts_by_wahlperiode,
    "12_speech_counts_by_wahlperiode",
)

write_csv(
    speech_counts_by_party_wahlperiode,
    "13_speech_counts_by_party_wahlperiode",
)


# ------------------------------------------------------------
# Party-interval level: overall scores
# ------------------------------------------------------------

write_csv(
    party_interval_populism,
    "14_party_interval_populism",
)

write_csv(
    party_interval_gal_tan,
    "15_party_interval_gal_tan",
)

write_csv(
    party_interval_overall_stats,
    "16_party_interval_overall",
)

write_csv(
    party_interval_overall_by_party,
    "17_party_interval_by_party",
)


# ------------------------------------------------------------
# Party-interval level: subdimensions
# ------------------------------------------------------------

write_csv(
    party_interval_pop_subdimensions,
    "18_party_interval_populism_subdimensions",
)

write_csv(
    party_interval_gal_subdimensions,
    "19_party_interval_gal_tan_subdimensions",
)

write_csv(
    party_interval_subdimension_stats,
    "20_party_interval_all_subdimensions",
)

write_csv(
    party_interval_subdimensions_by_party,
    "21_party_interval_subdimensions_by_party",
)


# ------------------------------------------------------------
# Change in support
# ------------------------------------------------------------

write_csv(
    support_change_descriptives,
    "22_support_change_descriptives",
)

write_csv(
    support_change_categories,
    "23_support_change_categories",
)


# ------------------------------------------------------------
# Aggregated mainstream level: overall scores
# ------------------------------------------------------------

write_csv(
    mainstream_interval_populism,
    "24_mainstream_interval_populism",
)

write_csv(
    mainstream_interval_gal_tan,
    "25_mainstream_interval_gal_tan",
)

write_csv(
    mainstream_interval_overall_stats,
    "26_mainstream_interval_overall",
)


# ------------------------------------------------------------
# Aggregated mainstream level: subdimensions
# ------------------------------------------------------------

write_csv(
    mainstream_interval_pop_subdimensions,
    "27_mainstream_interval_populism_subdimensions",
)

write_csv(
    mainstream_interval_gal_subdimensions,
    "28_mainstream_interval_gal_tan_subdimensions",
)

write_csv(
    mainstream_interval_subdimension_stats,
    "29_mainstream_interval_all_subdimensions",
)


# ============================================================
# 25. CHECK OUTPUTS
# ============================================================

print("\n" + "=" * 78)
print("KEY DESCRIPTIVE STATISTICS")
print("=" * 78)


print("\n1. Populism overall at speech level:")

print(
    speech_scores_overall.loc[
        speech_scores_overall["construct"]
        == "Populism",
        [
            "variable_label",
            "n_valid",
            "mean",
            "sd",
            "min",
            "median",
            "max",
        ],
    ]
    .round(DECIMALS)
    .to_string(index=False)
)


print("\n2. Populism prevalence:")

print(
    populism_prevalence[
        [
            "variable_label",
            "n_valid",
            "n_score_1_or_2",
            "percent_score_1_or_2",
            "mean",
            "sd",
        ]
    ]
    .round(DECIMALS)
    .to_string(index=False)
)


print("\n3. GAL-TAN overall at speech level:")

print(
    speech_scores_overall.loc[
        speech_scores_overall["construct"]
        == "GAL-TAN",
        [
            "variable_label",
            "n_valid",
            "mean",
            "sd",
            "min",
            "median",
            "max",
        ],
    ]
    .round(DECIMALS)
    .to_string(index=False)
)


print("\n4. GAL-TAN topic frequencies:")

print(
    gal_tan_topic_prevalence[
        [
            "variable_label",
            "n_topic_mentioned",
            "percent_topic_mentioned",
            "mean",
            "sd",
        ]
    ]
    .round(DECIMALS)
    .to_string(index=False)
)


print("\n5. Populism by party:")

print(
    populism_party_rankings[
        [
            "variable_label",
            "party_clean",
            "n_valid",
            "mean",
            "sd",
            "rank_highest_value",
        ]
    ]
    .round(DECIMALS)
    .to_string(index=False)
)


print("\n6. GAL-TAN by party:")

print(
    gal_tan_party_rankings[
        [
            "variable_label",
            "party_clean",
            "n_valid",
            "mean",
            "sd",
            "rank_highest_value",
        ]
    ]
    .round(DECIMALS)
    .to_string(index=False)
)


print("\n7. Mainstream values at speech level:")

print(
    mainstream_speech_stats[
        [
            "construct",
            "variable_label",
            "n_valid",
            "mean",
            "sd",
        ]
    ]
    .round(DECIMALS)
    .to_string(index=False)
)


print(
    "\n8. Party-interval level: "
    "overall scores and distances"
)

print(
    party_interval_overall_stats[
        [
            "construct",
            "measure_type",
            "variable",
            "n_valid",
            "mean",
            "sd",
        ]
    ]
    .round(DECIMALS)
    .to_string(index=False)
)


print(
    "\n9. Party-interval level: "
    "subdimension values and distances"
)

print(
    party_interval_subdimension_stats[
        [
            "construct",
            "subdimension_label",
            "measure_type",
            "n_valid",
            "mean",
            "sd",
        ]
    ]
    .round(DECIMALS)
    .to_string(index=False)
)


print("\n10. Change in support:")

print(
    support_change_descriptives
    .round(DECIMALS)
    .to_string(index=False)
)


print(
    "\n11. Losses, unchanged values "
    "and gains:"
)

print(
    support_change_categories
    .round(DECIMALS)
    .to_string(index=False)
)


print(
    "\n12. Aggregated mainstream level: "
    "overall scores and distances"
)

print(
    mainstream_interval_overall_stats[
        [
            "construct",
            "measure_type",
            "variable",
            "n_valid",
            "mean",
            "sd",
        ]
    ]
    .round(DECIMALS)
    .to_string(index=False)
)


print(
    "\n13. Aggregated mainstream level: "
    "subdimension values and distances"
)

print(
    mainstream_interval_subdimension_stats[
        [
            "construct",
            "subdimension_label",
            "measure_type",
            "n_valid",
            "mean",
            "sd",
        ]
    ]
    .round(DECIMALS)
    .to_string(index=False)
)


print("\n14. Number of speeches per party:")

print(
    speech_counts_by_party
    .round(DECIMALS)
    .to_string(index=False)
)


print("\n15. Number of speeches per electoral term:")

print(
    speech_counts_by_wahlperiode
    .round(DECIMALS)
    .to_string(index=False)
)


print("\n16. Word count:")

print(
    word_count_overall
    .round(DECIMALS)
    .to_string(index=False)
)


print("\n" + "=" * 78)
print("ANALYSIS COMPLETE")
print("=" * 78)

print(
    f"\nCSV files saved to:\n"
    f"{OUTPUT_DIR.resolve()}"
)