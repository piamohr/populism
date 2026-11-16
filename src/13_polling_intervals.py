from pathlib import Path
import numpy as np
import pandas as pd

# =====================================================
# SETTINGS
# =====================================================

SPEECH_FILE = Path("data/processed/final_llm_coding/final_analysis_dataset_merged.csv")
POLL_FILE = Path("data/raw/infratest_dimap.xlsx")

OUTPUT_DIR = Path("data/processed/final_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_SPEECH_POLL = OUTPUT_DIR / "02_speech_level_with_polling.csv"
OUTPUT_PARTY_INTERVAL = OUTPUT_DIR / "03_party_polling_interval.csv"
OUTPUT_MAINSTREAM_INTERVAL = OUTPUT_DIR / "04_mainstream_polling_interval.csv"
OUTPUT_PARTY_DISTANCE = OUTPUT_DIR / "05_party_polling_interval_distances.csv"
OUTPUT_MAINSTREAM_DISTANCE = OUTPUT_DIR / "06_mainstream_polling_interval_distances.csv"

MAINSTREAM_PARTIES = ["linke", "grüne", "spd", "cdu/csu", "fdp"]
ALL_PARTIES = ["linke", "grüne", "spd", "cdu/csu", "fdp", "afd"]

POP_COLS = [
    "P1_people",
    "P2_anti_elite",
    "P3_outgroup"
]

GAL_TAN_COLS = [
    "G1_environment",
    "G2_migration",
    "G3_society",
    "G4_civil_rights",
    "G5_cosmopolitanism"
]

GAL_TAN_TOPIC_LABELS = {
    "G1_environment": "environment",
    "G2_migration": "migration",
    "G3_society": "society",
    "G4_civil_rights": "civil_rights",
    "G5_cosmopolitanism": "cosmopolitanism"
}

TOPIC_COUNT_COLS = [
    "n_speeches_environment",
    "n_speeches_migration",
    "n_speeches_society",
    "n_speeches_civil_rights",
    "n_speeches_cosmopolitanism"
]

TOPIC_MAP = {
    "environment": "n_speeches_environment",
    "migration": "n_speeches_migration",
    "society": "n_speeches_society",
    "civil_rights": "n_speeches_civil_rights",
    "cosmopolitanism": "n_speeches_cosmopolitanism"
}


def check_required_columns(data: pd.DataFrame, required_columns: list[str], file_label: str) -> None:
    """Check that all expected variables are present."""
    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Expected variables are missing in {file_label}: "
            f"{missing_columns}"
        )

# =====================================================
# LOAD DATA
# =====================================================

speeches = pd.read_csv(
    SPEECH_FILE,
    dtype={
        "rede_id": str,
        "redner_id": str,
        "redner_rolle_kurz": str
    },
    low_memory=False
)

polls = pd.read_excel(POLL_FILE)

print("Speeches:", speeches.shape)
print("Polls:", polls.shape)

required_speech_columns = [
    "rede_id",
    "redner_id",
    "redner_rolle_kurz",
    "party_clean",
    "sitzung_datum",
    "wahlperiode",
    "government_label",
    "word_count",
    "has_gal_tan_score",
    "populism_score",
    "populism_sum",
    "gal_tan_score",
] + POP_COLS + GAL_TAN_COLS

check_required_columns(
    data=speeches,
    required_columns=required_speech_columns,
    file_label=str(SPEECH_FILE),
)

# =====================================================
# CLEANING
# =====================================================

# Harmonize column names
polls.columns = [c.strip().lower() for c in polls.columns]

# Check expected polling columns after harmonization
required_poll_columns = ["datum"] + ALL_PARTIES
check_required_columns(
    data=polls,
    required_columns=required_poll_columns,
    file_label=str(POLL_FILE),
)

# PHarmonize party names
speeches["party_clean"] = (
    speeches["party_clean"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# Date variables
speeches["sitzung_datum"] = pd.to_datetime(
    speeches["sitzung_datum"],
    errors="coerce"
)

# Robustly parse Excel dates
if pd.api.types.is_numeric_dtype(polls["datum"]):
    polls["poll_date"] = pd.to_datetime(
        polls["datum"],
        origin="1899-12-30",
        unit="D"
    )
else:
    polls["poll_date"] = pd.to_datetime(
        polls["datum"],
        errors="coerce"
    )

# Convert polling values to numeric
for col in ALL_PARTIES:
    polls[col] = pd.to_numeric(polls[col], errors="coerce")

# Convert LLM scores to numeric
for col in POP_COLS + GAL_TAN_COLS + ["populism_score", "populism_sum", "gal_tan_score"]:
    speeches[col] = pd.to_numeric(speeches[col], errors="coerce")

# Sort
polls = polls.sort_values("poll_date").reset_index(drop=True)
speeches = speeches.sort_values("sitzung_datum").reset_index(drop=True)

# =====================================================
# CREATE POLLING INTERVALS
# =====================================================

polls["next_poll_date"] = polls["poll_date"].shift(-1)
polls["poll_interval_id"] = np.arange(len(polls))

# AfD support at the start of the interval
polls["afd_support_t"] = polls["afd"]

# Keep only complete intervals
poll_intervals = polls.dropna(subset=["next_poll_date"]).copy()

print("\nPolling period:")
print(
    poll_intervals["poll_date"].min(),
    "to",
    poll_intervals["next_poll_date"].max()
)

# =====================================================
# ASSIGN SPEECHES TO POLLING INTERVALS
# =====================================================

speeches_with_polls = pd.merge_asof(
    speeches,
    poll_intervals[
        [
            "poll_interval_id",
            "poll_date",
            "next_poll_date",
            "afd_support_t",
            "cdu/csu",
            "afd",
            "spd",
            "grüne",
            "linke",
            "fdp"
        ]
    ].sort_values("poll_date"),
    left_on="sitzung_datum",
    right_on="poll_date",
    direction="backward",
    allow_exact_matches=True
)

# Safety check:
# The speech must actually fall within the interval [poll_date, next_poll_date).
speeches_with_polls = speeches_with_polls[
    (speeches_with_polls["sitzung_datum"] >= speeches_with_polls["poll_date"]) &
    (speeches_with_polls["sitzung_datum"] < speeches_with_polls["next_poll_date"])
].copy()

print("\nSpeeches after polling match:", speeches_with_polls.shape)
print("Unmatched speeches:", len(speeches) - len(speeches_with_polls))

# =====================================================
# RESHAPE POLLING DATA TO LONG FORMAT
# =====================================================

polls_long = poll_intervals.melt(
    id_vars=[
        "poll_interval_id",
        "poll_date",
        "next_poll_date",
        "afd_support_t"
    ],
    value_vars=ALL_PARTIES,
    var_name="party_clean",
    value_name="party_support_t"
)

polls_long = polls_long.sort_values(
    ["party_clean", "poll_date"]
).reset_index(drop=True)

# =====================================================
# MODERATOR: SUPPORT CHANGE
# =====================================================

# Change compared with the previous Infratest dimap poll.
# Negative values = loss
# Positive values = gain

polls_long["party_support_t_minus_1"] = (
    polls_long
    .groupby("party_clean")["party_support_t"]
    .shift(1)
)

polls_long["party_support_change"] = (
    polls_long["party_support_t"] -
    polls_long["party_support_t_minus_1"]
)

polls_long["party_support_change_abs"] = (
    polls_long["party_support_change"].abs()
)

# =====================================================
# ATTACH PARTY-SPECIFIC SUPPORT VARIABLES TO SPEECH LEVEL
# =====================================================

speeches_with_polls = speeches_with_polls.merge(
    polls_long[
        [
            "poll_interval_id",
            "party_clean",
            "party_support_t",
            "party_support_t_minus_1",
            "party_support_change",
            "party_support_change_abs"
        ]
    ],
    on=["poll_interval_id", "party_clean"],
    how="left",
    validate="many_to_one"
)

# =====================================================
# GAL-TAN TOPIC PRESENCE AT SPEECH LEVEL
# =====================================================

# A GAL-TAN dimension counts as addressed if a valid score exists.
# NaN means: topic not substantively present in the speech / not codable.

for col, label in GAL_TAN_TOPIC_LABELS.items():
    speeches_with_polls[f"has_{label}"] = (
        speeches_with_polls[col]
        .notna()
        .astype(int)
    )

# =====================================================
# SAVE SPEECH LEVEL WITH POLLING
# =====================================================

speeches_with_polls.to_csv(OUTPUT_SPEECH_POLL, index=False)

print(f"\nSaved: {OUTPUT_SPEECH_POLL}")

# =====================================================
# COMPLETE PARTY × POLLING INTERVAL GRID
# =====================================================

complete_party_grid = (
    poll_intervals[
        [
            "poll_interval_id",
            "poll_date",
            "next_poll_date",
            "afd_support_t"
        ]
    ]
    .assign(key=1)
    .merge(
        pd.DataFrame({"party_clean": ALL_PARTIES, "key": 1}),
        on="key"
    )
    .drop(columns="key")
)

# Attach support variables to the complete grid
support_vars = polls_long[
    [
        "poll_interval_id",
        "party_clean",
        "party_support_t",
        "party_support_t_minus_1",
        "party_support_change",
        "party_support_change_abs"
    ]
].copy()

complete_party_grid = complete_party_grid.merge(
    support_vars,
    on=["poll_interval_id", "party_clean"],
    how="left",
    validate="one_to_one"
)

# =====================================================
# AGGREGATE SPEECH SCORES: PARTY × INTERVAL
# =====================================================

speech_agg = (
    speeches_with_polls
    .groupby(
        [
            "poll_interval_id",
            "party_clean"
        ],
        as_index=False
    )
    .agg({
        # Indices
        "populism_score": "mean",
        "populism_sum": "mean",
        "gal_tan_score": "mean",

        # Populism subdimensions
        "P1_people": "mean",
        "P2_anti_elite": "mean",
        "P3_outgroup": "mean",

        # GAL-TAN subdimensions
        "G1_environment": "mean",
        "G2_migration": "mean",
        "G3_society": "mean",
        "G4_civil_rights": "mean",
        "G5_cosmopolitanism": "mean",

        # GAL-TAN topic counts
        "has_environment": "sum",
        "has_migration": "sum",
        "has_society": "sum",
        "has_civil_rights": "sum",
        "has_cosmopolitanism": "sum",

        # Case counts / checks
        "rede_id": "count",
        "has_gal_tan_score": "sum",
        "word_count": "mean"
    })
)

speech_agg = speech_agg.rename(columns={
    "populism_score": "mean_populism_score",
    "populism_sum": "mean_populism_sum",
    "gal_tan_score": "mean_gal_tan_score",

    "P1_people": "mean_P1_people",
    "P2_anti_elite": "mean_P2_anti_elite",
    "P3_outgroup": "mean_P3_outgroup",

    "G1_environment": "mean_G1_environment",
    "G2_migration": "mean_G2_migration",
    "G3_society": "mean_G3_society",
    "G4_civil_rights": "mean_G4_civil_rights",
    "G5_cosmopolitanism": "mean_G5_cosmopolitanism",

    "has_environment": "n_speeches_environment",
    "has_migration": "n_speeches_migration",
    "has_society": "n_speeches_society",
    "has_civil_rights": "n_speeches_civil_rights",
    "has_cosmopolitanism": "n_speeches_cosmopolitanism",

    "rede_id": "n_speeches",
    "has_gal_tan_score": "n_speeches_with_gal_tan",
    "word_count": "mean_word_count"
})

# Attach aggregated speech scores to the complete party-interval grid
party_interval = complete_party_grid.merge(
    speech_agg,
    on=["poll_interval_id", "party_clean"],
    how="left",
    validate="one_to_one"
)

# Make empty party intervals visible
party_interval["n_speeches"] = (
    party_interval["n_speeches"]
    .fillna(0)
    .astype(int)
)

party_interval["n_speeches_with_gal_tan"] = (
    party_interval["n_speeches_with_gal_tan"]
    .fillna(0)
    .astype(int)
)

for col in TOPIC_COUNT_COLS:
    party_interval[col] = (
        party_interval[col]
        .fillna(0)
        .astype(int)
    )

party_interval["is_mainstream_party"] = (
    party_interval["party_clean"].isin(MAINSTREAM_PARTIES)
)

party_interval["is_afd"] = (
    party_interval["party_clean"].eq("afd")
)

# =====================================================
# MODERATOR: ORDINAL PROXIMITY TO THE AFD
# =====================================================

afd_proximity_map = {
    "linke": 0,
    "grüne": 1,
    "spd": 2,
    "fdp": 3,
    "cdu/csu": 4
}

party_interval["afd_proximity_ordinal"] = (
    party_interval["party_clean"]
    .map(afd_proximity_map)
)

# Label for readability
afd_proximity_label_map = {
    0: "Die Linke",
    1: "Bündnis 90/Die Grünen",
    2: "SPD",
    3: "FDP",
    4: "CDU/CSU"
}

party_interval["afd_proximity_label"] = (
    party_interval["afd_proximity_ordinal"]
    .map(afd_proximity_label_map)
)

# =====================================================
# GAL-TAN TOPIC SHARES WITHIN PARTY INTERVALS
# =====================================================

for topic, count_col in TOPIC_MAP.items():
    party_interval[f"share_{topic}_of_party_speeches"] = np.where(
        party_interval["n_speeches"] > 0,
        party_interval[count_col] / party_interval["n_speeches"],
        np.nan
    )

# =====================================================
# MAINSTREAM TOTALS PER INTERVAL
# =====================================================

mainstream_totals = (
    party_interval[
        party_interval["party_clean"].isin(MAINSTREAM_PARTIES)
    ]
    .groupby("poll_interval_id", as_index=False)
    .agg({
        "n_speeches": "sum",
        "n_speeches_environment": "sum",
        "n_speeches_migration": "sum",
        "n_speeches_society": "sum",
        "n_speeches_civil_rights": "sum",
        "n_speeches_cosmopolitanism": "sum"
    })
    .rename(columns={
        "n_speeches": "n_mainstream_speeches",
        "n_speeches_environment": "n_mainstream_speeches_environment",
        "n_speeches_migration": "n_mainstream_speeches_migration",
        "n_speeches_society": "n_mainstream_speeches_society",
        "n_speeches_civil_rights": "n_mainstream_speeches_civil_rights",
        "n_speeches_cosmopolitanism": "n_mainstream_speeches_cosmopolitanism"
    })
)

party_interval = party_interval.merge(
    mainstream_totals,
    on="poll_interval_id",
    how="left",
    validate="many_to_one"
)

# Share of a party's topic speeches among all mainstream speeches in the interval
for topic, count_col in TOPIC_MAP.items():
    party_interval[f"share_{topic}_of_mainstream_speeches"] = np.where(
        party_interval["n_mainstream_speeches"] > 0,
        party_interval[count_col] / party_interval["n_mainstream_speeches"],
        np.nan
    )

# Share of a party among all mainstream speeches on a given topic
for topic, count_col in TOPIC_MAP.items():
    mainstream_topic_col = f"n_mainstream_speeches_{topic}"

    party_interval[f"share_party_of_mainstream_{topic}_speeches"] = np.where(
        party_interval[mainstream_topic_col] > 0,
        party_interval[count_col] / party_interval[mainstream_topic_col],
        np.nan
    )

# =====================================================
# GOVERNMENT / OPPOSITION AT INTERVAL LEVEL
# =====================================================

gov_interval = (
    speeches_with_polls
    .groupby(["poll_interval_id", "party_clean"])["government_label"]
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
    .reset_index()
    .rename(columns={"government_label": "government_label_interval"})
)

party_interval = party_interval.merge(
    gov_interval,
    on=["poll_interval_id", "party_clean"],
    how="left",
    validate="one_to_one"
)

# Numeric dummy variables:
# 1 = yes, 0 = no, NaN = no speech in the interval / not determinable
party_interval["is_government"] = np.where(
    party_interval["government_label_interval"].eq("Regierung"),
    1,
    np.where(
        party_interval["government_label_interval"].eq("Opposition"),
        0,
        np.nan
    )
)

party_interval["is_opposition"] = np.where(
    party_interval["government_label_interval"].eq("Opposition"),
    1,
    np.where(
        party_interval["government_label_interval"].eq("Regierung"),
        0,
        np.nan
    )
)

# =====================================================
# ELECTORAL TERM AT PARTY × INTERVAL LEVEL
# =====================================================

wahlperiode_party_interval = (
    speeches_with_polls
    .groupby(["poll_interval_id", "party_clean"])["wahlperiode"]
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
    .reset_index()
    .rename(columns={"wahlperiode": "wahlperiode"})
)

party_interval = party_interval.merge(
    wahlperiode_party_interval,
    on=["poll_interval_id", "party_clean"],
    how="left",
    validate="one_to_one"
)

party_interval["wahlperiode"] = party_interval["wahlperiode"].astype("Int64")
party_interval["wahlperiode_cat"] = (
    "WP" + party_interval["wahlperiode"].astype("string")
)

# =====================================================
# TIME VARIABLES
# =====================================================

party_interval["poll_year"] = party_interval["poll_date"].dt.year
party_interval["poll_month"] = party_interval["poll_date"].dt.month
party_interval["interval_length_days"] = (
    party_interval["next_poll_date"] -
    party_interval["poll_date"]
).dt.days

# Sort and save
party_interval = party_interval.sort_values(
    ["poll_interval_id", "party_clean"]
).reset_index(drop=True)

party_interval.to_csv(OUTPUT_PARTY_INTERVAL, index=False)

print(f"Saved: {OUTPUT_PARTY_INTERVAL}")

# =====================================================
# COMPLETE MAINSTREAM_ALL × POLLING INTERVAL GRID
# =====================================================

complete_mainstream_grid = poll_intervals[
    [
        "poll_interval_id",
        "poll_date",
        "next_poll_date",
        "afd_support_t"
    ]
].copy()

# =====================================================
# AGGREGATE MAINSTREAM_ALL
# =====================================================
# Speech-weighted: parties with more speeches in the interval contribute more.

mainstream_agg = (
    speeches_with_polls[
        speeches_with_polls["party_clean"].isin(MAINSTREAM_PARTIES)
    ]
    .groupby("poll_interval_id", as_index=False)
    .agg({
        # Indices
        "populism_score": "mean",
        "populism_sum": "mean",
        "gal_tan_score": "mean",

        # Populism subdimensions
        "P1_people": "mean",
        "P2_anti_elite": "mean",
        "P3_outgroup": "mean",

        # GAL-TAN subdimensions
        "G1_environment": "mean",
        "G2_migration": "mean",
        "G3_society": "mean",
        "G4_civil_rights": "mean",
        "G5_cosmopolitanism": "mean",

        # GAL-TAN topic counts
        "has_environment": "sum",
        "has_migration": "sum",
        "has_society": "sum",
        "has_civil_rights": "sum",
        "has_cosmopolitanism": "sum",

        # Case counts / checks
        "rede_id": "count",
        "has_gal_tan_score": "sum",
        "word_count": "mean"
    })
)

mainstream_agg = mainstream_agg.rename(columns={
    "populism_score": "mean_populism_score",
    "populism_sum": "mean_populism_sum",
    "gal_tan_score": "mean_gal_tan_score",

    "P1_people": "mean_P1_people",
    "P2_anti_elite": "mean_P2_anti_elite",
    "P3_outgroup": "mean_P3_outgroup",

    "G1_environment": "mean_G1_environment",
    "G2_migration": "mean_G2_migration",
    "G3_society": "mean_G3_society",
    "G4_civil_rights": "mean_G4_civil_rights",
    "G5_cosmopolitanism": "mean_G5_cosmopolitanism",

    "has_environment": "n_speeches_environment",
    "has_migration": "n_speeches_migration",
    "has_society": "n_speeches_society",
    "has_civil_rights": "n_speeches_civil_rights",
    "has_cosmopolitanism": "n_speeches_cosmopolitanism",

    "rede_id": "n_speeches",
    "has_gal_tan_score": "n_speeches_with_gal_tan",
    "word_count": "mean_word_count"
})

mainstream_interval = complete_mainstream_grid.merge(
    mainstream_agg,
    on="poll_interval_id",
    how="left",
    validate="one_to_one"
)

mainstream_interval["n_speeches"] = (
    mainstream_interval["n_speeches"]
    .fillna(0)
    .astype(int)
)

mainstream_interval["n_speeches_with_gal_tan"] = (
    mainstream_interval["n_speeches_with_gal_tan"]
    .fillna(0)
    .astype(int)
)

for col in TOPIC_COUNT_COLS:
    mainstream_interval[col] = (
        mainstream_interval[col]
        .fillna(0)
        .astype(int)
    )

for topic, count_col in TOPIC_MAP.items():
    mainstream_interval[f"share_{topic}_of_mainstream_speeches"] = np.where(
        mainstream_interval["n_speeches"] > 0,
        mainstream_interval[count_col] / mainstream_interval["n_speeches"],
        np.nan
    )

mainstream_interval["unit"] = "mainstream_all"

# =====================================================
# ELECTORAL TERM AT MAINSTREAM_ALL × INTERVAL LEVEL
# =====================================================

wahlperiode_mainstream_interval = (
    speeches_with_polls[
        speeches_with_polls["party_clean"].isin(MAINSTREAM_PARTIES)
    ]
    .groupby("poll_interval_id")["wahlperiode"]
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
    .reset_index()
)

mainstream_interval = mainstream_interval.merge(
    wahlperiode_mainstream_interval,
    on="poll_interval_id",
    how="left",
    validate="one_to_one"
)

mainstream_interval["poll_year"] = mainstream_interval["poll_date"].dt.year
mainstream_interval["poll_month"] = mainstream_interval["poll_date"].dt.month
mainstream_interval["interval_length_days"] = (
    mainstream_interval["next_poll_date"] -
    mainstream_interval["poll_date"]
).dt.days

mainstream_interval = mainstream_interval.sort_values(
    "poll_interval_id"
).reset_index(drop=True)

mainstream_interval["wahlperiode"] = mainstream_interval["wahlperiode"].astype("Int64")
mainstream_interval["wahlperiode_cat"] = (
    "WP" + mainstream_interval["wahlperiode"].astype("string")
)

mainstream_interval.to_csv(OUTPUT_MAINSTREAM_INTERVAL, index=False)

print(f"Saved: {OUTPUT_MAINSTREAM_INTERVAL}")

# =====================================================
# AFD REFERENCE VALUES PER INTERVAL
# =====================================================
# No new substantive aggregation:
# The AfD row from party_interval is only used as a reference for distances.

afd_interval = party_interval[
    party_interval["party_clean"] == "afd"
][
    [
        "poll_interval_id",

        "mean_populism_score",
        "mean_populism_sum",
        "mean_gal_tan_score",

        "mean_P1_people",
        "mean_P2_anti_elite",
        "mean_P3_outgroup",

        "mean_G1_environment",
        "mean_G2_migration",
        "mean_G3_society",
        "mean_G4_civil_rights",
        "mean_G5_cosmopolitanism",

        "n_speeches",
        "n_speeches_with_gal_tan",

        "n_speeches_environment",
        "n_speeches_migration",
        "n_speeches_society",
        "n_speeches_civil_rights",
        "n_speeches_cosmopolitanism",

        "share_environment_of_party_speeches",
        "share_migration_of_party_speeches",
        "share_society_of_party_speeches",
        "share_civil_rights_of_party_speeches",
        "share_cosmopolitanism_of_party_speeches"
    ]
].copy()

afd_interval = afd_interval.rename(columns={
    "mean_populism_score": "afd_mean_populism_score",
    "mean_populism_sum": "afd_mean_populism_sum",
    "mean_gal_tan_score": "afd_mean_gal_tan_score",

    "mean_P1_people": "afd_mean_P1_people",
    "mean_P2_anti_elite": "afd_mean_P2_anti_elite",
    "mean_P3_outgroup": "afd_mean_P3_outgroup",

    "mean_G1_environment": "afd_mean_G1_environment",
    "mean_G2_migration": "afd_mean_G2_migration",
    "mean_G3_society": "afd_mean_G3_society",
    "mean_G4_civil_rights": "afd_mean_G4_civil_rights",
    "mean_G5_cosmopolitanism": "afd_mean_G5_cosmopolitanism",

    "n_speeches": "afd_n_speeches",
    "n_speeches_with_gal_tan": "afd_n_speeches_with_gal_tan",

    "n_speeches_environment": "afd_n_speeches_environment",
    "n_speeches_migration": "afd_n_speeches_migration",
    "n_speeches_society": "afd_n_speeches_society",
    "n_speeches_civil_rights": "afd_n_speeches_civil_rights",
    "n_speeches_cosmopolitanism": "afd_n_speeches_cosmopolitanism",

    "share_environment_of_party_speeches": "afd_share_environment_of_party_speeches",
    "share_migration_of_party_speeches": "afd_share_migration_of_party_speeches",
    "share_society_of_party_speeches": "afd_share_society_of_party_speeches",
    "share_civil_rights_of_party_speeches": "afd_share_civil_rights_of_party_speeches",
    "share_cosmopolitanism_of_party_speeches": "afd_share_cosmopolitanism_of_party_speeches"
})

# =====================================================
# DISTANCES: MAINSTREAM PARTIES TO THE AFD
# =====================================================

party_distance = party_interval.merge(
    afd_interval,
    on="poll_interval_id",
    how="left",
    validate="many_to_one"
)

# Only mainstream parties for hypothesis tests
party_distance = party_distance[
    party_distance["party_clean"].isin(MAINSTREAM_PARTIES)
].copy()

# Central distances
party_distance["populism_distance_to_afd"] = (
    party_distance["mean_populism_score"] -
    party_distance["afd_mean_populism_score"]
).abs()

party_distance["populism_sum_distance_to_afd"] = (
    party_distance["mean_populism_sum"] -
    party_distance["afd_mean_populism_sum"]
).abs()

party_distance["gal_tan_distance_to_afd"] = (
    party_distance["mean_gal_tan_score"] -
    party_distance["afd_mean_gal_tan_score"]
).abs()

# Populismus subdimensions
party_distance["P1_people_distance_to_afd"] = (
    party_distance["mean_P1_people"] -
    party_distance["afd_mean_P1_people"]
).abs()

party_distance["P2_anti_elite_distance_to_afd"] = (
    party_distance["mean_P2_anti_elite"] -
    party_distance["afd_mean_P2_anti_elite"]
).abs()

party_distance["P3_outgroup_distance_to_afd"] = (
    party_distance["mean_P3_outgroup"] -
    party_distance["afd_mean_P3_outgroup"]
).abs()

# GAL-TAN subdimensions
party_distance["G1_environment_distance_to_afd"] = (
    party_distance["mean_G1_environment"] -
    party_distance["afd_mean_G1_environment"]
).abs()

party_distance["G2_migration_distance_to_afd"] = (
    party_distance["mean_G2_migration"] -
    party_distance["afd_mean_G2_migration"]
).abs()

party_distance["G3_society_distance_to_afd"] = (
    party_distance["mean_G3_society"] -
    party_distance["afd_mean_G3_society"]
).abs()

party_distance["G4_civil_rights_distance_to_afd"] = (
    party_distance["mean_G4_civil_rights"] -
    party_distance["afd_mean_G4_civil_rights"]
).abs()

party_distance["G5_cosmopolitanism_distance_to_afd"] = (
    party_distance["mean_G5_cosmopolitanism"] -
    party_distance["afd_mean_G5_cosmopolitanism"]
).abs()

party_distance = party_distance.sort_values(
    ["poll_interval_id", "party_clean"]
).reset_index(drop=True)

party_distance.to_csv(OUTPUT_PARTY_DISTANCE, index=False)

print(f"Saved: {OUTPUT_PARTY_DISTANCE}")

# =====================================================
# DISTANCES: MAINSTREAM_ALL TO THE AFD
# =====================================================

mainstream_distance = mainstream_interval.merge(
    afd_interval,
    on="poll_interval_id",
    how="left",
    validate="one_to_one"
)

mainstream_distance["populism_distance_to_afd"] = (
    mainstream_distance["mean_populism_score"] -
    mainstream_distance["afd_mean_populism_score"]
).abs()

mainstream_distance["populism_sum_distance_to_afd"] = (
    mainstream_distance["mean_populism_sum"] -
    mainstream_distance["afd_mean_populism_sum"]
).abs()

mainstream_distance["gal_tan_distance_to_afd"] = (
    mainstream_distance["mean_gal_tan_score"] -
    mainstream_distance["afd_mean_gal_tan_score"]
).abs()

mainstream_distance["P1_people_distance_to_afd"] = (
    mainstream_distance["mean_P1_people"] -
    mainstream_distance["afd_mean_P1_people"]
).abs()

mainstream_distance["P2_anti_elite_distance_to_afd"] = (
    mainstream_distance["mean_P2_anti_elite"] -
    mainstream_distance["afd_mean_P2_anti_elite"]
).abs()

mainstream_distance["P3_outgroup_distance_to_afd"] = (
    mainstream_distance["mean_P3_outgroup"] -
    mainstream_distance["afd_mean_P3_outgroup"]
).abs()

mainstream_distance["G1_environment_distance_to_afd"] = (
    mainstream_distance["mean_G1_environment"] -
    mainstream_distance["afd_mean_G1_environment"]
).abs()

mainstream_distance["G2_migration_distance_to_afd"] = (
    mainstream_distance["mean_G2_migration"] -
    mainstream_distance["afd_mean_G2_migration"]
).abs()

mainstream_distance["G3_society_distance_to_afd"] = (
    mainstream_distance["mean_G3_society"] -
    mainstream_distance["afd_mean_G3_society"]
).abs()

mainstream_distance["G4_civil_rights_distance_to_afd"] = (
    mainstream_distance["mean_G4_civil_rights"] -
    mainstream_distance["afd_mean_G4_civil_rights"]
).abs()

mainstream_distance["G5_cosmopolitanism_distance_to_afd"] = (
    mainstream_distance["mean_G5_cosmopolitanism"] -
    mainstream_distance["afd_mean_G5_cosmopolitanism"]
).abs()

mainstream_distance = mainstream_distance.sort_values(
    "poll_interval_id"
).reset_index(drop=True)

mainstream_distance.to_csv(OUTPUT_MAINSTREAM_DISTANCE, index=False)

print(f"Saved: {OUTPUT_MAINSTREAM_DISTANCE}")

# =====================================================
# CHECK OUTPUTS
# =====================================================

print("\n=====================================================")
print("CHECKS")
print("=====================================================")

print("\nSpeech level with polling:")
print(speeches_with_polls.shape)

print("\nParty × Polling interval:")
print(party_interval.shape)
print(party_interval["party_clean"].value_counts())

print("\nMainstream × Polling interval:")
print(mainstream_interval.shape)

print("\nParty distances:")
print(party_distance.shape)
print(party_distance["party_clean"].value_counts())

print("\nMainstream distances:")
print(mainstream_distance.shape)

print("\nTime period of matched speeches:")
print(
    speeches_with_polls["sitzung_datum"].min(),
    "to",
    speeches_with_polls["sitzung_datum"].max()
)

print("\nComplete polling intervals:")
print("Number of poll_intervals:", len(poll_intervals))
print("Expected party interval rows:", len(poll_intervals) * len(ALL_PARTIES))
print("Actual party interval rows:", len(party_interval))

print("\nEmpty party intervals without speeches:")
print(
    party_interval
    .groupby("party_clean")["n_speeches"]
    .apply(lambda x: (x == 0).sum())
)

print("\nSpeeches per party overall across all intervals:")
print(
    party_interval
    .groupby("party_clean")["n_speeches"]
    .sum()
    .sort_values(ascending=False)
)

print("\nMissing party_support_t by party:")
print(
    party_interval
    .groupby("party_clean")["party_support_t"]
    .apply(lambda x: x.isna().sum())
)

print("\nMissing party_support_change by party:")
print(
    party_interval
    .groupby("party_clean")["party_support_change"]
    .apply(lambda x: x.isna().sum())
)

print("\nSupport change, descriptive for mainstream parties:")
print(
    party_interval[
        party_interval["party_clean"].isin(MAINSTREAM_PARTIES)
    ]["party_support_change"].describe()
)

print("\nTopic counts across all party intervals:")
print(
    party_interval[
        [
            "n_speeches_environment",
            "n_speeches_migration",
            "n_speeches_society",
            "n_speeches_civil_rights",
            "n_speeches_cosmopolitanism"
        ]
    ].sum()
)

print("\nAfD intervals without speeches:")
print(
    party_interval[
        party_interval["party_clean"].eq("afd")
    ]["n_speeches"].eq(0).sum()
)

print("\nDistances with missing AfD reference value:")
print(
    party_distance[
        [
            "afd_mean_populism_score",
            "afd_mean_gal_tan_score",
            "populism_distance_to_afd",
            "gal_tan_distance_to_afd"
        ]
    ].isna().sum()
)

print("\nExample: party distance file")
print(party_distance.head())

print("\nDone.")