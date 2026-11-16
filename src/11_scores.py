from pathlib import Path
import pandas as pd


# =====================================================
# SETTINGS
# =====================================================

INPUT_FILE = Path("data/processed/final_llm_coding/final_llm_scores_analysis_ready.csv")
OUTPUT_FILE = Path("data/processed/final_llm_coding/final_llm_scores_with_indices.csv")

POP_COLS = [
    "P1_people",
    "P2_anti_elite",
    "P3_outgroup",
]

GAL_TAN_COLS = [
    "G1_environment",
    "G2_migration",
    "G3_society",
    "G4_civil_rights",
    "G5_cosmopolitanism",
]

REQUIRED_COLS = POP_COLS + GAL_TAN_COLS


# =====================================================
# FUNCTIONS
# =====================================================

def validate_columns(df: pd.DataFrame) -> None:
    """Check that all columns required for index construction are present."""

    missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]

    if missing_cols:
        raise ValueError(f"These columns are missing in the input: {missing_cols}")


def add_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Compute populism and GAL-TAN indices without altering the original columns."""

    df = df.copy()

    # Convert scores to numeric.
    for col in REQUIRED_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Populism index
    # Mean across the three populism dimensions.
    # Scale: 0 to 2
    df["populism_score"] = df[POP_COLS].mean(axis=1, skipna=True)

    # Sum index across the three populism dimensions.
    # Scale: 0 to 6
    df["populism_sum"] = df[POP_COLS].sum(axis=1, skipna=True)

    # Number of valid populism dimensions
    df["populism_n_valid"] = df[POP_COLS].notna().sum(axis=1)

    # GAL-TAN index
    # Mean across all substantively coded GAL-TAN dimensions.
    # Scale: 0 to 10
    # Important: missing values are not imputed but ignored.
    df["gal_tan_score"] = df[GAL_TAN_COLS].mean(axis=1, skipna=True)

    # Number of valid GAL-TAN dimensions per speech
    # Scale: 0 to 5
    df["gal_tan_n_valid"] = df[GAL_TAN_COLS].notna().sum(axis=1)

    # Flag indicating whether any GAL-TAN value is present
    df["has_gal_tan_score"] = df["gal_tan_n_valid"] > 0

    return df


def print_checks(df: pd.DataFrame) -> None:
    """Print the same control outputs as the original script."""

    print("\nPopulism score:")
    print(df["populism_score"].describe())

    print("\nGAL-TAN score:")
    print(df["gal_tan_score"].describe())

    print("\nNumber of valid GAL-TAN dimensions per speech:")
    print(df["gal_tan_n_valid"].value_counts().sort_index())

    print("\nShare of speeches without GAL-TAN score:")
    print(df["gal_tan_score"].isna().mean())


# =====================================================
# MAIN
# =====================================================

def main() -> None:
    df = pd.read_csv(INPUT_FILE)

    validate_columns(df)

    print("Number of speeches:", len(df))
    print("Number of columns:", len(df.columns))

    df = add_indices(df)

    print_checks(df)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()