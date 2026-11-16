from pathlib import Path
import numpy as np
import pandas as pd


INPUT_FILE = Path("data/processed/03_cppbt_government.csv")
OUTPUT_FILE = Path("data/processed/validation/validation_sample.csv")

PARTY_COLUMN = "party_clean"
DATE_COLUMN = "sitzung_datum"
N_PER_PARTY = 15

REQUIRED_COLUMNS = [
    PARTY_COLUMN,
    DATE_COLUMN,
]


def check_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Abort with a clear error message if expected columns are missing."""
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            "The following expected columns are missing in the input dataset: "
            + ", ".join(missing_columns)
        )


def draw_balanced_sample(group: pd.DataFrame, n: int = N_PER_PARTY) -> pd.DataFrame:
    """
    Draw up to n speeches per party, evenly distributed over the party's time span.

    Selection is deterministic, using equidistant positions in the party dataset
    sorted by date. This preserves the behavior of the original script.
    """
    group = group.sort_values(DATE_COLUMN)

    positions = np.linspace(
        0,
        len(group) - 1,
        min(n, len(group)),
        dtype=int,
    )

    return group.iloc[positions]


def main() -> None:
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    check_required_columns(df, REQUIRED_COLUMNS)

    validation_sample = (
        df.groupby(PARTY_COLUMN, group_keys=True)
        .apply(draw_balanced_sample, n=N_PER_PARTY)
        .reset_index(drop=True)
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    validation_sample.to_csv(OUTPUT_FILE, index=False)

    print("\nSample size:")
    print(len(validation_sample))

    print("\nDistribution by party:")
    print(validation_sample.groupby(PARTY_COLUMN).size())

    print("\nTime span per party:")
    print(validation_sample.groupby(PARTY_COLUMN)[DATE_COLUMN].agg(["min", "max"]))

    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()