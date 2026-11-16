# ============================================================
# RQ3 / RQ4: Reciprocal relationship between rhetorical
# and positional development at the aggregated mainstream level
#
# RQ3: Does the rhetorical development of the mainstream
#      influence its later positional development?
#
# RQ4: Does the positional development of the mainstream
#      influence its later rhetorical development?
#
# Level of analysis:
# - one aggregated mainstream observation per polling interval
#
# Two operationalizations are analyzed:
#
# A) DISTANCE MODELS
#    - populism_distance_to_afd
#    - gal_tan_distance_to_afd
#
# B) MODELS WITH ABSOLUTE MAINSTREAM VALUES
#    - mean_populism_score
#    - mean_gal_tan_score
#
# Procedure:
# - dynamic linear time-series regressions
# - electoral-term fixed effects
# - HAC/Newey-West standard errors with maxlags = 1
# - control for the lagged dependent variable
# - control for lagged AfD support
#
# All tabular outputs use three decimal places.
# ============================================================

from pathlib import Path
import warnings

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# 1. PATHS AND VARIABLES
# ============================================================

INPUT_FILE = Path(
    "data/processed/final_analysis/06_mainstream_polling_interval_distances.csv"
)

OUTPUT_DIR = Path(
    "results/research_question3_4_crosslagged"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Structural and control variables
INTERVAL_ID = "poll_interval_id"
TIME_ORDER = "poll_date"
NEXT_POLL_DATE = "next_poll_date"
PERIOD = "wahlperiode"
AFD_SUPPORT = "afd_support_t"

# Distance values to the AfD
POSITION_DISTANCE = "gal_tan_distance_to_afd"
RHETORIC_DISTANCE = "populism_distance_to_afd"

# Absolute values of the aggregated mainstream
POSITION_ABSOLUTE = "mean_gal_tan_score"
RHETORIC_ABSOLUTE = "mean_populism_score"

# HAC/Newey-West setting
HAC_MAXLAGS = 1


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def make_numeric(series: pd.Series) -> pd.Series:
    """
    Robustly converts numeric values. Also supports
    German decimal separators and percent signs.
    """
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def check_required_columns(
    data: pd.DataFrame,
    required_columns: list[str],
) -> None:
    """Checks whether all variables required for the analysis are present."""
    missing = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing:
        available = "\n".join(
            f"  - {column}" for column in data.columns
        )
        missing_text = "\n".join(
            f"  - {column}" for column in missing
        )

        raise KeyError(
            "\nThe following required variables are missing:\n"
            f"{missing_text}\n\n"
            "Available columns:\n"
            f"{available}"
        )


def create_lag(
    data: pd.DataFrame,
    source: str,
    target: str,
) -> None:
    """
    Creates a first-order lag.

    Because the dataset contains exactly one aggregated mainstream
    time series, no grouping by party is applied.
    """
    data[target] = data[source].shift(1)


def fit_time_series_model(
    data: pd.DataFrame,
    outcome: str,
    predictor_lag: str,
    outcome_lag: str,
    afd_support_lag: str,
    model_name: str,
):
    """
    Estimates a dynamic model at the aggregated mainstream level:

    Y_t = b0
          + b1 * X_(t-1)
          + b2 * Y_(t-1)
          + b3 * AfD support_(t-1)
          + electoral-term FE
          + error_t

    Inference is based on HAC/Newey-West standard errors.
    """

    formula = (
        f"{outcome} ~ "
        f"{predictor_lag} + "
        f"{outcome_lag} + "
        f"{afd_support_lag} + "
        f"C({PERIOD})"
    )

    model_columns = [
        outcome,
        predictor_lag,
        outcome_lag,
        afd_support_lag,
        PERIOD,
    ]

    model_df = (
        data[model_columns]
        .dropna()
        .copy()
    )

    # Patsy/statsmodels cannot reliably process the pandas-native
    # nullable dtype Int64Dtype as a categorical variable.
    # After dropping missing values, the electoral term is therefore
    # converted to ordinary Python strings. C(wahlperiode) then
    # treats it unambiguously as a categorical fixed-effects variable.
    model_df[PERIOD] = (
        model_df[PERIOD]
        .astype(int)
        .astype(str)
    )

    if model_df.empty:
        raise ValueError(
            f"{model_name}: No observations remain after "
            "listwise deletion."
        )

    if len(model_df) <= len(model_df.columns):
        raise ValueError(
            f"{model_name}: There are too few complete "
            "observations for the model."
        )

    result = smf.ols(
        formula=formula,
        data=model_df,
    ).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": HAC_MAXLAGS},
    )

    return result, model_df, formula


def tidy_coefficient_table(
    result,
    focal_term: str,
    model_name: str,
    research_question: str,
    operationalization: str,
) -> pd.DataFrame:
    """Creates a compact coefficient table."""
    conf = result.conf_int()

    table = pd.DataFrame({
        "research_question": research_question,
        "operationalization": operationalization,
        "model": model_name,
        "term": result.params.index,
        "focal_term": result.params.index == focal_term,
        "b": result.params.values,
        "SE": result.bse.values,
        "t": result.tvalues.values,
        "p": result.pvalues.values,
        "CI_low": conf.iloc[:, 0].values,
        "CI_high": conf.iloc[:, 1].values,
    })

    return table


def extract_focal_test(
    result,
    focal_term: str,
    research_question: str,
    operationalization: str,
    effect: str,
    n_obs: int,
) -> dict:
    """Extracts the central time-lagged effect."""
    ci_low, ci_high = result.conf_int().loc[focal_term]

    return {
        "research_question": research_question,
        "operationalization": operationalization,
        "effect": effect,
        "focal_term": focal_term,
        "N": int(n_obs),
        "b": result.params[focal_term],
        "SE": result.bse[focal_term],
        "t": result.tvalues[focal_term],
        "p_raw": result.pvalues[focal_term],
        "CI_low": ci_low,
        "CI_high": ci_high,
        "R2": result.rsquared,
        "R2_adj": result.rsquared_adj,
    }


def format_model_report(
    title: str,
    result,
    model_df: pd.DataFrame,
    formula: str,
    focal_term: str,
    interpretation_type: str,
) -> str:
    """Creates a detailed report for a single model."""
    b = result.params[focal_term]
    se = result.bse[focal_term]
    t_value = result.tvalues[focal_term]
    p_value = result.pvalues[focal_term]
    ci_low, ci_high = result.conf_int().loc[focal_term]

    lines = [
        "=" * 88,
        title,
        "=" * 88,
        f"Formula: {formula}",
        f"N observations: {int(result.nobs)}",
        f"Number of electoral terms: {model_df[PERIOD].nunique()}",
        f"HAC/Newey-West standard errors: maxlags = {HAC_MAXLAGS}",
        f"R²: {result.rsquared:.3f}",
        f"Adjusted R²: {result.rsquared_adj:.3f}",
        "",
        "Central time-lagged effect:",
        (
            f"b = {b:.3f}, SE = {se:.3f}, t = {t_value:.3f}, "
            f"p = {p_value:.3f}, "
            f"95% CI [{ci_low:.3f}, {ci_high:.3f}]"
        ),
        "",
    ]

    if interpretation_type == "distance":
        lines.extend([
            "Note on interpreting the distance models:",
            (
                "- A positive coefficient means that larger distances "
                "in t-1 are associated with larger distances in t. "
                "Conversely, stronger convergence in t-1 goes along with "
                "stronger convergence in t."
            ),
            (
                "- A negative coefficient means that larger distances "
                "in t-1 are associated with smaller distances in t. "
                "The two forms of convergence then develop in opposite "
                "directions."
            ),
            "",
        ])

    elif interpretation_type == "absolute":
        lines.extend([
            "Note on interpreting the absolute values:",
            (
                "- A positive coefficient means that a higher value "
                "of the explanatory dimension in t-1 is associated with "
                "a higher value of the dependent dimension in t."
            ),
            (
                "- A negative coefficient means that a higher value "
                "of the explanatory dimension in t-1 is associated with "
                "a lower value of the dependent dimension in t."
            ),
            (
                "- The substantive direction of the GAL–TAN score depends "
                "on the coding in the original dataset. Before any "
                "substantive interpretation, it should therefore be checked "
                "whether higher values denote the TAN or the GAL pole."
            ),
            "",
        ])

    lines.extend([
        "Full model summary:",
        result.summary().as_text(),
        "",
    ])

    return "\n".join(lines)


# ============================================================
# 3. LOAD AND PREPARE DATA
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        "The input file was not found:\n"
        f"{INPUT_FILE.resolve()}"
    )

df = pd.read_csv(INPUT_FILE)

required_columns = [
    TIME_ORDER,
    PERIOD,
    AFD_SUPPORT,
    POSITION_DISTANCE,
    RHETORIC_DISTANCE,
    POSITION_ABSOLUTE,
    RHETORIC_ABSOLUTE,
]

check_required_columns(
    data=df,
    required_columns=required_columns,
)

# Clean numeric variables
numeric_columns = [
    PERIOD,
    AFD_SUPPORT,
    POSITION_DISTANCE,
    RHETORIC_DISTANCE,
    POSITION_ABSOLUTE,
    RHETORIC_ABSOLUTE,
]

for column in numeric_columns:
    df[column] = make_numeric(df[column])

# Keep the electoral term numeric for now. The final conversion
# into an ordinary categorical variable takes place after listwise
# deletion inside the model function.
df[PERIOD] = pd.to_numeric(
    df[PERIOD],
    errors="coerce",
)

# Convert date variables
df[TIME_ORDER] = pd.to_datetime(
    df[TIME_ORDER],
    errors="coerce",
)

if NEXT_POLL_DATE in df.columns:
    df[NEXT_POLL_DATE] = pd.to_datetime(
        df[NEXT_POLL_DATE],
        errors="coerce",
    )

if df[TIME_ORDER].isna().all():
    raise ValueError(
        f"The time variable '{TIME_ORDER}' could not be parsed "
        "as a date."
    )

# Sort chronologically
df = (
    df.sort_values(TIME_ORDER)
    .reset_index(drop=True)
)

# Check uniqueness of the aggregated time series
if df[TIME_ORDER].duplicated().any():
    duplicated_dates = (
        df.loc[df[TIME_ORDER].duplicated(keep=False), TIME_ORDER]
        .dt.strftime("%Y-%m-%d")
        .tolist()
    )
    raise ValueError(
        "The aggregated mainstream file contains multiple observations "
        "for at least one poll_date. The following need to be checked:\n"
        + "\n".join(f"  - {date}" for date in duplicated_dates)
    )

# ============================================================
# 4. CREATE LAGGED VARIABLES
# ============================================================

# Shared control variable
create_lag(
    data=df,
    source=AFD_SUPPORT,
    target="afd_support_lag1",
)

# Distance models
create_lag(
    data=df,
    source=RHETORIC_DISTANCE,
    target="rhetoric_distance_lag1",
)
create_lag(
    data=df,
    source=POSITION_DISTANCE,
    target="position_distance_lag1",
)

# Models with absolute values
create_lag(
    data=df,
    source=RHETORIC_ABSOLUTE,
    target="rhetoric_absolute_lag1",
)
create_lag(
    data=df,
    source=POSITION_ABSOLUTE,
    target="position_absolute_lag1",
)

# Diagnostic file for checking the lag assignment
diagnostic_columns = [
    INTERVAL_ID,
    TIME_ORDER,
    NEXT_POLL_DATE,
    PERIOD,
    AFD_SUPPORT,
    "afd_support_lag1",
    RHETORIC_DISTANCE,
    "rhetoric_distance_lag1",
    POSITION_DISTANCE,
    "position_distance_lag1",
    RHETORIC_ABSOLUTE,
    "rhetoric_absolute_lag1",
    POSITION_ABSOLUTE,
    "position_absolute_lag1",
]

diagnostic_columns = [
    column for column in diagnostic_columns
    if column in df.columns
]

df[diagnostic_columns].to_csv(
    OUTPUT_DIR / "rq3_rq4_lag_diagnostics.csv",
    index=False,
    float_format="%.3f",
)


# ============================================================
# 5. MODEL DEFINITIONS
# ============================================================

model_specs = [
    {
        "model_name": "RQ3_Distance",
        "research_question": "RQ3",
        "operationalization": "Distance to the AfD",
        "outcome": POSITION_DISTANCE,
        "predictor_lag": "rhetoric_distance_lag1",
        "outcome_lag": "position_distance_lag1",
        "effect": (
            "rhetorical distance to the AfD t-1 "
            "-> positional distance to the AfD t"
        ),
        "title": (
            "RQ3 – Distance model: Effect of the previous rhetorical "
            "distance on the subsequent positional distance"
        ),
        "interpretation_type": "distance",
    },
    {
        "model_name": "RQ4_Distance",
        "research_question": "RQ4",
        "operationalization": "Distance to the AfD",
        "outcome": RHETORIC_DISTANCE,
        "predictor_lag": "position_distance_lag1",
        "outcome_lag": "rhetoric_distance_lag1",
        "effect": (
            "positional distance to the AfD t-1 "
            "-> rhetorical distance to the AfD t"
        ),
        "title": (
            "RQ4 – Distance model: Effect of the previous positional "
            "distance on the subsequent rhetorical distance"
        ),
        "interpretation_type": "distance",
    },
    {
        "model_name": "RQ3_Absolute",
        "research_question": "RQ3",
        "operationalization": "Absolute mainstream values",
        "outcome": POSITION_ABSOLUTE,
        "predictor_lag": "rhetoric_absolute_lag1",
        "outcome_lag": "position_absolute_lag1",
        "effect": (
            "absolute populism score t-1 "
            "-> absolute GAL–TAN score t"
        ),
        "title": (
            "RQ3 – Absolute model: Effect of the previous absolute "
            "populism score on the subsequent absolute GAL–TAN score"
        ),
        "interpretation_type": "absolute",
    },
    {
        "model_name": "RQ4_Absolute",
        "research_question": "RQ4",
        "operationalization": "Absolute mainstream values",
        "outcome": RHETORIC_ABSOLUTE,
        "predictor_lag": "position_absolute_lag1",
        "outcome_lag": "rhetoric_absolute_lag1",
        "effect": (
            "absolute GAL–TAN score t-1 "
            "-> absolute populism score t"
        ),
        "title": (
            "RQ4 – Absolute model: Effect of the previous absolute "
            "GAL–TAN score on the subsequent absolute populism score"
        ),
        "interpretation_type": "absolute",
    },
]


# ============================================================
# 6. ESTIMATE MODELS
# ============================================================

model_results = {}
coefficient_tables = []
focal_tests = []
model_reports = []

for spec in model_specs:
    result, model_df, formula = fit_time_series_model(
        data=df,
        outcome=spec["outcome"],
        predictor_lag=spec["predictor_lag"],
        outcome_lag=spec["outcome_lag"],
        afd_support_lag="afd_support_lag1",
        model_name=spec["model_name"],
    )

    model_results[spec["model_name"]] = {
        "result": result,
        "data": model_df,
        "formula": formula,
        "spec": spec,
    }

    coefficient_tables.append(
        tidy_coefficient_table(
            result=result,
            focal_term=spec["predictor_lag"],
            model_name=spec["model_name"],
            research_question=spec["research_question"],
            operationalization=spec["operationalization"],
        )
    )

    focal_tests.append(
        extract_focal_test(
            result=result,
            focal_term=spec["predictor_lag"],
            research_question=spec["research_question"],
            operationalization=spec["operationalization"],
            effect=spec["effect"],
            n_obs=result.nobs,
        )
    )

    model_reports.append(
        format_model_report(
            title=spec["title"],
            result=result,
            model_df=model_df,
            formula=formula,
            focal_term=spec["predictor_lag"],
            interpretation_type=spec["interpretation_type"],
        )
    )


# ============================================================
# 7. HOLM CORRECTIONS
# ============================================================

focal_tests_df = pd.DataFrame(focal_tests)

# Holm correction separately within each operationalization:
# RQ3 and RQ4 together in each case.
focal_tests_df["p_Holm_family"] = pd.NA
focal_tests_df["significant_Holm_family_005"] = False

for operationalization, indices in focal_tests_df.groupby(
    "operationalization"
).groups.items():
    p_values = focal_tests_df.loc[indices, "p_raw"].astype(float)

    reject, corrected_p, _, _ = multipletests(
        p_values,
        alpha=0.05,
        method="holm",
    )

    focal_tests_df.loc[indices, "p_Holm_family"] = corrected_p
    focal_tests_df.loc[
        indices,
        "significant_Holm_family_005",
    ] = reject

focal_tests_df["p_Holm_family"] = pd.to_numeric(
    focal_tests_df["p_Holm_family"],
    errors="coerce",
)

# Additional conservative Holm correction across all four tests
reject_all, corrected_all, _, _ = multipletests(
    focal_tests_df["p_raw"].astype(float),
    alpha=0.05,
    method="holm",
)

focal_tests_df["p_Holm_all"] = corrected_all
focal_tests_df["significant_Holm_all_005"] = reject_all


# ============================================================
# 8. SAVE RESULTS
# ============================================================

all_coefficients = pd.concat(
    coefficient_tables,
    ignore_index=True,
)

all_coefficients.to_csv(
    OUTPUT_DIR / "rq3_rq4_all_coefficients.csv",
    index=False,
    float_format="%.3f",
)

focal_tests_df.to_csv(
    OUTPUT_DIR / "rq3_rq4_focal_effects_holm.csv",
    index=False,
    float_format="%.3f",
)

# Additional separate tables
distance_focal = focal_tests_df[
    focal_tests_df["operationalization"] == "Distance to the AfD"
].copy()

absolute_focal = focal_tests_df[
    focal_tests_df["operationalization"] == "Absolute mainstream values"
].copy()

distance_focal.to_csv(
    OUTPUT_DIR / "rq3_rq4_distance_focal_effects.csv",
    index=False,
    float_format="%.3f",
)

absolute_focal.to_csv(
    OUTPUT_DIR / "rq3_rq4_absolute_focal_effects.csv",
    index=False,
    float_format="%.3f",
)

# Full text report
with open(
    OUTPUT_DIR / "rq3_rq4_model_report.txt",
    "w",
    encoding="utf-8",
) as file:
    file.write(
        "\n".join(model_reports)
    )

    file.write("\n")
    file.write("=" * 88 + "\n")
    file.write("HOLM-CORRECTED CENTRAL TESTS\n")
    file.write("=" * 88 + "\n")
    file.write(
        focal_tests_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )
    file.write("\n\n")

    file.write(
        "Explanation of the correction variables:\n"
        "- p_Holm_family corrects RQ3 and RQ4 separately within "
        "the distance and absolute models, respectively.\n"
        "- p_Holm_all corrects conservatively across all four central "
        "tests jointly.\n"
    )


# ============================================================
# 9. CONSOLE OUTPUT
# ============================================================

print("\n" + "=" * 88)
print("RQ3 / RQ4: CROSS-LAGGED MODELS AT THE MAINSTREAM LEVEL")
print("=" * 88)
print(f"Input file: {INPUT_FILE}")
print(f"Rows in the original dataset: {len(df)}")
print(f"Time variable: {TIME_ORDER}")
print(f"Electoral-term FE: {PERIOD}")
print(f"AfD control variable: {AFD_SUPPORT} (lagged)")
print(f"HAC/Newey-West maxlags: {HAC_MAXLAGS}")

print("\nDistance variables used:")
print(f"- Rhetoric: {RHETORIC_DISTANCE}")
print(f"- Position: {POSITION_DISTANCE}")

print("\nAbsolute variables used:")
print(f"- Rhetoric: {RHETORIC_ABSOLUTE}")
print(f"- Position: {POSITION_ABSOLUTE}")

print("\nCentral results:")
print(
    focal_tests_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.3f}",
    )
)

print("\nSaved files:")
for output_path in sorted(OUTPUT_DIR.glob("*")):
    print(f"- {output_path}")