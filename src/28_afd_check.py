# ============================================================
# SUPPLEMENTARY ANALYSIS:
# DOES THE AfD MOVE MORE STRONGLY TOWARD TAN AND COMMUNICATE
# IN A MORE POPULIST WAY AS SUPPORT RISES?
#
# Unit of analysis:
# AfD × polling interval
#
# MODEL A:
# Absolute GAL-TAN position of the AfD
#
# MODEL B:
# Absolute populism of the AfD
#
# For both models:
#   - Electoral-term fixed effects
#   - Newey-West/HAC standard errors
#   - HAC main model with lag 4
#   - Sensitivity analysis with several lag lengths
#   - Predicted values and figures
#
# No models without electoral-term FE are estimated.
# All numeric results are printed and saved
# with three decimal places.
# ============================================================

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# ============================================================
# 1. SETTINGS
# ============================================================

INPUT_FILE = Path(
    "data/processed/final_analysis/"
    "05_party_polling_interval_distances.csv"
)

OUTPUT_DIR = Path(
    "results/additional_afd_position_and_populism"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PARTY = "party_clean"
AFD_SUPPORT = "afd_support_t"
PERIOD = "wahlperiode_cat"
CENTERED_SUPPORT = "afd_support_c"

PRIMARY_HAC_LAG = 4
HAC_LAGS = [0, 1, 2, 3, 4, 5, 6, 8]

ALPHA = 0.05
DECIMALS = 3

TIME_ORDER = "poll_date"

OUTCOMES = {
    "gal_tan": "afd_mean_gal_tan_score",
    "populism": "afd_mean_populism_score",
}

MODEL_SPECS = {
    "gal_tan": {
        "label": "Absolute GAL-TAN position of the AfD",
        "expected_sign": 1,
        "output_prefix": "afd_gal_tan",
        "prediction_column": "predicted_gal_tan",
        "ylabel": "Predicted absolute GAL–TAN position of the AfD",
        "title": (
            "AfD support and absolute GAL–TAN position of the AfD"
        ),
    },
    "populism": {
        "label": "Absolute populism of the AfD",
        "expected_sign": 1,
        "output_prefix": "afd_populism",
        "prediction_column": "predicted_populism",
        "ylabel": "Predicted absolute populism of the AfD",
        "title": (
            "AfD support and populist communication of the AfD"
        ),
    },
}


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def save_dataframe(
    dataframe: pd.DataFrame,
    output_file: Path,
) -> None:
    dataframe.to_csv(
        output_file,
        index=False,
        float_format="%.3f",
    )


def prepare_afd_sample(
    data: pd.DataFrame,
    outcomes: dict,
) -> tuple[pd.DataFrame, str]:
    required = [
        PARTY,
        AFD_SUPPORT,
        PERIOD,
        TIME_ORDER,
        *outcomes.values(),
    ]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            "The following variables are missing:\n"
            + "\n".join(
                f"- {column}"
                for column in missing
            )
        )

    sample = data.copy()
    party_clean = (
        sample[PARTY]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    sample = sample.loc[party_clean.eq("afd")].copy()

    for column in [
        AFD_SUPPORT,
        *outcomes.values(),
    ]:
        sample[column] = pd.to_numeric(
            sample[column],
            errors="coerce",
        )

    sample = sample.dropna(
        subset=required
    ).copy()

    if sample.empty:
        raise ValueError(
            "The AfD analysis sample is empty."
        )

    sample[PERIOD] = (
        sample[PERIOD]
        .astype(str)
        .astype("category")
    )

    sample["_time_order"] = pd.to_datetime(
        sample[TIME_ORDER],
        errors="coerce",
    )

    if sample["_time_order"].isna().any():
        raise ValueError(
            f"The time variable '{TIME_ORDER}' contains values that cannot be parsed."
        )

    # Removes duplicate AfD values from the same interval.
    dedupe_columns = [
        "_time_order",
        PERIOD,
        AFD_SUPPORT,
        *outcomes.values(),
    ]

    sample = (
        sample
        .sort_values("_time_order")
        .drop_duplicates(
            subset=dedupe_columns
        )
        .reset_index(drop=True)
    )

    sample[CENTERED_SUPPORT] = (
        sample[AFD_SUPPORT]
        - sample[AFD_SUPPORT].mean()
    )

    return sample, TIME_ORDER


def fit_hac_model(
    formula: str,
    data: pd.DataFrame,
    lag: int,
):
    model = smf.ols(
        formula=formula,
        data=data,
    )

    result = model.fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags": lag,
            "use_correction": True,
        },
        use_t=True,
    )

    return result


def extract_parameter(
    result,
    parameter: str,
) -> dict:
    ci = result.conf_int(
        alpha=ALPHA
    ).loc[parameter]

    return {
        "coefficient": float(
            result.params.loc[parameter]
        ),
        "hac_se": float(
            result.bse.loc[parameter]
        ),
        "t_value": float(
            result.tvalues.loc[parameter]
        ),
        "p_value": float(
            result.pvalues.loc[parameter]
        ),
        "ci_95_low": float(
            ci.iloc[0]
        ),
        "ci_95_high": float(
            ci.iloc[1]
        ),
    }


def create_predictions(
    sample: pd.DataFrame,
    result,
    prediction_column: str,
) -> pd.DataFrame:
    support_grid = np.linspace(
        sample[AFD_SUPPORT].min(),
        sample[AFD_SUPPORT].max(),
        100,
    )

    reference_period = (
        sample[PERIOD]
        .cat.categories[0]
    )

    prediction_data = pd.DataFrame(
        {
            AFD_SUPPORT: support_grid,
            PERIOD: reference_period,
        }
    )

    prediction_data[CENTERED_SUPPORT] = (
        prediction_data[AFD_SUPPORT]
        - sample[AFD_SUPPORT].mean()
    )

    prediction = result.get_prediction(
        prediction_data
    ).summary_frame(alpha=ALPHA)

    prediction_data[prediction_column] = (
        prediction["mean"]
    )
    prediction_data["ci_95_low"] = (
        prediction["mean_ci_lower"]
    )
    prediction_data["ci_95_high"] = (
        prediction["mean_ci_upper"]
    )

    return prediction_data


def make_plot(
    prediction_data: pd.DataFrame,
    prediction_column: str,
    ylabel: str,
    title: str,
    output_file: Path,
) -> None:
    plt.figure(figsize=(9, 6))

    plt.plot(
        prediction_data[AFD_SUPPORT],
        prediction_data[prediction_column],
    )

    plt.fill_between(
        prediction_data[AFD_SUPPORT],
        prediction_data["ci_95_low"],
        prediction_data["ci_95_high"],
        alpha=0.2,
    )

    plt.xlabel(
        "AfD support in percentage points"
    )
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(
        output_file,
        dpi=300,
    )
    plt.close()


def run_model(
    sample: pd.DataFrame,
    model_key: str,
    outcome: str,
    spec: dict,
) -> dict:
    label = spec["label"]
    prefix = spec["output_prefix"]

    formula = f"""
        {outcome}
        ~ {CENTERED_SUPPORT}
        + C({PERIOD})
    """

    print("\n\n" + "#" * 72)
    print(
        f"STARTING MODEL: {label.upper()}"
    )
    print("#" * 72)

    print(
        f"Analysis units: {len(sample)}"
    )
    print(
        f"Electoral terms: {sample[PERIOD].nunique()}"
    )
    print(
        f"Mean AfD support: "
        f"{sample[AFD_SUPPORT].mean():.3f}"
    )

    print("\nModel specification:")
    print(formula.strip())

    primary_result = fit_hac_model(
        formula=formula,
        data=sample,
        lag=PRIMARY_HAC_LAG,
    )

    primary_stats = extract_parameter(
        result=primary_result,
        parameter=CENTERED_SUPPORT,
    )

    supported = (
        primary_stats["coefficient"]
        * spec["expected_sign"] > 0
        and primary_stats["p_value"] < ALPHA
    )

    print("\n" + "=" * 72)
    print(
        f"HAC main model – {label}"
    )
    print("=" * 72)

    print(
        f"Coefficient:  "
        f"{primary_stats['coefficient']:.3f}"
    )
    print(
        f"HAC SE:       "
        f"{primary_stats['hac_se']:.3f}"
    )
    print(
        f"t-value:      "
        f"{primary_stats['t_value']:.3f}"
    )
    print(
        f"p-value:      "
        f"{primary_stats['p_value']:.3f}"
    )
    print(
        f"95% CI:       "
        f"[{primary_stats['ci_95_low']:.3f}, "
        f"{primary_stats['ci_95_high']:.3f}]"
    )
    print(
        f"Expectation confirmed: "
        f"{'Yes' if supported else 'No'}"
    )

    primary_output = pd.DataFrame(
        [
            {
                "analysis": "hac_primary",
                "model_key": model_key,
                "model_label": label,
                "outcome": outcome,
                "predictor": AFD_SUPPORT,
                "model": (
                    "wahlperiode_fixed_effects"
                ),
                "hac_lag": PRIMARY_HAC_LAG,
                **primary_stats,
                "n_observations": int(
                    primary_result.nobs
                ),
                "n_periods": int(
                    sample[PERIOD].nunique()
                ),
                "r_squared": float(
                    primary_result.rsquared
                ),
                "adjusted_r_squared": float(
                    primary_result.rsquared_adj
                ),
                "afd_support_mean": float(
                    sample[AFD_SUPPORT].mean()
                ),
                "supported": supported,
            }
        ]
    )

    sensitivity_rows = []

    for lag in HAC_LAGS:
        sensitivity_result = fit_hac_model(
            formula=formula,
            data=sample,
            lag=lag,
        )

        stats = extract_parameter(
            result=sensitivity_result,
            parameter=CENTERED_SUPPORT,
        )

        sensitivity_rows.append(
            {
                "model_key": model_key,
                "model_label": label,
                "outcome": outcome,
                "hac_lag": lag,
                **stats,
                "n_observations": int(
                    sensitivity_result.nobs
                ),
            }
        )

    sensitivity = pd.DataFrame(
        sensitivity_rows
    )

    print("\nHAC sensitivity analysis:")
    print(
        sensitivity[
            [
                "hac_lag",
                "coefficient",
                "hac_se",
                "t_value",
                "p_value",
                "ci_95_low",
                "ci_95_high",
            ]
        ]
        .round(DECIMALS)
        .to_string(index=False)
    )

    predictions = create_predictions(
        sample=sample,
        result=primary_result,
        prediction_column=spec[
            "prediction_column"
        ],
    )

    make_plot(
        prediction_data=predictions,
        prediction_column=spec[
            "prediction_column"
        ],
        ylabel=spec["ylabel"],
        title=spec["title"],
        output_file=(
            OUTPUT_DIR
            / f"{prefix}_prediction_plot.png"
        ),
    )

    save_dataframe(
        primary_output,
        OUTPUT_DIR
        / f"{prefix}_primary_model.csv",
    )

    save_dataframe(
        sensitivity,
        OUTPUT_DIR
        / f"{prefix}_hac_sensitivity.csv",
    )

    save_dataframe(
        predictions,
        OUTPUT_DIR
        / f"{prefix}_predicted_values.csv",
    )

    return {
        "primary": primary_output,
        "sensitivity": sensitivity,
    }


# ============================================================
# 3. LOAD DATA
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"File not found:\n"
        f"{INPUT_FILE.resolve()}"
    )

df = pd.read_csv(
    INPUT_FILE
)

outcomes = OUTCOMES

afd_sample, time_variable = prepare_afd_sample(
    data=df,
    outcomes=outcomes,
)

print("=" * 72)
print(
    "SUPPLEMENTARY AfD ANALYSIS"
)
print("=" * 72)
print(
    f"File: {INPUT_FILE}"
)
print(
    f"Rows in the full dataset: {len(df)}"
)
print(
    f"AfD intervals after deduplication: "
    f"{len(afd_sample)}"
)
print(
    f"Time variable: {time_variable}"
)

print("\nOutcomes used:")
for key, outcome in outcomes.items():
    print(
        f"- {MODEL_SPECS[key]['label']}: "
        f"{outcome}"
    )

afd_sample.to_csv(
    OUTPUT_DIR
    / "afd_analysis_sample.csv",
    index=False,
    float_format="%.3f",
)


# ============================================================
# 4. ESTIMATE MODELS
# ============================================================

primary_results = []
sensitivity_results = []

for model_key, spec in MODEL_SPECS.items():
    result = run_model(
        sample=afd_sample,
        model_key=model_key,
        outcome=outcomes[model_key],
        spec=spec,
    )

    primary_results.append(
        result["primary"]
    )
    sensitivity_results.append(
        result["sensitivity"]
    )


# ============================================================
# 5. COMBINED RESULT TABLES
# ============================================================

combined_primary = pd.concat(
    primary_results,
    ignore_index=True,
)

combined_sensitivity = pd.concat(
    sensitivity_results,
    ignore_index=True,
)

save_dataframe(
    combined_primary,
    OUTPUT_DIR
    / "afd_combined_primary_models.csv",
)

save_dataframe(
    combined_sensitivity,
    OUTPUT_DIR
    / "afd_combined_hac_sensitivity.csv",
)


# ============================================================
# 6. FINAL OUTPUT
# ============================================================

print("\n\n" + "=" * 72)
print("ALL AfD ANALYSES COMPLETED")
print("=" * 72)

print("\nHAC main models compared:")
print(
    combined_primary[
        [
            "model_label",
            "outcome",
            "coefficient",
            "hac_se",
            "t_value",
            "p_value",
            "ci_95_low",
            "ci_95_high",
            "n_observations",
            "supported",
        ]
    ]
    .round(DECIMALS)
    .to_string(index=False)
)

print(
    "\nNotes on interpretation:\n"
    "- A positive coefficient in the GAL-TAN model means "
    "that the AfD is positioned more strongly toward "
    "the TAN pole when support is higher.\n"
    "- A positive coefficient in the populism model means "
    "that the AfD communicates in a more populist way when "
    "support is higher.\n"
    "- The models test statistical associations and "
    "do not allow unambiguous causal statements.\n"
    "- All models include electoral-term fixed effects."
)