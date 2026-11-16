# =====================================================
# H2: RHETORICAL CONVERGENCE AND ABSOLUTE POPULISM
#
# This script tests H2 in two ways:
#
# MODEL A: Rhetorical convergence toward the AfD
#   DV: populism_distance_to_afd
#   Expectation: negative coefficient
#   Interpretation:
#   As AfD support rises, the absolute populism distance
#   between a mainstream party and the AfD becomes smaller.
#
# MODEL B: Absolute populism of the mainstream parties
#   DV: mean_populism_score
#   Expectation: positive coefficient
#   Interpretation:
#   As AfD support rises, mainstream parties
#   communicate in a more populist way.
#
# For both models:
#   - Party fixed effects
#   - Electoral-term fixed effects
#   - Cluster-robust standard errors at the party level
#   - Small-sample correction
#   - Wild cluster bootstrap WCR11 and WCR31
#
# All result values are printed and saved
# with three decimal places.
# =====================================================

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from wildboottest.wildboottest import wildboottest


# =====================================================
# 1. SETTINGS
# =====================================================

INPUT_FILE = Path(
    "data/processed/final_analysis/05_party_polling_interval_distances.csv"
)

OUTPUT_DIR = Path("results/hypothesis2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTOR = "afd_support_t"
PARTY = "party_clean"
PERIOD = "wahlperiode_cat"
MAINSTREAM = "is_mainstream_party"

BOOTSTRAP_REPLICATIONS = 9999
SEED = 12345
DECIMALS = 3

# Both H2 operationalizations.
MODEL_SPECS = {
    "distance": {
        "label": "Rhetorical convergence",
        "outcome": "populism_distance_to_afd",
        "expected_direction": "negative",
        "expected_sign": -1,
        "output_prefix": "h2_distance",
        "descriptive_variables": [
            "populism_distance_to_afd",
            "afd_support_t",
            "mean_populism_score",
            "afd_mean_populism_score",
            "n_speeches",
            "afd_n_speeches",
        ],
    },
    "absolute": {
        "label": "Absolute populism",
        "outcome": "mean_populism_score",
        "expected_direction": "positive",
        "expected_sign": 1,
        "output_prefix": "h2_absolute_populism",
        "descriptive_variables": [
            "mean_populism_score",
            "afd_support_t",
            "afd_mean_populism_score",
            "populism_distance_to_afd",
            "n_speeches",
            "afd_n_speeches",
        ],
    },
}


# =====================================================
# 2. HELPER FUNCTIONS
# =====================================================

def is_true_like(series):
    """
    Converts various possible representations of a boolean
    variable into a boolean mask.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "ja"])
    )


def save_bootstrap_result(result, output_path):
    """
    Saves the wildboottest result as a CSV file.
    """
    if isinstance(result, pd.DataFrame):
        result.reset_index().to_csv(
            output_path,
            index=False,
            float_format="%.3f",
        )
    else:
        pd.DataFrame(
            {"result": [str(result)]}
        ).to_csv(output_path, index=False)


def save_bootstrap_error(error, output_path):
    """
    Saves the error type and error message of a bootstrap run.
    """
    pd.DataFrame(
        {
            "error_type": [type(error).__name__],
            "error_message": [str(error)],
        }
    ).to_csv(output_path, index=False)


def extract_cluster_result(
    fitted_ols,
    fitted_cluster,
    parameter,
):
    """
    Extracts the coefficient, cluster-robust standard error,
    t-value, p-value and 95% confidence interval.
    """
    parameter_names = fitted_ols.model.exog_names

    if parameter not in parameter_names:
        raise ValueError(
            f"The parameter '{parameter}' was not found "
            "in the model matrix."
        )

    parameter_index = parameter_names.index(parameter)
    confidence_interval = fitted_cluster.conf_int()[parameter_index]

    return {
        "coefficient": float(
            fitted_cluster.params[parameter_index]
        ),
        "cluster_robust_se": float(
            fitted_cluster.bse[parameter_index]
        ),
        "t_value": float(
            fitted_cluster.tvalues[parameter_index]
        ),
        "p_value": float(
            fitted_cluster.pvalues[parameter_index]
        ),
        "ci_95_low": float(confidence_interval[0]),
        "ci_95_high": float(confidence_interval[1]),
    }


def interpret_result(
    coefficient,
    p_value,
    expected_sign,
    model_label,
):
    """
    Creates an automatic, direction-based interpretation.
    """
    direction_matches = (
        coefficient < 0
        if expected_sign == -1
        else coefficient > 0
    )

    if direction_matches and p_value < 0.05:
        return (
            f"{model_label}: The coefficient points in the expected "
            "direction and is statistically significant."
        )

    if direction_matches:
        return (
            f"{model_label}: The coefficient points in the expected "
            "direction but is not statistically significant."
        )

    if p_value < 0.05:
        return (
            f"{model_label}: The coefficient is statistically significant "
            "but points in the direction opposite to the hypothesis."
        )

    return (
        f"{model_label}: The coefficient does not point in the expected "
        "direction and is not statistically significant."
    )


def run_wild_cluster_bootstrap(
    model,
    parameter,
    cluster_array,
    bootstrap_type,
    output_path,
    heading,
):
    """
    Runs a wild cluster bootstrap and saves the result.
    """
    print("\n" + "=" * 70)
    print(heading)
    print("=" * 70)

    try:
        result = wildboottest(
            model,
            param=parameter,
            cluster=cluster_array,
            B=BOOTSTRAP_REPLICATIONS,
            bootstrap_type=bootstrap_type,
            impose_null=True,
            seed=SEED,
            show=False,
        )

        if isinstance(result, pd.DataFrame):
            print(result.round(DECIMALS))
        else:
            print(result)

        save_bootstrap_result(result, output_path)
        return result

    except Exception as error:
        print(
            f"The wild cluster bootstrap {bootstrap_type} "
            "could not be computed."
        )
        print(f"Error type: {type(error).__name__}")
        print(f"Error message: {error}")

        save_bootstrap_error(error, output_path)
        return None


def prepare_analysis_sample(
    data,
    outcome,
):
    """
    Creates the complete mainstream analysis sample
    for a dependent variable.
    """
    required_columns = [
        outcome,
        PREDICTOR,
        PARTY,
        PERIOD,
        MAINSTREAM,
        "poll_interval_id",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "The following required variables are missing from the dataset:\n"
            + "\n".join(
                f"- {column}"
                for column in missing_columns
            )
        )

    analysis_filter = (
        data[outcome].notna()
        & data[PREDICTOR].notna()
        & data[PARTY].notna()
        & data[PERIOD].notna()
    )

    analysis_filter &= is_true_like(data[MAINSTREAM])

    analysis_df = data.loc[analysis_filter].copy()

    if analysis_df.empty:
        raise ValueError(
            f"After applying the filters, the sample "
            f"for '{outcome}' contains no observations."
        )

    analysis_df[PARTY] = analysis_df[PARTY].astype("category")
    analysis_df[PERIOD] = analysis_df[PERIOD].astype("category")

    analysis_df["cluster_id"] = (
        analysis_df[PARTY]
        .cat.codes
        .astype(np.int64)
    )

    if (analysis_df["cluster_id"] < 0).any():
        raise ValueError(
            "Invalid values were created while generating "
            "the numeric cluster ID."
        )

    return analysis_df


def check_panel_structure(
    analysis_df,
    model_label,
):
    """
    Prints key information on the panel structure and checks
    for duplicate party-interval combinations.
    """
    print("\n" + "=" * 70)
    print(f"H2: Analysis sample – {model_label}")
    print("=" * 70)

    print(f"Analysis units: {len(analysis_df)}")
    print(
        "Number of parties/clusters: "
        f"{analysis_df[PARTY].nunique()}"
    )
    print(
        "Number of electoral terms: "
        f"{analysis_df[PERIOD].nunique()}"
    )

    print(
        "Number of polling intervals: "
        f"{analysis_df['poll_interval_id'].nunique()}"
    )

    print("\nObservations per party:")
    print(
        analysis_df.groupby(
            PARTY,
            observed=True,
        ).size()
    )

    print("\nObservations per electoral term:")
    print(
        analysis_df.groupby(
            PERIOD,
            observed=True,
        ).size()
    )

    print("\nMapping of numeric cluster IDs:")
    print(
        analysis_df[
            [PARTY, "cluster_id"]
        ]
        .drop_duplicates()
        .sort_values("cluster_id")
        .to_string(index=False)
    )

    duplicate_rows = analysis_df.duplicated(
        subset=[PARTY, "poll_interval_id"],
        keep=False,
    )

    n_duplicate_rows = int(duplicate_rows.sum())

    print(
        "\nDuplicate party-interval rows: "
        f"{n_duplicate_rows}"
    )

    if n_duplicate_rows > 0:
        duplicate_examples = (
            analysis_df.loc[
                duplicate_rows,
                [PARTY, "poll_interval_id"],
            ]
            .sort_values([PARTY, "poll_interval_id"])
        )

        print("\nExamples of duplicate combinations:")
        print(
            duplicate_examples
            .head(20)
            .to_string(index=False)
        )

        raise ValueError(
            "The analysis file contains duplicate combinations "
            "of party and polling interval."
        )


def fit_cluster_model(
    analysis_df,
    formula,
):
    """
    Estimates an OLS model with dummy variables for the fixed effects
    and computes cluster-robust inference at the party level.
    """
    model = smf.ols(
        formula=formula,
        data=analysis_df,
    )

    fitted_ols = model.fit()

    cluster_array = analysis_df["cluster_id"].to_numpy(
        dtype=np.int64
    )

    n_model_rows = model.exog.shape[0]
    n_cluster_rows = len(cluster_array)

    print("\nRows in the analysis dataset:", len(analysis_df))
    print("Rows in the model matrix:", n_model_rows)
    print("Rows in the cluster array:", n_cluster_rows)

    if not (
        len(analysis_df)
        == n_model_rows
        == n_cluster_rows
    ):
        raise ValueError(
            "The number of model rows does not match the number "
            "of cluster identifiers."
        )

    fitted_cluster = (
        fitted_ols.get_robustcov_results(
            cov_type="cluster",
            groups=cluster_array,
            use_correction=True,
            df_correction=True,
            use_t=True,
        )
    )

    return (
        model,
        fitted_ols,
        fitted_cluster,
        cluster_array,
    )


def run_model_specification(
    data,
    model_key,
    model_spec,
):
    """
    Runs the main model and bootstrap tests in full for one
    H2 operationalization.
    """
    label = model_spec["label"]
    outcome = model_spec["outcome"]
    expected_direction = model_spec["expected_direction"]
    expected_sign = model_spec["expected_sign"]
    prefix = model_spec["output_prefix"]

    output_analysis_sample = (
        OUTPUT_DIR / f"{prefix}_analysis_sample.csv"
    )
    output_model_results = (
        OUTPUT_DIR / f"{prefix}_model_results.csv"
    )
    output_bootstrap_11 = (
        OUTPUT_DIR / f"{prefix}_wild_cluster_bootstrap_11.csv"
    )
    output_bootstrap_31 = (
        OUTPUT_DIR / f"{prefix}_wild_cluster_bootstrap_31.csv"
    )

    analysis_df = prepare_analysis_sample(
        data=data,
        outcome=outcome,
    )

    check_panel_structure(
        analysis_df=analysis_df,
        model_label=label,
    )

    print("\n" + "=" * 70)
    print(f"H2: Descriptive statistics – {label}")
    print("=" * 70)

    descriptive_variables = model_spec["descriptive_variables"]

    print(
        analysis_df[
            descriptive_variables
        ]
        .describe()
        .round(3)
    )

    # -------------------------------------------------
    # Main model with party and electoral-term FE
    # -------------------------------------------------

    formula_main = f"""
        {outcome}
        ~ {PREDICTOR}
        + C({PARTY})
        + C({PERIOD})
    """

    print("\n" + "=" * 70)
    print(f"H2: Main model – {label}")
    print("=" * 70)
    print(formula_main.strip())

    (
        model_main,
        fitted_main_ols,
        fitted_main_cluster,
        cluster_array,
    ) = fit_cluster_model(
        analysis_df=analysis_df,
        formula=formula_main,
    )

    estimates_main = extract_cluster_result(
        fitted_ols=fitted_main_ols,
        fitted_cluster=fitted_main_cluster,
        parameter=PREDICTOR,
    )

    interpretation_main = interpret_result(
        coefficient=estimates_main["coefficient"],
        p_value=estimates_main["p_value"],
        expected_sign=expected_sign,
        model_label=label,
    )

    n_observations = int(fitted_main_ols.nobs)
    n_clusters = int(analysis_df[PARTY].nunique())

    print(f"Observations:            {n_observations}")
    print(f"Parties/clusters:        {n_clusters}")
    print(
        "Coefficient:             "
        f"{estimates_main['coefficient']:.3f}"
    )
    print(
        "Cluster-robust SE:       "
        f"{estimates_main['cluster_robust_se']:.3f}"
    )
    print(
        "t-value:                 "
        f"{estimates_main['t_value']:.3f}"
    )
    print(
        "p-value:                 "
        f"{estimates_main['p_value']:.3f}"
    )
    print(
        "95% confidence interval: "
        f"[{estimates_main['ci_95_low']:.3f}, "
        f"{estimates_main['ci_95_high']:.3f}]"
    )

    print("\nInterpretation:")
    print(interpretation_main)

    main_result = pd.DataFrame(
        [
            {
                "hypothesis": "H2",
                "operationalization": model_key,
                "model_label": label,
                "model": "party_and_period_fixed_effects",
                "outcome": outcome,
                "predictor": PREDICTOR,
                "expected_direction": expected_direction,
                **estimates_main,
                "n_observations": n_observations,
                "n_clusters": n_clusters,
                "r_squared": float(fitted_main_ols.rsquared),
                "adjusted_r_squared": float(
                    fitted_main_ols.rsquared_adj
                ),
                "interpretation": interpretation_main,
            }
        ]
    )

    main_result.to_csv(
        output_model_results,
        index=False,
        float_format="%.3f",
    )

    analysis_df.to_csv(
        output_analysis_sample,
        index=False,
    )

    run_wild_cluster_bootstrap(
        model=model_main,
        parameter=PREDICTOR,
        cluster_array=cluster_array,
        bootstrap_type="11",
        output_path=output_bootstrap_11,
        heading=f"H2: WCR11 – {label}",
    )

    run_wild_cluster_bootstrap(
        model=model_main,
        parameter=PREDICTOR,
        cluster_array=cluster_array,
        bootstrap_type="31",
        output_path=output_bootstrap_31,
        heading=f"H2: WCR31 – {label}",
    )

    print("\nSaved files for this model:")
    print(f"- Analysis sample: {output_analysis_sample}")
    print(f"- Main model:      {output_model_results}")
    print(f"- Bootstrap WCR11: {output_bootstrap_11}")
    print(f"- Bootstrap WCR31: {output_bootstrap_31}")

    return main_result



# =====================================================
# 3. LOAD DATA
# =====================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"The input file was not found:\n"
        f"{INPUT_FILE.resolve()}"
    )

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("H2: Data loaded")
print("=" * 70)
print(f"File: {INPUT_FILE}")
print(f"Rows in the full dataset: {len(df)}")
print(f"Columns in the full dataset: {len(df.columns)}")


# =====================================================
# 4. ESTIMATE BOTH H2 OPERATIONALIZATIONS
# =====================================================

all_main_results = []

for model_key, model_spec in MODEL_SPECS.items():
    print("\n\n" + "#" * 70)
    print(
        f"STARTING H2 MODEL: "
        f"{model_spec['label'].upper()}"
    )
    print("#" * 70)

    model_results = run_model_specification(
        data=df,
        model_key=model_key,
        model_spec=model_spec,
    )

    all_main_results.append(model_results)


# =====================================================
# 5. SAVE COMBINED RESULTS TABLE
# =====================================================

combined_main_results = pd.concat(
    all_main_results,
    ignore_index=True,
)


output_combined_main = (
    OUTPUT_DIR / "h2_combined_main_models.csv"
)

combined_main_results.to_csv(
    output_combined_main,
    index=False,
    float_format="%.3f",
)



# =====================================================
# 6. FINAL OUTPUT
# =====================================================

print("\n\n" + "=" * 70)
print("H2: ALL ANALYSES COMPLETED")
print("=" * 70)

print("\nMain models compared:")
print(
    combined_main_results[
        [
            "model_label",
            "outcome",
            "expected_direction",
            "coefficient",
            "cluster_robust_se",
            "t_value",
            "p_value",
            "ci_95_low",
            "ci_95_high",
            "n_observations",
            "n_clusters",
        ]
    ]
    .round(DECIMALS)
    .to_string(index=False)
)

print("\nCombined results file:")
print(f"- Main models:       {output_combined_main}")

print(
    "\nNote on interpretation:\n"
    "- In the distance model, a negative coefficient supports H2.\n"
    "- In the absolute populism model, a positive coefficient supports "
    "H2.\n"
    "- The two models answer different but complementary questions."
)