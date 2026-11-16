# ============================================================
# H1b: MODERATION BY SUPPORT LOSSES
#
# H1b:
# The more support a mainstream party has lost, the stronger
# the positional convergence effect.
#
# The hypothesis is tested with two complementary
# operationalizations:
#
# MODEL A: Positional distance to the AfD
#   DV: gal_tan_distance_to_afd
#   Expectation:
#   Interaction afd_support_c × electoral_loss_c < 0
#
# MODEL B: Absolute GAL-TAN position
#   DV: mean_gal_tan_score
#   Expectation:
#   Interaction afd_support_c × electoral_loss_c > 0
#
# Prerequisite for Model B:
#   Higher GAL-TAN values indicate more traditionalist,
#   authoritarian and nationalist positions.
#
# IMPORTANT:
# party_support_change is coded as:
# current poll value minus previous poll value.
#
# Therefore:
# electoral_loss = -party_support_change
#
# Positive values of electoral_loss indicate losses.
# Negative values indicate gains.
#
# For both models:
#   - Party fixed effects
#   - Electoral-term fixed effects
#   - Cluster-robust standard errors at the party level
#   - Small-sample correction
#   - Wild cluster bootstrap WCR11 and WCR31
#   - Simple-slope analyses
#   - Predicted values and interaction plots
#
# All result values are printed and saved with three
# decimal places.
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from wildboottest.wildboottest import wildboottest


# ============================================================
# 1. SETTINGS
# ============================================================

INPUT_FILE = Path(
    "data/processed/final_analysis/"
    "05_party_polling_interval_distances.csv"
)

OUTPUT_DIR = Path("results/hypothesis1b")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AFD_SUPPORT = "afd_support_t"
SUPPORT_CHANGE = "party_support_change"
PARTY = "party_clean"
PERIOD = "wahlperiode_cat"
MAINSTREAM = "is_mainstream_party"

BOOTSTRAP_REPLICATIONS = 9999
SEED = 12345
ALPHA = 0.05
DECIMALS = 3

CENTERED_SUPPORT = "afd_support_c"
ELECTORAL_LOSS = "electoral_loss"
CENTERED_LOSS = "electoral_loss_c"
INTERACTION_NAME = "afd_x_loss"

MODEL_SPECS = {
    "distance": {
        "label": "Positional distance to the AfD",
        "outcome": "gal_tan_distance_to_afd",
        "expected_direction": "negative",
        "expected_sign": -1,
        "output_prefix": "h1b_distance",
        "prediction_column": "predicted_distance",
        "y_axis_label": "Predicted GAL–TAN distance to the AfD",
        "figure_title": (
            "H1b: Support losses and positional distance to the AfD"
        ),
        "descriptive_variables": [
            "gal_tan_distance_to_afd",
            "mean_gal_tan_score",
            "afd_mean_gal_tan_score",
            "afd_support_t",
            "party_support_change",
        ],
    },
    "absolute": {
        "label": "Absolute GAL-TAN position",
        "outcome": "mean_gal_tan_score",
        "expected_direction": "positive",
        "expected_sign": 1,
        "output_prefix": "h1b_absolute_gal_tan",
        "prediction_column": "predicted_gal_tan",
        "y_axis_label": "Predicted absolute GAL–TAN position",
        "figure_title": (
            "H1b: Support losses and absolute GAL–TAN position"
        ),
        "descriptive_variables": [
            "mean_gal_tan_score",
            "gal_tan_distance_to_afd",
            "afd_mean_gal_tan_score",
            "afd_support_t",
            "party_support_change",
        ],
    },
}


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def mainstream_mask(series: pd.Series) -> pd.Series:
    """
    Create a robust filter for the mainstream variable.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().any():
        return numeric.eq(1)

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "ja"])
    )


def save_bootstrap_result(
    result,
    output_file: Path,
) -> None:
    """
    Save a wildboottest result as CSV with three
    decimal places.
    """
    if isinstance(result, pd.DataFrame):
        result.reset_index().to_csv(
            output_file,
            index=False,
            float_format="%.3f",
        )
    else:
        pd.DataFrame(
            {"result": [str(result)]}
        ).to_csv(
            output_file,
            index=False,
        )


def save_bootstrap_error(
    error: Exception,
    output_file: Path,
) -> None:
    """
    Save the error type and error message of a failed
    bootstrap run.
    """
    pd.DataFrame(
        {
            "error_type": [type(error).__name__],
            "error_message": [str(error)],
        }
    ).to_csv(
        output_file,
        index=False,
    )


def save_result_dataframe(
    dataframe: pd.DataFrame,
    output_file: Path,
) -> None:
    """
    Save numeric result values with three decimal places.
    """
    dataframe.to_csv(
        output_file,
        index=False,
        float_format="%.3f",
    )


def direction_matches(
    coefficient: float,
    expected_sign: int,
) -> bool:
    """
    Check whether a coefficient points in the expected direction.
    """
    if expected_sign == -1:
        return coefficient < 0

    if expected_sign == 1:
        return coefficient > 0

    raise ValueError(
        "expected_sign must be either -1 or 1."
    )


def interpret_interaction(
    coefficient: float,
    p_value: float,
    expected_sign: int,
    model_label: str,
) -> str:
    """
    Create an automatic interpretation of the
    interaction effect.
    """
    matches = direction_matches(
        coefficient=coefficient,
        expected_sign=expected_sign,
    )

    significant = p_value < ALPHA

    if matches and significant:
        return (
            f"{model_label}: The interaction effect points in the "
            "expected direction and is statistically significant."
        )

    if matches and not significant:
        return (
            f"{model_label}: The interaction effect points in the "
            "expected direction but is not statistically significant."
        )

    if not matches and significant:
        return (
            f"{model_label}: The interaction effect is statistically "
            "significant but points in the direction opposite to "
            "the hypothesis."
        )

    return (
        f"{model_label}: The interaction effect does not point in "
        "the expected direction and is not statistically significant."
    )


def prepare_analysis_sample(
    data: pd.DataFrame,
    outcome: str,
) -> pd.DataFrame:
    """
    Create the complete mainstream analysis sample
    for a dependent variable.
    """
    required_columns = [
        outcome,
        AFD_SUPPORT,
        SUPPORT_CHANGE,
        PARTY,
        PERIOD,
        MAINSTREAM,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "The following variables are missing:\n"
            + "\n".join(
                f"- {column}"
                for column in missing_columns
            )
        )

    mask = (
        data[outcome].notna()
        & data[AFD_SUPPORT].notna()
        & data[SUPPORT_CHANGE].notna()
        & data[PARTY].notna()
        & data[PERIOD].notna()
        & data[MAINSTREAM].notna()
        & mainstream_mask(data[MAINSTREAM])
    )

    analysis_df = data.loc[mask].copy()

    for column in [
        outcome,
        AFD_SUPPORT,
        SUPPORT_CHANGE,
    ]:
        analysis_df[column] = pd.to_numeric(
            analysis_df[column],
            errors="coerce",
        )

    analysis_df = analysis_df.dropna(
        subset=[
            outcome,
            AFD_SUPPORT,
            SUPPORT_CHANGE,
            PARTY,
            PERIOD,
        ]
    ).copy()

    if analysis_df.empty:
        raise ValueError(
            f"The analysis sample for '{outcome}' is empty."
        )

    analysis_df[PARTY] = (
        analysis_df[PARTY]
        .astype("category")
    )

    analysis_df[PERIOD] = (
        analysis_df[PERIOD]
        .astype("category")
    )

    analysis_df["cluster_id"] = (
        analysis_df[PARTY]
        .cat.codes
        .astype(np.int64)
    )

    if (analysis_df["cluster_id"] < 0).any():
        raise ValueError(
            "Invalid values occurred while creating "
            "the cluster ID."
        )

    return analysis_df


def create_loss_variables(
    analysis_df: pd.DataFrame,
) -> tuple[pd.DataFrame, float, float, float]:
    """
    Create the loss variable, center the predictors
    and create the interaction term.
    """
    centered_df = analysis_df.copy()

    centered_df[ELECTORAL_LOSS] = (
        -centered_df[SUPPORT_CHANGE]
    )

    afd_support_mean = centered_df[
        AFD_SUPPORT
    ].mean()

    electoral_loss_mean = centered_df[
        ELECTORAL_LOSS
    ].mean()

    electoral_loss_sd = centered_df[
        ELECTORAL_LOSS
    ].std(ddof=1)

    centered_df[CENTERED_SUPPORT] = (
        centered_df[AFD_SUPPORT]
        - afd_support_mean
    )

    centered_df[CENTERED_LOSS] = (
        centered_df[ELECTORAL_LOSS]
        - electoral_loss_mean
    )

    centered_df[INTERACTION_NAME] = (
        centered_df[CENTERED_SUPPORT]
        * centered_df[CENTERED_LOSS]
    )

    return (
        centered_df,
        float(afd_support_mean),
        float(electoral_loss_mean),
        float(electoral_loss_sd),
    )


def extract_parameter_result(
    ols_result,
    cluster_result,
    parameter_name: str,
) -> dict:
    """
    Extract coefficient, cluster-robust standard error,
    t-value, p-value and 95% confidence interval.
    """
    parameter_names = ols_result.model.exog_names

    if parameter_name not in parameter_names:
        raise ValueError(
            f"Parameter '{parameter_name}' was not found "
            "in the model."
        )

    parameter_index = parameter_names.index(
        parameter_name
    )

    confidence_interval = (
        cluster_result.conf_int()[parameter_index]
    )

    return {
        "coefficient": float(
            cluster_result.params[parameter_index]
        ),
        "cluster_robust_se": float(
            cluster_result.bse[parameter_index]
        ),
        "t_value": float(
            cluster_result.tvalues[parameter_index]
        ),
        "p_value": float(
            cluster_result.pvalues[parameter_index]
        ),
        "ci_95_low": float(
            confidence_interval[0]
        ),
        "ci_95_high": float(
            confidence_interval[1]
        ),
    }


def run_wild_cluster_bootstrap(
    model,
    cluster_array: np.ndarray,
    bootstrap_type: str,
    output_file: Path,
    heading: str,
):
    """
    Run a wild cluster bootstrap for the
    interaction effect.
    """
    print("\n" + "=" * 72)
    print(heading)
    print("=" * 72)

    try:
        result = wildboottest(
            model,
            param=INTERACTION_NAME,
            cluster=cluster_array,
            B=BOOTSTRAP_REPLICATIONS,
            bootstrap_type=bootstrap_type,
            impose_null=True,
            seed=SEED,
            show=False,
        )

        if isinstance(result, pd.DataFrame):
            print(
                result.round(DECIMALS)
            )
        else:
            print(result)

        save_bootstrap_result(
            result=result,
            output_file=output_file,
        )

        return result

    except Exception as error:
        print(
            f"The wild cluster bootstrap {bootstrap_type} "
            "could not be computed."
        )
        print(
            f"Error type: {type(error).__name__}"
        )
        print(
            f"Error message: {error}"
        )

        save_bootstrap_error(
            error=error,
            output_file=output_file,
        )

        return None


def calculate_simple_slopes(
    ols_result,
    cluster_result,
    electoral_loss_mean: float,
    electoral_loss_sd: float,
) -> pd.DataFrame:
    """
    Compute the effect of AfD support at low, medium
    and high support loss.
    """
    loss_levels = {
        "Low loss / rather gain": (
            electoral_loss_mean
            - electoral_loss_sd
        ),
        "Medium loss": (
            electoral_loss_mean
        ),
        "High loss": (
            electoral_loss_mean
            + electoral_loss_sd
        ),
    }

    parameter_names = (
        ols_result.model.exog_names
    )

    afd_index = parameter_names.index(
        CENTERED_SUPPORT
    )

    interaction_index = parameter_names.index(
        INTERACTION_NAME
    )

    simple_slope_rows = []

    for label, loss_value in loss_levels.items():
        loss_centered = (
            loss_value
            - electoral_loss_mean
        )

        restriction = np.zeros(
            len(parameter_names)
        )

        restriction[afd_index] = 1
        restriction[interaction_index] = (
            loss_centered
        )

        slope_test = cluster_result.t_test(
            restriction
        )

        slope = float(
            np.asarray(
                slope_test.effect
            ).squeeze()
        )

        slope_se = float(
            np.asarray(
                slope_test.sd
            ).squeeze()
        )

        slope_t = float(
            np.asarray(
                slope_test.tvalue
            ).squeeze()
        )

        slope_p = float(
            np.asarray(
                slope_test.pvalue
            ).squeeze()
        )

        slope_ci = np.asarray(
            slope_test.conf_int(
                alpha=ALPHA
            )
        ).squeeze()

        simple_slope_rows.append(
            {
                "loss_level": label,
                ELECTORAL_LOSS: loss_value,
                "effect_afd_support": slope,
                "cluster_robust_se": slope_se,
                "t_value": slope_t,
                "p_value": slope_p,
                "ci_95_low": float(
                    slope_ci[0]
                ),
                "ci_95_high": float(
                    slope_ci[1]
                ),
                "significant_05": (
                    slope_p < ALPHA
                ),
            }
        )

    return pd.DataFrame(
        simple_slope_rows
    )


def create_predictions(
    analysis_df: pd.DataFrame,
    ols_result,
    afd_support_mean: float,
    electoral_loss_mean: float,
    electoral_loss_sd: float,
    prediction_column: str,
) -> pd.DataFrame:
    """
    Create predicted values for low, medium and high
    support losses.
    """
    loss_levels = {
        "Low loss / rather gain": (
            electoral_loss_mean
            - electoral_loss_sd
        ),
        "Medium loss": (
            electoral_loss_mean
        ),
        "High loss": (
            electoral_loss_mean
            + electoral_loss_sd
        ),
    }

    afd_grid = np.linspace(
        analysis_df[AFD_SUPPORT].min(),
        analysis_df[AFD_SUPPORT].max(),
        100,
    )

    reference_party = (
        analysis_df[PARTY]
        .cat.categories[0]
    )

    reference_period = (
        analysis_df[PERIOD]
        .cat.categories[0]
    )

    prediction_frames = []

    for label, loss_value in loss_levels.items():
        prediction_data = pd.DataFrame(
            {
                AFD_SUPPORT: afd_grid,
                ELECTORAL_LOSS: loss_value,
                PARTY: reference_party,
                PERIOD: reference_period,
            }
        )

        prediction_data[CENTERED_SUPPORT] = (
            prediction_data[AFD_SUPPORT]
            - afd_support_mean
        )

        prediction_data[CENTERED_LOSS] = (
            prediction_data[ELECTORAL_LOSS]
            - electoral_loss_mean
        )

        prediction_data[INTERACTION_NAME] = (
            prediction_data[CENTERED_SUPPORT]
            * prediction_data[CENTERED_LOSS]
        )

        prediction = (
            ols_result
            .get_prediction(prediction_data)
            .summary_frame(alpha=ALPHA)
        )

        prediction_data["loss_level"] = label

        prediction_data[prediction_column] = (
            prediction["mean"]
        )

        prediction_data["ci_95_low"] = (
            prediction["mean_ci_lower"]
        )

        prediction_data["ci_95_high"] = (
            prediction["mean_ci_upper"]
        )

        prediction_frames.append(
            prediction_data
        )

    return pd.concat(
        prediction_frames,
        ignore_index=True,
    )


def create_interaction_plot(
    predictions: pd.DataFrame,
    prediction_column: str,
    y_axis_label: str,
    figure_title: str,
    output_file: Path,
) -> None:
    """
    Save an interaction plot.
    """
    plt.figure(
        figsize=(9, 6)
    )

    for label, group in predictions.groupby(
        "loss_level",
        sort=False,
    ):
        plt.plot(
            group[AFD_SUPPORT],
            group[prediction_column],
            label=label,
        )

    plt.xlabel(
        "AfD support in percentage points"
    )

    plt.ylabel(
        y_axis_label
    )

    plt.title(
        figure_title
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
    )

    plt.close()


def run_model_specification(
    data: pd.DataFrame,
    model_key: str,
    model_spec: dict,
) -> dict:
    """
    Run the complete H1b model for one
    operationalization.
    """
    label = model_spec["label"]
    outcome = model_spec["outcome"]
    expected_direction = (
        model_spec["expected_direction"]
    )
    expected_sign = (
        model_spec["expected_sign"]
    )
    prefix = (
        model_spec["output_prefix"]
    )
    prediction_column = (
        model_spec["prediction_column"]
    )

    output_model = (
        OUTPUT_DIR
        / f"{prefix}_model_results.csv"
    )

    output_bootstrap_11 = (
        OUTPUT_DIR
        / f"{prefix}_bootstrap_wcr11.csv"
    )

    output_bootstrap_31 = (
        OUTPUT_DIR
        / f"{prefix}_bootstrap_wcr31.csv"
    )

    output_simple_slopes = (
        OUTPUT_DIR
        / f"{prefix}_simple_slopes.csv"
    )

    output_predictions = (
        OUTPUT_DIR
        / f"{prefix}_predicted_values.csv"
    )

    output_figure = (
        OUTPUT_DIR
        / f"{prefix}_interaction_plot.png"
    )

    output_sample = (
        OUTPUT_DIR
        / f"{prefix}_analysis_sample.csv"
    )

    analysis_df = prepare_analysis_sample(
        data=data,
        outcome=outcome,
    )

    (
        analysis_df,
        afd_support_mean,
        electoral_loss_mean,
        electoral_loss_sd,
    ) = create_loss_variables(
        analysis_df=analysis_df,
    )

    cluster_array = (
        analysis_df["cluster_id"]
        .to_numpy(dtype=np.int64)
    )

    print("\n\n" + "#" * 72)
    print(
        f"STARTING H1b MODEL: "
        f"{label.upper()}"
    )
    print("#" * 72)

    print(
        f"Units of analysis: "
        f"{len(analysis_df)}"
    )
    print(
        f"Parties/clusters: "
        f"{analysis_df[PARTY].nunique()}"
    )
    print(
        f"Electoral terms: "
        f"{analysis_df[PERIOD].nunique()}"
    )

    print("\nOriginal change variable:")
    print(
        analysis_df[SUPPORT_CHANGE]
        .describe()
        .round(DECIMALS)
    )

    print("\nNewly created loss variable:")
    print(
        analysis_df[ELECTORAL_LOSS]
        .describe()
        .round(DECIMALS)
    )

    print("\nInterpretation of the loss variable:")
    print("- positive value = loss of support")
    print("- zero           = no change")
    print("- negative value = gain in support")

    print("\nShares:")
    print(
        pd.Series(
            {
                "Loss": (
                    analysis_df[ELECTORAL_LOSS] > 0
                ).mean(),
                "Unchanged": (
                    analysis_df[ELECTORAL_LOSS] == 0
                ).mean(),
                "Gain": (
                    analysis_df[ELECTORAL_LOSS] < 0
                ).mean(),
            }
        ).round(DECIMALS)
    )

    print("\nMeans used for centering:")
    print(
        f"AfD support: "
        f"{afd_support_mean:.3f}"
    )
    print(
        f"Support loss: "
        f"{electoral_loss_mean:.3f}"
    )
    print(
        f"SD of support loss: "
        f"{electoral_loss_sd:.3f}"
    )

    descriptive_variables = model_spec["descriptive_variables"]

    print("\nDescriptive values:")
    print(
        analysis_df[descriptive_variables]
        .describe()
        .round(DECIMALS)
    )

    # --------------------------------------------------------
    # Interaction model with party and electoral-term FE
    # --------------------------------------------------------

    formula = f"""
        {outcome}
        ~ {CENTERED_SUPPORT}
        + {CENTERED_LOSS}
        + {INTERACTION_NAME}
        + C({PARTY})
        + C({PERIOD})
    """

    model = smf.ols(
        formula=formula,
        data=analysis_df,
    )

    ols_result = model.fit()

    if len(cluster_array) != int(
        ols_result.nobs
    ):
        raise ValueError(
            "The number of cluster identifiers does not match "
            "the number of model observations."
        )

    cluster_result = (
        ols_result
        .get_robustcov_results(
            cov_type="cluster",
            groups=cluster_array,
            use_correction=True,
            df_correction=True,
            use_t=True,
        )
    )

    interaction_stats = extract_parameter_result(
        ols_result=ols_result,
        cluster_result=cluster_result,
        parameter_name=INTERACTION_NAME,
    )

    supported = (
        direction_matches(
            coefficient=interaction_stats[
                "coefficient"
            ],
            expected_sign=expected_sign,
        )
        and interaction_stats["p_value"] < ALPHA
    )

    interpretation = interpret_interaction(
        coefficient=interaction_stats[
            "coefficient"
        ],
        p_value=interaction_stats[
            "p_value"
        ],
        expected_sign=expected_sign,
        model_label=label,
    )

    print("\nModel specification:")
    print(
        formula.strip()
    )

    print("\n" + "=" * 72)
    print(
        f"H1b: Interaction model – {label}"
    )
    print("=" * 72)

    print(
        "Interaction coefficient: "
        f"{interaction_stats['coefficient']:.3f}"
    )
    print(
        "Cluster-robust SE:       "
        f"{interaction_stats['cluster_robust_se']:.3f}"
    )
    print(
        "t-value:                 "
        f"{interaction_stats['t_value']:.3f}"
    )
    print(
        "p-value:                 "
        f"{interaction_stats['p_value']:.3f}"
    )
    print(
        "95% CI:                  "
        f"[{interaction_stats['ci_95_low']:.3f}, "
        f"{interaction_stats['ci_95_high']:.3f}]"
    )
    print(
        "H1b supported:           "
        f"{'Yes' if supported else 'No'}"
    )

    print("\nInterpretation:")
    print(
        interpretation
    )

    # --------------------------------------------------------
    # Wild cluster bootstrap
    # --------------------------------------------------------

    run_wild_cluster_bootstrap(
        model=model,
        cluster_array=cluster_array,
        bootstrap_type="11",
        output_file=output_bootstrap_11,
        heading=f"H1b: WCR11 – {label}",
    )

    run_wild_cluster_bootstrap(
        model=model,
        cluster_array=cluster_array,
        bootstrap_type="31",
        output_file=output_bootstrap_31,
        heading=f"H1b: WCR31 – {label}",
    )

    # --------------------------------------------------------
    # Simple slopes
    # --------------------------------------------------------

    simple_slopes = calculate_simple_slopes(
        ols_result=ols_result,
        cluster_result=cluster_result,
        electoral_loss_mean=electoral_loss_mean,
        electoral_loss_sd=electoral_loss_sd,
    )

    print("\n" + "=" * 72)
    print(
        f"Simple slopes – {label}"
    )
    print("=" * 72)

    print(
        simple_slopes
        .round(DECIMALS)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Predicted values and plot
    # --------------------------------------------------------

    predictions = create_predictions(
        analysis_df=analysis_df,
        ols_result=ols_result,
        afd_support_mean=afd_support_mean,
        electoral_loss_mean=electoral_loss_mean,
        electoral_loss_sd=electoral_loss_sd,
        prediction_column=prediction_column,
    )

    create_interaction_plot(
        predictions=predictions,
        prediction_column=prediction_column,
        y_axis_label=model_spec[
            "y_axis_label"
        ],
        figure_title=model_spec[
            "figure_title"
        ],
        output_file=output_figure,
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    model_output = pd.DataFrame(
        [
            {
                "hypothesis": "H1b",
                "operationalization": model_key,
                "model_label": label,
                "model": (
                    "party_and_period_fixed_effects"
                ),
                "outcome": outcome,
                "predictor": AFD_SUPPORT,
                "moderator": ELECTORAL_LOSS,
                "interaction": INTERACTION_NAME,
                "expected_direction": (
                    expected_direction
                ),
                **interaction_stats,
                "n_observations": int(
                    ols_result.nobs
                ),
                "n_clusters": int(
                    analysis_df[PARTY].nunique()
                ),
                "n_periods": int(
                    analysis_df[PERIOD].nunique()
                ),
                "r_squared": float(
                    ols_result.rsquared
                ),
                "adjusted_r_squared": float(
                    ols_result.rsquared_adj
                ),
                "afd_support_mean": (
                    afd_support_mean
                ),
                "electoral_loss_mean": (
                    electoral_loss_mean
                ),
                "electoral_loss_sd": (
                    electoral_loss_sd
                ),
                "supported": supported,
                "interpretation": (
                    interpretation
                ),
            }
        ]
    )

    save_result_dataframe(
        dataframe=model_output,
        output_file=output_model,
    )

    save_result_dataframe(
        dataframe=simple_slopes,
        output_file=output_simple_slopes,
    )

    save_result_dataframe(
        dataframe=predictions,
        output_file=output_predictions,
    )

    analysis_df.to_csv(
        output_sample,
        index=False,
        float_format="%.3f",
    )

    print("\nSaved files:")
    print(f"- {output_model}")
    print(f"- {output_bootstrap_11}")
    print(f"- {output_bootstrap_31}")
    print(f"- {output_simple_slopes}")
    print(f"- {output_predictions}")
    print(f"- {output_figure}")
    print(f"- {output_sample}")

    return {
        "main": model_output,
        "simple_slopes": simple_slopes,
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

print("=" * 72)
print("H1b: Moderation by support losses")
print("=" * 72)
print(f"File: {INPUT_FILE}")
print(f"Rows in the full dataset: {len(df)}")
print(f"Columns in the full dataset: {len(df.columns)}")


# ============================================================
# 4. ESTIMATE BOTH H1b OPERATIONALIZATIONS
# ============================================================

all_main_results = []
all_simple_slopes = []

for model_key, model_spec in MODEL_SPECS.items():
    results = run_model_specification(
        data=df,
        model_key=model_key,
        model_spec=model_spec,
    )

    all_main_results.append(
        results["main"]
    )

    slopes_with_model = (
        results["simple_slopes"]
        .copy()
    )

    slopes_with_model.insert(
        0,
        "operationalization",
        model_key,
    )

    slopes_with_model.insert(
        1,
        "model_label",
        model_spec["label"],
    )

    all_simple_slopes.append(
        slopes_with_model
    )


# ============================================================
# 5. COMBINED RESULTS TABLES
# ============================================================

combined_main_results = pd.concat(
    all_main_results,
    ignore_index=True,
)

combined_simple_slopes = pd.concat(
    all_simple_slopes,
    ignore_index=True,
)

OUTPUT_COMBINED_MODELS = (
    OUTPUT_DIR
    / "h1b_combined_models.csv"
)

OUTPUT_COMBINED_SIMPLE_SLOPES = (
    OUTPUT_DIR
    / "h1b_combined_simple_slopes.csv"
)

save_result_dataframe(
    dataframe=combined_main_results,
    output_file=OUTPUT_COMBINED_MODELS,
)

save_result_dataframe(
    dataframe=combined_simple_slopes,
    output_file=OUTPUT_COMBINED_SIMPLE_SLOPES,
)


# ============================================================
# 6. FINAL OUTPUT
# ============================================================

print("\n\n" + "=" * 72)
print("H1b: ALL ANALYSES COMPLETED")
print("=" * 72)

print("\nInteraction models compared:")
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
            "supported",
        ]
    ]
    .round(DECIMALS)
    .to_string(index=False)
)

print("\nCombined results files:")
print(
    f"- Interaction models: "
    f"{OUTPUT_COMBINED_MODELS}"
)
print(
    f"- Simple slopes:      "
    f"{OUTPUT_COMBINED_SIMPLE_SLOPES}"
)

print(
    "\nNote on interpretation:\n"
    "- In the distance model, a negative interaction coefficient "
    "supports H1b.\n"
    "- In the absolute GAL-TAN position model, a positive "
    "interaction coefficient supports H1b, provided that higher "
    "values represent the TAN pole.\n"
    "- Positive values of the electoral_loss variable indicate "
    "support losses; negative values indicate gains."
)