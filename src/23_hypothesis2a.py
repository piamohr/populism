# ============================================================
# H2a: MODERATION BY IDEOLOGICAL PROXIMITY TO THE AfD
#
# H2a:
# The closer a mainstream party is to the AfD, the stronger
# the rhetorical convergence effect.
#
# MODEL A: Rhetorical distance to the AfD
#   DV: populism_distance_to_afd
#   Expectation: afd_support_c × proximity_c < 0
#
# MODEL B: Absolute populism
#   DV: mean_populism_score
#   Expectation: afd_support_c × proximity_c > 0
#
# # For both models:
#   - Party fixed effects
#   - Electoral-term fixed effects
#   - Cluster-robust standard errors at the party level
#   - Small-sample correction
#   - Wild cluster bootstrap WCR11 and WCR31
#   - Simple slope analyses
#   - Predictions and interaction plots
#
# The time-invariant main effect of proximity is absorbed by the
# party fixed effects and is not estimated separately.
# All result values are printed with three decimal places.
# ============================================================

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from wildboottest.wildboottest import wildboottest


# ============================================================
# 1. SETTINGS
# ============================================================

INPUT_FILE = Path(
    "data/processed/final_analysis/"
    "05_party_polling_interval_distances.csv"
)

OUTPUT_DIR = Path("results/hypothesis2a")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AFD_SUPPORT = "afd_support_t"
MODERATOR = "afd_proximity_ordinal"
PARTY = "party_clean"
PERIOD = "wahlperiode_cat"
MAINSTREAM = "is_mainstream_party"

CENTERED_SUPPORT = "afd_support_c"
CENTERED_PROXIMITY = "proximity_c"
INTERACTION = "afd_x_proximity"

BOOTSTRAP_REPLICATIONS = 9999
SEED = 12345
ALPHA = 0.05
DECIMALS = 3

MODEL_SPECS = {
    "distance": {
        "label": "Rhetorical distance to the AfD",
        "outcome": "populism_distance_to_afd",
        "expected_direction": "negative",
        "expected_sign": -1,
        "output_prefix": "h2a_distance",
        "prediction_column": "predicted_distance",
        "y_axis_label": "Predicted rhetorical distance to the AfD",
        "figure_title": (
            "H2a: Ideological proximity and rhetorical distance to the AfD"
        ),
    },
    "absolute": {
        "label": "Absolute populism",
        "outcome": "mean_populism_score",
        "expected_direction": "positive",
        "expected_sign": 1,
        "output_prefix": "h2a_absolute_populism",
        "prediction_column": "predicted_populism",
        "y_axis_label": "Predicted absolute populism",
        "figure_title": (
            "H2a: Ideological proximity and absolute populism"
        ),
    },
}


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def mainstream_mask(series: pd.Series) -> pd.Series:
    """Creates a robust filter for the mainstream variable."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.eq(1)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "ja"])
    )


def save_dataframe(dataframe: pd.DataFrame, output_file: Path) -> None:
    """Saves numeric values with three decimal places."""
    dataframe.to_csv(
        output_file,
        index=False,
        float_format="%.3f",
    )


def save_bootstrap_result(result, output_file: Path) -> None:
    """Saves wild cluster bootstrap results."""
    if isinstance(result, pd.DataFrame):
        result.reset_index().to_csv(
            output_file,
            index=False,
            float_format="%.3f",
        )
    else:
        pd.DataFrame({"result": [str(result)]}).to_csv(
            output_file,
            index=False,
        )


def save_bootstrap_error(error: Exception, output_file: Path) -> None:
    """Saves the error type and error message."""
    pd.DataFrame(
        {
            "error_type": [type(error).__name__],
            "error_message": [str(error)],
        }
    ).to_csv(output_file, index=False)


def direction_matches(coefficient: float, expected_sign: int) -> bool:
    """Checks whether the interaction effect is as expected."""
    if expected_sign == -1:
        return coefficient < 0
    if expected_sign == 1:
        return coefficient > 0
    raise ValueError("expected_sign must be -1 or 1.")


def interpret_interaction(
    coefficient: float,
    p_value: float,
    expected_sign: int,
    label: str,
) -> str:
    """Creates a direction-based interpretation."""
    matches = direction_matches(coefficient, expected_sign)
    significant = p_value < ALPHA

    if matches and significant:
        return (
            f"{label}: The interaction effect points in the expected "
            "direction and is statistically significant."
        )
    if matches:
        return (
            f"{label}: The interaction effect points in the expected "
            "direction but is not statistically significant."
        )
    if significant:
        return (
            f"{label}: The interaction effect is statistically "
            "significant but points in the opposite direction."
        )
    return (
        f"{label}: The interaction effect does not point in the expected "
        "direction and is not statistically significant."
    )


def prepare_analysis_sample(data: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Creates the mainstream analysis sample."""
    required_columns = [
        outcome,
        AFD_SUPPORT,
        MODERATOR,
        PARTY,
        PERIOD,
        MAINSTREAM,
    ]

    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(
            "The following variables are missing:\n"
            + "\n".join(f"- {column}" for column in missing_columns)
        )

    mask = (
        data[outcome].notna()
        & data[AFD_SUPPORT].notna()
        & data[MODERATOR].notna()
        & data[PARTY].notna()
        & data[PERIOD].notna()
    )

    mask &= mainstream_mask(data[MAINSTREAM])

    analysis_df = data.loc[mask].copy()

    for column in [outcome, AFD_SUPPORT, MODERATOR]:
        analysis_df[column] = pd.to_numeric(
            analysis_df[column],
            errors="coerce",
        )

    analysis_df = analysis_df.dropna(
        subset=[outcome, AFD_SUPPORT, MODERATOR, PARTY, PERIOD]
    ).copy()

    if analysis_df.empty:
        raise ValueError(f"The analysis sample for '{outcome}' is empty.")

    analysis_df[PARTY] = analysis_df[PARTY].astype("category")
    analysis_df[PERIOD] = analysis_df[PERIOD].astype("category")
    analysis_df["cluster_id"] = (
        analysis_df[PARTY].cat.codes.astype(np.int64)
    )

    if (analysis_df["cluster_id"] < 0).any():
        raise ValueError("Invalid cluster IDs were created.")

    return analysis_df


def center_variables(
    analysis_df: pd.DataFrame,
) -> tuple[pd.DataFrame, float, float]:
    """Centers predictor and moderator and creates the interaction."""
    analysis_df = analysis_df.copy()
    afd_support_mean = float(analysis_df[AFD_SUPPORT].mean())
    proximity_mean = float(analysis_df[MODERATOR].mean())

    analysis_df[CENTERED_SUPPORT] = (
        analysis_df[AFD_SUPPORT] - afd_support_mean
    )
    analysis_df[CENTERED_PROXIMITY] = (
        analysis_df[MODERATOR] - proximity_mean
    )
    analysis_df[INTERACTION] = (
        analysis_df[CENTERED_SUPPORT]
        * analysis_df[CENTERED_PROXIMITY]
    )

    return analysis_df, afd_support_mean, proximity_mean


def extract_parameter_result(
    ols_result,
    cluster_result,
    parameter_name: str,
) -> dict:
    """Extracts the model parameter of interest."""
    parameter_names = ols_result.model.exog_names
    if parameter_name not in parameter_names:
        raise ValueError(f"Parameter '{parameter_name}' is missing from the model.")

    parameter_index = parameter_names.index(parameter_name)
    ci = cluster_result.conf_int()[parameter_index]

    return {
        "coefficient": float(cluster_result.params[parameter_index]),
        "cluster_robust_se": float(cluster_result.bse[parameter_index]),
        "t_value": float(cluster_result.tvalues[parameter_index]),
        "p_value": float(cluster_result.pvalues[parameter_index]),
        "ci_95_low": float(ci[0]),
        "ci_95_high": float(ci[1]),
    }


def run_wild_cluster_bootstrap(
    model,
    cluster_array: np.ndarray,
    bootstrap_type: str,
    output_file: Path,
    heading: str,
):
    """Runs the wild cluster bootstrap for the interaction."""
    print("\n" + "=" * 72)
    print(heading)
    print("=" * 72)

    try:
        result = wildboottest(
            model,
            param=INTERACTION,
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

        save_bootstrap_result(result, output_file)
        return result

    except Exception as error:
        print(
            f"The wild cluster bootstrap {bootstrap_type} "
            "could not be computed."
        )
        print(f"Error type: {type(error).__name__}")
        print(f"Error message: {error}")
        save_bootstrap_error(error, output_file)
        return None


def calculate_simple_slopes(
    analysis_df: pd.DataFrame,
    ols_result,
    cluster_result,
    proximity_mean: float,
) -> pd.DataFrame:
    """Calculates the AfD support effect for each proximity value."""
    parameter_names = ols_result.model.exog_names
    afd_index = parameter_names.index(CENTERED_SUPPORT)
    interaction_index = parameter_names.index(INTERACTION)

    rows = []
    for proximity_value in sorted(analysis_df[MODERATOR].unique()):
        proximity_centered = proximity_value - proximity_mean

        restriction = np.zeros(len(parameter_names))
        restriction[afd_index] = 1
        restriction[interaction_index] = proximity_centered

        slope_test = cluster_result.t_test(restriction)
        slope_ci = np.asarray(
            slope_test.conf_int(alpha=ALPHA)
        ).squeeze()

        slope = float(np.asarray(slope_test.effect).squeeze())
        slope_se = float(np.asarray(slope_test.sd).squeeze())
        slope_t = float(np.asarray(slope_test.tvalue).squeeze())
        slope_p = float(np.asarray(slope_test.pvalue).squeeze())

        rows.append(
            {
                MODERATOR: float(proximity_value),
                "effect_afd_support": slope,
                "cluster_robust_se": slope_se,
                "t_value": slope_t,
                "p_value": slope_p,
                "ci_95_low": float(slope_ci[0]),
                "ci_95_high": float(slope_ci[1]),
                "significant_05": slope_p < ALPHA,
            }
        )

    return pd.DataFrame(rows)


def create_predictions(
    analysis_df: pd.DataFrame,
    ols_result,
    afd_support_mean: float,
    proximity_mean: float,
    prediction_column: str,
) -> pd.DataFrame:
    """Creates predictions for all observed proximity values."""
    afd_grid = np.linspace(
        analysis_df[AFD_SUPPORT].min(),
        analysis_df[AFD_SUPPORT].max(),
        100,
    )

    reference_party = analysis_df[PARTY].cat.categories[0]
    reference_period = analysis_df[PERIOD].cat.categories[0]
    frames = []

    for proximity_value in sorted(analysis_df[MODERATOR].unique()):
        prediction_data = pd.DataFrame(
            {
                AFD_SUPPORT: afd_grid,
                MODERATOR: proximity_value,
                PARTY: reference_party,
                PERIOD: reference_period,
            }
        )

        prediction_data[CENTERED_SUPPORT] = (
            prediction_data[AFD_SUPPORT] - afd_support_mean
        )
        prediction_data[CENTERED_PROXIMITY] = (
            prediction_data[MODERATOR] - proximity_mean
        )
        prediction_data[INTERACTION] = (
            prediction_data[CENTERED_SUPPORT]
            * prediction_data[CENTERED_PROXIMITY]
        )

        prediction = ols_result.get_prediction(
            prediction_data
        ).summary_frame(alpha=ALPHA)

        prediction_data[prediction_column] = prediction["mean"]
        prediction_data["ci_95_low"] = prediction["mean_ci_lower"]
        prediction_data["ci_95_high"] = prediction["mean_ci_upper"]
        frames.append(prediction_data)

    return pd.concat(frames, ignore_index=True)


def create_interaction_plot(
    predictions: pd.DataFrame,
    prediction_column: str,
    y_axis_label: str,
    figure_title: str,
    output_file: Path,
) -> None:
    """Saves the interaction plot."""
    plt.figure(figsize=(9, 6))

    for proximity_value, group in predictions.groupby(MODERATOR):
        plt.plot(
            group[AFD_SUPPORT],
            group[prediction_column],
            label=f"AfD proximity = {int(proximity_value)}",
        )

    plt.xlabel("AfD support in percentage points")
    plt.ylabel(y_axis_label)
    plt.title(figure_title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def run_model_specification(
    data: pd.DataFrame,
    model_key: str,
    model_spec: dict,
) -> dict:
    """Runs one complete H2a operationalization."""
    label = model_spec["label"]
    outcome = model_spec["outcome"]
    expected_direction = model_spec["expected_direction"]
    expected_sign = model_spec["expected_sign"]
    prefix = model_spec["output_prefix"]
    prediction_column = model_spec["prediction_column"]

    output_model = OUTPUT_DIR / f"{prefix}_model_results.csv"
    output_bootstrap_11 = OUTPUT_DIR / f"{prefix}_bootstrap_wcr11.csv"
    output_bootstrap_31 = OUTPUT_DIR / f"{prefix}_bootstrap_wcr31.csv"
    output_simple_slopes = OUTPUT_DIR / f"{prefix}_simple_slopes.csv"
    output_predictions = OUTPUT_DIR / f"{prefix}_predicted_values.csv"
    output_figure = OUTPUT_DIR / f"{prefix}_interaction_plot.png"
    output_sample = OUTPUT_DIR / f"{prefix}_analysis_sample.csv"

    analysis_df = prepare_analysis_sample(data, outcome)
    analysis_df, afd_support_mean, proximity_mean = center_variables(
        analysis_df
    )
    cluster_array = analysis_df["cluster_id"].to_numpy(dtype=np.int64)

    print("\n\n" + "#" * 72)
    print(f"STARTING H2a MODEL: {label.upper()}")
    print("#" * 72)
    print(f"Analysis units: {len(analysis_df)}")
    print(f"Parties/clusters: {analysis_df[PARTY].nunique()}")
    print(f"Electoral terms: {analysis_df[PERIOD].nunique()}")

    print("\nProximity values by party:")
    print(
        analysis_df[[PARTY, MODERATOR]]
        .drop_duplicates()
        .sort_values(MODERATOR)
        .to_string(index=False)
    )

    print("\nMeans used for centering:")
    print(f"AfD support:   {afd_support_mean:.3f}")
    print(f"AfD proximity: {proximity_mean:.3f}")

    formula = f"""
        {outcome}
        ~ {CENTERED_SUPPORT}
        + {INTERACTION}
        + C({PARTY})
        + C({PERIOD})
    """

    model = smf.ols(formula=formula, data=analysis_df)
    ols_result = model.fit()

    if len(cluster_array) != int(ols_result.nobs):
        raise ValueError(
            "Cluster array and model observations do not match."
        )

    cluster_result = ols_result.get_robustcov_results(
        cov_type="cluster",
        groups=cluster_array,
        use_correction=True,
        df_correction=True,
        use_t=True,
    )

    stats = extract_parameter_result(
        ols_result,
        cluster_result,
        INTERACTION,
    )

    supported = (
        direction_matches(stats["coefficient"], expected_sign)
        and stats["p_value"] < ALPHA
    )

    interpretation = interpret_interaction(
        stats["coefficient"],
        stats["p_value"],
        expected_sign,
        label,
    )

    print("\nModel specification:")
    print(formula.strip())
    print("\n" + "=" * 72)
    print(f"H2a: Interaction model – {label}")
    print("=" * 72)
    print(f"Interaction coefficient: {stats['coefficient']:.3f}")
    print(f"Cluster-robust SE:       {stats['cluster_robust_se']:.3f}")
    print(f"t-value:                 {stats['t_value']:.3f}")
    print(f"p-value:                 {stats['p_value']:.3f}")
    print(
        f"95% CI:                  "
        f"[{stats['ci_95_low']:.3f}, {stats['ci_95_high']:.3f}]"
    )
    print(f"H2a supported:           {'Yes' if supported else 'No'}")
    print("\nInterpretation:")
    print(interpretation)

    run_wild_cluster_bootstrap(
        model,
        cluster_array,
        "11",
        output_bootstrap_11,
        f"H2a: WCR11 – {label}",
    )
    run_wild_cluster_bootstrap(
        model,
        cluster_array,
        "31",
        output_bootstrap_31,
        f"H2a: WCR31 – {label}",
    )

    simple_slopes = calculate_simple_slopes(
        analysis_df,
        ols_result,
        cluster_result,
        proximity_mean,
    )

    print("\n" + "=" * 72)
    print(f"Simple slopes – {label}")
    print("=" * 72)
    print(simple_slopes.round(DECIMALS).to_string(index=False))

    predictions = create_predictions(
        analysis_df,
        ols_result,
        afd_support_mean,
        proximity_mean,
        prediction_column,
    )

    create_interaction_plot(
        predictions,
        prediction_column,
        model_spec["y_axis_label"],
        model_spec["figure_title"],
        output_figure,
    )

    model_output = pd.DataFrame(
        [
            {
                "hypothesis": "H2a",
                "operationalization": model_key,
                "model_label": label,
                "model": "party_and_period_fixed_effects",
                "outcome": outcome,
                "predictor": AFD_SUPPORT,
                "moderator": MODERATOR,
                "interaction": INTERACTION,
                "expected_direction": expected_direction,
                **stats,
                "n_observations": int(ols_result.nobs),
                "n_clusters": int(analysis_df[PARTY].nunique()),
                "n_periods": int(analysis_df[PERIOD].nunique()),
                "r_squared": float(ols_result.rsquared),
                "adjusted_r_squared": float(ols_result.rsquared_adj),
                "afd_support_mean": afd_support_mean,
                "proximity_mean": proximity_mean,
                "supported": supported,
                "interpretation": interpretation,
            }
        ]
    )

    save_dataframe(model_output, output_model)
    save_dataframe(simple_slopes, output_simple_slopes)
    save_dataframe(predictions, output_predictions)
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
        f"File not found:\n{INPUT_FILE.resolve()}"
    )

df = pd.read_csv(INPUT_FILE)

print("=" * 72)
print("H2a: Moderation of rhetorical convergence by ideological proximity")
print("=" * 72)
print(f"File: {INPUT_FILE}")
print(f"Rows in the full dataset: {len(df)}")
print(f"Columns in the full dataset: {len(df.columns)}")


# ============================================================
# 4. ESTIMATE BOTH H2a OPERATIONALIZATIONS
# ============================================================

all_main_results = []
all_simple_slopes = []

for model_key, model_spec in MODEL_SPECS.items():
    results = run_model_specification(
        data=df,
        model_key=model_key,
        model_spec=model_spec,
    )

    all_main_results.append(results["main"])

    slopes = results["simple_slopes"].copy()
    slopes.insert(0, "operationalization", model_key)
    slopes.insert(1, "model_label", model_spec["label"])
    all_simple_slopes.append(slopes)


# ============================================================
# 5. COMBINED RESULT TABLES
# ============================================================

combined_main_results = pd.concat(
    all_main_results,
    ignore_index=True,
)
combined_simple_slopes = pd.concat(
    all_simple_slopes,
    ignore_index=True,
)

OUTPUT_COMBINED_MODELS = OUTPUT_DIR / "h2a_combined_models.csv"
OUTPUT_COMBINED_SIMPLE_SLOPES = (
    OUTPUT_DIR / "h2a_combined_simple_slopes.csv"
)

save_dataframe(combined_main_results, OUTPUT_COMBINED_MODELS)
save_dataframe(combined_simple_slopes, OUTPUT_COMBINED_SIMPLE_SLOPES)


# ============================================================
# 6. FINAL OUTPUT
# ============================================================

print("\n\n" + "=" * 72)
print("H2a: ALL ANALYSES COMPLETED")
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

print("\nCombined result files:")
print(f"- Interaction models: {OUTPUT_COMBINED_MODELS}")
print(f"- Simple slopes:      {OUTPUT_COMBINED_SIMPLE_SLOPES}")

print(
    "\nNote on interpretation:\n"
    "- In the distance model, a negative interaction coefficient "
    "supports H2a.\n"
    "- In the absolute populism position model, a positive interaction "
    "coefficient supports H2a, provided that higher values represent "
    "the TAN pole.\n"
    "- The main effect of the time-invariant proximity variable is "
    "absorbed by the party fixed effects."
)