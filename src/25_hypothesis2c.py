# ============================================================
# H2c: MODERATION OF RHETORICAL CONVERGENCE
# BY OPPOSITION STATUS
#
# is_opposition: 0 = government, 1 = opposition
#
# MODEL A: populism_distance_to_afd
# Expectation: afd_support_c × is_opposition < 0
#
# MODEL B: mean_populism_score
# Expectation: afd_support_c × is_opposition > 0.
#
# Both models include party and electoral-term FE.
# All result values are printed with three decimal places.
# ============================================================

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from wildboottest.wildboottest import wildboottest

INPUT_FILE = Path(
    "data/processed/final_analysis/"
    "05_party_polling_interval_distances.csv"
)
OUTPUT_DIR = Path("results/hypothesis2c")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AFD_SUPPORT = "afd_support_t"
OPPOSITION = "is_opposition"
PARTY = "party_clean"
PERIOD = "wahlperiode_cat"
MAINSTREAM = "is_mainstream_party"

AFD_SUPPORT_C = "afd_support_c"
INTERACTION = "afd_x_opposition"

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
        "prefix": "h2c_distance",
        "prediction": "predicted_distance",
        "ylabel": "Predicted rhetorical distance to the AfD",
    },
    "absolute": {
        "label": "Absolute populism",
        "outcome": "mean_populism_score",
        "expected_direction": "positive",
        "expected_sign": 1,
        "prefix": "h2c_absolute_populism",
        "prediction": "predicted_populism",
        "ylabel": "Predicted absolute populism",
    },
}


def mainstream_mask(series: pd.Series) -> pd.Series:
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


def save_df(dataframe: pd.DataFrame, path: Path) -> None:
    dataframe.to_csv(
        path,
        index=False,
        float_format="%.3f",
    )


def save_bootstrap(result, path: Path) -> None:
    if isinstance(result, pd.DataFrame):
        result.reset_index().to_csv(
            path,
            index=False,
            float_format="%.3f",
        )
    else:
        pd.DataFrame({"result": [str(result)]}).to_csv(
            path,
            index=False,
        )


def direction_matches(coefficient: float, expected_sign: int) -> bool:
    return coefficient < 0 if expected_sign == -1 else coefficient > 0


def prepare_sample(data: pd.DataFrame, outcome: str) -> pd.DataFrame:
    required = [outcome, AFD_SUPPORT, OPPOSITION, PARTY, PERIOD, MAINSTREAM]
    missing = [column for column in required if column not in data.columns]

    if missing:
        raise ValueError(
            "The following variables are missing:\n"
            + "\n".join(f"- {column}" for column in missing)
        )

    mask = data[required].notna().all(axis=1)
    mask &= mainstream_mask(data[MAINSTREAM])

    sample = data.loc[mask].copy()

    for column in [outcome, AFD_SUPPORT, OPPOSITION]:
        sample[column] = pd.to_numeric(sample[column], errors="coerce")

    sample = sample.dropna(subset=required).copy()

    invalid = ~sample[OPPOSITION].isin([0, 1])
    if invalid.any():
        values = sorted(sample.loc[invalid, OPPOSITION].unique())
        raise ValueError(
            "is_opposition contains values other than 0 and 1: "
            f"{values}"
        )

    sample[OPPOSITION] = sample[OPPOSITION].astype(int)
    sample[PARTY] = sample[PARTY].astype("category")
    sample[PERIOD] = sample[PERIOD].astype("category")
    sample["cluster_id"] = sample[PARTY].cat.codes.astype(np.int64)

    afd_support_mean = sample[AFD_SUPPORT].mean()
    sample[AFD_SUPPORT_C] = sample[AFD_SUPPORT] - afd_support_mean
    sample[INTERACTION] = sample[AFD_SUPPORT_C] * sample[OPPOSITION]

    return sample


def parameter_stats(ols_result, robust_result, parameter: str) -> dict:
    names = ols_result.model.exog_names
    if parameter not in names:
        raise ValueError(f"Parameter '{parameter}' was not found.")

    index = names.index(parameter)
    ci = robust_result.conf_int()[index]

    return {
        "coefficient": float(robust_result.params[index]),
        "cluster_robust_se": float(robust_result.bse[index]),
        "t_value": float(robust_result.tvalues[index]),
        "p_value": float(robust_result.pvalues[index]),
        "ci_95_low": float(ci[0]),
        "ci_95_high": float(ci[1]),
    }


def run_bootstrap(model, clusters, bootstrap_type: str, output: Path, label: str):
    print("\n" + "=" * 72)
    print(f"H2c: WCR{bootstrap_type} – {label}")
    print("=" * 72)

    try:
        result = wildboottest(
            model,
            param=INTERACTION,
            cluster=clusters,
            B=BOOTSTRAP_REPLICATIONS,
            bootstrap_type=bootstrap_type,
            impose_null=True,
            seed=SEED,
            show=False,
        )
        print(result.round(DECIMALS) if isinstance(result, pd.DataFrame) else result)
        save_bootstrap(result, output)
        return result
    except Exception as error:
        print(f"Bootstrap failed: {type(error).__name__}: {error}")
        pd.DataFrame(
            {
                "error_type": [type(error).__name__],
                "error_message": [str(error)],
            }
        ).to_csv(output, index=False)
        return None


def simple_slopes(ols_result, robust_result) -> pd.DataFrame:
    names = ols_result.model.exog_names
    afd_index = names.index(AFD_SUPPORT_C)
    interaction_index = names.index(INTERACTION)
    rows = []

    for label, opposition_value in {"Government": 0, "Opposition": 1}.items():
        restriction = np.zeros(len(names))
        restriction[afd_index] = 1
        restriction[interaction_index] = opposition_value

        test = robust_result.t_test(restriction)
        ci = np.asarray(test.conf_int(alpha=ALPHA)).squeeze()
        slope = float(np.asarray(test.effect).squeeze())
        se = float(np.asarray(test.sd).squeeze())
        t_value = float(np.asarray(test.tvalue).squeeze())
        p_value = float(np.asarray(test.pvalue).squeeze())

        rows.append(
            {
                "status": label,
                OPPOSITION: opposition_value,
                "effect_afd_support": slope,
                "cluster_robust_se": se,
                "t_value": t_value,
                "p_value": p_value,
                "ci_95_low": float(ci[0]),
                "ci_95_high": float(ci[1]),
                "significant_05": p_value < ALPHA,
            }
        )

    return pd.DataFrame(rows)


def predictions(
    sample: pd.DataFrame,
    ols_result,
    afd_support_mean: float,
    prediction_column: str,
) -> pd.DataFrame:
    afd_grid = np.linspace(
        sample[AFD_SUPPORT].min(),
        sample[AFD_SUPPORT].max(),
        100,
    )
    reference_party = sample[PARTY].cat.categories[0]
    reference_period = sample[PERIOD].cat.categories[0]
    frames = []

    for label, opposition_value in {"Government": 0, "Opposition": 1}.items():
        prediction_data = pd.DataFrame(
            {
                AFD_SUPPORT: afd_grid,
                OPPOSITION: opposition_value,
                PARTY: reference_party,
                PERIOD: reference_period,
            }
        )
        prediction_data[AFD_SUPPORT_C] = (
            prediction_data[AFD_SUPPORT] - afd_support_mean
        )
        prediction_data[INTERACTION] = (
            prediction_data[AFD_SUPPORT_C]
            * prediction_data[OPPOSITION]
        )

        prediction = ols_result.get_prediction(
            prediction_data
        ).summary_frame(alpha=ALPHA)

        prediction_data["status"] = label
        prediction_data[prediction_column] = prediction["mean"]
        prediction_data["ci_95_low"] = prediction["mean_ci_lower"]
        prediction_data["ci_95_high"] = prediction["mean_ci_upper"]
        frames.append(prediction_data)

    return pd.concat(frames, ignore_index=True)


def make_plot(
    prediction_data: pd.DataFrame,
    prediction_column: str,
    ylabel: str,
    title: str,
    output: Path,
) -> None:
    plt.figure(figsize=(9, 6))

    for label, group in prediction_data.groupby("status", sort=False):
        plt.plot(
            group[AFD_SUPPORT],
            group[prediction_column],
            label=label,
        )

    plt.xlabel("AfD support in percentage points")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()


def run_model(data: pd.DataFrame, model_key: str, spec: dict) -> dict:
    label = spec["label"]
    outcome = spec["outcome"]
    prefix = spec["prefix"]
    expected_sign = spec["expected_sign"]

    outputs = {
        "model": OUTPUT_DIR / f"{prefix}_model_results.csv",
        "bootstrap_11": OUTPUT_DIR / f"{prefix}_bootstrap_wcr11.csv",
        "bootstrap_31": OUTPUT_DIR / f"{prefix}_bootstrap_wcr31.csv",
        "slopes": OUTPUT_DIR / f"{prefix}_simple_slopes.csv",
        "predictions": OUTPUT_DIR / f"{prefix}_predicted_values.csv",
        "figure": OUTPUT_DIR / f"{prefix}_interaction_plot.png",
        "sample": OUTPUT_DIR / f"{prefix}_analysis_sample.csv",
    }

    sample = prepare_sample(data, outcome)
    afd_support_mean = float(sample[AFD_SUPPORT].mean())
    clusters = sample["cluster_id"].to_numpy(dtype=np.int64)

    print("\n\n" + "#" * 72)
    print(f"STARTING H2c MODEL: {label.upper()}")
    print("#" * 72)
    print(f"Analysis units: {len(sample)}")
    print(f"Parties/clusters: {sample[PARTY].nunique()}")
    print(f"Electoral terms: {sample[PERIOD].nunique()}")

    print("\nObservations by status:")
    print(
        sample[OPPOSITION]
        .value_counts()
        .sort_index()
        .rename(index={0: "Government", 1: "Opposition"})
    )

    print("\nStatus by party:")
    print(
        pd.crosstab(sample[PARTY], sample[OPPOSITION]).rename(
            columns={0: "Government", 1: "Opposition"}
        )
    )

    print("\nMean used for centering:")
    print(f"AfD support: {afd_support_mean:.3f}")

    formula = f"""
        {outcome}
        ~ {AFD_SUPPORT_C}
        + {OPPOSITION}
        + {INTERACTION}
        + C({PARTY})
        + C({PERIOD})
    """

    model = smf.ols(formula=formula, data=sample)
    ols_result = model.fit()

    robust_result = ols_result.get_robustcov_results(
        cov_type="cluster",
        groups=clusters,
        use_correction=True,
        df_correction=True,
        use_t=True,
    )

    stats = parameter_stats(
        ols_result,
        robust_result,
        INTERACTION,
    )

    supported = (
        direction_matches(stats["coefficient"], expected_sign)
        and stats["p_value"] < ALPHA
    )

    print("\nModel specification:")
    print(formula.strip())
    print("\n" + "=" * 72)
    print(f"H2c: Interaction model – {label}")
    print("=" * 72)
    print(f"Interaction coefficient: {stats['coefficient']:.3f}")
    print(f"Cluster-robust SE:       {stats['cluster_robust_se']:.3f}")
    print(f"t-value:                 {stats['t_value']:.3f}")
    print(f"p-value:                 {stats['p_value']:.3f}")
    print(
        "95% CI:                  "
        f"[{stats['ci_95_low']:.3f}, {stats['ci_95_high']:.3f}]"
    )
    print(f"H2c supported:           {'Yes' if supported else 'No'}")

    run_bootstrap(
        model,
        clusters,
        "11",
        outputs["bootstrap_11"],
        label,
    )
    run_bootstrap(
        model,
        clusters,
        "31",
        outputs["bootstrap_31"],
        label,
    )

    slopes = simple_slopes(ols_result, robust_result)
    print("\n" + "=" * 72)
    print(f"Simple slopes – {label}")
    print("=" * 72)
    print(slopes.round(DECIMALS).to_string(index=False))

    prediction_data = predictions(
        sample,
        ols_result,
        afd_support_mean,
        spec["prediction"],
    )

    make_plot(
        prediction_data,
        spec["prediction"],
        spec["ylabel"],
        f"H2c: Opposition status – {label}",
        outputs["figure"],
    )

    direction_ok = direction_matches(
        stats["coefficient"],
        expected_sign,
    )
    significant = stats["p_value"] < ALPHA

    if direction_ok and significant:
        interpretation = (
            f"{label}: The interaction effect points in the expected "
            "direction and is statistically significant."
        )
    elif direction_ok:
        interpretation = (
            f"{label}: The interaction effect points in the expected "
            "direction but is not statistically significant."
        )
    elif significant:
        interpretation = (
            f"{label}: The interaction effect is statistically "
            "significant but points in the opposite direction."
        )
    else:
        interpretation = (
            f"{label}: The interaction effect does not point in the "
            "expected direction and is not statistically significant."
        )

    print("\nInterpretation:")
    print(interpretation)

    model_output = pd.DataFrame(
        [
            {
                "hypothesis": "H2c",
                "operationalization": model_key,
                "model_label": label,
                "model": "party_and_period_fixed_effects",
                "outcome": outcome,
                "predictor": AFD_SUPPORT,
                "moderator": OPPOSITION,
                "interaction": INTERACTION,
                "expected_direction": spec["expected_direction"],
                **stats,
                "n_observations": int(ols_result.nobs),
                "n_clusters": int(sample[PARTY].nunique()),
                "n_periods": int(sample[PERIOD].nunique()),
                "r_squared": float(ols_result.rsquared),
                "adjusted_r_squared": float(ols_result.rsquared_adj),
                "afd_support_mean": afd_support_mean,
                "supported": supported,
                "interpretation": interpretation,
            }
        ]
    )

    save_df(model_output, outputs["model"])
    save_df(slopes, outputs["slopes"])
    save_df(prediction_data, outputs["predictions"])
    sample.to_csv(
        outputs["sample"],
        index=False,
        float_format="%.3f",
    )

    print("\nSaved files:")
    for output in outputs.values():
        print(f"- {output}")

    slopes = slopes.copy()
    slopes.insert(0, "operationalization", model_key)
    slopes.insert(1, "model_label", label)

    return {
        "main": model_output,
        "slopes": slopes,
    }


if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"File not found:\n{INPUT_FILE.resolve()}"
    )

df = pd.read_csv(INPUT_FILE)

print("=" * 72)
print("H2c: Moderation of rhetorical convergence by opposition status")
print("=" * 72)
print(f"File: {INPUT_FILE}")
print(f"Rows in the full dataset: {len(df)}")
print(f"Columns in the full dataset: {len(df.columns)}")

main_results = []
slope_results = []

for model_key, model_spec in MODEL_SPECS.items():
    result = run_model(df, model_key, model_spec)
    main_results.append(result["main"])
    slope_results.append(result["slopes"])

combined_main = pd.concat(main_results, ignore_index=True)
combined_slopes = pd.concat(slope_results, ignore_index=True)

OUTPUT_COMBINED_MODELS = OUTPUT_DIR / "h2c_combined_models.csv"
OUTPUT_COMBINED_SLOPES = OUTPUT_DIR / "h2c_combined_simple_slopes.csv"

save_df(combined_main, OUTPUT_COMBINED_MODELS)
save_df(combined_slopes, OUTPUT_COMBINED_SLOPES)

print("\n\n" + "=" * 72)
print("H2c: ALL ANALYSES COMPLETED")
print("=" * 72)
print("\nInteraction models compared:")
print(
    combined_main[
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
print(f"- Simple slopes:      {OUTPUT_COMBINED_SLOPES}")

print(
    "\nNote on interpretation:\n"
    "- In the distance model, a negative interaction "
    "coefficient supports H2c.\n"
    "- In the absolute populism position model, a positive "
    "interaction coefficient supports H2c, provided that higher "
    "values represent stronger populist communication.\n"
    "- is_opposition = 0 denotes government parties and "
    "is_opposition = 1 denotes opposition parties."
)