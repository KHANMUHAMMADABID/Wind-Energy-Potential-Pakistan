#!/usr/bin/env python3
"""Reanalysis metrics, Perkins PDF skill, multi-metric ranking, and plots.

Expected input:
    WindData.csv

Required columns:
    PMD, ERA5, JRA55, NCAR, ENSEMBLE

Outputs are written to:
    reanalysis_metrics_output/
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import weibull_min

INPUT = Path("WindData.csv")
OUT = Path("reanalysis_metrics_output")
OUT.mkdir(exist_ok=True)
OBS = "PMD"
MODELS = ["ERA5", "JRA55", "NCAR", "ENSEMBLE"]
N_BINS = 10


def read_data():
    if not INPUT.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT.resolve()}")
    df = pd.read_csv(INPUT)
    needed = [OBS] + MODELS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df[needed].apply(pd.to_numeric, errors="coerce")
    return df


def paired_values(obs, model):
    pair = pd.concat([obs, model], axis=1).dropna()
    return pair.iloc[:, 0].to_numpy(float), pair.iloc[:, 1].to_numpy(float)


def perkins_skill(obs, model, bins):
    """Calculate Perkins et al. PDF skill score using common bins."""
    obs = np.asarray(obs, float)
    model = np.asarray(model, float)
    obs = obs[np.isfinite(obs)]
    model = model[np.isfinite(model)]
    hist_obs, _ = np.histogram(obs, bins=bins, density=True)
    hist_model, _ = np.histogram(model, bins=bins, density=True)
    widths = np.diff(bins)
    return float(np.sum(np.minimum(hist_obs, hist_model) * widths))


def weibull_parameters(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) < 2:
        return np.nan, np.nan
    shape, location, scale = weibull_min.fit(values, floc=0)
    return float(shape), float(scale)


def calculate_metrics(obs, model, bins):
    """Calculate all requested metrics from paired observations."""
    obs, model = paired_values(obs, model)
    if len(obs) < 3:
        return {"N": len(obs)}

    mean_obs = np.mean(obs)
    mean_model = np.mean(model)
    std_obs = np.std(obs, ddof=1)
    std_model = np.std(model, ddof=1)
    error = model - obs
    abs_obs = np.maximum(np.abs(obs), np.finfo(float).eps)

    ss_total = np.sum((obs - mean_obs) ** 2)
    ss_res = np.sum(error ** 2)
    denominator_d = np.sum((np.abs(model - mean_obs) + np.abs(obs - mean_obs)) ** 2)

    correlation = np.corrcoef(obs, model)[0, 1]
    relative_error = np.sum(np.abs(error)) / np.maximum(np.sum(np.abs(obs)), np.finfo(float).eps)
    fractional_bias = 2 * (mean_model - mean_obs) / np.maximum(abs(mean_model + mean_obs), np.finfo(float).eps)

    return {
        "N": len(obs),
        "Observed_mean": mean_obs,
        "Model_mean": mean_model,
        "Observed_std": std_obs,
        "Model_std": std_model,
        "Mean_difference_model_minus_observed": mean_model - mean_obs,
        "MAE": np.mean(np.abs(error)),
        "RMSE": np.sqrt(np.mean(error ** 2)),
        "MBE": np.mean(error),
        "MAPE_percent": np.mean(np.abs(error) / abs_obs) * 100,
        "Relative_error": relative_error,
        "Fractional_bias": fractional_bias,
        "Correlation": correlation,
        "R_squared_correlation": correlation ** 2,
        "NSE": 1 - ss_res / ss_total if ss_total > 0 else np.nan,
        "Willmott_d": 1 - ss_res / denominator_d if denominator_d > 0 else np.nan,
        "Perkins_PDF_skill": perkins_skill(obs, model, bins),
    }


def minmax_score(series, higher_is_better):
    values = series.astype(float)
    lo, hi = values.min(), values.max()
    if not np.isfinite(lo) or hi == lo:
        return pd.Series(1.0, index=series.index)
    score = (values - lo) / (hi - lo)
    return score if higher_is_better else 1 - score


def make_ranking(metrics):
    """Create transparent equal-weight multi-metric ranking."""
    ranked = metrics.copy()
    directions = {
        "Model_mean": False,
        "Model_std": False,
        "MAE": False,
        "RMSE": False,
        "MBE": False,
        "MAPE_percent": False,
        "Relative_error": False,
        "Fractional_bias": False,
        "Correlation": True,
        "Willmott_d": True,
        "Perkins_PDF_skill": True,
    }
    score_cols = []
    for metric, higher in directions.items():
        if metric in ranked:
            # Bias metrics are scored by absolute magnitude.
            source = ranked[metric].abs() if metric in {"MBE", "Fractional_bias"} else ranked[metric]
            col = f"Score_{metric}"
            ranked[col] = minmax_score(source, higher)
            score_cols.append(col)
    ranked["Composite_score"] = ranked[score_cols].mean(axis=1)
    ranked["Rank"] = ranked["Composite_score"].rank(ascending=False, method="min").astype(int)
    return ranked.sort_values("Rank")


def save_bar_plot(ranking):
    plot = ranking.sort_values("Composite_score")
    plt.figure(figsize=(8, 5))
    plt.barh(plot["Model"], plot["Composite_score"], color="#2c7fb8")
    plt.xlabel("Equal-weight composite score")
    plt.ylabel("Reanalysis product")
    plt.title("Multi-metric reanalysis ranking")
    plt.tight_layout()
    plt.savefig(OUT / "reanalysis_composite_ranking.png", dpi=300)
    plt.close()


def save_weibull_plot(df):
    x = np.linspace(df[OBS].min(), df[OBS].max(), 300)
    plt.figure(figsize=(8, 5))
    for col, color in zip([OBS] + MODELS, ["black", "#1b9e77", "#d95f02", "#7570b3", "#e7298a"]):
        values = df[col].dropna().to_numpy(float)
        shape, scale = weibull_parameters(values)
        if np.isfinite(shape):
            plt.plot(x, weibull_min.pdf(x, shape, loc=0, scale=scale), label=col, color=color)
    plt.xlabel("Wind speed")
    plt.ylabel("Density")
    plt.title("Weibull probability density comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "weibull_pdf_comparison.png", dpi=300)
    plt.close()


def main():
    df = read_data()
    combined = pd.concat([df[c] for c in [OBS] + MODELS])
    combined = combined[np.isfinite(combined)]
    bins = np.linspace(combined.min(), combined.max(), N_BINS + 1)

    rows = []
    for model in MODELS:
        result = calculate_metrics(df[OBS], df[model], bins)
        result["Model"] = model
        rows.append(result)

    metrics = pd.DataFrame(rows)
    ranking = make_ranking(metrics)

    metrics.to_csv(OUT / "reanalysis_metrics.csv", index=False)
    ranking.to_csv(OUT / "reanalysis_multimetric_ranking.csv", index=False)
    pd.DataFrame({"Bin_edge": bins}).to_csv(OUT / "perkins_common_bins.csv", index=False)

    with pd.ExcelWriter(OUT / "reanalysis_metrics_and_ranking.xlsx", engine="openpyxl") as writer:
        metrics.to_excel(writer, index=False, sheet_name="metrics")
        ranking.to_excel(writer, index=False, sheet_name="ranking")
        pd.DataFrame({"Bin_edge": bins}).to_excel(writer, index=False, sheet_name="PDF_bins")

    save_bar_plot(ranking)
    save_weibull_plot(df)

    print("\nMETRICS\n")
    print(metrics.round(5).to_string(index=False))
    print("\nMULTI-METRIC RANKING\n")
    print(ranking[["Model", "Composite_score", "Rank"]].round(5).to_string(index=False))
    print(f"\nOutputs saved to: {OUT.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
