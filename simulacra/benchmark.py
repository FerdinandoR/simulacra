import os
import pickle
from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sdv.metadata import SingleTableMetadata
from sdv.single_table import (
    CTGANSynthesizer,
    TVAESynthesizer,
    GaussianCopulaSynthesizer,
)


def _log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")


def train_ridge_classifier_fair(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 42,
) -> Tuple[Pipeline, Dict[str, Any]]:
    # Clean NaNs
    train_mask = (~X_train.isna().any(axis=1)) & y_train.notna()
    test_mask = (~X_test.isna().any(axis=1)) & y_test.notna()

    X_train_clean = X_train.loc[train_mask]
    y_train_clean = y_train.loc[train_mask]
    X_test_clean = X_test.loc[test_mask]
    y_test_clean = y_test.loc[test_mask]

    model = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("ridge", RidgeClassifier(random_state=random_state)),
    ])
    model.fit(X_train_clean, y_train_clean)

    y_pred = model.predict(X_test_clean)
    metrics: Dict[str, Any] = {
        "accuracy": accuracy_score(y_test_clean, y_pred),
        "f1_macro": f1_score(y_test_clean, y_pred, average="macro"),
        "report": classification_report(y_test_clean, y_pred, output_dict=False),
        "y_true": y_test_clean,
        "y_pred": y_pred,
        "test_size": len(y_test_clean),
    }
    return model, metrics


def _normalize_multipliers(multipliers: Union[Tuple[Union[int, float], ...], int, float]) -> Tuple[float, ...]:
    if isinstance(multipliers, (int, float)):
        multipliers = (multipliers,)  # type: ignore[assignment]
    if None in multipliers:  # type: ignore[operator]
        raise ValueError("None is not allowed in multipliers tuple. Baseline is always included.")
    normalized: List[float] = []
    for m in multipliers:  # type: ignore[assignment]
        if not isinstance(m, (int, float)) or m <= 0:
            raise ValueError(f"All multipliers must be positive numbers, got: {m}")
        normalized.append(float(m))
    return tuple(normalized)


def benchmark_classifiers(
    accession: str,
    test_fraction: float = 0.2,
    seed: int = 42,
    multipliers: Union[Tuple[Union[int, float], ...], int, float] = (1, 2),
) -> Dict[str, Dict[str, Any]]:
    multipliers_t = _normalize_multipliers(multipliers)

    _log(f"Starting benchmark for {accession} with seed {seed} and multipliers {multipliers_t}...")

    data_dir = "2_poc_simulacra"
    combined_path = os.path.join(data_dir, f"{accession}_embeddings_with_target.csv")

    # Cache key includes full multiplier signature to avoid partial loads
    multipliers_key = "-".join([
        str(int(m)) if float(m).is_integer() else str(m) for m in sorted(multipliers_t)
    ])
    benchmark_cache_path = os.path.join(
        data_dir, f"{accession}_benchmark_seed_{seed}_mults_{multipliers_key}.pkl"
    )

    if os.path.exists(benchmark_cache_path):
        _log(f"Loading existing benchmark results from {benchmark_cache_path}...")
        with open(benchmark_cache_path, "rb") as f:
            results = pickle.load(f)
        expected_methods = ["baseline"] + [
            f"{s}_{m}x" for s in ["GaussianCopula", "CTGAN", "TVAE"] for m in multipliers_t
        ]
        missing = [m for m in expected_methods if m not in results]
        if not missing:
            _log("Benchmark results loaded successfully")
            return results
        _log(f"Cache missing methods {missing}; recomputing this seed.")

    # Load data
    _log("Loading combined embeddings with target...")
    df = pd.read_csv(combined_path, index_col=0)

    # Split
    X = df.drop(columns=["disease"])  # features
    y = df["disease"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_fraction, random_state=seed, stratify=y
    )
    _log(f"Split sizes - Train: {len(X_train)}, Test: {len(X_test)}")

    # Train baseline
    train_df = pd.concat([y_train, X_train], axis=1)
    train_df.columns = ["disease"] + list(X_train.columns)

    _log("=== Training Baseline Classifier (no augmentation) ===")
    baseline_model, baseline_metrics = train_ridge_classifier_fair(
        X_train, y_train, X_test, y_test, random_state=seed
    )
    results: Dict[str, Dict[str, Any]] = {
        "baseline": {
            "model": baseline_model,
            "metrics": baseline_metrics,
            "train_size": len(X_train),
            "multiplier": None,
        }
    }

    # Synthesizers
    synthesizers = {
        "GaussianCopula": GaussianCopulaSynthesizer,
        "CTGAN": CTGANSynthesizer,
        "TVAE": TVAESynthesizer,
    }

    for synth_name, synth_class in synthesizers.items():
        _log(f"=== Training {synth_name} Augmented Classifiers ===")
        step_start = datetime.now()

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(train_df)

        _log(f"Training {synth_name} synthesizer (LONG OPERATION)...")
        synth = synth_class(metadata)
        synth.fit(train_df)
        _log(
            f"{synth_name} synthesizer training completed in {(datetime.now() - step_start).total_seconds():.2f} seconds"
        )

        max_multiplier = max(multipliers_t) if multipliers_t else 1
        _log(f"Generating {max_multiplier}x synthetic data with {synth_name} (LONG OPERATION)...")
        synthetic_data_all = synth.sample(num_rows=len(train_df) * int(max_multiplier))
        _log(
            f"Synthetic data generation completed in {(datetime.now() - step_start).total_seconds():.2f} seconds"
        )

        for multiplier in multipliers_t:
            _log(f"Testing {synth_name} with {multiplier}x augmentation...")
            num_needed = int(len(train_df) * multiplier)
            synthetic_subset = synthetic_data_all.iloc[:num_needed]

            augmented_train = pd.concat([train_df, synthetic_subset], axis=0, ignore_index=True)
            X_aug = augmented_train.drop(columns=["disease"])
            y_aug = augmented_train["disease"]

            augmented_model, augmented_metrics = train_ridge_classifier_fair(
                X_aug, y_aug, X_test, y_test, random_state=seed
            )
            results[f"{synth_name}_{int(multiplier) if float(multiplier).is_integer() else multiplier}x"] = {
                "model": augmented_model,
                "metrics": augmented_metrics,
                "train_size": len(augmented_train),
                "multiplier": multiplier,
                "synthesizer": synth,
            }

        _log(
            f"Total {synth_name} processing time: {(datetime.now() - step_start).total_seconds():.2f} seconds"
        )

    _log(f"Saving benchmark results to {benchmark_cache_path}...")
    with open(benchmark_cache_path, "wb") as f:
        pickle.dump(results, f)
    _log("Benchmark results saved successfully")
    return results


def run_benchmark_experiment(
    accession: str,
    seeds: List[int] = [42, 931782, 8481962],
    multipliers: Union[Tuple[Union[int, float], ...], int, float] = (1, 2),
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    multipliers_t = _normalize_multipliers(multipliers)

    _log(
        f"Starting benchmark experiment for {accession} with seeds: {seeds} and multipliers: {multipliers_t}"
    )

    all_results: Dict[int, Dict[str, Any]] = {}
    for seed in seeds:
        _log(f"\n=== Running benchmark with seed {seed} ===")
        results = benchmark_classifiers(accession, test_fraction=0.2, seed=seed, multipliers=multipliers_t)
        all_results[seed] = results

    _log("\n=== Computing Statistics Across Seeds ===")

    method_sets = [set(results.keys()) for results in all_results.values()]
    all_methods = set().union(*method_sets)
    methods = ["baseline"] + sorted([m for m in all_methods if m != "baseline"])

    summary_stats: Dict[str, Any] = {}
    for method in methods:
        acc_values: List[float] = []
        f1_values: List[float] = []
        missing_for_seeds: List[int] = []
        for seed in seeds:
            seed_results = all_results.get(seed, {})
            if method in seed_results:
                acc_values.append(seed_results[method]["metrics"]["accuracy"])  # type: ignore[index]
                f1_values.append(seed_results[method]["metrics"]["f1_macro"])  # type: ignore[index]
            else:
                missing_for_seeds.append(seed)

        if not acc_values:
            _log(f"Warning: Method '{method}' missing for all seeds. Skipping.")
            continue
        if missing_for_seeds:
            _log(
                f"Note: Method '{method}' missing for seeds: {missing_for_seeds}. Stats computed over {len(acc_values)} seeds."
            )

        summary_stats[method] = {
            "accuracy": {"mean": float(np.mean(acc_values)), "std": float(np.std(acc_values)), "values": acc_values},
            "f1_macro": {"mean": float(np.mean(f1_values)), "std": float(np.std(f1_values)), "values": f1_values},
        }

    _log("\n=== BENCHMARK RESULTS SUMMARY ===")
    _log("Method                Accuracy (mean ± std)    F1 Macro (mean ± std)")
    _log("-" * 70)
    for method in methods:
        if method not in summary_stats:
            continue
        acc_mean = summary_stats[method]["accuracy"]["mean"]
        acc_std = summary_stats[method]["accuracy"]["std"]
        f1_mean = summary_stats[method]["f1_macro"]["mean"]
        f1_std = summary_stats[method]["f1_macro"]["std"]
        _log(f"{method:<20} {acc_mean:.4f} ± {acc_std:.4f}        {f1_mean:.4f} ± {f1_std:.4f}")

    return all_results, summary_stats


def save_results_to_csv(
    all_results: Dict[int, Dict[str, Any]],
    summary_stats: Dict[str, Any],
    accession: str,
    target_column: str = "disease",
    csv_path: str | None = None,
) -> str:
    _log("Saving results to CSV file...")

    data_dir = "2_poc_simulacra"
    dnam_path = os.path.join(data_dir, "dnam.csv")
    metadata_path = os.path.join(data_dir, "metadata.csv")

    csv_data: List[Dict[str, Any]] = []
    for seed, results in all_results.items():
        for method, result_data in results.items():
            metrics = result_data["metrics"]
            csv_data.append(
                {
                    "accession": accession,
                    "dnam_path": dnam_path,
                    "metadata_path": metadata_path,
                    "target_column": target_column,
                    "seed": seed,
                    "method": method,
                    "multiplier": result_data.get("multiplier", None),
                    "accuracy": metrics["accuracy"],
                    "f1_macro": metrics["f1_macro"],
                    "test_size": metrics["test_size"],
                    "train_size": result_data["train_size"],
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

    for method, stats in summary_stats.items():
        csv_data.append(
            {
                "accession": accession,
                "dnam_path": dnam_path,
                "metadata_path": metadata_path,
                "target_column": target_column,
                "seed": "SUMMARY",
                "method": method,
                "multiplier": "N/A",
                "accuracy": stats["accuracy"]["mean"],
                "f1_macro": stats["f1_macro"]["mean"],
                "test_size": "N/A",
                "train_size": "N/A",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    results_df = pd.DataFrame(csv_data)
    if csv_path is None:
        csv_path = os.path.join("2_poc_simulacra", f"{accession}_benchmark_results.csv")
    results_df.to_csv(csv_path, index=False)
    _log(f"Results saved to {csv_path}")
    return csv_path


