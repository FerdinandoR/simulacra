import os
import argparse
import secrets
import pickle
from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple, Union, Optional, cast

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sdv.metadata import Metadata, SingleTableMetadata
from sdv.single_table import (
    CTGANSynthesizer,
    TVAESynthesizer,
    GaussianCopulaSynthesizer,
)
from sdv.evaluation.single_table import evaluate_quality
from simulacra.quality_metrics import evaluate_synthetic_quality
from simulacra.log import _log
from simulacra.embeddings import (
    generate_embeddings_with_pythae,
    create_embeddings_with_target,
)

# Embedding dimension configuration
DEFAULT_EMBEDDING_DIM: int = 512
EMBEDDING_DIM_INFER_THRESHOLD: int = 1000

try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False

import warnings


def _noop() -> None:
    return None


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


def optimize_synthesizer_hyperparameters(
    synthesizer_type: str,
    train_data: pd.DataFrame,
    metadata: Metadata,
    num_candidates: int = 5,
    sample_size: int = 2000,
    use_cuda: bool = False,
) -> Tuple[Any, str]:
    """
    Optimize synthesizer by testing different configurations and selecting the best.
    
    Args:
        synthesizer_type: Type of synthesizer ('GaussianCopula', 'CTGAN', 'TVAE')
        train_data: Training data with target column
        metadata: SDV metadata object
        num_candidates: Number of configurations to test
        sample_size: Size of synthetic samples for evaluation
        use_cuda: Whether to use CUDA for GPU-enabled synthesizers
        
    Returns:
        Tuple of (best_synthesizer, best_config_name)
    """
    
    synthesizers = {}
    
    if synthesizer_type == 'GaussianCopula':
        # Test different default distributions
        dist_candidates = ['gaussian_kde', 'norm', 'truncnorm', 'gamma', 'beta']
        candidates = dist_candidates[:num_candidates]
        
        for dist in candidates:
            # Create fresh metadata for each synthesizer
            synth_metadata = SingleTableMetadata()
            synth_metadata.detect_from_dataframe(train_data)
            synth = GaussianCopulaSynthesizer(synth_metadata, default_distribution=dist)
            synth.fit(train_data)
            synthesizers[dist] = synth
            
    elif synthesizer_type == 'CTGAN':
        # Expanded CTGAN configurations
        full_candidates = [
            ('default', {}),
            ('epochs_100', {'epochs': 100}),
            ('epochs_200', {'epochs': 200}),
            ('batch_500', {'batch_size': 500}),
            ('batch_1000', {'batch_size': 1000}),
            ('pac_2', {'pac': 2}),
            ('pac_4', {'pac': 4}),
            ('disc_steps_5', {'discriminator_steps': 5}),
            ('disc_steps_10', {'discriminator_steps': 10}),
            ('dims_small', {'generator_dim': (64, 64), 'discriminator_dim': (64, 64)}),
            ('dims_large', {'generator_dim': (256, 256), 'discriminator_dim': (256, 256)}),
        ]
        selected = full_candidates[:max(1, num_candidates)]
        
        for config, kwargs in selected:
            # Create fresh metadata for each synthesizer
            synth_metadata = SingleTableMetadata()
            synth_metadata.detect_from_dataframe(train_data)
            synth = CTGANSynthesizer(synth_metadata, cuda=use_cuda, **kwargs)
            synth.fit(train_data)
            synthesizers[config] = synth
            
    elif synthesizer_type == 'TVAE':
        # Expanded TVAE configurations
        full_candidates = [
            ('default', {}),
            ('epochs_100', {'epochs': 100}),
            ('epochs_200', {'epochs': 200}),
            ('batch_500', {'batch_size': 500}),
            ('batch_1000', {'batch_size': 1000}),
            ('latent_32', {'embedding_dim': 32}),
            ('latent_128', {'embedding_dim': 128}),
            ('compress_small', {'compress_dims': (64, 32), 'decompress_dims': (32, 64)}),
            ('compress_large', {'compress_dims': (256, 128), 'decompress_dims': (128, 256)}),
        ]
        selected = full_candidates[:max(1, num_candidates)]
        
        for config, kwargs in selected:
            # Create fresh metadata for each synthesizer
            synth_metadata = SingleTableMetadata()
            synth_metadata.detect_from_dataframe(train_data)
            synth = TVAESynthesizer(synth_metadata, cuda=use_cuda, **kwargs)
            synth.fit(train_data)
            synthesizers[config] = synth
    else:
        raise ValueError(f"Unknown synthesizer type: {synthesizer_type}")
    
    # Evaluate each synthesizer: SDV quality + simple utility on validation split
    quality_scores: Dict[str, float] = {}
    combined_scores: Dict[str, float] = {}
    # Prepare a small validation split from train_data
    try:
        from sklearn.model_selection import train_test_split as _tts
        from sklearn.linear_model import RidgeClassifier as _Ridge
        from sklearn.preprocessing import StandardScaler as _SS
        from sklearn.pipeline import Pipeline as _Pipe
        
        y_col = 'disease'
        X_all = train_data.drop(columns=[y_col])
        y_all = train_data[y_col]
        X_tr, X_val, y_tr, y_val = _tts(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all if y_all.nunique() > 1 else None)
        base_model = _Pipe([('scaler', _SS(with_mean=False)), ('ridge', _Ridge(random_state=42))])
        base_model.fit(X_tr, y_tr)
        base_acc = float(accuracy_score(y_val, base_model.predict(X_val)))
    except Exception as _e:
        base_acc = 0.0
    
    for config, synth in synthesizers.items():
        try:
            synth_samples = synth.sample(num_rows=int(min(sample_size, len(train_data))))
            # Create a fresh metadata for evaluation to avoid issues
            eval_metadata = SingleTableMetadata()
            eval_metadata.detect_from_dataframe(train_data)
            report = evaluate_quality(real_data=train_data, synthetic_data=synth_samples, metadata=eval_metadata)
            raw_score = report.get_score()
            sdv_score = float(raw_score) if raw_score is not None else 0.0
            quality_scores[config] = sdv_score
            
            # Utility: train on augmented mini-train, eval on val
            try:
                aug_tr = pd.concat([X_tr.assign(disease=y_tr.values), synth_samples], axis=0, ignore_index=True)
                X_aug = aug_tr.drop(columns=[y_col])
                y_aug = aug_tr[y_col]
                aug_model = _Pipe([('scaler', _SS(with_mean=False)), ('ridge', _Ridge(random_state=42))])
                aug_model.fit(X_aug, y_aug)
                util_acc = float(accuracy_score(y_val, aug_model.predict(X_val)))
            except Exception:
                util_acc = base_acc
            
            # Combine scores (weights can be tuned)
            combined = 0.5 * sdv_score + 0.5 * max(0.0, util_acc - base_acc)
            combined_scores[config] = combined
            _log(f"{synthesizer_type} - {config}: SDV={sdv_score:.4f}, UtilityΔ={util_acc - base_acc:.4f}, Combined={combined:.4f}")
        except Exception as e:
            _log(f"{synthesizer_type} - {config}: evaluation failed with error {e}")
            quality_scores[config] = 0.0
            combined_scores[config] = -1.0
    
    # Select best configuration
    # Prefer combined score when available
    score_source = combined_scores if combined_scores else quality_scores
    best_config = sorted(score_source.items(), key=lambda kv: kv[1], reverse=True)[0][0]
    best_synth = synthesizers[best_config]
    _log(f"Best {synthesizer_type} configuration: {best_config}")
    
    return best_synth, best_config


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
    use_cuda: Optional[bool] = None,
    embedding_dim: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    multipliers_t = _normalize_multipliers(multipliers)
    
    # Determine CUDA usage
    if use_cuda is None:
        use_cuda = CUDA_AVAILABLE
    elif use_cuda and not CUDA_AVAILABLE:
        raise RuntimeError("CUDA requested but not available. Please install CUDA-enabled PyTorch or set use_cuda=False.")
    
    _log(f"Starting benchmark for {accession} with seed {seed} and multipliers {multipliers_t}...")
    _log(f"CUDA usage: {'Enabled' if use_cuda else 'Disabled'}")

    data_dir = "2_poc_simulacra"
    combined_path = os.path.join(data_dir, f"{accession}_embeddings_with_target.csv")
    embeddings_path = os.path.join(data_dir, f"{accession}_embeddings.csv")

    # Ensure embeddings exist and match desired dimension
    desired_embedding_dim: Optional[int] = embedding_dim
    if desired_embedding_dim is None:
        # Infer from existing embeddings if available; otherwise use standard default (512)
        if os.path.exists(combined_path):
            sample_df = pd.read_csv(combined_path, nrows=1, index_col=0)
            current_dim = max(0, sample_df.shape[1] - 1)
            desired_embedding_dim = (
                current_dim if current_dim < EMBEDDING_DIM_INFER_THRESHOLD else DEFAULT_EMBEDDING_DIM
            )
            _log(
                f"Inferred embedding dim from combined file: {current_dim} -> using {desired_embedding_dim}"
            )
        elif os.path.exists(embeddings_path):
            sample_df = pd.read_csv(embeddings_path, nrows=1, index_col=0)
            current_dim = sample_df.shape[1]
            desired_embedding_dim = (
                current_dim if current_dim < EMBEDDING_DIM_INFER_THRESHOLD else DEFAULT_EMBEDDING_DIM
            )
            _log(
                f"Inferred embedding dim from embeddings file: {current_dim} -> using {desired_embedding_dim}"
            )
        else:
            desired_embedding_dim = DEFAULT_EMBEDDING_DIM
            _log(f"No existing embeddings found; using standard embedding dim {DEFAULT_EMBEDDING_DIM}")

    # Generate embeddings if they don't exist
    if not os.path.exists(embeddings_path):
        _log(
            f"Embeddings not found. Generating with latent_dim={desired_embedding_dim} (LONG OPERATION)..."
        )
        generate_embeddings_with_pythae(
            accession=accession,
            latent_dim=int(desired_embedding_dim),
        )
    else:
        # Check existing dimension for informational purposes
        sample_df = pd.read_csv(embeddings_path, nrows=1, index_col=0)
        existing_dim = sample_df.shape[1]
        if embedding_dim is not None and existing_dim != embedding_dim:
            _log(
                f"Note: existing embeddings dim {existing_dim} differs from requested {embedding_dim}. "
                "Re-generation is skipped to avoid overwriting existing results."
            )

    # Ensure combined file exists
    if not os.path.exists(combined_path):
        _log("Combined embeddings with target not found. Creating it now...")
        create_embeddings_with_target(accession=accession, target_col="disease")

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
    X = cast(pd.DataFrame, df.drop(columns=["disease"]))  # features
    y = cast(pd.Series, df["disease"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_fraction, random_state=seed, stratify=y
    )
    X_train = cast(pd.DataFrame, X_train)
    X_test = cast(pd.DataFrame, X_test)
    y_train = cast(pd.Series, y_train)
    y_test = cast(pd.Series, y_test)
    _log(f"Split sizes - Train: {len(X_train)}, Test: {len(X_test)}")

    # Feature scaling for synthesis (fit on train only)
    synth_scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_scaled_arr = synth_scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled_arr, columns=X_train.columns, index=X_train.index)

    # Train baseline
    # Concatenate target and features into a single training DataFrame
    train_df: pd.DataFrame = pd.concat([y_train, X_train], axis=1)
    train_df.columns = ["disease"] + list(X_train.columns)

    # Scaled version for synthesizer training
    train_df_scaled: pd.DataFrame = pd.concat([y_train, X_train_scaled], axis=1)
    train_df_scaled.columns = ["disease"] + list(X_train.columns)

    _log("=== Training Baseline Classifier (no augmentation) ===")
    baseline_model, baseline_metrics = train_ridge_classifier_fair(  # type: ignore[arg-type]
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

    # Synthesizers with CUDA support
    synthesizers = {
        "GaussianCopula": GaussianCopulaSynthesizer,
        "CTGAN": CTGANSynthesizer,
        "TVAE": TVAESynthesizer,
    }
    
    # CUDA-enabled synthesizers
    cuda_synthesizers = {"CTGAN", "TVAE"}

    for synth_name, synth_class in synthesizers.items():
        _log(f"=== Training {synth_name} Augmented Classifiers ===")
        step_start = datetime.now()

        metadata_path = os.path.join(data_dir, f"{accession}_{synth_name}_metadata_seed_{seed}.json")
        synthesizer_path = os.path.join(data_dir, f"{accession}_{synth_name}_synthesizer_seed_{seed}.pkl")
        
        if os.path.exists(synthesizer_path) and os.path.exists(metadata_path):
            _log(f"Loading existing synthesizer from {synthesizer_path}")
            with open(synthesizer_path, "rb") as f:
                synth_data = pickle.load(f)
            
            # Handle both old format (direct synthesizer) and new format (with hyperparameters)
            if isinstance(synth_data, dict) and 'synthesizer' in synth_data:
                synth = synth_data['synthesizer']
                best_config = synth_data.get('best_config', 'unknown')
                _log(f"Pre-trained {synth_name} synthesizer loaded successfully (config: {best_config})")
            else:
                # Old format - direct synthesizer object
                synth = synth_data
                best_config = 'N/A'
                _log(f"Pre-trained {synth_name} synthesizer loaded successfully (legacy format)")
        else:
            _log(f"Creating new synthesizer for {synth_name}...")
            
            # Always create fresh metadata to ensure proper structure
            _log(f"Creating metadata for {synth_name}...")
            metadata = Metadata()
            # Detect from scaled data to match synthesizer training space
            metadata.detect_from_dataframe(train_df_scaled)
            
            # Save metadata for reproducibility
            metadata.save_to_json(metadata_path)
            _log(f"Metadata saved to {metadata_path}")

            _log(f"Optimizing {synth_name} synthesizer hyperparameters (LONG OPERATION)...")
            
            # Use hyperparameter optimization
            synth: Any
            synth, best_config = optimize_synthesizer_hyperparameters(
                synthesizer_type=synth_name,
                train_data=train_df_scaled,
                metadata=metadata,
                num_candidates=5,
                sample_size=2000,
                use_cuda=use_cuda and synth_name in cuda_synthesizers
            )
            # synth is now guaranteed to be a synthesizer object
            
            # Save the trained synthesizer with config info
            synth_data = {
                'synthesizer': synth,
                'best_config': best_config,
                'hyperparameters': best_config
            }
            _log(f"Saving trained synthesizer to {synthesizer_path}")
            with open(synthesizer_path, "wb") as f:
                pickle.dump(synth_data, f)
            _log(f"Synthesizer saved successfully")
        _log(
            f"{synth_name} synthesizer training completed in {(datetime.now() - step_start).total_seconds():.2f} seconds"
        )

        # Conditional synthesis per class to preserve label distribution
        max_multiplier = max(multipliers_t) if multipliers_t else 1
        _log(f"Generating {max_multiplier}x synthetic data with {synth_name} using class-conditional sampling (LONG OPERATION)...")
        class_counts = y_train.value_counts()
        synthetic_pools_scaled: Dict[Any, pd.DataFrame] = {}
        for cls_label, cls_count in class_counts.items():
            desired_rows = int(cls_count * max_multiplier)
            if desired_rows <= 0:
                continue
            try:
                # Use conditional sampling if available
                known = pd.DataFrame({"disease": [cls_label] * desired_rows})
                synth_cls = synth.sample_remaining_columns(known_columns=known)
            except Exception:
                # Fallback: sample many rows and filter
                synth_many = synth.sample(num_rows=max(desired_rows * 2, desired_rows + 50))
                if "disease" in synth_many.columns:
                    synth_cls = synth_many[synth_many["disease"] == cls_label].head(desired_rows)
                    if len(synth_cls) < desired_rows:
                        # Top up if insufficient
                        synth_extra = synth.sample(num_rows=desired_rows - len(synth_cls))
                        synth_extra["disease"] = cls_label
                        synth_cls = pd.concat([synth_cls, synth_extra], ignore_index=True)
                else:
                    synth_many["disease"] = cls_label
                    synth_cls = synth_many.head(desired_rows)
            synthetic_pools_scaled[cls_label] = synth_cls

        # Merge class pools
        synthetic_data_all_scaled = pd.concat(list(synthetic_pools_scaled.values()), axis=0, ignore_index=True)
        _log(
            f"Synthetic data generation completed in {(datetime.now() - step_start).total_seconds():.2f} seconds"
        )

        for multiplier in multipliers_t:
            _log(f"Testing {synth_name} with {multiplier}x augmentation...")
            # Build per-class subset for this multiplier
            synth_parts = []
            for cls_label, cls_count in class_counts.items():
                need = int(cls_count * multiplier)
                pool_cls = synthetic_pools_scaled.get(cls_label, pd.DataFrame(columns=train_df_scaled.columns))
                synth_parts.append(pool_cls.iloc[:need])
            synthetic_subset_scaled = pd.concat(synth_parts, axis=0, ignore_index=True)

            # Inverse scale features back to original space
            if not synthetic_subset_scaled.empty:
                synth_y = synthetic_subset_scaled["disease"].reset_index(drop=True)
                synth_X_scaled = synthetic_subset_scaled.drop(columns=["disease"])  # same columns as X_train
                synth_X_inv = pd.DataFrame(
                    synth_scaler.inverse_transform(synth_X_scaled),
                    columns=X_train.columns,
                )
                synthetic_subset = pd.concat([synth_y, synth_X_inv], axis=1)
                synthetic_subset.columns = ["disease"] + list(X_train.columns)
            else:
                synthetic_subset = pd.DataFrame(columns=["disease"] + list(X_train.columns))

            # Validation-guided accept/reject of synthetic augmentation
            # Create a small validation split from the original train
            from sklearn.model_selection import train_test_split as _tts
            X_tr0, X_val0, y_tr0, y_val0 = _tts(
                X_train, y_train, test_size=0.2, random_state=seed, stratify=y_train
            )
            # Baseline on val
            base_model_tmp, _ = train_ridge_classifier_fair(X_tr0, y_tr0, X_val0, y_val0, random_state=seed)
            base_val_acc = float(
                accuracy_score(y_val0, base_model_tmp.predict(X_val0))
            )

            # Augmented on val (merge synthetic with original training partition only)
            aug_tr_df = pd.concat(
                [pd.concat([y_tr0, X_tr0], axis=1).set_axis(["disease"] + list(X_train.columns), axis=1), synthetic_subset],
                axis=0,
                ignore_index=True,
            )
            X_tr_aug = cast(pd.DataFrame, aug_tr_df.drop(columns=["disease"]))
            y_tr_aug = cast(pd.Series, aug_tr_df["disease"]) 
            aug_model_tmp, _ = train_ridge_classifier_fair(X_tr_aug, y_tr_aug, X_val0, y_val0, random_state=seed)
            aug_val_acc = float(
                accuracy_score(y_val0, aug_model_tmp.predict(X_val0))
            )

            use_augmentation = aug_val_acc >= base_val_acc

            augmented_train = pd.concat([train_df, synthetic_subset], axis=0, ignore_index=True) if use_augmentation else train_df
            X_aug = cast(pd.DataFrame, augmented_train.drop(columns=["disease"]))
            y_aug = cast(pd.Series, augmented_train["disease"])

            augmented_model, augmented_metrics = train_ridge_classifier_fair(  # type: ignore[arg-type]
                X_aug, y_aug, X_test, y_test, random_state=seed
            )
            # Compute quality metrics on synthetic subset vs real train
            try:
                quality_metrics = evaluate_synthetic_quality(real_data=train_df, synthetic_data=synthetic_subset, target_col="disease")
            except Exception as _e:
                quality_metrics = {"quality_error": str(_e)}
            results[f"{synth_name}_{int(multiplier) if float(multiplier).is_integer() else multiplier}x"] = {
                "model": augmented_model,
                "metrics": augmented_metrics,
                "train_size": len(augmented_train),
                "multiplier": multiplier,
                "synthesizer": synth,
                "hyperparameters": best_config,
                "quality": quality_metrics,
                "val_guided": {"used": use_augmentation, "base_val_acc": base_val_acc, "aug_val_acc": aug_val_acc},
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
    use_cuda: Optional[bool] = None,
    embedding_dim: Optional[int] = None,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    multipliers_t = _normalize_multipliers(multipliers)

    _log(
        f"Starting benchmark experiment for {accession} with seeds: {seeds} and multipliers: {multipliers_t}"
    )

    all_results: Dict[int, Dict[str, Any]] = {}
    for seed in seeds:
        _log(f"\n=== Running benchmark with seed {seed} ===")
        results = benchmark_classifiers(
            accession,
            test_fraction=0.2,
            seed=seed,
            multipliers=multipliers_t,
            use_cuda=use_cuda,
            embedding_dim=embedding_dim,
        )
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
    quality_keys = [
        "discriminator_accuracy",
        "discriminator_precision",
        "discriminator_recall",
        "correlation_mad",
        "avg_nn_distance",
        "min_nn_distance",
        "class_distribution_difference",
    ]

    for seed, results in all_results.items():
        for method, result_data in results.items():
            metrics = result_data["metrics"]
            quality = result_data.get("quality", {}) or {}

            row: Dict[str, Any] = {
                "accession": accession,
                "dnam_path": dnam_path,
                "metadata_path": metadata_path,
                "target_column": target_column,
                "seed": seed,
                "method": method,
                "multiplier": result_data.get("multiplier", None),
                "hyperparameters": result_data.get("hyperparameters", None),
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "test_size": metrics["test_size"],
                "train_size": result_data["train_size"],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            # Flatten selected quality metrics into CSV
            for k in quality_keys:
                row[f"quality_{k}"] = quality.get(k)

            csv_data.append(row)

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
                "hyperparameters": "N/A",
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



# -----------------------------
# CLI utilities
# -----------------------------

def generate_random_seeds(count: int) -> List[int]:
    """Generate cryptographically secure random seeds."""
    return [secrets.randbelow(2**31) for _ in range(count)]


def parse_multipliers(multipliers_str: str) -> Tuple[Union[int, float], ...]:
    """Parse multipliers string into tuple of numbers."""
    try:
        multipliers: List[Union[int, float]] = []
        for mult_str in multipliers_str.split(','):
            mult_str = mult_str.strip()
            if '.' in mult_str:
                multipliers.append(float(mult_str))
            else:
                multipliers.append(int(mult_str))
        return tuple(multipliers)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid multipliers format: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run benchmark experiments with synthetic data augmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default seeds and multipliers
  python -m simulacra.benchmark GSE42861

  # Specify custom seeds and multipliers
  python -m simulacra.benchmark GSE42861 --seeds 42,931782,8481962 --multipliers 1,2,5

  # Generate random seeds
  python -m simulacra.benchmark GSE42861 --random-seeds 3 --multipliers 1,2,5

  # Force CUDA usage
  python -m simulacra.benchmark GSE42861 --use-cuda

  # Force CPU usage
  python -m simulacra.benchmark GSE42861 --no-cuda
        """
    )

    parser.add_argument(
        'accession',
        help='Dataset accession number (e.g., GSE42861)'
    )

    parser.add_argument(
        '--seeds',
        type=parse_multipliers,
        help='Comma-separated list of seeds (e.g., "42,931782,8481962")'
    )

    parser.add_argument(
        '--random-seeds',
        type=int,
        metavar='COUNT',
        help='Generate COUNT cryptographically secure random seeds'
    )

    parser.add_argument(
        '--multipliers',
        type=parse_multipliers,
        default=(1, 2, 5),
        help='Comma-separated list of augmentation multipliers (default: 1,2,5)'
    )

    parser.add_argument(
        '--use-cuda',
        action='store_true',
        help='Force CUDA usage (raises error if CUDA not available)'
    )

    parser.add_argument(
        '--no-cuda',
        action='store_true',
        help='Force CPU usage (disable CUDA even if available)'
    )

    parser.add_argument(
        '--target-column',
        default='disease',
        help='Name of the target column (default: disease)'
    )

    parser.add_argument(
        '--output',
        help='Output CSV file path (default: auto-generated)'
    )

    parser.add_argument(
        '--embedding-dim',
        type=int,
        help=(
            'Embedding dimension for Pythae VAE latent space. '
            'If provided, embeddings are generated/used with this latent size. '
            f"If omitted, the dimension is inferred: use current embeddings' dimension when it is < {EMBEDDING_DIM_INFER_THRESHOLD} columns; "
            f'otherwise fall back to the standard {DEFAULT_EMBEDDING_DIM}.'
        )
    )

    args = parser.parse_args()

    # Validate arguments
    if args.seeds and args.random_seeds:
        parser.error("Cannot specify both --seeds and --random-seeds")

    if args.use_cuda and args.no_cuda:
        parser.error("Cannot specify both --use-cuda and --no-cuda")

    # Determine seeds
    if args.random_seeds:
        seeds = generate_random_seeds(args.random_seeds)
        _log(f"Generated {len(seeds)} random seeds: {seeds}")
    elif args.seeds:
        seeds = list(args.seeds)
        _log(f"Using specified seeds: {seeds}")
    else:
        seeds = [42, 931782, 8481962]
        _log(f"Using default seeds: {seeds}")

    # Determine CUDA usage
    if args.use_cuda:
        use_cuda = True
    elif args.no_cuda:
        use_cuda = False
    else:
        use_cuda = None  # Auto-detect

    try:
        _log(f"Starting benchmark experiment for {args.accession}")
        _log(f"Seeds: {seeds}")
        _log(f"Multipliers: {args.multipliers}")
        _log(f"Target column: {args.target_column}")

        # Run the benchmark experiment
        all_results, summary_stats = run_benchmark_experiment(
            accession=args.accession,
            seeds=seeds,
            multipliers=args.multipliers,
            use_cuda=use_cuda,
            embedding_dim=args.embedding_dim,
        )

        # Save results to CSV
        csv_path = save_results_to_csv(
            all_results=all_results,
            summary_stats=summary_stats,
            accession=args.accession,
            target_column=args.target_column,
            csv_path=args.output,
        )

        _log(f"Benchmark completed successfully!")
        _log(f"Results saved to: {csv_path}")

    except Exception as e:
        _log(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
