"""
Quality assessment metrics for synthetic data evaluation.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from simulacra.log import _log


# TODO: all of these functions should be implemented with object oriented programming, as children of an abstract base class that implements the common functionality
def calculate_statistical_distances(
    real_data: pd.DataFrame, 
    synthetic_data: pd.DataFrame,
    numerical_cols: List[str]
) -> Dict[str, float]:
    """
    Calculate statistical distances between real and synthetic data.
    
    Returns:
        Dict with per-feature Wasserstein distance and Jensen-Shannon divergence
    """
    metrics = {}
    
    for col in numerical_cols:
        if col not in real_data.columns or col not in synthetic_data.columns:
            continue
            
        real_vals = real_data[col].dropna()
        synth_vals = synthetic_data[col].dropna()
        
        if len(real_vals) < 2 or len(synth_vals) < 2:
            continue
            
        # Wasserstein distance
        try:
            wd = wasserstein_distance(real_vals, synth_vals)
            metrics[f"wasserstein_{col}"] = float(wd)
        except Exception as e:
            _log(f"Warning: Could not compute Wasserstein for {col}: {e}")
        
        # Jensen-Shannon divergence (histogram-based)
        try:
            # Create histograms with same bins
            min_val = min(real_vals.min(), synth_vals.min())
            max_val = max(real_vals.max(), synth_vals.max())
            bins = np.linspace(min_val, max_val, min(50, len(real_vals) // 2))
            
            hist_real, _ = np.histogram(real_vals, bins=bins, density=True)
            hist_synth, _ = np.histogram(synth_vals, bins=bins, density=True)
            
            # Add small epsilon to avoid zeros
            hist_real = hist_real + 1e-10
            hist_synth = hist_synth + 1e-10
            
            # Normalize
            hist_real = hist_real / hist_real.sum()
            hist_synth = hist_synth / hist_synth.sum()
            
            js_div = jensenshannon(hist_real, hist_synth)
            metrics[f"jensen_shannon_{col}"] = float(js_div)
        except Exception as e:
            _log(f"Warning: Could not compute JS divergence for {col}: {e}")
    
    return metrics


def calculate_discriminator_score(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    numerical_cols: List[str]
) -> Dict[str, Any]:
    """
    Train a classifier to distinguish real from synthetic data.
    Lower accuracy is better (real and synthetic should be indistinguishable).
    
    Returns:
        Dict with accuracy, precision, recall of the discriminator
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        
        # Combine data with labels
        real_data_subset = real_data[numerical_cols].fillna(0)
        synth_data_subset = synthetic_data[numerical_cols].fillna(0)
        
        if real_data_subset.empty or synth_data_subset.empty:
            return {"discriminator_accuracy": 1.0, "note": "insufficient_data"}
        
        X = pd.concat([real_data_subset, synth_data_subset], axis=0)
        y = [0] * len(real_data_subset) + [1] * len(synth_data_subset)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train discriminator
        discriminator = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
        discriminator.fit(X_train, y_train)
        
        # Evaluate
        y_pred = discriminator.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        
        return {
            "discriminator_accuracy": float(acc),
            "discriminator_precision": float(prec),
            "discriminator_recall": float(rec),
            "note": "lower_is_better"
        }
    except Exception as e:
        _log(f"Warning: Could not compute discriminator score: {e}")
        return {"discriminator_accuracy": 1.0, "note": f"error: {str(e)}"}


def calculate_correlation_similarity(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    numerical_cols: List[str]
) -> float:
    """
    Compare correlation matrices between real and synthetic data.
    
    Returns:
        Mean absolute difference between correlation matrices
    """
    try:
        real_subset = real_data[numerical_cols].dropna()
        synth_subset = synthetic_data[numerical_cols].dropna()
        
        if len(real_subset) < 2 or len(synth_subset) < 2:
            return 1.0
        
        # Compute correlation matrices
        corr_real = real_subset.corr().abs()
        corr_synth = synth_subset.corr().abs()
        
        # Find common columns
        common_cols = corr_real.index.intersection(corr_synth.index)
        corr_real = corr_real.loc[common_cols, common_cols]
        corr_synth = corr_synth.loc[common_cols, common_cols]
        
        # Mean absolute difference
        diff = (corr_real - corr_synth).abs().mean().mean()
        
        return float(diff)
    except Exception as e:
        _log(f"Warning: Could not compute correlation similarity: {e}")
        return 1.0


def calculate_privacy_risk(
    train_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    k: int = 5
) -> Dict[str, float]:
    """
    Calculate privacy risk by finding nearest neighbors of synthetic samples in training data.
    
    Args:
        train_data: Original training data
        synthetic_data: Generated synthetic data
        k: Number of nearest neighbors to consider
    
    Returns:
        Dict with average distance to nearest neighbors
    """
    try:
        # Convert to numeric-only
        train_numeric = train_data.select_dtypes(include=[np.number]).fillna(0)
        synth_numeric = synthetic_data.select_dtypes(include=[np.number]).fillna(0)
        
        if len(train_numeric) < k or len(synth_numeric) == 0:
            return {"avg_nn_distance": np.inf}
        
        # Common columns
        common_cols = train_numeric.columns.intersection(synth_numeric.columns)
        train_numeric = train_numeric[common_cols]
        synth_numeric = synth_numeric[common_cols]
        
        # Fit nearest neighbors on training data
        nn = NearestNeighbors(n_neighbors=min(k, len(train_numeric)), metric='euclidean')
        nn.fit(train_numeric)
        
        # Find distances for synthetic samples
        distances, _ = nn.kneighbors(synth_numeric)
        avg_distance = distances.mean()
        
        return {
            "avg_nn_distance": float(avg_distance),
            "min_nn_distance": float(distances.min()),
        }
    except Exception as e:
        _log(f"Warning: Could not compute privacy risk: {e}")
        return {"avg_nn_distance": np.inf}


def evaluate_synthetic_quality(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    target_col: str = "disease"
) -> Dict[str, Any]:
    """
    Comprehensive quality assessment of synthetic data.
    
    Args:
        real_data: Real training data
        synthetic_data: Generated synthetic data
        target_col: Target column name
    
    Returns:
        Dict with all quality metrics
    """
    _log("Computing synthetic data quality metrics...")
    
    # Get numerical columns (exclude target)
    numerical_cols = [col for col in real_data.columns 
                     if col != target_col and 
                     pd.api.types.is_numeric_dtype(real_data[col])]
    
    # Get categorical columns
    categorical_cols = [col for col in real_data.columns 
                       if col != target_col and 
                       col in synthetic_data.columns]
    
    metrics = {}
    
    # Statistical distances
    _log("Computing statistical distances...")
    stats_metrics = calculate_statistical_distances(real_data, synthetic_data, numerical_cols)
    metrics.update(stats_metrics)
    
    # Discriminator score
    _log("Computing discriminator score...")
    disc_metrics = calculate_discriminator_score(real_data, synthetic_data, numerical_cols)
    metrics.update(disc_metrics)
    
    # Correlation similarity
    _log("Computing correlation similarity...")
    corr_sim = calculate_correlation_similarity(real_data, synthetic_data, numerical_cols)
    metrics["correlation_mad"] = corr_sim
    
    # Privacy risk
    _log("Computing privacy risk...")
    privacy_metrics = calculate_privacy_risk(real_data, synthetic_data, k=5)
    metrics.update(privacy_metrics)
    
    # Class distribution preservation
    if target_col in real_data.columns and target_col in synthetic_data.columns:
        _log("Computing class distribution preservation...")
        try:
            real_counts = real_data[target_col].value_counts(normalize=True).sort_index()
            synth_counts = synthetic_data[target_col].value_counts(normalize=True).sort_index()
            
            # Find common classes
            common_classes = real_counts.index.intersection(synth_counts.index)
            real_counts_aligned = real_counts[common_classes]
            synth_counts_aligned = synth_counts[common_classes]
            
            # Chi-square-like metric
            class_dist_diff = (real_counts_aligned - synth_counts_aligned).abs().mean()
            metrics["class_distribution_difference"] = float(class_dist_diff)
        except Exception as e:
            _log(f"Warning: Could not compute class distribution: {e}")
    
    _log("Quality metrics computation completed")
    return metrics


