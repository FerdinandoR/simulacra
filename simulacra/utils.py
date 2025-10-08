from typing import Tuple, Optional, Dict, Any
from functools import lru_cache

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from biolearn.data_library import DataLibrary
from sdv.metadata import Metadata
from scipy.stats import norm, truncnorm, beta, gamma, gaussian_kde
from sklearn.metrics import mean_squared_error


@lru_cache(maxsize=1)
def get_library():
    """Get a cached instance of DataLibrary."""
    return DataLibrary()


def load_acc(acc: str):
    """
    Load data from a given accession.
    
    Args:
        acc: The accession identifier
        
    Returns:
        The loaded data object
        
    Raises:
        ValueError: If the accession is not found in the data library
    """
    lib = get_library()
    data = lib.get(acc)
    if data is None:
        raise ValueError(f"Accession {acc} not found in the data library.")
    return data.load()


def fit_distributions(data: pd.DataFrame, target_col: str = 'age', 
                     plot_title: str = 'Fitted Distributions vs Original Data', 
                     save_plot: bool = True, plot_filename: Optional[str] = None) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """
    Fit multiple distributions to data and return MSE scores and fitted PDFs.
    
    Args:
        data: DataFrame containing the data
        target_col: Column name to analyze (default: 'age')
        plot_title: Title for the plot
        save_plot: Whether to save the plot to file
        plot_filename: Custom filename for saved plot
        
    Returns:
        tuple: (mse_dict, fits_dict)
    """
    # Sample data and setup
    data.iloc[:,:-1].sample(10, random_state=42, axis=1).plot.kde()
    target_data = data.iloc[:, -1].dropna().values
    x = np.linspace(target_data.min(), target_data.max(), 200)
    hist_data = np.histogram(target_data, bins=200, density=True)

    # Fit distributions
    distributions = {
        'norm': norm,
        'gaussian_kde': gaussian_kde,
        'truncnorm': truncnorm,
        'gamma': gamma
    }

    fits, mse = {}, {}
    for name, dist in distributions.items():
        if name == 'gaussian_kde':
            kde = dist(target_data)
            pdf = kde(x)
            mse[name] = mean_squared_error(hist_data[0], kde(hist_data[1][:-1]))
        elif name == 'truncnorm':
            a, b = (target_data.min() - target_data.mean()) / target_data.std(), (target_data.max() - target_data.mean()) / target_data.std()
            params = dist.fit(target_data, floc=target_data.min(), fscale=target_data.std())
            pdf = dist.pdf(x, *params)
            mse[name] = mean_squared_error(hist_data[0], dist.pdf(hist_data[1][:-1], *params))
        elif name == 'gamma':
            params = dist.fit(target_data, floc=0)
            pdf = dist.pdf(x, *params)
            mse[name] = mean_squared_error(hist_data[0], dist.pdf(hist_data[1][:-1], *params))
        else:  # norm
            params = dist.fit(target_data)
            pdf = dist.pdf(x, *params)
            mse[name] = mean_squared_error(hist_data[0], dist.pdf(hist_data[1][:-1], *params))
        
        fits[name] = pdf

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.hist(target_data, bins=50, density=True, alpha=0.3, label='Original Data')
    for dist_name, pdf in fits.items():
        plt.plot(x, pdf, label=dist_name)
    plt.legend()
    plt.title(plot_title)
    plt.xlabel('Value')
    plt.ylabel('Density')
    
    if save_plot:
        if plot_filename is None:
            plot_filename = f'distribution_fits_{target_col}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as: {plot_filename}")
    
    plt.show()
    
    return mse, fits


def get_biolearn_df(accession: str, 
                    target_col: Optional[str] | None = None, 
                    stop_cols: Optional[int] | None = 0) -> Tuple[pd.DataFrame, Metadata]:
    """
    Load a Biolearn dataset given a GEO accession, optionally include a target
    column from the metadata, and return a tuple of (dataframe, SDV Metadata).

    The returned dataframe contains the transposed DNAm matrix (samples as rows),
    optionally joined with a single target column from the metadata.
    """
    data_obj = DataLibrary().get(accession)
    if data_obj is None:
        raise ValueError(f"Accession {accession} not found in the data library.")
    data = data_obj.load()
    if target_col is not None and target_col not in data.metadata.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in metadata. "
            f"Available columns: {data.metadata.columns.tolist()}"
        )

    dnam = data.dnam.T.iloc[:, :stop_cols]
    df = dnam.join(data.metadata[target_col]) if target_col else dnam

    return (
        df,
        Metadata.detect_from_dataframe(
            data=df,
            table_name=accession,
        ),
    )
