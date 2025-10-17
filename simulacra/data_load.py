import os
from datetime import datetime
from typing import Tuple

import pandas as pd

from biolearn.data_library import DataLibrary
from simulacra.benchmark import _log


def load_gse_data(accession: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load GSE data using biolearn and save locally as CSV files.

    Args:
        accession (str): GSE accession number (e.g., 'GSE42861')

    Returns:
        tuple: (dnam_df, metadata_df) - methylation data and metadata as pandas DataFrames
    """
    _log(f"Starting {accession} data loading...")

    # Create local data directory
    data_dir = "2_poc_simulacra"
    os.makedirs(data_dir, exist_ok=True)

    # Check if data already exists
    dnam_path = os.path.join(data_dir, f"{accession}_dnam.csv")
    metadata_path = os.path.join(data_dir, f"{accession}_metadata.csv")

    if os.path.exists(dnam_path) and os.path.exists(metadata_path):
        _log(f"Loading existing {accession} data from local files...")
        dnam_df = pd.read_csv(dnam_path, index_col=0)
        metadata_df = pd.read_csv(metadata_path, index_col=0)
        _log(f"Loaded {accession}_dnam.csv: {dnam_df.shape[0]} rows, {dnam_df.shape[1]} columns")
        _log(f"Loaded {accession}_metadata.csv: {metadata_df.shape[0]} rows, {metadata_df.shape[1]} columns")
        return dnam_df, metadata_df

    _log(f"Downloading {accession} data from biolearn (LONG OPERATION)...")
    start_time = datetime.now()

    # Load data using biolearn
    library = DataLibrary()
    data = library.get(accession)
    if data is None:
        raise ValueError(f"{accession} dataset not found in biolearn library")

    data = data.load()

    # Extract methylation data (transpose to have samples as rows)
    dnam_df = data.dnam.T
    metadata_df = data.metadata.copy()

    # Ensure index alignment
    common_samples = dnam_df.index.intersection(metadata_df.index)
    dnam_df = dnam_df.loc[common_samples]
    metadata_df = metadata_df.loc[common_samples]

    download_duration = datetime.now() - start_time
    _log(f"Data download completed in {download_duration.total_seconds():.2f} seconds")

    _log(f"Saving {accession} data to local CSV files (LONG OPERATION)...")
    save_start = datetime.now()

    # Save methylation data
    dnam_df.to_csv(dnam_path)
    _log(f"Saved {accession}_dnam.csv: {dnam_df.shape[0]} rows, {dnam_df.shape[1]} columns")

    # Save metadata
    metadata_df.to_csv(metadata_path)
    _log(f"Saved {accession}_metadata.csv: {metadata_df.shape[0]} rows, {metadata_df.shape[1]} columns")

    save_duration = datetime.now() - save_start
    _log(f"Data saving completed in {save_duration.total_seconds():.2f} seconds")

    return dnam_df, metadata_df


