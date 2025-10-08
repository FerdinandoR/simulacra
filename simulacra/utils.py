from typing import Tuple, Optional
from functools import lru_cache

import pandas as pd
from biolearn.data_library import DataLibrary
from sdv.metadata import Metadata


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
