import os
import warnings
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import IterableDataset, Dataset
from typing import Optional, List, Union, Callable, Dict
from sklearn.preprocessing import LabelEncoder
from pythae.data.datasets import DatasetOutput

class PythaeIterableDataset(IterableDataset):
    def __init__(
        self,
        folder: Optional[str] = None,
        dnam_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        input_metadata_cols: Optional[List[str]] = None,
        input_metadata_types: Optional[Dict[str, str]] = None,  # e.g. {"sex": "categorical", "age": "numeric"}
        target_metadata_col: Optional[str] = None,
        target_metadata_type: Optional[str] = None,  # "categorical" or "numeric"
        scaler: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        n_dnam_cols: Optional[int] = None,
        n_rows: Optional[int] = None,
        join_on: Optional[str] = "id",
        on_unmatched: str = "warn",  # "ignore", "warn", "error"
        chunk_size: Optional[int] = 512,
        device: Optional[Union[str, torch.device]] = None,
        shuffle: bool = False,
        seed: Optional[int] = None,
        row_offset: int = 0,
    ):
        super().__init__()
        # File/folder logic
        if folder is not None:
            dnam_path = dnam_path or os.path.join(folder, "dnam.csv")
            metadata_path = metadata_path or os.path.join(folder, "metadata.csv")
        if dnam_path is None or metadata_path is None:
            raise ValueError("Must provide either folder or both dnam_path and metadata_path.")
        self.dnam_path = dnam_path
        self.metadata_path = metadata_path
        self.input_metadata_cols = input_metadata_cols or []
        self.input_metadata_types = input_metadata_types or {}
        self.target_metadata_col = target_metadata_col
        self.target_metadata_type = target_metadata_type
        self.scaler = scaler
        self.n_dnam_cols = n_dnam_cols
        self.n_rows = n_rows
        self.join_on = join_on
        self.on_unmatched = on_unmatched
        self.chunk_size = chunk_size
        self.device = device
        self.shuffle = shuffle
        self.seed = seed
        self.row_offset = row_offset
        
        # Set random seed if provided
        if self.shuffle and self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
        
        # Prepare metadata
        self._prepare_metadata()
        # Prepare dnam column names
        self._prepare_dnam_header()

    def _prepare_metadata(self):
        # Read metadata (optionally limit rows)
        self.metadata = pd.read_csv(self.metadata_path, nrows=self.n_rows)
        self.metadata.set_index(self.join_on, inplace=True)
        # Prepare encoders for categorical columns
        self.label_encoders = {}
        for col, typ in self.input_metadata_types.items():
            if typ == "categorical":
                nans_before = self.metadata[col].isna().sum()
                le = LabelEncoder()
                # Convert to string to avoid issues with NaN
                self.metadata[col] = le.fit_transform(self.metadata[col].astype(str))
                nans_after = self.metadata[col].isna().sum()
                if nans_after > nans_before:
                    warnings.warn(
                        f"Column '{col}': Number of NaN values increased from {nans_before} to {nans_after} after encoding. "
                        "This may be due to unseen or missing values during label encoding."
                    )
                self.label_encoders[col] = le
        if self.target_metadata_col and self.target_metadata_type == "categorical":
            nans_before = self.metadata[self.target_metadata_col].isna().sum()
            le = LabelEncoder()
            self.metadata[self.target_metadata_col] = le.fit_transform(self.metadata[self.target_metadata_col].astype(str))
            nans_after = self.metadata[self.target_metadata_col].isna().sum()
            if nans_after > nans_before:
                warnings.warn(
                    f"Target column '{self.target_metadata_col}': Number of NaN values increased from {nans_before} to {nans_after} after encoding. "
                    "This may be due to unseen or missing values during label encoding."
                )
            self.label_encoders[self.target_metadata_col] = le
        # Numeric columns: scaling will be done on the fly

    def _prepare_dnam_header(self):
        # Read only the header to get column names
        if self.dnam_path.endswith('.gz'):
            import gzip
            with gzip.open(self.dnam_path, 'rt') as f:
                header = f.readline().strip().split(',')
        else:
            with open(self.dnam_path, 'r') as f:
                header = f.readline().strip().split(',')
        self.dnam_columns = header[1:self.n_dnam_cols+1] if self.n_dnam_cols else header[1:]
        self.dnam_id_col = header[0]

    def __iter__(self):
        dnam_iter = pd.read_csv(
            self.dnam_path,
            usecols = range(self.n_dnam_cols + 1) if self.n_dnam_cols is not None else None,
            index_col = 0,
            iterator=True,
            chunksize=self.chunk_size,
            nrows=self.n_rows + self.row_offset if self.n_rows else None
        )
        count = 0
        skipped = 0
        
        for chunk in dnam_iter:
            # Skip rows if we have a row_offset
            if skipped < self.row_offset:
                skip_in_chunk = min(self.row_offset - skipped, len(chunk))
                chunk = chunk.iloc[skip_in_chunk:]
                skipped += skip_in_chunk
                if len(chunk) == 0:
                    continue
            
            # Inner join with metadata
            joined = chunk.join(self.metadata, how='inner', lsuffix='_dnam', rsuffix='_meta')
            # Handle unmatched rows
            n_unmatched = len(chunk) - len(joined)
            if n_unmatched > 0:
                msg = f"{n_unmatched} rows in dnam.csv not matched in metadata.csv."
                if self.on_unmatched == "error":
                    raise ValueError(msg)
                elif self.on_unmatched == "warn":
                    warnings.warn(msg)
                # else ignore
            # Optionally limit rows
            if self.n_rows is not None:
                if count + len(joined) > self.n_rows:
                    joined = joined.iloc[:self.n_rows - count]
                count += len(joined)
                done = count >= self.n_rows
            else:
                done = False
            # Prepare X
            dnam_data = joined[self.dnam_columns].to_numpy(dtype=np.float32)
            meta_inputs = []
            for col in self.input_metadata_cols:
                typ = self.input_metadata_types.get(col, "numeric")
                arr = joined[col].to_numpy()
                if typ == "numeric" and self.scaler is not None:
                    arr = self.scaler(arr.astype(np.float32))
                meta_inputs.append(arr.reshape(-1, 1))
            if meta_inputs:
                meta_inputs = np.concatenate(meta_inputs, axis=1).astype(np.float32)
                X = np.concatenate([dnam_data, meta_inputs], axis=1)
            else:
                X = dnam_data
            # Prepare y
            if self.target_metadata_col:
                y = joined[self.target_metadata_col].to_numpy()
                if self.target_metadata_type == "numeric" and self.scaler is not None:
                    y = self.scaler(y.astype(np.float32))
                y = torch.tensor(y, dtype=torch.float32 if self.target_metadata_type == "numeric" else torch.long, device=self.device)
            else:
                y = None
            
            # Create indices for shuffling if enabled
            indices = list(range(X.shape[0]))
            if self.shuffle:
                random.shuffle(indices)
            
            # Yield row by row (shuffled if enabled)
            for i in indices:
                Xi = torch.tensor(X[i], dtype=torch.float32, device=self.device)
                yi = y[i] if y is not None else None
                yield Xi, yi
            if done:
                break

    @staticmethod
    def split_dataset(
        split_points: Union[int, float, List[Union[int, float]]],
        shuffle: bool = False,
        seed: Optional[int] = None,
        dataset_class: Optional[type] = None,
        **kwargs
    ) -> List[Union['PythaeIterableDataset', 'DNAmDataset']]:
        """
        Create multiple PythaeIterableDataset instances based on split points.
        
        Args:
            split_points: Can be:
                - int: number of records before split
                - float: fraction of data before split (0.0 to 1.0)
                - list: multiple split points (e.g., [100, 0.4] means first 100 rows, then 40% of remaining, then rest)
            shuffle: Whether to shuffle data before splitting
            seed: Random seed for shuffling (if shuffle=True)
            dataset_class: Class to use for creating datasets (defaults to PythaeIterableDataset)
            **kwargs: All other arguments for PythaeIterableDataset constructor
        
        Returns:
            List of dataset instances, one per split
        """
        # Use the calling class if dataset_class is not specified
        if dataset_class is None:
            dataset_class = PythaeIterableDataset
        
        # Convert single split point to list
        if not isinstance(split_points, list):
            split_points = [split_points]
        
        # Get total row count
        total_rows = PythaeIterableDataset._get_total_rows(**kwargs)
        
        # Calculate actual split indices
        split_indices = []
        current_pos = 0
        
        for split_point in split_points:
            if isinstance(split_point, int):
                # Absolute number of rows
                split_indices.append(min(split_point, total_rows))
            elif isinstance(split_point, float):
                # Fraction of remaining data
                if split_point <= 0.0 or split_point >= 1.0:
                    raise ValueError("Float split points must be positive and less than 1.0")
                remaining_rows = total_rows - current_pos
                split_indices.append(current_pos + int(remaining_rows * split_point))
            else:
                raise ValueError("Split points must be integers or floats")
            current_pos = split_indices[-1]
        
        # Add final split if needed
        if split_indices[-1] < total_rows:
            split_indices.append(total_rows)
        
        # Create datasets for each split
        datasets = []
        for i in range(len(split_indices)):
            start_idx = 0 if i == 0 else split_indices[i-1]
            end_idx = split_indices[i]
            
            # Create kwargs for this split
            split_kwargs = kwargs.copy()
            split_kwargs['n_rows'] = end_idx - start_idx
            split_kwargs['row_offset'] = start_idx  # New parameter for row offset
            
            # Create dataset for this split
            dataset = dataset_class(
                shuffle=shuffle,
                seed=seed,
                **split_kwargs
            )
            datasets.append(dataset)
        
        return datasets
    
    @staticmethod
    def _get_total_rows(**kwargs) -> int:
        """Helper method to get total number of rows efficiently."""
        metadata_path = kwargs.get('metadata_path')
        folder = kwargs.get('folder')
        if folder is not None:
            metadata_path = metadata_path or os.path.join(folder, "metadata.csv")
        
        if metadata_path is None:
            raise ValueError("Must provide either folder or metadata_path to determine total rows")
        
        # Read metadata to get total count
        metadata = pd.read_csv(metadata_path, nrows=kwargs.get('n_rows'))
        return len(metadata)


class DNAmDataset(Dataset):
    """
    A dataset implementation compatible with Pythae that loads data from CSV files.
    This implementation inherits from torch.utils.data.Dataset instead of IterableDataset
    to be fully compatible with Pythae's requirements.
    """
    def __init__(
        self,
        folder: Optional[str] = None,
        dnam_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        input_metadata_cols: Optional[List[str]] = None,
        input_metadata_types: Optional[Dict[str, str]] = None,
        target_metadata_col: Optional[str] = None,
        target_metadata_type: Optional[str] = None,
        scaler: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        n_dnam_cols: Optional[int] = None,
        n_rows: Optional[int] = None,
        join_on: Optional[str] = "Unnamed: 0",
        on_unmatched: str = "warn",
        chunk_size: Optional[int] = 512,
        device: Optional[Union[str, torch.device]] = None,
        shuffle: bool = False,
        seed: Optional[int] = None,
        row_offset: int = 0,
    ):
        super().__init__()
        # File/folder logic
        if folder is not None:
            dnam_path = dnam_path or os.path.join(folder, "dnam.csv")
            metadata_path = metadata_path or os.path.join(folder, "metadata.csv")
        if dnam_path is None or metadata_path is None:
            raise ValueError("Must provide either folder or both dnam_path and metadata_path.")
        
        self.dnam_path = dnam_path
        self.metadata_path = metadata_path
        self.input_metadata_cols = input_metadata_cols or []
        self.input_metadata_types = input_metadata_types or {}
        self.target_metadata_col = target_metadata_col
        self.target_metadata_type = target_metadata_type
        self.scaler = scaler
        self.n_dnam_cols = n_dnam_cols
        self.n_rows = n_rows
        self.join_on = join_on
        self.on_unmatched = on_unmatched
        self.device = device
        self.row_offset = row_offset
        
        # Set random seed if provided
        if shuffle and seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Load the entire dataset into memory
        self._load_data()
        
        # Shuffle data if requested
        if shuffle:
            indices = np.random.permutation(len(self.data))
            self.data = self.data[indices]
            if hasattr(self, 'targets') and self.targets is not None:
                self.targets = self.targets[indices]

    def _load_data(self):
        """Load the entire dataset into memory for random access"""
        # Read metadata (don't limit rows here, we'll split after join)
        metadata = pd.read_csv(self.metadata_path)
        metadata.set_index(self.join_on, inplace=True)
        
        # Prepare encoders for categorical columns
        label_encoders = {}
        for col, typ in self.input_metadata_types.items():
            if typ == "categorical":
                le = LabelEncoder()
                metadata[col] = le.fit_transform(metadata[col].astype(str))
                label_encoders[col] = le
        if self.target_metadata_col and self.target_metadata_type == "categorical":
            le = LabelEncoder()
            metadata[self.target_metadata_col] = le.fit_transform(metadata[self.target_metadata_col].astype(str))
            label_encoders[self.target_metadata_col] = le
        
        # Read dnam data (don't apply row limits here either)
        dnam = pd.read_csv(
            self.dnam_path,
            usecols=range(self.n_dnam_cols + 1) if self.n_dnam_cols is not None else None,
            index_col=0
        )
        
        # Inner join with metadata first
        joined_full = dnam.join(metadata, how='inner', lsuffix='_dnam', rsuffix='_meta')
        
        # Handle unmatched rows (calculate this before applying splits)
        n_unmatched = len(dnam) - len(joined_full)
        if n_unmatched > 0:
            msg = f"{n_unmatched} rows in dnam.csv not matched in metadata.csv."
            if self.on_unmatched == "error":
                raise ValueError(msg)
            elif self.on_unmatched == "warn":
                warnings.warn(msg)
        
        # Now apply row_offset and n_rows to the joined data
        joined = joined_full
        if self.row_offset > 0:
            joined = joined.iloc[self.row_offset:]
        if self.n_rows is not None:
            joined = joined.iloc[:self.n_rows]
        
        # Get dnam column names
        if self.dnam_path.endswith('.gz'):
            import gzip
            with gzip.open(self.dnam_path, 'rt') as f:
                header = f.readline().strip().split(',')
        else:
            with open(self.dnam_path, 'r') as f:
                header = f.readline().strip().split(',')
        dnam_columns = header[1:self.n_dnam_cols+1] if self.n_dnam_cols else header[1:]
        
        # Prepare data
        dnam_data = joined[dnam_columns].to_numpy(dtype=np.float32)
        meta_inputs = []
        for col in self.input_metadata_cols:
            typ = self.input_metadata_types.get(col, "numeric")
            arr = joined[col].to_numpy()
            if typ == "numeric" and self.scaler is not None:
                arr = self.scaler(arr.astype(np.float32))
            meta_inputs.append(arr.reshape(-1, 1))
        
        if meta_inputs:
            meta_inputs = np.concatenate(meta_inputs, axis=1).astype(np.float32)
            self.data = np.concatenate([dnam_data, meta_inputs], axis=1)
        else:
            self.data = dnam_data
        
        # Prepare targets
        if self.target_metadata_col:
            targets = joined[self.target_metadata_col].to_numpy()
            if self.target_metadata_type == "numeric" and self.scaler is not None:
                targets = self.scaler(targets.astype(np.float32))
            self.targets = torch.tensor(
                targets, 
                dtype=torch.float32 if self.target_metadata_type == "numeric" else torch.long,
                device=self.device
            )
        else:
            self.targets = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """Get a sample from the dataset at the given index"""
        data = torch.tensor(self.data[index], dtype=torch.float32, device=self.device)
        return DatasetOutput(data=data)

    @staticmethod
    def split_dataset(
        split_points: Union[int, float, List[Union[int, float]]],
        shuffle: bool = False,
        seed: Optional[int] = None,
        **kwargs
    ) -> List['DNAmDataset']:
        """
        Create multiple DNAmDataset instances based on split points.
        
        Args:
            split_points: Can be:
                - int: number of records before split
                - float: fraction of data before split (0.0 to 1.0)
                - list: multiple split points (e.g., [100, 0.4] means first 100 rows, then 40% of remaining, then rest)
            shuffle: Whether to shuffle data before splitting
            seed: Random seed for shuffling (if shuffle=True)

            **kwargs: All other arguments for DNAmDataset constructor
        
        Returns:
            List of DNAmDataset instances, one per split
        """
        # Always use DNAmDataset for this method
        # Get total row count by actually checking the joined data
        # We need to do a sample join to get the actual number of rows that will be available
        
        # File/folder logic
        folder = kwargs.get('folder')
        dnam_path = kwargs.get('dnam_path')
        metadata_path = kwargs.get('metadata_path')
        
        if folder is not None:
            dnam_path = dnam_path or os.path.join(folder, "dnam.csv")
            metadata_path = metadata_path or os.path.join(folder, "metadata.csv")
        
        if dnam_path is None or metadata_path is None:
            raise ValueError("Must provide either folder or both dnam_path and metadata_path.")
        
        # Read metadata
        n_dnam_cols = kwargs.get('n_dnam_cols')
        n_rows = kwargs.get('n_rows')
        join_on = kwargs.get('join_on', 'id')
        
        metadata = pd.read_csv(metadata_path, nrows=n_rows)
        metadata.set_index(join_on, inplace=True)
        
        # Read a sample of dnam data to determine actual joined size
        dnam_sample = pd.read_csv(
            dnam_path,
            usecols=range(n_dnam_cols + 1) if n_dnam_cols is not None else None,
            index_col=0,
            nrows=n_rows if n_rows else None
        )
        
        # Do the join to see actual available rows
        joined = dnam_sample.join(metadata, how='inner')
        total_rows = len(joined)
        
        # Convert single split point to list
        if not isinstance(split_points, list):
            split_points = [split_points]
        
        # Calculate actual split indices
        split_indices = []
        current_pos = 0
        
        for split_point in split_points:
            if isinstance(split_point, int):
                # Absolute number of rows
                next_pos = min(current_pos + split_point, total_rows)
                split_indices.append(next_pos)
            elif isinstance(split_point, float):
                # Fraction of total data
                if split_point <= 0.0 or split_point >= 1.0:
                    raise ValueError("Float split points must be positive and less than 1.0")
                next_pos = int(total_rows * split_point)
                split_indices.append(next_pos)
            else:
                raise ValueError("Split points must be integers or floats")
            current_pos = split_indices[-1]
        
        # Add final split if needed (for the remaining data)
        if split_indices[-1] < total_rows:
            split_indices.append(total_rows)
        
        # Create datasets for each split
        datasets = []
        for i in range(len(split_indices)):
            start_idx = 0 if i == 0 else split_indices[i-1]
            end_idx = split_indices[i]
            
            # Skip empty splits
            if start_idx >= end_idx:
                continue
                
            # Create kwargs for this split
            split_kwargs = kwargs.copy()
            split_kwargs['n_rows'] = end_idx - start_idx
            split_kwargs['row_offset'] = start_idx
            
            # Create dataset for this split
            dataset = DNAmDataset(
                shuffle=shuffle,
                seed=seed,
                **split_kwargs
            )
            datasets.append(dataset)
        
        return datasets
