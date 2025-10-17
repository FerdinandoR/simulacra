import io
import os
from typing import List, Optional, Tuple

import pandas as pd


def load_dataframe_from_file(file_obj) -> pd.DataFrame:
    """Load a DataFrame from a gradio file object (CSV or Parquet)."""
    if file_obj is None:
        raise ValueError("No file provided")
    path = getattr(file_obj, 'name', None) or getattr(file_obj, 'orig_name', None)
    if not path:
        # Try bytes
        if hasattr(file_obj, 'read'):
            data = file_obj.read()
            try:
                return pd.read_csv(io.BytesIO(data))
            except Exception:
                return pd.read_parquet(io.BytesIO(data))
        raise ValueError("Unsupported file object")

    lower = path.lower()
    if lower.endswith('.csv'):
        return pd.read_csv(path)
    if lower.endswith('.parquet') or lower.endswith('.pq'):
        return pd.read_parquet(path)
    # Fallback: try CSV then Parquet
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_parquet(path)


def dataframe_shape_str(df: pd.DataFrame) -> str:
    return f"shape: {df.shape[0]} rows × {df.shape[1]} columns"


def compact_preview(df: pd.DataFrame, n: int = 5) -> Tuple[pd.DataFrame, bool, bool]:
    """
    Return a compact preview showing first/last n rows and first/last n columns.
    Insert ellipsis rows/cols when middle is elided.
    Returns: (preview_df, elided_rows, elided_cols)
    """
    rows, cols = df.shape
    left_cols = list(df.columns[:n])
    right_cols = list(df.columns[-n:]) if cols > n else []

    col_elided = cols > 2 * n
    if col_elided:
        selected_cols = left_cols + right_cols
    else:
        selected_cols = list(df.columns)

    top_rows = df.iloc[:n]
    bottom_rows = df.iloc[-n:] if rows > n else pd.DataFrame(columns=df.columns)
    row_elided = rows > 2 * n

    # Build compact table
    if row_elided:
        top = top_rows[selected_cols]
        bottom = bottom_rows[selected_cols]
        # Insert an ellipsis row
        ellipsis_row = pd.DataFrame(
            {c: ['…'] for c in selected_cols}
        )
        compact = pd.concat([top, ellipsis_row, bottom], ignore_index=True)
    else:
        compact = df[selected_cols]

    # Insert an ellipsis column if needed
    if col_elided:
        left_part = compact[left_cols]
        right_part = compact[right_cols]
        ellipsis_col = pd.Series(['…'] * len(compact), name='…')
        compact = pd.concat([left_part, ellipsis_col, right_part], axis=1)

    return compact, row_elided, col_elided


def highlight_headers_html(df: pd.DataFrame, highlight_cols: Optional[List[str]] = None, max_height: int = 360) -> str:
    """
    Render a simple HTML table for df with highlighted header cells for columns in highlight_cols.
    Adds a scrollable container if content overflows.
    """
    highlight_set = set(highlight_cols or [])
    # Build header
    header_cells = []
    for col in df.columns:
        style = "background-color: #fff59d;" if col in highlight_set else ""
        header_cells.append(f'<th style="{style} position: sticky; top: 0; background-clip: padding-box;">{col}</th>')
    thead = f"<thead><tr>{''.join(header_cells)}</tr></thead>"

    # Build body
    body_rows = []
    for _, row in df.iterrows():
        tds = ''.join([f"<td>{row[c]}</td>" for c in df.columns])
        body_rows.append(f"<tr>{tds}</tr>")
    tbody = f"<tbody>{''.join(body_rows)}</tbody>"

    table_html = f"<div style='max-height:{max_height}px; overflow:auto; border:1px solid #ddd'>" \
                 f"<table style='border-collapse: collapse; width: 100%'>" \
                 f"{thead}{tbody}</table></div>"
    return table_html


def perform_join(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_keys: List[str],
    right_keys: List[str],
    how: str = 'left',
) -> pd.DataFrame:
    if len(left_keys) != len(right_keys):
        raise ValueError("Left and right key lists must be the same length")
    return left_df.merge(right_df, left_on=left_keys, right_on=right_keys, how=how)


def save_working_dataset(
    df: pd.DataFrame,
    target_column: str,
    data_dir: str = "2_poc_simulacra",
    accession: str = "USER",
) -> Tuple[str, str]:
    """Save working dataset into expected files for benchmark flow."""
    os.makedirs(data_dir, exist_ok=True)
    dnam_path = os.path.join(data_dir, "dnam.csv")
    metadata_path = os.path.join(data_dir, "metadata.csv")

    # Ensure target exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")

    # Split into metadata (target only) and features (rest)
    metadata = df[[target_column]].copy()
    features = df.drop(columns=[target_column]).copy()

    metadata.to_csv(metadata_path)
    features.to_csv(dnam_path)
    return dnam_path, metadata_path


def write_embeddings_as_is(
    df: pd.DataFrame,
    target_column: str,
    data_dir: str = "2_poc_simulacra",
    accession: str = "USER",
) -> str:
    """Create an embeddings file equal to current features and a combined file for benchmark flow."""
    os.makedirs(data_dir, exist_ok=True)
    embeddings_path = os.path.join(data_dir, f"{accession}_embeddings.csv")
    combined_path = os.path.join(data_dir, f"{accession}_embeddings_with_target.csv")

    features = df.drop(columns=[target_column]).copy()
    # Rename columns to emb_*
    features = features.rename(columns={c: f"emb_{i}" for i, c in enumerate(features.columns)})
    features.to_csv(embeddings_path)

    combined = pd.concat([df[[target_column]], features], axis=1)
    combined.to_csv(combined_path)
    return combined_path


