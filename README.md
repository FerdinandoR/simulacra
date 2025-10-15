## Simulacra

Synthetic patient data generation and benchmarking for health and longevity applications.

`simulacra` provides a lightweight framework to synthesize tabular patient data that follows the distribution of a given real dataset, and to benchmark simple classifiers trained on (i) real data and (ii) real data augmented with synthetic samples. Current focus is on omics-like embeddings (e.g., DNA methylation, proteomics), but the tooling is generic and can be applied to other tabular contexts.

### Why synthetic data?
- **Abundance**: Real datasets are often too small. Synthetic augmentation lets you scale up training data.
- **Privacy**: Synthetic samples are not tied to specific individuals and are easier to share and store.
- **Time to access**: Synthetic samples can be freely shared with the public, so that researchers don't need to spend time and money to write and fund applications to access them. This accelerates scientific research.
- **Space-saving**: Synthetic data can be generated reproducibly, cached, and discarded when not needed, without taking space on the hard drive.

### Key features
- Benchmark baseline vs. augmented training using SDV synthesizers: GaussianCopula, CTGAN, TVAE.
- Reproducible runs across multiple seeds and augmentation multipliers.
- Optional CUDA acceleration for GAN/VAE models when PyTorch with CUDA is available.
- Results caching and CSV export for downstream analysis.

## Installation

Requirements: Python >= 3.8

```bash
# To be run from the project's home directory
pip install -e .
# Required for synthesis
pip install sdv
# Optional (PyTorch for CTGAN/TVAE; choose ONE)
# If the code below does not work, check the pytorch.org website for updated install information
# CPU-only
pip install torch --index-url https://download.pytorch.org/whl/cpu
# or CUDA (pick a wheel matching your CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Notes:
- The PyPI extras are not pinned here; use versions compatible with your environment.
- If you do not have a CUDA-capable setup, install CPU-only PyTorch or skip it; the code will run on CPU.

## Quickstart

The benchmark utilities live in `simulacra/benchmark.py`. You can run them via the Python API or the CLI.

```python
from simulacra.benchmark import run_benchmark_experiment, save_results_to_csv

# Accession refers to the dataset prefix used on disk, e.g. "GSE42861"
accession = "GSE42861"

# Run benchmarks for multiple seeds and augmentation multipliers (1x, 2x shown here)
all_results, summary = run_benchmark_experiment(
    accession=accession,
    seeds=[42, 931782, 8481962],
    multipliers=(1, 2),
    use_cuda=None,  # set to True to force CUDA (if available), False for CPU
)

# Persist a tidy CSV with per-seed metrics and a SUMMARY row per method
csv_path = save_results_to_csv(all_results, summary, accession)
print("Saved:", csv_path)
```

CLI usage:

```bash
# From the project root (ensure PYTHONPATH includes the package or install in editable mode)
python -m simulacra.benchmark GSE42861 \
  --seeds 42,931782,8481962 \
  --multipliers 1,2 \
  --no-cuda

# Or generate random seeds
python -m simulacra.benchmark GSE42861 --random-seeds 3 --multipliers 1,2,5
```

### Expected data layout

By default, the code expects to find data under a working directory containing a folder named `2_poc_simulacra` with:
- `dnam.csv` and `metadata.csv` (for reference in the exported CSV), and
- `<ACCESSION>_embeddings_with_target.csv` (features + target), where the target column is named `disease`.

In this repository, those files are located under `notebooks/2_poc_simulacra/`. The current implementation uses a relative path `2_poc_simulacra`, so you have two options:
- Run your scripts/notebooks with the working directory set to `notebooks/`, or
- Copy/symlink `notebooks/2_poc_simulacra` to a sibling path accessible as `./2_poc_simulacra` from your working directory.

### Outputs and caching

Running a benchmark will:
- Train a baseline classifier (Ridge) on the original train split.
- Train/Load synthesizers and generate augmented samples at the requested multipliers.
- Train classifiers on augmented data and evaluate on the same test split.
- Cache trained synthesizers and per‑seed benchmark results in `2_poc_simulacra/` using informative filenames (seed and multiplier signature included).
- Produce a consolidated CSV via `save_results_to_csv(...)` with both per‑seed rows and SUMMARY rows.

### GPU/CUDA

- CTGAN and TVAE can optionally use CUDA. Set `use_cuda=True` when calling the APIs to force GPU use (will error if CUDA is unavailable). By default, the code auto‑detects CUDA if PyTorch is installed with CUDA support.
- GaussianCopula runs on CPU, as it's unlikely to benefit from CUDA.

## Development

Install dev dependencies and run tests:

```bash
pip install -e .[dev]
pytest -q
```

Code lives under `simulacra/`. Notebooks and example data are under `notebooks/`.

## Notes and limitations

- Synthetic data is for augmentation and experimentation. Do not use synthetic‑only performance to estimate real‑world performance.
- Current scripts under `scripts/` are placeholders; the primary usage is via the Python API in `simulacra/benchmark.py` and notebooks under `notebooks/`.

## License

This project is dual-licensed:

- Open Source: AGPL-3.0 — see `LICENSE` (link to the official text included there).
- Commercial: A paid proprietary license is available — see `LICENSES/COMMERCIAL-LICENSE.txt` and contact `ferdinando.randisi@gmail.com`.
