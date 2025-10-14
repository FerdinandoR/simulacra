#!/usr/bin/env python3
"""
Command line interface for running benchmark experiments with synthetic data augmentation.
"""

import argparse
import secrets
import sys
from typing import List, Union, Tuple

from simulacra.benchmark import (
    run_benchmark_experiment,
    save_results_to_csv,
    _log,
)


def generate_random_seeds(count: int) -> List[int]:
    """Generate cryptographically secure random seeds."""
    return [secrets.randbelow(2**31) for _ in range(count)]


def parse_multipliers(multipliers_str: str) -> Tuple[Union[int, float], ...]:
    """Parse multipliers string into tuple of numbers."""
    try:
        multipliers = []
        for mult_str in multipliers_str.split(','):
            mult_str = mult_str.strip()
            if '.' in mult_str:
                multipliers.append(float(mult_str))
            else:
                multipliers.append(int(mult_str))
        return tuple(multipliers)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid multipliers format: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmark experiments with synthetic data augmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default seeds and multipliers
  python simulacra.py GSE42861

  # Specify custom seeds and multipliers
  python simulacra.py GSE42861 --seeds 42,931782,8481962 --multipliers 1,2,5

  # Generate random seeds
  python simulacra.py GSE42861 --random-seeds 3 --multipliers 1,2,5

  # Force CUDA usage
  python simulacra.py GSE42861 --use-cuda

  # Force CPU usage
  python simulacra.py GSE42861 --no-cuda
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
        sys.exit(1)


if __name__ == '__main__':
    main()
