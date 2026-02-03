#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Ensure local imports work when running as a script.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from diversity import task_similarity


def _load_embeddings(path: Path):
    embeddings = np.load(path, allow_pickle=True)
    if isinstance(embeddings, np.ndarray):
        return list(embeddings)
    return embeddings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Load a .npy embedding file and compute the diversity coefficient."
    )
    parser.add_argument(
        "--embeddings",
        required=True,
        type=Path,
        help="Path to .npy embeddings file saved by main.py/main_files.py",
    )
    parser.add_argument(
        "--distance",
        default="cosine",
        help="Distance metric to use (default: cosine)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory to write outputs (default: embeddings file directory)",
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Do not write distance_matrix.npy or diversity_results.json",
    )
    args = parser.parse_args()

    embeddings_path = args.embeddings.expanduser().resolve()
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    embeddings = _load_embeddings(embeddings_path)
    if len(embeddings) < 2:
        raise ValueError("Need at least 2 embeddings to compute diversity coefficient")

    distance_matrix = task_similarity.pdist(embeddings, distance=args.distance)
    div_coeff, div_coeff_ci = task_similarity.stats_of_distance_matrix(distance_matrix)

    print(f"diversity_coefficient={div_coeff}")
    print(f"diversity_ci_std={div_coeff_ci}")

    if not args.no_save:
        output_dir = (
            args.output_dir.expanduser().resolve()
            if args.output_dir is not None
            else embeddings_path.parent
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "distance_matrix.npy", distance_matrix)
        results = {
            "diversity_coefficient": float(div_coeff),
            "diversity_ci_std": float(div_coeff_ci),
            "num_embeddings": int(len(embeddings)),
            "distance": args.distance,
            "embeddings_file": str(embeddings_path),
        }
        with open(output_dir / "diversity_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
