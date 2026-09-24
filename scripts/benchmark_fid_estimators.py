"""Benchmark the NumPy FID estimators on synthetic feature arrays.

Run from the repository root with, for example:

    python scripts/benchmark_fid_estimators.py --repeats 3
"""

import gc
import json
import sys
import tracemalloc
import timeit
from pathlib import Path

import click
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fid import compute_statistics, frechet_distance
from src.fid_infty import fid_infinity_from_feats
from src.rmt_fid import rmt_frechet_distance_analytical


DEFAULT_DIMENSIONS = (64, 128, 512, 1024, 2048)


def benchmark_call(function, *args, repeats: int, warmup: int, **kwargs) -> tuple[float, float, float, float, float, list[float]]:
    """Return runtime summary statistics and individual measurements."""
    for _ in range(warmup):
        function(*args, **kwargs)

    def call_estimator():
        value = function(*args, **kwargs)
        if not np.isfinite(value):
            raise ValueError(f"{function.__name__} returned a non-finite value: {value}")
        return value

    timings = timeit.repeat(call_estimator, repeat=repeats, number=1)

    return (
        float(np.mean(timings)),
        float(np.median(timings)),
        float(np.min(timings)),
        float(np.max(timings)),
        float(np.std(timings)),
        list(timings),
    )


def measure_peak_memory(function, *args, **kwargs) -> int:
    """Return the peak Python-tracked allocation during one estimator call."""
    tracemalloc.start()
    try:
        current_before, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        value = function(*args, **kwargs)
        if not np.isfinite(value):
            raise ValueError(f"{function.__name__} returned a non-finite value: {value}")
        _, peak_memory = tracemalloc.get_traced_memory()
        return max(0, peak_memory - current_before)
    finally:
        tracemalloc.stop()


def benchmark_dimension(
    dimension: int,
    n_samples: int,
    repeats: int,
    warmup: int,
    seed: int,
    fid_infinity_points: int,
) -> list[dict[str, float | int | str]]:
    """Benchmark all estimators for one feature dimension."""
    rng = np.random.default_rng(seed)
    reference_feats = rng.standard_normal((n_samples, dimension))
    synthetic_feats = rng.standard_normal((n_samples, dimension))
    reference_mean, reference_covariance = compute_statistics(reference_feats)
    synthetic_mean, synthetic_covariance = compute_statistics(synthetic_feats)

    estimators = (
        (
            "fid_from_feats_efficient",
            frechet_distance,
            {
                "efficient": True,
                "mu1": reference_mean,
                "sigma1": reference_covariance,
                "mu2": synthetic_mean,
                "sigma2": synthetic_covariance,
            },
        ),
        (
            "fid_from_feats_standard",
            frechet_distance,
            {
                "efficient": False,
                "mu1": reference_mean,
                "sigma1": reference_covariance,
                "mu2": synthetic_mean,
                "sigma2": synthetic_covariance,
            },
        ),
        (
            "fid_infinity_from_feats",
            fid_infinity_from_feats,
            {
                "reference_feats": reference_feats,
                "synthetic_feats": synthetic_feats,
                "reference_statistics": (reference_mean, reference_covariance),
                "num_points": fid_infinity_points,
                "min_samples": dimension + 1,
                "seed": seed,
            },
        ),
        (
            "rmt_fid_from_feats",
            rmt_frechet_distance_analytical,
            {
                "mu1": reference_mean,
                "sigma1": reference_covariance,
                "mu2": synthetic_mean,
                "sigma2": synthetic_covariance,
                "n": n_samples,
            },
        ),
    )

    results = []
    for name, function, kwargs in estimators:
        average_seconds, median_seconds, minimum_seconds, maximum_seconds, std_seconds, timings = benchmark_call(
            repeats=repeats,
            warmup=warmup,
            function=function,
            **kwargs,
        )
        peak_memory_bytes = measure_peak_memory(
            function=function,
            **kwargs,
        )
        results.append(
            {
                "dimension": dimension,
                "n_samples": n_samples,
                "estimator": name,
                "average_seconds": average_seconds,
                "median_seconds": median_seconds,
                "minimum_seconds": minimum_seconds,
                "maximum_seconds": maximum_seconds,
                "std_seconds": std_seconds,
                "timings_seconds": timings,
                "peak_memory_bytes": peak_memory_bytes,
            }
        )

    del reference_feats, synthetic_feats
    gc.collect()
    return results


def save_boxplot(results: list[dict], output_path: Path) -> None:
    """Save a boxplot of individual benchmark timings."""
    import matplotlib.pyplot as plt

    labels = [f"{result['estimator']}\np={result['dimension']}" for result in results]
    timings = [result["timings_seconds"] for result in results]

    figure_width = max(10, 0.7 * len(labels))
    figure, axis = plt.subplots(figsize=(figure_width, 6))
    axis.boxplot(timings, tick_labels=labels)
    axis.set_ylabel("Runtime (seconds)")
    axis.set_title("FID estimator benchmark")
    axis.tick_params(axis="x", labelrotation=45)
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


@click.command()
@click.option(
        "--dimensions",
        multiple=True,
        type=click.IntRange(1,),
        default=DEFAULT_DIMENSIONS,
        help="Feature dimensions to benchmark (default: 64 128 512 1024 2048).",
    )
@click.option(
        "--n-samples",
        type=click.IntRange(1,),
        default=50000,
        help="Number of samples to estimate the covariances.",
    )
@click.option("--repeats", type=click.IntRange(1,), default=3, show_default=True)
@click.option("--warmup", type=click.IntRange(0,), default=1, show_default=True)
@click.option("--fid-infinity-points", type=click.IntRange(2,), default=15, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option(
    "--output-json",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path("benchmark_results.json"),
    show_default=True,
    help="JSON file in which to store benchmark results.",
)
@click.option(
    "--boxplot",
    "boxplot_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Save a matplotlib boxplot to this path.",
)
def main(
    dimensions: tuple[int, ...],
    n_samples: int,
    repeats: int,
    warmup: int,
    fid_infinity_points: int,
    seed: int,
    output_json: Path,
    boxplot_path: Path | None,
) -> None:
    """Benchmark the NumPy FID estimators on synthetic feature arrays."""

    results = []
    print("dimension,n_samples,estimator,average_seconds,median_seconds,minimum_seconds,maximum_seconds,std_seconds,peak_memory_mib")
    for dimension in dimensions:
        if dimension <= 0:
            raise click.BadParameter("must be positive", param_hint="--dimensions")
        for result in benchmark_dimension(
            dimension=dimension,
            n_samples=n_samples,
            repeats=repeats,
            warmup=warmup,
            seed=seed + dimension,
            fid_infinity_points=fid_infinity_points,
        ):
            results.append(result)
            print(
                f"{result['dimension']},{result['n_samples']},{result['estimator']},"
                f"{result['average_seconds']:.6f},{result['median_seconds']:.6f},"
                f"{result['minimum_seconds']:.6f},"
                f"{result['maximum_seconds']:.6f},"
                f"{result['std_seconds']:.6e},"
                f"{float(result['peak_memory_bytes']) / 1024**2:.6f}"
            )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(
            {
                "dimensions": list(dimensions),
                "n_samples": n_samples,
                "repeats": repeats,
                "warmup": warmup,
                "fid_infinity_points": fid_infinity_points,
                "seed": seed,
                "results": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if boxplot_path is not None:
        save_boxplot(results, boxplot_path)


if __name__ == "__main__":
    main()