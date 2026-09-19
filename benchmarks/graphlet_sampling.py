"""Small reproducible benchmark for GraphletSampling."""
import argparse
import json
import platform
import statistics
import sys
import time

try:
    import resource
except ImportError:
    resource = None

import numpy as np
import scipy
import sklearn

from grakel.datasets import generate_dataset
from grakel.kernels import GraphletSampling


def peak_rss_mb():
    """Return the process peak resident set size in MiB."""
    if resource is None:
        return None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 ** 2)
    return rss / 1024


def run(args):
    exhaustive = args.mode == "exhaustive"
    graphs = args.graphs if not exhaustive else min(args.graphs, 8)
    test_graphs = min(args.test_graphs, graphs - 1)
    train, test = generate_dataset(
        n_graphs=graphs,
        r_vertices=(args.min_vertices, args.max_vertices),
        r_connectivity=(0.4, 0.8),
        r_weight_edges=(1, 1),
        n_graphs_test=test_graphs,
        random_state=123,
        features=None,
    )

    runs = []
    for _ in range(args.repeat):
        kernel = GraphletSampling(
            random_state=42,
            k=args.k,
            sampling=None if exhaustive else {"n_samples": args.samples},
        )

        start = time.perf_counter()
        fitted = kernel.fit_transform(train)
        fit_transform_seconds = time.perf_counter() - start

        start = time.perf_counter()
        transformed = kernel.transform(test)
        transform_seconds = time.perf_counter() - start

        runs.append({
            "fit_transform_seconds": fit_transform_seconds,
            "transform_seconds": transform_seconds,
            "bins": len(kernel._graph_bins),
            "fit_shape": list(fitted.shape),
            "transform_shape": list(transformed.shape),
            "fit_checksum": float(np.sum(fitted)),
            "transform_checksum": float(np.sum(transformed)),
        })

    return {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "mode": args.mode,
        "graphs": graphs,
        "test_graphs": test_graphs,
        "vertices": [args.min_vertices, args.max_vertices],
        "k": args.k,
        "samples": None if exhaustive else args.samples,
        "repeat": args.repeat,
        "runs": runs,
        "median_fit_transform_seconds": statistics.median(
            run["fit_transform_seconds"] for run in runs
        ),
        "median_transform_seconds": statistics.median(
            run["transform_seconds"] for run in runs
        ),
        "process_peak_rss_mib": peak_rss_mb(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("probabilistic", "exhaustive"),
                        default="probabilistic")
    parser.add_argument("--graphs", type=int, default=24)
    parser.add_argument("--test-graphs", type=int, default=6)
    parser.add_argument("--min-vertices", type=int, default=10)
    parser.add_argument("--max-vertices", type=int, default=20)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=3)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
