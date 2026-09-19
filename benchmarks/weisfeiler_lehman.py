"""Small reproducible benchmark for the WeisfeilerLehman subtree kernel.

Runs the same kernel through the native relabelling extension and through the
legacy pure-Python relabelling loops (selected with the ``_NATIVE`` module
flag) in the same process, alternating order between repeats. Full Gram
matrices are compared outside the timed sections at every repeat.
"""
import argparse
import gc
import json
import platform
import statistics
import sys
import time
from unittest.mock import patch

try:
    import resource
except ImportError:
    resource = None

import numpy as np
import scipy
import sklearn

import grakel.kernels.weisfeiler_lehman as wl_module
from grakel.datasets import generate_dataset
from grakel.kernels import WeisfeilerLehman


def peak_rss_mb():
    """Return the process peak resident set size in MiB."""
    if resource is None:
        return None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 ** 2)
    return rss / 1024


def run(args):
    native_available = bool(wl_module._NATIVE)
    if not native_available:
        raise RuntimeError("Build the native WL extension before benchmarking")
    train, test = generate_dataset(
        n_graphs=args.graphs,
        r_vertices=(args.min_vertices, args.max_vertices),
        r_connectivity=(0.4, 0.8),
        r_weight_edges=(1, 1),
        n_graphs_test=args.test_graphs,
        random_state=123,
        features=("nl", 3),
    )

    rows = []
    for h in args.heights:
        samples = {"native": [], "legacy": []}
        for rep in range(args.warmup + args.repeat):
            outputs = {}
            order = ["native", "legacy"] if rep % 2 == 0 else ["legacy", "native"]
            for name in order:
                with patch.object(wl_module, "_NATIVE", name == "native"):
                    gc.collect()
                    start = time.perf_counter()
                    kernel = WeisfeilerLehman(n_iter=h, normalize=args.normalize)
                    K = kernel.fit_transform(train)
                    fit_transform_seconds = time.perf_counter() - start
                    start = time.perf_counter()
                    Kt = kernel.transform(test)
                    transform_seconds = time.perf_counter() - start
                outputs[name] = (K, Kt)
                if rep >= args.warmup:
                    samples[name].append(
                        (fit_transform_seconds, transform_seconds))
            for native, legacy in zip(outputs["native"], outputs["legacy"]):
                if args.normalize:
                    np.testing.assert_allclose(native, legacy, rtol=1e-13,
                                               atol=1e-13, equal_nan=False)
                else:
                    np.testing.assert_array_equal(native, legacy)
        medians = {
            name: {
                "fit_transform_seconds": statistics.median(
                    run_[0] for run_ in samples[name]),
                "transform_seconds": statistics.median(
                    run_[1] for run_ in samples[name]),
            }
            for name in samples
        }
        rows.append({
            "h": h,
            "checksums": [float(np.sum(K)) for K in outputs["native"]],
            "median_seconds": medians,
            "speedup": {
                "fit_transform": medians["legacy"]["fit_transform_seconds"]
                / medians["native"]["fit_transform_seconds"],
                "transform": medians["legacy"]["transform_seconds"]
                / medians["native"]["transform_seconds"],
            },
        })

    return {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "native_extension": native_available,
        },
        "graphs": args.graphs,
        "test_graphs": args.test_graphs,
        "vertices": [args.min_vertices, args.max_vertices],
        "normalize": args.normalize,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "rows": rows,
        "process_peak_rss_mib": peak_rss_mb(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", type=int, default=200)
    parser.add_argument("--test-graphs", type=int, default=40)
    parser.add_argument("--min-vertices", type=int, default=10)
    parser.add_argument("--max-vertices", type=int, default=20)
    parser.add_argument("--heights", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--normalize", action="store_true")
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
