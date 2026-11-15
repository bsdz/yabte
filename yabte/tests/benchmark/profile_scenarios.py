"""Profile where time goes in each engine scenario.

Runs the scenarios from :mod:`benchmark_cpp_vs_py` (same data, same strategy) under
cProfile and prints the most expensive functions by cumulative time. Time spent inside
the C++ engine shows up under the extension's ``run`` call.

Usage (C++ extension on PYTHONPATH for a dev build)::

    PYTHONPATH=cpp/build/Release/pybind python -m yabte.tests.benchmark.profile_scenarios
    python -m yabte.tests.benchmark.profile_scenarios --assets 50 --years 5 --top 15 \\
        --scenarios python/numpy,cpp/numpy
"""

import argparse
import cProfile
import io
import pstats
import time

from yabte.backtest.runner_cpp import yabte_cpp
from yabte.tests.benchmark.benchmark_cpp_vs_py import (
    SCENARIOS,
    generate_data,
    run_scenario,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--assets", type=int, default=50)
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--top", type=int, default=20, help="functions to show")
    parser.add_argument(
        "--scenarios",
        default=",".join(f"{e}/{k}" for e, k in SCENARIOS),
        help="comma-separated engine/strategy pairs",
    )
    args = parser.parse_args()

    data, names = generate_data(args.assets, args.years)
    print(f"{args.assets} assets, {args.years} years ({len(data)} steps)")

    for label in args.scenarios.split(","):
        engine, kind = label.split("/")
        if engine == "cpp" and yabte_cpp is None:
            print(f"\n=== {label}: skipped, C++ extension not found")
            continue

        profiler = cProfile.Profile()
        start = time.perf_counter()
        profiler.enable()
        run_scenario(data, names, engine, kind)
        profiler.disable()
        elapsed = time.perf_counter() - start

        out = io.StringIO()
        pstats.Stats(profiler, stream=out).sort_stats("cumulative").print_stats(
            args.top
        )
        print(f"\n=== {label}: {elapsed:.3f}s (under cProfile)")
        print(out.getvalue())


if __name__ == "__main__":
    main()
