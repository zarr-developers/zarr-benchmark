# Zarr Benchmark

Benchmark the performance of Zarr implementations.

This repo defines Zarr-specific workloads and datasets. These are run by our general-purpose,
IO-centric benchmarking framework, [`perfcapture`](https://github.com/zarr-developers/perfcapture).

# Code maturity

This project started in early October 2023, and is currently in the planning stage!
This code does not run yet.

# Usage

1. Install [`perfcapture`](https://github.com/zarr-developers/perfcapture).
2. `pip install zarr`
3. Clone the `zarr-benchmark` GitHub repository.
4. Run `perfcapture/scripts/cli.py` with `--recipe-path` set to your `zarr-benchmark/recipes` path.
