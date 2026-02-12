import argparse
import os
import runpy
import sys


def main() -> None:
    ap = argparse.ArgumentParser(description="Run PyIQA benchmark_results with your arguments")
    ap.add_argument("--metrics", required=True, type=str, help="Comma-separated metric names")
    ap.add_argument("--datasets", required=True, type=str, help="Comma-separated dataset names")
    ap.add_argument("--data-opt", type=str, default=None, help="Custom data option YAML")
    ap.add_argument("--metric-opt", type=str, default=None, help="Custom metric option YAML")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--split-file", type=str, default=None)
    ap.add_argument("--test-phase", type=str, default=None)
    ap.add_argument("--save-result-path", type=str, default=None)
    ap.add_argument("--use-gpu", action="store_true")
    args = ap.parse_args()

    argv = []
    argv += ["-m"] + [m.strip() for m in args.metrics.split(",") if m.strip()]
    argv += ["-d"] + [d.strip() for d in args.datasets.split(",") if d.strip()]
    if args.metric_opt:
        argv += ["--metric_opt", args.metric_opt]
    if args.data_opt:
        argv += ["--data_opt", args.data_opt]
    if args.batch_size is not None:
        argv += ["--batch_size", str(args.batch_size)]
    if args.split_file:
        argv += ["--split_file", args.split_file]
    if args.test_phase:
        argv += ["--test_phase", args.test_phase]
    if args.save_result_path:
        argv += ["--save_result_path", args.save_result_path]
    if args.use_gpu:
        argv += ["--use_gpu"]

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "IQA-PyTorch"))
    bench_path = os.path.join(repo_root, "benchmark_results.py")
    if not os.path.isfile(bench_path):
        raise FileNotFoundError(f"benchmark_results.py not found at {bench_path}")

    sys.argv = ["benchmark_results"] + argv
    runpy.run_path(bench_path, run_name="__main__")


if __name__ == "__main__":
    main()
