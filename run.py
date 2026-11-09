#!/usr/bin/env python3
"""Command line for the baseline. See MANUAL.md for step-by-step instructions."""

from __future__ import annotations

import argparse
import sys
import tarfile
import tempfile
from dataclasses import replace
from pathlib import Path

from stego import cache
from stego.config import ConfigError, build_config, load_config
from stego.data import DataError, discover_images
from stego.mapping import MappingError
from stego.pipeline import resolve_paths, run_baseline
from stego.tracking import build_dashboard

REPO_ROOT = Path(__file__).resolve().parent


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Path to a YAML file in configs/")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Change a setting for this run")
    parser.add_argument("--data-root", help="Folder containing the dataset. Default: /kaggle/input on Kaggle, else data/")
    parser.add_argument("--cache-dir", default=str(REPO_ROOT / "cache"), help="Where new caches are written")
    parser.add_argument("--cache-read", action="append", default=[], help="Extra folder to search for a finished cache")
    parser.add_argument("--runs-dir", default=str(REPO_ROOT / "experiments" / "runs"), help="Where runs are saved")


def paths_from(args):
    return resolve_paths(args.data_root, args.cache_dir, args.runs_dir, args.cache_read)


def command_check_data(args) -> int:
    config = load_config(args.config, args.set)
    paths = paths_from(args)
    ids, _ = discover_images(config, paths.data_root)
    print(f"PASS: found all {len(ids)} images for {config.dataset} under {paths.data_root}")
    return 0


def command_extract(args) -> int:
    from stego.extract import run_extraction

    config = load_config(args.config, args.set)
    paths = paths_from(args)
    ids, image_paths = discover_images(config, paths.data_root)
    print(f"[extract] {len(ids)} images")
    run_extraction(config, ids, image_paths, paths.cache_dir, paths.read_dirs, args.batch_size, args.limit)
    return 0


def command_verify(args) -> int:
    from stego.extract import verify_detections

    config = load_config(args.config, args.set)
    paths = paths_from(args)
    ids, image_paths = discover_images(config, paths.data_root)
    extraction = cache.load_extraction(ids, config.dataset, paths.cache_dir, paths.read_dirs)
    result = verify_detections(config, extraction, image_paths, args.sample, config.seed)
    return 0 if result["matching"] == result["sampled"] else 1


def command_baseline(args) -> int:
    config = load_config(args.config, args.set)
    run_baseline(config, paths_from(args), args.label, args.allow_dirty)
    return 0


def command_sweep(args) -> int:
    base = load_config(args.config, args.set)
    if args.param not in ("n", "M", "seed", "k", "conf_threshold", "ssim_mode"):
        raise ConfigError(f"Cannot sweep {args.param!r}")
    failures = []
    for value in args.values:
        values = base.to_dict()
        values[args.param] = value
        config = build_config(values)
        print(f"\n===== {args.param} = {value} =====")
        try:
            run_baseline(config, paths_from(args), args.label, args.allow_dirty)
        except (MappingError, RuntimeError) as error:
            failures.append(f"{args.param}={value}: {error}")
            print(f"[sweep] FAILED {args.param}={value}: {error}")
    if failures:
        print("\n[sweep] Some settings failed:\n" + "\n".join(failures))
        return 1
    print("\n[sweep] All settings finished")
    return 0


def command_dashboard(args) -> int:
    runs = Path(args.runs_dir)
    output = build_dashboard(runs, runs.parent / "RESULTS_DASHBOARD.md")
    print(f"Dashboard written: {output}")
    return 0


def command_bundle(args) -> int:
    runs = Path(args.runs_dir)
    experiments = runs.parent
    build_dashboard(runs, experiments / "RESULTS_DASHBOARD.md")
    output = Path(args.output)
    with tarfile.open(output, "w:gz") as archive:
        archive.add(experiments, arcname="experiments")
    print(f"Results bundle written: {output}")
    return 0


def command_smoke(args) -> int:
    from stego.synthetic import create_synthetic, write_synthetic_cache

    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        ids, data_root = create_synthetic(root / "data")
        write_synthetic_cache(ids, root / "cache")
        base = {
            "dataset": "synthetic",
            "k": 8,
            "capacity_bytes": [10, 100],
            "capacity_trials": 2,
            "robustness_trials": 2,
            "robustness_segments": 8,
            "bootstrap_samples": 100,
        }
        settings = [{"n": 6, "M": 4}, {"n": 3, "M": 4}, {"n": 6, "M": 6}]
        paths = resolve_paths(str(data_root), str(root / "cache"), str(root / "experiments" / "runs"), [])
        paths.read_dirs = []
        failed = []
        for setting in settings:
            config = build_config({**base, **setting})
            run_folder = run_baseline(config, paths, allow_dirty=True, log=lambda message: None)
            import json

            checks = json.loads((run_folder / "checks.json").read_text(encoding="utf-8"))
            failed += [f"{config.tag}: {name}" for name, item in checks.items() if not item["passed"]]
            print(f"Checked {config.tag}")
    if failed:
        print("SMOKE TEST FAILED:\n" + "\n".join(failed))
        return 1
    print("SMOKE TEST PASSED")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    sub = commands.add_parser("smoke", help="Test the pipeline on made-up data. No GPU or dataset needed")
    sub.set_defaults(func=command_smoke)

    sub = commands.add_parser("check-data", help="Check that the dataset is complete")
    add_common(sub)
    sub.set_defaults(func=command_check_data)

    sub = commands.add_parser("extract", help="GPU pass: run the detector and ResNet-101 on every image")
    add_common(sub)
    sub.add_argument("--batch-size", type=int, default=1, help="Keep at 1: batching pads images and can change detections")
    sub.add_argument("--limit", type=int, default=0, help="Preview this many images and save nothing")
    sub.set_defaults(func=command_extract)

    sub = commands.add_parser("verify-detections", help="Re-run the detector on a sample and compare with the cache")
    add_common(sub)
    sub.add_argument("--sample", type=int, default=200)
    sub.set_defaults(func=command_verify)

    sub = commands.add_parser("baseline", help="Run capacity and robustness once and save the results")
    add_common(sub)
    sub.add_argument("--label", default="", help="Short name added to the run folder")
    sub.add_argument("--allow-dirty", action="store_true", help="Allow uncommitted code. The run is NOT official")
    sub.set_defaults(func=command_baseline)

    sub = commands.add_parser("sweep", help="Run the baseline for several values of one setting")
    add_common(sub)
    sub.add_argument("--param", required=True, help="Setting to change: n, M, seed, k, conf_threshold or ssim_mode")
    sub.add_argument("--values", required=True, nargs="+", help="Values to run, such as 3 4 5 6")
    sub.add_argument("--label", default="")
    sub.add_argument("--allow-dirty", action="store_true")
    sub.set_defaults(func=command_sweep)

    sub = commands.add_parser("dashboard", help="Rebuild experiments/RESULTS_DASHBOARD.md")
    sub.add_argument("--runs-dir", default=str(REPO_ROOT / "experiments" / "runs"))
    sub.set_defaults(func=command_dashboard)

    sub = commands.add_parser("bundle", help="Pack the results into one file for download")
    sub.add_argument("--runs-dir", default=str(REPO_ROOT / "experiments" / "runs"))
    sub.add_argument("--output", default="results_bundle.tar.gz")
    sub.set_defaults(func=command_bundle)

    args = parser.parse_args()
    try:
        return args.func(args)
    except ModuleNotFoundError as error:
        if error.name in ("torch", "torchvision"):
            print("ERROR: PyTorch is needed for this command. Run it on Kaggle, which includes PyTorch.", file=sys.stderr)
            return 2
        raise
    except (ConfigError, DataError, MappingError, cache.CacheError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
