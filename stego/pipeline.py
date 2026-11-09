"""Runs one complete baseline experiment and saves it."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import cache, paper_refs, tracking
from .attacks import NO_ATTACK, TABLE_IX_ORDER
from .data import discover_images, expected_count
from .evaluate import (
    attack_group_means,
    run_capacity,
    run_robustness,
    summarise_robustness,
)
from .mapping import ImageIndex, build_index, label_image_counts, run_clustering, segment_to_bits
from .recovery import Retriever


@dataclass
class Paths:
    data_root: Path
    cache_dir: Path
    runs_dir: Path
    read_dirs: list[Path]


def resolve_paths(data_root: str | None, cache_dir: str, runs_dir: str, cache_read: list[str] | None) -> Paths:
    kaggle_input = Path("/kaggle/input")
    if data_root:
        root = Path(data_root)
    else:
        root = kaggle_input if kaggle_input.is_dir() else Path("data")
    read_dirs = [Path(item) for item in (cache_read or [])] + cache.default_read_dirs()
    return Paths(root, Path(cache_dir), Path(runs_dir), read_dirs)


def load_clusters_or_build(config, extraction, paths: Paths, log=print) -> np.ndarray:
    key = cache.cluster_key(extraction.key, config)
    clusters = cache.load_clusters(key, paths.cache_dir, paths.read_dirs)
    if clusters is not None:
        log(f"[clusters] Using saved clusters: {key}")
        return clusters
    log(f"[clusters] Clustering {len(extraction.ids)} images into {config.k} groups. This can take a while.")
    started = time.time()
    clusters = run_clustering(config, extraction.features)
    cache.save_clusters(key, paths.cache_dir, clusters, {"k": config.k, "seconds": round(time.time() - started, 1)})
    log(f"[clusters] Finished in {time.time() - started:.0f} s")
    return clusters


def dictionary_frame(index: ImageIndex, label_counts: dict[str, int]):
    import pandas as pd

    rows = [
        {
            "object_label": label,
            "group": group,
            "binary_sequence": segment_to_bits(group, index.n),
            "images_with_label": label_counts[label],
        }
        for label, group in sorted(index.dictionary.items())
    ]
    return pd.DataFrame(rows)


def evaluate_checks(config, index, label_counts, capacity, capacity_checks, summary, images_expected) -> dict:
    results = {}

    def add(name: str, passed: bool, detail: str) -> None:
        results[name] = {"passed": bool(passed), "detail": detail}

    expected_groups = set(range(2**config.n))
    add(
        "dictionary_complete",
        set(index.dictionary.values()) == expected_groups and set(index.dictionary) == set(label_counts),
        f"{len(index.dictionary)} labels in {2**config.n} groups",
    )
    ordered = [index.dictionary[label] for label in sorted(index.dictionary)]
    add("dictionary_alphabetical", ordered == sorted(ordered), "groups rise in alphabetical label order")
    add("index_complete", bool(capacity["complete"].all()), f"{capacity['complete'].sum()}/{len(capacity)} messages fully hidden")
    add(
        "bytes_round_trip",
        capacity_checks["round_trips_ok"] == capacity_checks["round_trips_total"] > 0,
        f"{capacity_checks['round_trips_ok']}/{capacity_checks['round_trips_total']} exact",
    )
    add(
        "bytes_round_trip_with_retrieval",
        capacity_checks["retrieval_round_trips_ok"] == capacity_checks["retrieval_round_trips_total"] > 0,
        f"{capacity_checks['retrieval_round_trips_ok']}/{capacity_checks['retrieval_round_trips_total']} exact",
    )
    table = summary.set_index("attack")
    add(
        "all_attacks_present",
        list(summary["attack"]) == [NO_ATTACK, *ordered_attacks()],
        f"{len(summary) - 1} attacks plus no-attack control",
    )
    no_attack_acc = float(table.loc[NO_ATTACK, "acc_percent"])
    no_attack_retrieval = float(table.loc[NO_ATTACK, "retrieval_percent"])
    add("no_attack_acc_100", no_attack_acc == 100.0, f"ACC {no_attack_acc}%")
    add("no_attack_retrieval_100", no_attack_retrieval == 100.0, f"retrieval {no_attack_retrieval}%")
    add("image_count", images_expected == len(index.ids), f"{len(index.ids)} images used, {images_expected} expected")
    return results


def ordered_attacks() -> list[str]:
    """Table IX order, which is also the order attacks are run in."""
    return list(TABLE_IX_ORDER)


def run_baseline(config, paths: Paths, label: str = "", allow_dirty: bool = False, log=print) -> Path:
    git = tracking.git_state()
    if git["dirty"] and not allow_dirty:
        raise RuntimeError(
            "The code has uncommitted changes, so this run could not be traced to a commit. "
            "Commit your change first, or add --allow-dirty for a test run that will be marked NOT official.\n"
            + "\n".join(git["changes"][:20])
        )

    started = time.time()
    log(f"[run] {config.tag}: finding images in {paths.data_root}")
    ids, image_paths = discover_images(config, paths.data_root)
    extraction = cache.load_extraction(ids, config.dataset, paths.cache_dir, paths.read_dirs)
    clusters = load_clusters_or_build(config, extraction, paths, log)
    counts = label_image_counts(extraction.objects, config.conf_threshold, config.M)
    index = build_index(ids, extraction.objects, clusters, config.n, config.M, config.conf_threshold)
    log(
        f"[run] {len(ids)} images, {index.usable_images} with objects, {len(counts)} labels, "
        f"{len(index.cluster_members)} clusters"
    )
    retriever = Retriever(index, image_paths, config.ssim_mode, config.ssim_size)

    log("[run] Capacity experiment")
    capacity, capacity_checks = run_capacity(config, index, retriever, image_paths)
    log("[run] Robustness experiment")
    segments = run_robustness(config, index, retriever, image_paths, log)

    reference = paper_refs.lookup(config.dataset, config.n, config.M)
    paper_tables, paper_capacity, paper_robustness = reference if reference else (None, None, None)
    summary = summarise_robustness(segments, config, paper_robustness)
    checks = evaluate_checks(config, index, counts, capacity, capacity_checks, summary, expected_count(config, paths.data_root))
    checks_passed = all(item["passed"] for item in checks.values())

    means = attack_group_means(summary)
    complete = capacity[capacity["complete"]]
    group = complete.groupby("message_bytes")
    mean_images = {str(key): round(float(value), 2) for key, value in group["images"].mean().items()}
    mean_bits = {str(key): round(float(value), 2) for key, value in group["message_bits_per_image"].mean().items()}
    mean_utilisation = {str(key): round(float(value), 2) for key, value in group["utilisation_percent"].mean().items()}
    sizes = np.array([len(members) for members in index.cluster_members.values()])
    official = not git["dirty"] and checks_passed

    run_folder = tracking.create_run_folder(paths.runs_dir, config.tag, git["commit"], label)
    tracking.write_source_archive(run_folder)
    tracking.write_json(run_folder / "config.json", config.to_dict())
    tracking.write_json(
        run_folder / "environment.json",
        {"git": git, "packages": tracking.package_versions(), "gpu": tracking.gpu_name(), "extraction": extraction.meta},
    )
    tracking.write_json(run_folder / "checks.json", checks)
    capacity.to_csv(run_folder / "capacity.csv", index=False)
    summary.to_csv(run_folder / "attacks.csv", index=False)
    segments.to_csv(run_folder / "segments.csv.gz", index=False)
    dictionary_frame(index, counts).to_csv(run_folder / "dictionary.csv", index=False)
    tracking.write_json(
        run_folder / "summary.json",
        {
            "config": config.to_dict(),
            "official": official,
            "checks_passed": checks_passed,
            "git_commit": git["commit"],
            "paper_tables": paper_tables,
            "means": means,
            "capacity_mean_images": mean_images,
            "capacity_paper_images": {str(k): v for k, v in (paper_capacity or {}).items()},
            "capacity_bits_per_image": mean_bits,
            "capacity_utilisation_percent": mean_utilisation,
            "capacity_paper_utilisation_percent": {
                str(k): v for k, v in (paper_refs.lookup_utilisation(config.dataset, config.n, config.M) or {}).items()
            },
            "aux_bits_fixed": float(capacity["aux_bits_fixed"].iloc[0]),
            "aux_bits_tight": round(float(capacity["aux_bits_tight"].iloc[0]), 2),
            "images_total": len(ids),
            "images_with_objects": index.usable_images,
            "labels_detected": len(counts),
            "cluster_size_mean": round(float(sizes.mean()), 2),
            "cluster_size_max": int(sizes.max()),
            "seconds": round(time.time() - started, 1),
        },
    )
    tracking.build_dashboard(paths.runs_dir, paths.runs_dir.parent / "RESULTS_DASHBOARD.md")

    log("")
    for name, item in checks.items():
        log(f"[check] {'PASS' if item['passed'] else 'FAIL'}  {name}: {item['detail']}")
    log(f"[run] Mean images: {mean_images}  paper: {paper_capacity}")
    log(f"[run] Mean ACC over 22 attacks: {means['mean_acc_22_attacks']}%")
    log(f"[run] Status: {'OFFICIAL' if official else 'NOT OFFICIAL'}")
    log(f"[run] Saved: {run_folder}")
    return run_folder
