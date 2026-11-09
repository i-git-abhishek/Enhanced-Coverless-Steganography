"""Runs the Phase 2 ordering experiment on cached detections and saves it like a baseline run."""

from __future__ import annotations

import json
import math
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import cache, tracking
from .data import discover_images
from .mapping import build_index
from .ordering import INFINITY, SCHEMES_FIXED, build_lookup, canonical_sequences, order_array_bits, run_scheme

DEFAULT_LAMBDAS = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, INFINITY)


def scheme_name(scheme: str, lam: float) -> str:
    if scheme != "C":
        return scheme
    return "C (lambda=inf)" if math.isinf(lam) else f"C (lambda={lam:g})"


def _scheme_rng(seed: int, length: int, trial: int, scheme: str, lam: float) -> np.random.Generator:
    tag = zlib.crc32(f"{scheme}|{lam}".encode("utf-8"))
    return np.random.default_rng([seed, 4, length, trial, tag])


def run_messages(config, index, fixed_ltr, area, lookups, lambdas, trials, log=print) -> pd.DataFrame:
    """Every scheme hides the same messages. paper-greedy repeats the baseline's capacity search exactly."""
    cluster_bits = math.log2(config.k) if config.k > 1 else 0.0
    rows = []
    for length in config.capacity_bytes:
        for trial in range(trials):
            message_rng = np.random.default_rng([config.seed, 1, length, trial])
            message = message_rng.bytes(length)
            runs = [(scheme, INFINITY) for scheme in SCHEMES_FIXED] + [("C", lam) for lam in lambdas]
            for scheme, lam in runs:
                # paper-greedy continues the baseline's random stream, so its images match capacity.csv.
                rng = message_rng if scheme == "paper-greedy" else _scheme_rng(config.seed, length, trial, scheme, lam)
                outcome = run_scheme(message, scheme, lam, index, fixed_ltr, area, lookups, rng)
                message_bits = 8 * length
                row = {
                    "message_bytes": length,
                    "trial": trial,
                    "scheme": scheme_name(scheme, lam),
                    "base_scheme": scheme,
                    "lambda": lam,
                    "complete": outcome.complete,
                    "round_trip": outcome.round_trip,
                    "images": outcome.images,
                    "segments": outcome.segments,
                    "repeated_images": outcome.repeated_images,
                    "order_bits": round(outcome.order_bits, 4),
                    "header_bits": outcome.header_bits,
                    "cluster_bits": round(outcome.images * cluster_bits, 4),
                    "fixed_order_percent": outcome.fixed_share,
                }
                if outcome.complete and outcome.images:
                    extra_no_cluster = outcome.order_bits + outcome.header_bits
                    row.update(
                        {
                            "segments_per_image": outcome.segments / outcome.images,
                            "message_bits_per_image": message_bits / outcome.images,
                            "order_bits_per_image": outcome.order_bits / outcome.images,
                            "net_bits_per_image_without_cluster": (message_bits - extra_no_cluster) / outcome.images,
                            "net_bits_per_image_with_cluster": (message_bits - extra_no_cluster) / outcome.images
                            - cluster_bits,
                        }
                    )
                rows.append(row)
        log(f"[ordering] {length} B: {trials} messages done")
    return pd.DataFrame(rows)


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    order = {name: position for position, name in enumerate(dict.fromkeys(frame["scheme"]))}
    grouped = frame.groupby(["message_bytes", "scheme"], sort=False)
    summary = grouped.agg(
        messages=("trial", "count"),
        complete_percent=("complete", lambda s: 100.0 * s.mean()),
        images=("images", "mean"),
        segments_per_image=("segments_per_image", "mean"),
        message_bits_per_image=("message_bits_per_image", "mean"),
        order_bits_per_image=("order_bits_per_image", "mean"),
        net_bits_per_image_without_cluster=("net_bits_per_image_without_cluster", "mean"),
        net_bits_per_image_with_cluster=("net_bits_per_image_with_cluster", "mean"),
        fixed_order_percent=("fixed_order_percent", "mean"),
        repeated_images=("repeated_images", "mean"),
    ).reset_index()
    summary["_order"] = summary["scheme"].map(order)
    summary = summary.sort_values(["message_bytes", "_order"]).drop(columns="_order")
    return summary.round(3)


def evaluate_checks(frame: pd.DataFrame, lambdas) -> dict:
    results = {}

    def add(name, passed, detail):
        results[name] = {"passed": bool(passed), "detail": detail}

    done = frame[frame["complete"]]
    add("round_trip_exact", bool(done["round_trip"].all()) and len(done) > 0,
        f"{int(done['round_trip'].sum())}/{len(done)} hidden messages decode exactly")
    fixed_only = frame[frame["base_scheme"].isin(["A", "A-area"])]
    add("fixed_order_sends_no_order_bits", bool((fixed_only["order_bits"] == 0).all()),
        "schemes A and A-area send 0 order bits")
    keyed = frame.set_index(["message_bytes", "trial", "scheme"])
    pairs = [(b, t) for b, t in keyed.index.droplevel(2).unique()]

    def images(b, t, s):
        row = keyed.loc[(b, t, s)]
        return int(row["images"]) if bool(row["complete"]) else math.inf

    greedy_ok = all(images(b, t, "paper-dp") <= images(b, t, "paper-greedy") for b, t in pairs)
    add("dp_not_worse_than_greedy", greedy_ok, "paper-dp never needs more images than the paper's own search")
    c_inf = scheme_name("C", INFINITY)
    if math.inf in [float(x) for x in lambdas]:
        best = all(images(b, t, c_inf) <= min(images(b, t, "A"), images(b, t, "B")) for b, t in pairs)
        add("c_fewest_images", best, "C at lambda=inf never needs more images than A or B")
    ordered = sorted(float(x) for x in lambdas)
    monotone = True
    for b, t in pairs:
        counts = [images(b, t, scheme_name("C", lam)) for lam in ordered]
        monotone &= all(later <= earlier for earlier, later in zip(counts, counts[1:]))
    add("lambda_trade_off_monotone", monotone, "a higher image price never uses more images")
    return results


def build_ordering_dashboard(runs_dir: Path, output: Path) -> Path:
    sections = []
    for folder in sorted(Path(runs_dir).glob("*/ordering_summary.csv")):
        run = folder.parent
        summary_json = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        table = pd.read_csv(folder)
        table = table[table["message_bytes"] == table["message_bytes"].max()]
        table = table[["scheme", "complete_percent", "images", "segments_per_image", "order_bits_per_image",
                       "net_bits_per_image_without_cluster", "net_bits_per_image_with_cluster", "fixed_order_percent"]]
        table.columns = ["Scheme", "Hidden %", "Images", "Segments per image", "Order bits per image",
                         "Net bits per image (no cluster ID)", "Net bits per image (with cluster ID)", "Fixed order %"]
        status = "official" if summary_json["official"] else "NOT official"
        checks = "pass" if summary_json["checks_passed"] else "FAIL"
        sections.append(
            f"### {run.name}\n\nStatus: {status}. Checks: {checks}. "
            f"Message: {int(summary_json['largest_message_bytes'])} B. "
            f"Bound log2(usable images) = {summary_json['bound_bits_per_image']} bits per image. "
            f"Groups with no single-object image: {len(summary_json['groups_without_single_object_image'])}.\n\n"
            f"{tracking._markdown_table(table)}\n"
        )
    lines = [
        "# Ordering dashboard (Phase 2)",
        "",
        "Generated from `experiments/ordering_runs/`. Do not edit by hand.",
        "Net bits = message bits minus order bits minus the 16-bit length header, per image.",
        "",
        *(sections or ["No ordering runs yet."]),
    ]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output


def run_ordering(config, paths, lambdas=DEFAULT_LAMBDAS, trials: int = 20, label: str = "",
                 allow_dirty: bool = False, runs_dir: Path | None = None, log=print) -> Path:
    git = tracking.git_state()
    if git["dirty"] and not allow_dirty:
        raise RuntimeError(
            "The code has uncommitted changes, so this run could not be traced to a commit. "
            "Commit your change first, or add --allow-dirty for a test run that will be marked NOT official.\n"
            + "\n".join(git["changes"][:20])
        )
    started = time.time()
    lambdas = tuple(float(x) for x in lambdas)
    ids, _ = discover_images(config, paths.data_root)
    extraction = cache.load_extraction(ids, config.dataset, paths.cache_dir, paths.read_dirs)
    # Clusters are not needed: ordering does not change retrieval. Cluster ID bits are log2(k).
    index = build_index(ids, extraction.objects, np.zeros(len(ids), dtype=np.int32), config.n, config.M,
                        config.conf_threshold)
    fixed_ltr = canonical_sequences(index, extraction.objects, config.conf_threshold, "left_to_right")
    area = list(index.sequences)
    lookups = {"left_to_right": build_lookup(fixed_ltr, area), "area": build_lookup(area, area)}
    usable = index.usable_images
    # Scheme A can only carry a lone segment g with an image whose only object maps to g.
    missing_single = [g for g in range(index.groups) if (g,) not in lookups["left_to_right"].exact]
    log(f"[ordering] {config.tag}: {len(ids)} images, {usable} with objects, n={config.n}, M={config.M}")
    log(f"[ordering] Groups with no single-object image: {len(missing_single)} of {index.groups}")

    frame = run_messages(config, index, fixed_ltr, area, lookups, lambdas, trials, log)
    summary = summarise(frame)
    checks = evaluate_checks(frame, lambdas)
    checks_passed = all(item["passed"] for item in checks.values())
    official = not git["dirty"] and checks_passed

    runs_dir = Path(runs_dir) if runs_dir else Path(paths.runs_dir).parent / "ordering_runs"
    folder = tracking.create_run_folder(runs_dir, config.tag + "__ordering", git["commit"], label)
    tracking.write_source_archive(folder)
    tracking.write_json(folder / "config.json", {**config.to_dict(), "lambdas": list(lambdas), "trials": trials})
    tracking.write_json(folder / "environment.json", {"git": git, "packages": tracking.package_versions(),
                                                       "extraction": extraction.meta})
    tracking.write_json(folder / "checks.json", checks)
    frame.to_csv(folder / "ordering_messages.csv", index=False)
    summary.to_csv(folder / "ordering_summary.csv", index=False)
    tracking.write_json(
        folder / "summary.json",
        {
            "config": config.to_dict(),
            "official": official,
            "checks_passed": checks_passed,
            "git_commit": git["commit"],
            "images_total": len(ids),
            "images_usable": usable,
            "bound_bits_per_image": round(math.log2(usable), 3) if usable else 0.0,
            "order_array_bits": round(order_array_bits(config.M), 3),
            "cluster_bits": round(math.log2(config.k), 3) if config.k > 1 else 0.0,
            "largest_message_bytes": max(config.capacity_bytes),
            "groups_without_single_object_image": missing_single,
            "lambdas": [str(x) for x in lambdas],
            "trials": trials,
            "seconds": round(time.time() - started, 1),
        },
    )
    build_ordering_dashboard(runs_dir, runs_dir.parent / "ORDERING_DASHBOARD.md")

    log("")
    for name, item in checks.items():
        log(f"[check] {'PASS' if item['passed'] else 'FAIL'}  {name}: {item['detail']}")
    largest = summary[summary["message_bytes"] == summary["message_bytes"].max()]
    for _, row in largest.iterrows():
        log(f"[ordering] {row['scheme']:18s} images {row['images']:8.1f}  hidden {row['complete_percent']:5.1f}%  "
            f"net bits/image (no cluster ID) {row['net_bits_per_image_without_cluster']:6.2f}")
    log(f"[ordering] Status: {'OFFICIAL' if official else 'NOT OFFICIAL'}")
    log(f"[ordering] Saved: {folder}")
    return folder
