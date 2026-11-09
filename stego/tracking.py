"""Run folders that cannot be overwritten, and a dashboard rebuilt from them."""

from __future__ import annotations

import json
import platform
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_ITEMS = ("stego", "configs", "tests", "run.py", "requirements.txt")


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True, timeout=30
        )
        return result.stdout.rstrip("\n")  # the leading space of status lines is meaningful
    except (OSError, subprocess.SubprocessError):
        return None


def git_state() -> dict:
    commit = _git("rev-parse", "--short", "HEAD")
    status = _git("status", "--porcelain")
    if commit is None or status is None:
        return {"commit": "nogit", "dirty": True, "changes": []}
    changes = [line for line in status.splitlines() if not _is_output_path(line)]
    return {"commit": commit, "dirty": bool(changes), "changes": changes}


def _is_output_path(status_line: str) -> bool:
    path = status_line[3:].split(" -> ")[-1].strip().strip('"')
    return path.startswith(("experiments/", "cache/", "data/"))


def package_versions() -> dict:
    from importlib import metadata

    versions = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "scikit-learn", "scikit-image", "opencv-python-headless", "opencv-python",
                 "torch", "torchvision", "PyYAML"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return versions


def gpu_name() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=30,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def create_run_folder(runs_dir: Path, tag: str, commit: str, label: str = "") -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = f"__{label}" if label else ""
    folder = Path(runs_dir) / f"{tag}{suffix}__{stamp}__{commit}"
    # exist_ok=False is deliberate: a finished run must never be replaced.
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def write_source_archive(folder: Path) -> None:
    with tarfile.open(folder / "source.tar.gz", "w:gz") as archive:
        for item in SOURCE_ITEMS:
            path = REPO_ROOT / item
            if path.exists():
                archive.add(
                    path,
                    arcname=item,
                    filter=lambda info: None if "__pycache__" in info.name or ".pytest_cache" in info.name else info,
                )


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    def text(value) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "-"
        return str(value).replace("|", "\\|")

    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(text(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def load_summaries(runs_dir: Path) -> list[tuple[Path, dict]]:
    summaries = []
    for path in sorted(Path(runs_dir).glob("*/summary.json")):
        summaries.append((path.parent, json.loads(path.read_text(encoding="utf-8"))))
    return summaries


def build_dashboard(runs_dir: Path, output: Path) -> Path:
    """Rebuilds the dashboard. Every number comes from a run folder."""
    summaries = load_summaries(runs_dir)
    overview = []
    sections = []
    for folder, summary in summaries:
        config = summary["config"]
        capacity = summary["capacity_mean_images"]
        overview.append(
            {
                "Run": folder.name,
                "Status": "official" if summary["official"] else "NOT official",
                "Checks": "pass" if summary["checks_passed"] else "FAIL",
                "Dataset": config["dataset"],
                "n": config["n"],
                "M": config["M"],
                "No attack ACC %": summary["means"]["no_attack_acc"],
                "22 attacks ACC %": summary["means"]["mean_acc_22_attacks"],
                "Geometric %": summary["means"]["mean_acc_geometric"],
                "Non-geometric %": summary["means"]["mean_acc_non_geometric"],
                "Images, 100 B": capacity.get("100"),
                "Bits per image, 100 B": summary["capacity_bits_per_image"].get("100"),
                "Aux bits per image, tight": summary["aux_bits_tight"],
            }
        )
        attacks_path = folder / "attacks.csv"
        if attacks_path.is_file():
            attacks = pd.read_csv(attacks_path)
            attacks = attacks.rename(
                columns={
                    "attack": "Attack",
                    "acc_percent": "ACC %",
                    "ci95_low": "95% low",
                    "ci95_high": "95% high",
                    "retrieval_percent": "Retrieval %",
                    "paper_acc_percent": "Paper %",
                    "difference_from_paper": "Difference",
                }
            ).drop(columns=["segments"])
            reference = summary.get("paper_tables") or "no paper values for this setting"
            paper_images = summary.get("capacity_paper_images") or {}
            capacity_table = pd.DataFrame(
                [
                    {
                        "Message bytes": size,
                        "Images": images,
                        "Paper images": paper_images.get(size),
                        "Bits per image": summary["capacity_bits_per_image"].get(size),
                        "Label use %": summary["capacity_utilisation_percent"].get(size),
                        "Paper label use %": (summary.get("capacity_paper_utilisation_percent") or {}).get(size),
                    }
                    for size, images in capacity.items()
                ]
            )
            sections.append(
                f"### {folder.name}\n\nCompared with: {reference}\n\n"
                f"{_markdown_table(capacity_table)}\n\n{_markdown_table(attacks)}\n"
            )

    lines = [
        "# Results dashboard",
        "",
        "Generated by `python run.py dashboard` from `experiments/runs/`. Do not edit by hand.",
        "",
        "## Runs",
        "",
        _markdown_table(pd.DataFrame(overview)) if overview else "No runs yet.",
        "",
        "## Attack results",
        "",
        *sections,
    ]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output
