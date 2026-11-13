"""Tests for Phase 3: descriptors, Fourier–Mellin alignment, new retrieval modes, retrievability check."""

import json

import cv2
import numpy as np
import pytest
import skimage.data

from stego.attacks import build_attacks, attack_rng
from stego.config import ConfigError, build_config
from stego.data import discover_images
from stego.descriptors import (
    DescriptorError,
    DescriptorModel,
    descriptor_key,
    load_descriptors,
    run_descriptor_extraction,
)
from stego.hiding import hide_segments
from stego.mapping import build_index
from stego.pipeline import resolve_paths, run_baseline
from stego.recovery import DescriptorRetriever, read_image
from stego.register import align, fit_gray
from stego.retrievability import run_check, sendable_images, usable_mask
from stego.synthetic import create_synthetic, write_synthetic_cache
from stego import cache


def _bgr(name):
    image = getattr(skimage.data, name)()
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------- alignment


@pytest.mark.parametrize("name", ["astronaut", "coffee", "chelsea"])
@pytest.mark.parametrize("angle", [10, 30, 50])
def test_alignment_recovers_rotation(name, angle):
    image = _bgr(name)
    attacked = build_attacks()[f"Rotation {angle}°"](image, None)
    result = align(fit_gray(image, 256), attacked, 256)
    assert abs(result.angle - angle) < 1.5
    assert abs(result.scale - 1.0) < 0.02
    assert result.score > 0.95


@pytest.mark.parametrize("attack", ["No attack", "Translation (80, 50)", "Edge Cropping 20%", "Center Cropping 20%",
                                    "Scaling", "JPEG Compression", "Gauss Noise"])
def test_alignment_keeps_a_high_score_for_the_true_image(attack):
    image = _bgr("astronaut")
    attacked = build_attacks()[attack](image, attack_rng(0, attack, 0, 0))
    assert align(fit_gray(image, 256), attacked, 256).score > 0.95


def test_alignment_scores_a_different_image_low():
    attacked = build_attacks()["Rotation 30°"](_bgr("astronaut"), None)
    assert align(fit_gray(_bgr("coffee"), 256), attacked, 256).score < 0.6


# ---------------------------------------------------------------- shared synthetic data


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    root = tmp_path_factory.mktemp("phase3")
    ids, data_root = create_synthetic(root / "data")
    write_synthetic_cache(ids, root / "cache")
    config = build_config({"dataset": "synthetic", "descriptor": "pixels"})
    found, image_paths = discover_images(config, data_root)
    run_descriptor_extraction(config, found, image_paths, root / "cache", [], log=lambda message: None)
    return root, data_root, found, image_paths


def _config(**extra):
    return build_config({"dataset": "synthetic", "k": 8, "capacity_bytes": [10, 100], "capacity_trials": 2,
                         "robustness_trials": 1, "robustness_segments": 8, "bootstrap_samples": 50,
                         "descriptor": "pixels", **extra})


def _index(root, ids):
    extraction = cache.load_extraction(ids, "synthetic", root / "cache", [])
    return build_index(ids, extraction.objects, np.zeros(len(ids), dtype=np.int32), 6, 4, 0.9)


# ---------------------------------------------------------------- descriptors


def test_descriptor_cache_round_trip(synthetic):
    root, _, ids, image_paths = synthetic
    vectors = load_descriptors(ids, "synthetic", "pixels", root / "cache", [])
    assert vectors.shape == (len(ids), 256)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    fresh = DescriptorModel("pixels").describe([read_image(image_paths[ids[3]])])[0]
    assert np.allclose(fresh, vectors[3], atol=1e-5)
    # A second extraction reuses the finished cache.
    config = _config()
    again = run_descriptor_extraction(config, ids, image_paths, root / "cache", [], log=lambda message: None)
    assert again.name == descriptor_key("synthetic", ids, "pixels")


def test_descriptor_key_depends_on_model_and_images():
    assert descriptor_key("x", ["a", "b"], "pixels") != descriptor_key("x", ["a", "b"], "sscd_disc_mixup")
    assert descriptor_key("x", ["a", "b"], "pixels") != descriptor_key("x", ["a", "c"], "pixels")


def test_missing_descriptors_give_clear_error(synthetic, tmp_path):
    _, _, ids, _ = synthetic
    with pytest.raises(DescriptorError, match="run.py descriptors"):
        load_descriptors(ids, "synthetic", "pixels", tmp_path, [])


# ---------------------------------------------------------------- retrieval modes


def test_sscd_mode_finds_every_unattacked_image_and_ignores_the_cluster(synthetic):
    root, _, ids, image_paths = synthetic
    index = _index(root, ids)
    vectors = load_descriptors(ids, "synthetic", "pixels", root / "cache", [])
    retriever = DescriptorRetriever(index, image_paths, vectors, DescriptorModel("pixels"), "sscd", 10, 256)
    for image in range(0, len(ids), 37):
        found = retriever.retrieve(read_image(image_paths[ids[image]]), cluster=12345)
        assert found.image_index == image
        assert found.best_score > found.second_score


def test_search_set_is_respected(synthetic):
    root, _, ids, image_paths = synthetic
    index = _index(root, ids)
    vectors = load_descriptors(ids, "synthetic", "pixels", root / "cache", [])
    search = np.arange(100, 200)
    retriever = DescriptorRetriever(index, image_paths, vectors, DescriptorModel("pixels"), "sscd", 10, 256, search)
    assert retriever.retrieve(read_image(image_paths[ids[150]])).image_index == 150
    assert 100 <= retriever.retrieve(read_image(image_paths[ids[5]])).image_index < 200


def test_alignment_rescues_a_rotation_the_descriptor_misses(synthetic):
    root, _, ids, image_paths = synthetic
    index = _index(root, ids)
    vectors = load_descriptors(ids, "synthetic", "pixels", root / "cache", [])
    search = np.arange(60)
    model = DescriptorModel("pixels")
    plain = DescriptorRetriever(index, image_paths, vectors, model, "sscd", 60, 256, search)
    aligned = DescriptorRetriever(index, image_paths, vectors, model, "sscd_align", 60, 256, search)
    rotate = build_attacks()["Rotation 50°"]
    plain_hits = aligned_hits = 0
    for image in range(0, 60, 6):
        attacked = rotate(read_image(image_paths[ids[image]]), None)
        plain_hits += plain.retrieve(attacked).image_index == image
        aligned_hits += aligned.retrieve(attacked).image_index == image
    assert aligned_hits == 10
    assert aligned_hits > plain_hits


# ---------------------------------------------------------------- retrievability check


def test_usable_mask_threshold():
    hits = np.array([[1, 1, 1, 1], [1, 1, 0, 0], [0, 0, 0, 0], [1, 1, 1, 0]], dtype=bool)
    sendable = np.array([True, True, True, False])
    assert usable_mask(hits, sendable, 0.0).tolist() == [True, True, True, False]
    assert usable_mask(hits, sendable, 0.5).tolist() == [True, True, False, False]
    assert usable_mask(hits, sendable, 1.0).tolist() == [True, False, False, False]


def test_check_runs_caches_and_limits_the_sender(synthetic, tmp_path):
    root, _, all_ids, image_paths = synthetic
    ids = all_ids[:80]
    config = _config()
    # A small database of its own: descriptors for just these images.
    extraction = cache.load_extraction(all_ids, "synthetic", root / "cache", [])
    index = build_index(ids, extraction.objects[:80], np.zeros(80, dtype=np.int32), 2, 4, 0.9)
    model = DescriptorModel("pixels")
    vectors = model.describe([read_image(image_paths[i]) for i in ids])
    sendable = sendable_images(index)
    hits = run_check(config, ids, image_paths, vectors, model, sendable, tmp_path, [], log=lambda message: None)
    assert hits.shape == (80, 22)
    assert not hits[~sendable].any()
    assert hits[sendable].any()
    again = run_check(config, ids, image_paths, vectors, model, sendable, tmp_path, [], log=lambda message: None)
    assert (again == hits).all()

    usable = usable_mask(hits, sendable, 0.5)
    index.counts[~usable] = 0
    rng = np.random.default_rng(0)
    groups_left = np.flatnonzero(index.counts.sum(axis=0) > 0)
    for _ in range(20):
        segments = rng.choice(groups_left, size=12).tolist()
        result = hide_segments(segments, index, rng)
        assert result.complete
        assert all(usable[item.image_index] for item in result.stego)


# ---------------------------------------------------------------- whole runs


def test_config_checks_phase3_settings():
    with pytest.raises(ConfigError):
        build_config({"dataset": "voc2007", "retrieval": "nearest"})
    with pytest.raises(ConfigError):
        build_config({"dataset": "voc2007", "usable_threshold": 0.9})  # needs SSCD retrieval
    with pytest.raises(ConfigError):
        build_config({"dataset": "voc2007", "retrieval": "sscd", "usable_threshold": 1.5})
    assert build_config({"dataset": "voc2007"}).tag == "voc2007_n6_M4_seed0"
    assert build_config({"dataset": "voc2007", "retrieval": "sscd_align", "usable_threshold": 0.9}).tag == \
        "voc2007_n6_M4_seed0_sscd_align_usable0.9"


@pytest.mark.parametrize("extra", [{"retrieval": "sscd"}, {"retrieval": "sscd_align", "usable_threshold": 0.75}])
def test_phase3_run_passes_checks_and_sends_no_cluster_id(synthetic, extra):
    root, data_root, _, _ = synthetic
    name = "_".join(f"{k}{v}" for k, v in extra.items())
    paths = resolve_paths(str(data_root), str(root / "cache"), str(root / f"exp_{name}" / "runs"), [])
    paths.read_dirs = []
    folder = run_baseline(_config(**extra), paths, allow_dirty=True, log=lambda message: None)
    checks = json.loads((folder / "checks.json").read_text())
    assert all(item["passed"] for item in checks.values()), checks
    summary = json.loads((folder / "summary.json").read_text())
    assert summary["cluster_bits_sent"] == 0.0
    assert summary["retrieval"] == extra["retrieval"]
    assert summary["aux_bits_tight"] == pytest.approx(np.log2(sum(__import__("math").perm(4, u) for u in range(1, 5))),
                                                      abs=0.01)
    if "usable_threshold" in extra:
        report = summary["retrievability"]
        assert report["usable_images"] == summary["images_searched"] < report["sendable_images"]
    dashboard = (folder.parent.parent / "RESULTS_DASHBOARD.md").read_text()
    assert extra["retrieval"] in dashboard


def test_too_strict_threshold_is_explained(synthetic):
    root, data_root, _, _ = synthetic
    paths = resolve_paths(str(data_root), str(root / "cache"), str(root / "exp_strict" / "runs"), [])
    paths.read_dirs = []
    with pytest.raises(RuntimeError, match="lower usable_threshold"):
        run_baseline(_config(retrieval="sscd", usable_threshold=1.0), paths, allow_dirty=True, log=lambda m: None)
