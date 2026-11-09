import json

import cv2
import numpy as np
import pytest

from stego import attacks, cache, tracking
from stego.config import ConfigError, build_config, load_config
from stego.data import DataError, discover_images
from stego.evaluate import auxiliary_bits_per_image, bootstrap_interval
from stego.hiding import bytes_to_segments, hide_segments, segments_to_bytes
from stego.mapping import build_index, deep_hashes
from stego.pipeline import resolve_paths, run_baseline
from stego.recovery import decode_image, equation_2, windowed_ssim
from stego.synthetic import create_synthetic, write_synthetic_cache


def simple_index(n=2, m=3):
    labels = ["a", "b", "c", "d"]
    rng = np.random.default_rng(5)
    objects = []
    for _ in range(300):
        count = rng.integers(1, m + 2)
        objects.append(
            [{"label": labels[int(rng.integers(0, 4))], "score": 0.95, "box": [0.0, 0.0, float(10 + j), 10.0]} for j in range(count)]
        )
    clusters = rng.integers(0, 5, 300)
    return build_index([f"{i}.jpg" for i in range(300)], objects, clusters, n, m, 0.9)


def test_hiding_prefers_most_segments_and_decodes_exactly():
    index = simple_index()
    rng = np.random.default_rng(1)
    segments = rng.integers(0, 4, 200).tolist()
    result = hide_segments(segments, index, rng)
    assert result.complete
    recovered = []
    for item in result.stego:
        recovered.extend(decode_image(item.image_index, item.aux, index))
    assert recovered == segments
    assert max(item.aux.mapped_count for item in result.stego) <= 3
    # Greedy check: no selected image could have been replaced by one mapping more.
    position = 0
    for item in result.stego:
        k = item.aux.mapped_count
        position += k
        if position < len(segments) and k < 3:
            window = segments[position - k : position + 1]
            need = np.bincount(window, minlength=4)
            assert not (index.counts >= need).all(axis=1).any()


def test_hiding_avoids_repeats_when_possible():
    index = simple_index()
    rng = np.random.default_rng(2)
    result = hide_segments([0, 1, 2, 3] * 5, index, rng)
    chosen = [item.image_index for item in result.stego]
    assert len(chosen) == len(set(chosen)) and result.repeated_images == 0


def test_hiding_reports_incomplete_index():
    objects = [[{"label": "a", "score": 0.95, "box": [0.0, 0.0, 1.0, 1.0]}], [{"label": "b", "score": 0.95, "box": [0.0, 0.0, 1.0, 1.0]}]]
    index = build_index(["0.jpg", "1.jpg"], objects, np.zeros(2, dtype=int), 1, 2, 0.9)
    index.counts[:, 1] = 0
    result = hide_segments([0, 1, 0], index, np.random.default_rng(0))
    assert not result.complete and result.completion_percent == pytest.approx(100 / 3)


def test_wrong_image_with_fewer_objects_gives_lost_segment():
    index = simple_index()
    image = next(i for i, seq in enumerate(index.sequences) if len(seq) == 1)
    from stego.hiding import AuxInfo

    assert decode_image(image, AuxInfo((0, 1, 2), 0, 0), index) == [None, None]


def test_equation_2_properties():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 255, (64, 64))
    assert equation_2(x, x) == pytest.approx(1.0)
    assert windowed_ssim(x, x) == pytest.approx(1.0)
    y = rng.uniform(0, 255, (64, 64))
    assert equation_2(x, y) < 0.2
    assert equation_2(x, y) == pytest.approx(equation_2(y, x))
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mx, my, sx, sy = x.mean(), y.mean(), x.std(), y.std()
    sxy = ((x - mx) * (y - my)).mean()
    by_hand = ((2 * mx * my + c1) * (2 * sx * sy + c2) * (sxy + c2 / 2)) / (
        (mx**2 + my**2 + c1) * (sx**2 + sy**2 + c2) * (sx * sy + c2 / 2)
    )
    assert equation_2(x, y) == pytest.approx(by_hand)


def test_attacks_follow_table_ix():
    image = np.full((100, 200, 3), 128, dtype=np.uint8)
    rng = np.random.default_rng(0)
    assert np.mean(attacks.center_crop(image, 0.20)[:, :, 0] == 0) == pytest.approx(0.2, abs=0.02)
    assert np.mean(attacks.edge_crop(image, 0.20)[:, :, 0] == 0) == pytest.approx(0.2, abs=0.02)
    moved = attacks.translate(image, 36, 20)
    assert moved[0, 100, 0] == 0 and moved[50, 10, 0] == 0 and moved[50, 100, 0] == 128
    assert attacks.scale(image, 3.0).shape[:2] == (300, 600)
    noise = attacks.salt_and_pepper(image, 0.001, rng)
    assert np.sum(np.any(noise != 128, axis=2)) == 20
    assert attacks.gamma_correction(image, 0.8)[0, 0, 0] == round((128 / 255) ** 0.8 * 255)
    assert set(attacks.build_attacks()) == {attacks.NO_ATTACK, *attacks.TABLE_IX_ORDER}
    assert len(attacks.TABLE_IX_ORDER) == 22


def test_every_attack_keeps_type_and_noise_is_repeatable():
    rng = np.random.default_rng(1)
    image = rng.integers(0, 256, (90, 120, 3), dtype=np.uint8)
    for name, attack in attacks.build_attacks().items():
        one = attack(image, attacks.attack_rng(0, name, 1, 2))
        two = attack(image, attacks.attack_rng(0, name, 1, 2))
        assert one.dtype == np.uint8 and one.ndim == 3
        assert np.array_equal(one, two), name
        if name != attacks.NO_ATTACK:
            assert not np.array_equal(one, image), name


def test_hash_follows_equation_1():
    features = np.array([[0.0, 1.0, 2.0, 3.0, 4.0]])
    assert deep_hashes(features, 0.5).tolist() == [[0, 0, 0, 1, 1]]


def test_auxiliary_bits():
    bits = auxiliary_bits_per_image(4, 256)
    assert bits["aux_bits_fixed"] == 20
    assert bits["aux_bits_tight"] == pytest.approx(14.0)


def test_bootstrap_interval_contains_mean():
    values = np.array([90.0, 100.0, 95.0, 85.0])
    low, high = bootstrap_interval(values, 500, 0)
    assert low <= values.mean() <= high


def test_config_rejects_unknown_and_bad_values(tmp_path):
    with pytest.raises(ConfigError):
        build_config({"dataset": "voc2007", "ARCH_MODE": "RCNN_LSTM"})
    with pytest.raises(ConfigError):
        build_config({"dataset": "voc2007", "ssim_mode": "custom"})
    with pytest.raises(ConfigError):
        build_config({"dataset": "voc2007", "n": "2.5"})
    path = tmp_path / "c.yaml"
    path.write_text("dataset: voc2007\nn: 6\n")
    config = load_config(path, ["n=4", "M=5", "capacity_bytes=[10,100]"])
    assert (config.n, config.M, config.capacity_bytes) == (4, 5, (10, 100))


@pytest.mark.parametrize("name", ["voc2007", "voc2012", "coco2014"])
def test_repo_configs_load(name):
    from pathlib import Path

    config = load_config(Path(__file__).resolve().parent.parent / "configs" / f"{name}.yaml")
    assert config.n == 6 and config.M == 4 and config.k == (1024 if name == "coco2014" else 256)


def _touch_images(folder, names):
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / name).write_bytes(b"x")


def test_voc2007_discovery_combines_trainval_and_test(tmp_path):
    _touch_images(tmp_path / "VOCtrainval" / "VOCdevkit" / "VOC2007" / "JPEGImages", ["a.jpg", "b.jpg"])
    _touch_images(tmp_path / "VOCtest" / "VOCdevkit" / "VOC2007" / "JPEGImages", ["c.jpg"])
    _touch_images(tmp_path / "other" / "VOC2012" / "JPEGImages", ["z.jpg"])
    config = build_config({"dataset": "voc2007", "expected_images": 3})
    assert discover_images(config, tmp_path)[0] == ["a.jpg", "b.jpg", "c.jpg"]
    with pytest.raises(DataError):
        discover_images(build_config({"dataset": "voc2007"}), tmp_path)


def test_voc2012_uses_trainval_list(tmp_path):
    base = tmp_path / "VOC2012"
    _touch_images(base / "JPEGImages", ["a.jpg", "b.jpg", "seg_only.jpg"])
    (base / "ImageSets" / "Main").mkdir(parents=True)
    (base / "ImageSets" / "Main" / "trainval.txt").write_text("a\nb\n")
    config = build_config({"dataset": "voc2012", "expected_images": 2})
    assert discover_images(config, tmp_path)[0] == ["a.jpg", "b.jpg"]


def test_voc2012_handles_train_val_and_test_folders(tmp_path):
    """The layout of the gopalbhattrai Kaggle dataset, with a same-named test image."""
    train = tmp_path / "pascal-voc-2012-dataset" / "VOC2012_train_val" / "VOC2012_train_val"
    test = tmp_path / "pascal-voc-2012-dataset" / "VOC2012_test" / "VOC2012_test"
    _touch_images(train / "JPEGImages", ["a.jpg", "b.jpg", "extra.jpg"])
    _touch_images(test / "JPEGImages", ["a.jpg", "t.jpg"])
    (train / "ImageSets" / "Main").mkdir(parents=True)
    (train / "ImageSets" / "Main" / "trainval.txt").write_text("a\nb\n")
    config = build_config({"dataset": "voc2012", "expected_images": 2})
    ids, paths = discover_images(config, tmp_path)
    assert ids == ["a.jpg", "b.jpg"]
    assert "VOC2012_train_val" in str(paths["a.jpg"])


def test_voc2012_missing_list_names_the_problem(tmp_path):
    _touch_images(tmp_path / "VOC2012_test" / "JPEGImages", ["t.jpg"])
    config = build_config({"dataset": "voc2012", "expected_images": 1})
    with pytest.raises(DataError, match="trainval.txt.*VOC2012_test"):
        discover_images(config, tmp_path)


def test_voc2007_is_not_confused_with_voc2012(tmp_path):
    _touch_images(tmp_path / "VOC2007" / "JPEGImages", ["a.jpg"])
    _touch_images(tmp_path / "VOC2012_train_val" / "JPEGImages", ["b.jpg"])
    config = build_config({"dataset": "voc2007", "expected_images": 1})
    assert discover_images(config, tmp_path)[0] == ["a.jpg"]


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic")
    ids, data_root = create_synthetic(root / "data", images=700)
    write_synthetic_cache(ids, root / "cache")
    return root, data_root


def _config(**extra):
    return build_config(
        {
            "dataset": "synthetic",
            "k": 8,
            "capacity_bytes": [10, 100],
            "capacity_trials": 2,
            "robustness_trials": 2,
            "robustness_segments": 8,
            "bootstrap_samples": 50,
            **extra,
        }
    )


@pytest.mark.parametrize("n,m", [(6, 4), (5, 4), (4, 4), (3, 4), (6, 3), (6, 5), (6, 6)])
def test_full_pipeline_for_paper_sweeps(synthetic, n, m):
    root, data_root = synthetic
    paths = resolve_paths(str(data_root), str(root / "cache"), str(root / f"exp_{n}_{m}" / "runs"), [])
    paths.read_dirs = []
    folder = run_baseline(_config(n=n, M=m), paths, allow_dirty=True, log=lambda message: None)
    checks = json.loads((folder / "checks.json").read_text())
    assert all(item["passed"] for item in checks.values()), checks
    summary = json.loads((folder / "summary.json").read_text())
    # official means a clean tree and all checks passed, so it depends on where the tests run
    assert summary["official"] == (not tracking.git_state()["dirty"])
    assert summary["means"]["no_attack_acc"] == 100.0
    for name in ("capacity.csv", "attacks.csv", "segments.csv.gz", "dictionary.csv", "source.tar.gz", "environment.json"):
        assert (folder / name).is_file()
    assert (folder.parent.parent / "RESULTS_DASHBOARD.md").is_file()


def test_same_settings_give_same_numbers(synthetic):
    root, data_root = synthetic
    results = []
    for number in range(2):
        paths = resolve_paths(str(data_root), str(root / "cache"), str(root / f"repeat{number}" / "runs"), [])
        paths.read_dirs = []
        folder = run_baseline(_config(), paths, allow_dirty=True, log=lambda message: None)
        results.append((folder / "attacks.csv").read_text())
    assert results[0] == results[1]


def test_dirty_code_is_refused_without_flag(synthetic, monkeypatch):
    root, data_root = synthetic
    from stego import tracking

    monkeypatch.setattr(tracking, "git_state", lambda: {"commit": "x", "dirty": True, "changes": [" M stego/a.py"]})
    paths = resolve_paths(str(data_root), str(root / "cache"), str(root / "dirty" / "runs"), [])
    with pytest.raises(RuntimeError, match="uncommitted"):
        run_baseline(_config(), paths, log=lambda message: None)


def test_run_folder_cannot_be_replaced(tmp_path, monkeypatch):
    from stego import tracking

    class Frozen:
        @staticmethod
        def now(zone):
            import datetime as d

            return d.datetime(2026, 11, 5, tzinfo=zone)

    monkeypatch.setattr(tracking, "datetime", Frozen)
    tracking.create_run_folder(tmp_path, "tag", "abc")
    with pytest.raises(FileExistsError):
        tracking.create_run_folder(tmp_path, "tag", "abc")


def test_missing_cache_gives_clear_error(tmp_path):
    with pytest.raises(cache.CacheError, match="python run.py extract"):
        cache.load_extraction(["a.jpg"], "synthetic", tmp_path, [])


def test_cache_is_found_in_read_only_folder(synthetic, tmp_path):
    root, data_root = synthetic
    ids = sorted(p.name for p in (data_root / "images").iterdir())
    nested = tmp_path / "datasets" / "user" / "slug"
    nested.parent.mkdir(parents=True)
    import shutil

    shutil.copytree(root / "cache", nested)
    loaded = cache.load_extraction(ids, "synthetic", tmp_path / "empty", [tmp_path])
    assert len(loaded.ids) == len(ids) and loaded.features.shape[1] == 2048


def test_git_state_ignores_results_but_catches_code(tmp_path, monkeypatch):
    """Uses real git: a tracked, modified dashboard must not count as a code change."""
    import subprocess

    from stego import tracking

    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    (tmp_path / "experiments").mkdir()
    (tmp_path / "experiments" / "RESULTS_DASHBOARD.md").write_text("one\n")
    (tmp_path / "code.py").write_text("one\n")
    git("add", "-A")
    git("commit", "-m", "first")
    monkeypatch.setattr(tracking, "REPO_ROOT", tmp_path)
    assert tracking.git_state()["dirty"] is False

    (tmp_path / "experiments" / "RESULTS_DASHBOARD.md").write_text("two\n")
    (tmp_path / "experiments" / "new_run.txt").write_text("x\n")
    assert tracking.git_state()["dirty"] is False

    (tmp_path / "code.py").write_text("two\n")
    state = tracking.git_state()
    assert state["dirty"] is True and len(state["changes"]) == 1


def test_label_only_below_top_m_cannot_create_an_unusable_group():
    small = {"label": "rare", "score": 0.95, "box": [0.0, 0.0, 1.0, 1.0]}
    big = [{"label": name, "score": 0.95, "box": [0.0, 0.0, 10.0, 10.0]} for name in ("a", "b")]
    index = build_index(["0.jpg", "1.jpg"], [[*big, small], big[:1]], np.zeros(2, dtype=int), 1, 2, 0.9)
    assert set(index.dictionary) == {"a", "b"}
    assert (index.counts.sum(axis=0) > 0).all()
