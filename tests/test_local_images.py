import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from slugtrack import pipeline  # noqa: E402


class EmptyDetector:
    def __init__(self):
        self.frame_count = 0

    def detect(self, images, batch=8):
        self.frame_count += len(images)
        return [[] for _ in images]


def create_images(folder, names):
    folder.mkdir(parents=True)
    for index, name in enumerate(names):
        if Path(name).suffix.lower() not in pipeline.IMAGE_SUFFIXES:
            (folder / name).write_text("not an image")
            continue
        image = np.full((24, 32, 3), index, dtype=np.uint8)
        assert cv2.imwrite(str(folder / name), image)


def setup_cache_paths(monkeypatch, tmp_path):
    static = tmp_path / "static"
    monkeypatch.setattr(pipeline, "STATIC", static)
    monkeypatch.setattr(pipeline, "SESSIONS", static / "sessions")
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"test weights")
    monkeypatch.setattr(pipeline, "WEIGHTS", weights)


def test_scan_image_folder_naturally_sorts_and_uses_absolute_paths(tmp_path, monkeypatch):
    setup_cache_paths(monkeypatch, tmp_path)
    folder = tmp_path / "images"
    create_images(folder, ["frame_10.jpg", "frame_2.jpg", "notes.txt"])

    root, paths, fingerprint = pipeline.scan_image_folder(folder)

    assert root == folder
    assert [path.name for path in paths] == ["frame_2.jpg", "frame_10.jpg"]
    assert len(fingerprint) == 64

    with pytest.raises(ValueError, match="absolute path"):
        pipeline.scan_image_folder("relative-images")


def test_full_processing_reuses_pilot_detections(tmp_path, monkeypatch):
    setup_cache_paths(monkeypatch, tmp_path)
    folder = tmp_path / "images"
    create_images(folder, [f"frame_{index:04d}.jpg" for index in range(1, 7)])
    detector = EmptyDetector()

    pilot = pipeline.load_local_image_folder(folder, detector, limit=3, batch_size=2)
    full = pipeline.load_local_image_folder(folder, detector, batch_size=2)

    assert pilot["scope"] == "pilot"
    assert pilot["processed_images"] == 3
    assert full["scope"] == "full"
    assert full["processed_images"] == 6
    assert full["cached_images"] == 3
    assert detector.frame_count == 6
    assert len(full["detections"]) == len(full["urls"]) == 6
    assert [Path(path).name for path in full["names"]] == [
        f"frame_{index:04d}.jpg" for index in range(1, 7)
    ]


def test_player_template_is_rendered_as_utf8():
    html = pipeline.render_player_html({"frames": []})

    assert "← Flow" in html
    assert '"frames":[]' in html
