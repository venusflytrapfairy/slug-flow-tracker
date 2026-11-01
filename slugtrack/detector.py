"""YOLOv8 slug detector: runs the trained model over a frame sequence."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

FRAME_NUM_RE = re.compile(r"(\d+)(?!.*\d)")   # last group of digits in a name


def frame_number(name: str, fallback: int) -> int:
    m = FRAME_NUM_RE.search(Path(name).stem)
    return int(m.group(1)) if m else fallback


def natural_sort(names: Sequence[str]) -> List[str]:
    def key(s):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]
    return sorted(names, key=key)


class SlugDetector:
    """Thin wrapper around an Ultralytics YOLO model with one class, 'slug'."""

    def __init__(self, weights: str | Path, imgsz: int = 640, device: str = "cpu"):
        if device != "cpu":
            import torch
            if not torch.cuda.is_available():
                raise ValueError("CUDA was selected, but no CUDA-enabled PyTorch GPU is available.")
        from ultralytics import YOLO  # imported lazily: heavy
        self.model = YOLO(str(weights))
        self.imgsz = imgsz
        self.device = device

    def detect(self, images: Iterable[np.ndarray], low_conf: float = 0.10,
               batch: int = 8, progress: Optional[Callable[[int], None]] = None
               ) -> List[List[dict]]:
        """Detect slugs in BGR/gray numpy images. Keeps everything >= low_conf so
        the confidence threshold can be changed later without re-running."""
        out: List[List[dict]] = []
        buf: List[np.ndarray] = []

        def flush():
            if not buf:
                return
            res = self.model.predict(buf, imgsz=self.imgsz, conf=low_conf,
                                     device=self.device, verbose=False)
            for r in res:
                row = []
                for xyxy, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                    x1, y1, x2, y2 = (float(v) for v in xyxy)
                    row.append({"x1": round(x1, 1), "y1": round(y1, 1),
                                "x2": round(x2, 1), "y2": round(y2, 1),
                                "conf": round(float(c), 4)})
                out.append(row)
            if progress:
                progress(len(out))
            buf.clear()

        for img in images:
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            buf.append(img)
            if len(buf) >= batch:
                flush()
        flush()
        return out


def load_images(paths: Sequence[Path]) -> Tuple[List[np.ndarray], Tuple[int, int]]:
    import cv2
    imgs = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in paths]
    h, w = imgs[0].shape[:2]
    return imgs, (w, h)
