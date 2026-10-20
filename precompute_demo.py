"""Run the detector over the bundled demo frames once and save the raw
detections, so the hosted app can open the demo instantly.

Usage:  python precompute_demo.py
Re-run it whenever you replace the model weights or the demo frames.
"""
import json
import time
from pathlib import Path

from slugtrack.detector import SlugDetector, frame_number, load_images, natural_sort

ROOT = Path(__file__).parent
FRAMES = ROOT / "data" / "demo_frames"   # original 960x480 frames, if you have them
WEIGHTS = ROOT / "models" / "slug_yolov8s_500img.pt"
OUT = ROOT / "data" / "demo_detections.json"


def main():
    frames_dir = FRAMES
    if not any(FRAMES.glob("*.jpg")):          # originals not present: use the browser copies
        from slugtrack.pipeline import DEMO_WEB, ensure_demo_web_frames
        ensure_demo_web_frames()
        frames_dir = DEMO_WEB
    names = natural_sort([p.name for p in frames_dir.glob("*.jpg")])
    paths = [frames_dir / n for n in names]
    imgs, (w, h) = load_images(paths)
    det = SlugDetector(WEIGHTS)
    t0 = time.time()
    dets = det.detect(imgs, progress=lambda k: print(f"\r{k}/{len(imgs)}", end=""))
    print(f"\n{len(imgs)} frames in {time.time() - t0:.1f}s")
    OUT.write_text(json.dumps({
        "weights": WEIGHTS.name,
        "width": w, "height": h,
        "frames": [{"name": n, "frame": frame_number(n, i), "dets": d}
                   for i, (n, d) in enumerate(zip(names, dets))],
    }))
    print("saved", OUT)


if __name__ == "__main__":
    main()
