"""Glue between data sources, detector, tracker, metrics and the player."""
from __future__ import annotations

import hashlib
import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .detector import SlugDetector, frame_number, natural_sort
from .metrics import compute_metrics
from .tracker import TrackerConfig, track_slugs

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"
DEMO_WEB = STATIC / "demo"
DEMO_DETECTIONS = ROOT / "data" / "demo_detections.json"
DEMO_SOURCE = ROOT / "data" / "demo_frames"          # original frames (optional, not in repo)
DEMO_ZIP = ROOT / "data" / "demo_web_frames.zip"     # browser copies of the 500 frames
WEIGHTS = ROOT / "models" / "slug_yolov8s_500img.pt"
SESSIONS = STATIC / "sessions"
MAX_WIDTH = 960          # frames are shown (and detected) at most this wide
WEB_QUALITY = 78
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
LOCAL_CACHE_VERSION = 1


# ---------------------------------------------------------------- frames
def to_web_jpeg(img: np.ndarray, dest: Path) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Resize to MAX_WIDTH and write a light JPEG for the browser. Returns the
    resized image (used for detection so boxes match what is shown)."""
    h, w = img.shape[:2]
    if w > MAX_WIDTH:
        img = cv2.resize(img, (MAX_WIDTH, round(h * MAX_WIDTH / w)), interpolation=cv2.INTER_AREA)
    gray_like = img.ndim == 2 or np.allclose(img[..., 0], img[..., 1], atol=3)
    out = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if (img.ndim == 3 and gray_like) else img
    if not cv2.imwrite(str(dest), out, [cv2.IMWRITE_JPEG_QUALITY, WEB_QUALITY]):
        raise OSError(f"Could not write processed frame to {dest}.")
    return img, (img.shape[1], img.shape[0])


def new_session_dir() -> Path:
    cleanup_sessions()
    d = SESSIONS / uuid.uuid4().hex[:12]
    d.mkdir(parents=True, exist_ok=True)
    return d


def cleanup_sessions(max_age_s: int = 3 * 3600) -> None:
    if not SESSIONS.exists():
        return
    now = time.time()
    for d in SESSIONS.iterdir():
        try:
            if (d.is_dir() and not d.name.startswith("local-")
                    and now - d.stat().st_mtime > max_age_s):
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass


def url_for(path: Path) -> str:
    """Relative URL Streamlit's static file server uses for files in ./static."""
    return "app/static/" + path.relative_to(STATIC).as_posix()


# ---------------------------------------------------------------- sources
def load_demo(detector_factory: Callable[[], SlugDetector],
              progress: Optional[Callable[[float, str], None]] = None) -> Dict:
    """Bundled 500-frame demo. Uses cached detections when available."""
    if DEMO_DETECTIONS.exists():
        cache = json.loads(DEMO_DETECTIONS.read_text())
        frames = cache["frames"]
        missing = [f for f in frames if not (DEMO_WEB / f["name"]).exists()]
        if missing:
            ensure_demo_web_frames()
        return {
            "id": "demo-" + cache.get("weights", ""),
            "names": [f["name"] for f in frames],
            "frame_numbers": [f["frame"] for f in frames],
            "detections": [f["dets"] for f in frames],
            "urls": [url_for(DEMO_WEB / f["name"]) for f in frames],
            "size": (cache["width"], cache["height"]),
            "fps": None,
            "kind": "demo",
        }
    # No cache: detect now on the web frames.
    ensure_demo_web_frames()
    names = natural_sort([p.name for p in DEMO_WEB.glob("*.jpg")])
    imgs = [cv2.imread(str(DEMO_WEB / n), cv2.IMREAD_COLOR) for n in names]
    det = detector_factory().detect(imgs, progress=_wrap(progress, len(imgs), "Detecting slugs"))
    h, w = imgs[0].shape[:2]
    return {"id": "demo-live", "names": names,
            "frame_numbers": [frame_number(n, i) for i, n in enumerate(names)],
            "detections": det, "urls": [url_for(DEMO_WEB / n) for n in names],
            "size": (w, h), "fps": None, "kind": "demo"}


def ensure_demo_web_frames() -> None:
    """Unpack the bundled browser frames (or rebuild them from the originals)."""
    if DEMO_ZIP.exists():
        import zipfile
        DEMO_WEB.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(DEMO_ZIP) as z:
            for member in z.namelist():
                name = Path(member).name
                if name.lower().endswith(".jpg") and not (DEMO_WEB / name).exists():
                    (DEMO_WEB / name).write_bytes(z.read(member))
    elif DEMO_SOURCE.exists():
        build_demo_web_frames()


def build_demo_web_frames() -> None:
    DEMO_WEB.mkdir(parents=True, exist_ok=True)
    for p in DEMO_SOURCE.glob("*.jpg"):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        to_web_jpeg(img, DEMO_WEB / p.name)


def load_uploaded_images(files: Sequence, detector: SlugDetector,
                         progress: Optional[Callable[[float, str], None]] = None) -> Dict:
    """files: objects with .name and .getvalue() (Streamlit UploadedFile)."""
    sdir = new_session_dir()
    by_name = {f.name: f for f in files}
    names = natural_sort(list(by_name))
    imgs, urls, size = [], [], None
    for k, n in enumerate(names):
        arr = np.frombuffer(by_name[n].getvalue(), np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"{n} could not be read as an image.")
        dest = sdir / f"{k:06d}.jpg"
        img, sz = to_web_jpeg(img, dest)
        if size is None:
            size = sz
        elif sz != size:
            img = cv2.resize(img, size)
        imgs.append(img)
        urls.append(url_for(dest))
        if progress:
            progress(0.25 * (k + 1) / len(names), "Preparing frames")
    dets = detector.detect(imgs, progress=_wrap(progress, len(imgs), "Detecting slugs", 0.25))
    return {"id": sdir.name, "names": names,
            "frame_numbers": _monotonic([frame_number(n, i) for i, n in enumerate(names)]),
            "detections": dets, "urls": urls, "size": size, "fps": None, "kind": "images"}


def scan_image_folder(folder: str | Path) -> Tuple[Path, List[Path], str]:
    """Return naturally sorted supported images and a fingerprint for a local folder."""
    candidate = Path(folder).expanduser()
    if not candidate.is_absolute():
        raise ValueError(f"Enter an absolute path to the image folder: {folder}")
    try:
        root = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"Image folder does not exist or cannot be accessed: {folder}") from exc
    if not root.is_dir():
        raise ValueError(f"Image path is not a folder: {root}")

    try:
        paths = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
        metadata = [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    except OSError as exc:
        raise ValueError(f"Could not read the image folder: {root}") from exc

    paths = [root / name for name in natural_sort([item[0] for item in metadata])]
    by_name = {item[0]: item[1:] for item in metadata}
    ordered_metadata = [(p.name, *by_name[p.name]) for p in paths]
    weights_stat = WEIGHTS.stat()
    signature = json.dumps(
        {
            "version": LOCAL_CACHE_VERSION,
            "folder": str(root),
            "files": ordered_metadata,
            "weights": (weights_stat.st_size, weights_stat.st_mtime_ns),
            "max_width": MAX_WIDTH,
            "web_quality": WEB_QUALITY,
        },
        separators=(",", ":"),
    ).encode()
    return root, paths, hashlib.sha256(signature).hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, separators=(",", ":")))
    temp.replace(path)


def load_local_image_folder(
    folder: str | Path,
    detector: SlugDetector,
    limit: Optional[int] = None,
    batch_size: int = 8,
    progress: Optional[Callable[[float, str], None]] = None,
) -> Dict:
    """Detect a local image sequence incrementally and persist reusable checkpoints."""
    if batch_size < 1:
        raise ValueError("Batch size must be at least 1.")
    root, paths, fingerprint = scan_image_folder(folder)
    if not paths:
        raise ValueError(f"No supported images were found in {root}.")
    target = min(limit, len(paths)) if limit is not None else len(paths)
    if target < 1:
        raise ValueError("There are no images selected for processing.")

    cache_id = "local-" + fingerprint[:20]
    sdir = SESSIONS / cache_id
    sdir.mkdir(parents=True, exist_ok=True)
    manifest_path = sdir / "detections.json"
    frame_numbers = _monotonic([frame_number(p.name, i) for i, p in enumerate(paths)])
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"The saved processing cache is unreadable: {manifest_path}. "
                "Remove that cache file and run again."
            ) from exc
        if manifest.get("fingerprint") != fingerprint:
            raise ValueError("The saved processing cache does not match the selected image folder.")
    else:
        manifest = {
            "version": LOCAL_CACHE_VERSION,
            "fingerprint": fingerprint,
            "names": [p.name for p in paths],
            "frame_numbers": frame_numbers,
            "chunks": [],
            "completed_frames": 0,
            "size": None,
            "detection_seconds": 0.0,
            "pilot_detection_seconds": 0.0,
        }
        _write_json_atomic(manifest_path, manifest)

    if manifest.get("names") != [p.name for p in paths] or manifest.get("frame_numbers") != frame_numbers:
        raise ValueError("The saved processing cache does not match the selected image sequence.")
    completed = int(manifest.get("completed_frames", 0))
    chunks = manifest.get("chunks", [])
    detections = []
    expected_start = 0
    try:
        for chunk in chunks:
            if chunk["start"] != expected_start or chunk["stop"] <= chunk["start"]:
                raise ValueError("non-contiguous cache chunks")
            chunk_detections = json.loads((sdir / chunk["file"]).read_text())
            if len(chunk_detections) != chunk["stop"] - chunk["start"]:
                raise ValueError("cache chunk length mismatch")
            detections.extend(chunk_detections)
            expected_start = chunk["stop"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"The saved processing cache is incomplete or invalid: {manifest_path}.") from exc
    if completed != len(detections) or completed != expected_start or completed > len(paths):
        raise ValueError(f"The saved processing cache is incomplete or invalid: {manifest_path}.")
    size = tuple(manifest["size"]) if manifest.get("size") else None
    if any(not (sdir / f"{i:06d}.jpg").is_file() for i in range(completed)):
        completed = 0
        detections = []
        size = None
        manifest.update({
            "chunks": [],
            "completed_frames": 0,
            "size": None,
            "detection_seconds": 0.0,
            "pilot_detection_seconds": 0.0,
        })
        _write_json_atomic(manifest_path, manifest)

    cached_images = min(completed, target)
    started = time.monotonic()
    if size is None:
        first = cv2.imread(str(paths[0]), cv2.IMREAD_COLOR)
        if first is None:
            raise ValueError(f"{paths[0].name} could not be read as an image.")
        h, w = first.shape[:2]
        if w > MAX_WIDTH:
            h, w = round(h * MAX_WIDTH / w), MAX_WIDTH
        size = (w, h)

    while completed < target:
        stop = min(completed + batch_size, target)
        batch_images = []
        for index in range(completed, stop):
            img = cv2.imread(str(paths[index]), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"{paths[index].name} could not be read as an image.")
            if (img.shape[1], img.shape[0]) != size:
                img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            img, _ = to_web_jpeg(img, sdir / f"{index:06d}.jpg")
            batch_images.append(img)

        batch_started = time.monotonic()
        batch_detections = detector.detect(batch_images, batch=batch_size)
        detection_seconds = time.monotonic() - batch_started
        if len(batch_detections) != len(batch_images):
            raise RuntimeError("The detector returned a different number of results than input frames.")
        chunk_name = f"detections-{completed:06d}-{stop:06d}.json"
        _write_json_atomic(sdir / chunk_name, batch_detections)
        detections.extend(batch_detections)
        completed = stop
        manifest["chunks"].append({"start": stop - len(batch_detections), "stop": stop, "file": chunk_name})
        manifest["completed_frames"] = completed
        manifest["size"] = size
        manifest["detection_seconds"] = float(manifest.get("detection_seconds", 0.0)) + detection_seconds
        pilot_frames = max(0, min(stop, 1000) - (stop - len(batch_detections)))
        if pilot_frames:
            manifest["pilot_detection_seconds"] = (
                float(manifest.get("pilot_detection_seconds", 0.0))
                + detection_seconds * pilot_frames / len(batch_detections)
            )
        _write_json_atomic(manifest_path, manifest)
        if progress:
            progress(completed / target, f"Detected {completed:,}/{target:,} frames")

    selected_paths = paths[:target]
    selected_detections = detections[:target]
    selected_frame_numbers = frame_numbers[:target]
    elapsed = time.monotonic() - started
    return {
        "id": cache_id,
        "source_fingerprint": fingerprint,
        "source_folder": str(root),
        "names": [p.name for p in selected_paths],
        "frame_numbers": selected_frame_numbers,
        "detections": selected_detections,
        "urls": [url_for(sdir / f"{i:06d}.jpg") for i in range(target)],
        "size": size,
        "fps": None,
        "kind": "local-images",
        "scope": "pilot" if target < len(paths) else "full",
        "processed_images": target,
        "dataset_images": len(paths),
        "detection_seconds": float(manifest.get("detection_seconds", 0.0)),
        "pilot_detection_seconds": float(manifest.get("pilot_detection_seconds", 0.0)),
        "elapsed_seconds": elapsed,
        "cached_images": cached_images,
    }


def load_uploaded_video(data: bytes, suffix: str, detector: SlugDetector, step: int = 1,
                        max_frames: int = 1500,
                        progress: Optional[Callable[[float, str], None]] = None) -> Dict:
    sdir = new_session_dir()
    vpath = sdir / f"upload{suffix}"
    vpath.write_bytes(data)
    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        raise ValueError("This video could not be opened. Try an .mp4 (H.264) or .avi file.")
    fps = cap.get(cv2.CAP_PROP_FPS) or None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    imgs, urls, fnums, size, idx = [], [], [], None, 0
    while len(imgs) < max_frames:
        ok, img = cap.read()
        if not ok:
            break
        if idx % step == 0:
            dest = sdir / f"{idx:06d}.jpg"
            img, sz = to_web_jpeg(img, dest)
            size = size or sz
            imgs.append(img)
            urls.append(url_for(dest))
            fnums.append(idx)
            if progress and total:
                progress(0.25 * min(idx / total, 1.0), "Reading video")
        idx += 1
    cap.release()
    vpath.unlink(missing_ok=True)
    if not imgs:
        raise ValueError("No frames could be read from this video.")
    dets = detector.detect(imgs, progress=_wrap(progress, len(imgs), "Detecting slugs", 0.25))
    return {"id": sdir.name, "names": [f"frame {f}" for f in fnums], "frame_numbers": fnums,
            "detections": dets, "urls": urls, "size": size,
            "fps": round(fps, 3) if fps and fps > 0 else None, "kind": "video",
            "truncated": len(imgs) >= max_frames}


def _monotonic(nums: List[int]) -> List[int]:
    """Frame numbers must increase; fall back to positions if names are not ordered."""
    if all(b > a for a, b in zip(nums, nums[1:])):
        return nums
    return list(range(len(nums)))


def _wrap(progress, total, label, base=0.0):
    if not progress:
        return None
    return lambda k: progress(base + (1 - base) * k / total, f"{label} {k}/{total}")


# ---------------------------------------------------------------- analysis
def analyse(src: Dict, cfg: TrackerConfig, px_per_cm: float, fps: Optional[float]) -> Dict:
    w, h = src["size"]
    tr = track_slugs(src["frame_numbers"], src["detections"], w, cfg)
    m = compute_metrics(src["frame_numbers"], tr["per_frame"], tr["slugs"], tr["segments"],
                        px_per_cm, fps)
    return {"track": tr, "metrics": m}


def player_payload(src: Dict, result: Dict, px_per_cm: float, fps: Optional[float]) -> Dict:
    tr = result["track"]
    segs = tr["segments"]
    seg_info = []
    for i, s in enumerate(segs):
        if s >= len(seg_info):
            seg_info.append({"first": src["frame_numbers"][i], "last": src["frame_numbers"][i],
                             "first_idx": i})
        seg_info[s]["last"] = src["frame_numbers"][i]
    frames = []
    for i, url in enumerate(src["urls"]):
        frames.append({
            "url": url, "frame": src["frame_numbers"][i], "segment": segs[i],
            "boxes": [{k: b[k] for k in ("x1", "y1", "x2", "y2", "conf", "slug")}
                      for b in tr["per_frame"][i]],
            "rejected": len(tr["rejected"][i]),
        })
    slugs = [{k: s[k] for k in ("slug", "first_idx", "last_idx", "first_frame", "last_frame",
                                "front_velocity_px_per_frame", "track")} for s in tr["slugs"]]
    w, h = src["size"]
    return {"dataset_id": src["id"], "width": w, "height": h, "px_per_cm": px_per_cm,
            "fps": fps, "frames": frames, "slugs": slugs, "segments": seg_info}


def inline_frames(payload: Dict) -> Dict:
    """Fallback when Streamlit static file serving is off: embed frames as data URIs."""
    import base64
    for fr in payload["frames"]:
        path = STATIC / fr["url"].removeprefix("app/static/")
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        h, w = img.shape[:2]
        small = cv2.resize(img, (640, round(h * 640 / w)), interpolation=cv2.INTER_AREA) if w > 640 else img
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 62])
        fr["url"] = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
    return payload


def render_player_html(payload: Dict) -> str:
    tpl = (Path(__file__).parent / "player.html").read_text(encoding="utf-8")
    return tpl.replace("__DATA__", json.dumps(payload, separators=(",", ":")))
