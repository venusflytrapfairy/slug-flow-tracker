"""Slug tracking and counting: at most one slug per frame.

Physical rule
-------------
Slug flow is one continuous stream through the pipe, so a frame can show at
most one slug. A new slug is counted only after the previous one has ended,
i.e. after one or more frames in which no slug is detected. This is the same
rule used when the frames were annotated:

* consecutive frames containing a slug belong to the same slug;
* when frames without a slug interrupt the sequence, the next appearance of a
  slug is a new slug.

How detections become slugs
---------------------------
1. One box per frame. YOLO can return more than one box in a frame (for
   example two pieces of the same slug). All boxes above the confidence
   threshold are merged into a single box covering all of them (their union),
   with the highest confidence. The number of merged boxes is kept so it can be
   reported.
2. Slug events. Frames with a box are grouped into runs. A run continues across
   up to `gap_tolerance` frames without a box (missed detections); a longer
   break ends the slug, and the next box starts a new one. Set
   `gap_tolerance` to 0 to treat every empty frame as an interruption.
3. Noise. A run with fewer than `min_frames` detections is not counted.
4. Missing frame numbers. In sampled image sets the file numbers can jump.
   Jumps of up to `max_bridge` frame numbers are not evidence of an
   interruption (those frames were not observed), so they do not end a slug.
   Longer jumps split the footage into separate intervals; a slug never
   continues across intervals.

The slug front (the left edge of the box, since flow runs right to left) is
recorded in every frame and its speed is fitted per slug.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class TrackerConfig:
    conf: float = 0.40          # minimum detection confidence
    gap_tolerance: int = 3      # empty frames allowed inside one slug (missed detections)
    min_frames: int = 3         # detections needed before a run counts as a slug
    max_bridge: int = 30        # largest jump in frame numbers that does not split the footage
    edge_margin: int = 6        # px; a box this close to a frame edge is cut by the edge


def segment_ids(frame_numbers: List[int], max_bridge: int) -> List[int]:
    """Interval id per image; a new interval starts after a jump > max_bridge."""
    seg, out = 0, []
    for i, f in enumerate(frame_numbers):
        if i > 0 and f - frame_numbers[i - 1] > max_bridge:
            seg += 1
        out.append(seg)
    return out


def merge_frame(dets: List[dict], conf: float) -> Optional[dict]:
    """Merge all boxes at or above `conf` into one slug box (or None)."""
    kept = [d for d in dets if d["conf"] >= conf]
    if not kept:
        return None
    return {
        "x1": min(d["x1"] for d in kept),
        "y1": min(d["y1"] for d in kept),
        "x2": max(d["x2"] for d in kept),
        "y2": max(d["y2"] for d in kept),
        "conf": max(d["conf"] for d in kept),
        "merged": len(kept),
    }


def track_slugs(frame_numbers: List[int], detections: List[List[dict]],
                width: int, cfg: TrackerConfig) -> Dict:
    """Group per-frame detections into slugs (at most one per frame).

    frame_numbers : time index of each image (ascending).
    detections    : per image, list of {"x1","y1","x2","y2","conf"} in pixels.
    width         : frame width in px.

    Returns
      per_frame : per image, [] or [box + {"slug": id, "merged": n}]
      active    : per image, the slug id in progress (also on bridged empty frames) or None
      rejected  : per image, [] or [box] for runs too short to count
      slugs     : summary per counted slug
      segments  : interval id per image
    """
    n = len(frame_numbers)
    segs = segment_ids(frame_numbers, cfg.max_bridge)
    boxes = [merge_frame(d, cfg.conf) for d in detections]

    # ---- group frames into runs ------------------------------------------
    runs: List[dict] = []
    cur: Optional[dict] = None
    misses = 0
    for i in range(n):
        new_segment = i == 0 or segs[i] != segs[i - 1]
        if new_segment and cur is not None:
            cur["cut_at_end"] = True          # still in view when the interval ended
            runs.append(cur)
            cur = None
        if boxes[i] is not None:
            if cur is None:
                cur = {"idxs": [], "segment": segs[i], "cut_at_start": new_segment,
                       "cut_at_end": False}
            cur["idxs"].append(i)
            misses = 0
        elif cur is not None:
            misses += 1
            if misses > cfg.gap_tolerance:    # a real interruption: the slug has ended
                runs.append(cur)
                cur = None
    if cur is not None:
        cur["cut_at_end"] = True
        runs.append(cur)

    # ---- keep runs long enough to be a slug -------------------------------
    per_frame: List[List[dict]] = [[] for _ in range(n)]
    rejected: List[List[dict]] = [[] for _ in range(n)]
    active: List[Optional[int]] = [None] * n
    slugs: List[dict] = []
    for run in runs:
        idxs = run["idxs"]
        if len(idxs) < cfg.min_frames:
            for i in idxs:
                rejected[i].append(boxes[i])
            continue
        sid = len(slugs) + 1
        for i in idxs:
            per_frame[i].append({**boxes[i], "slug": sid})
        for i in range(idxs[0], idxs[-1] + 1):
            active[i] = sid
        slugs.append(_summarise(sid, run, boxes, frame_numbers, width, cfg))

    return {"per_frame": per_frame, "active": active, "rejected": rejected,
            "slugs": slugs, "segments": segs}


def _fit_velocity(frames: List[int], fronts: List[float], edge: int) -> Optional[float]:
    """Least-squares slope of front position vs frame number (front inside view only)."""
    pts = [(f, x) for f, x in zip(frames, fronts) if x > edge]
    if len(pts) < 3:
        return None
    fs = [p[0] for p in pts]
    xs = [p[1] for p in pts]
    fm, xm = sum(fs) / len(fs), sum(xs) / len(xs)
    den = sum((f - fm) ** 2 for f in fs)
    if den == 0:
        return None
    return sum((f - fm) * (x - xm) for f, x in zip(fs, xs)) / den


def _summarise(sid: int, run: dict, boxes, frame_numbers, width: int, cfg: TrackerConfig) -> dict:
    idxs = run["idxs"]
    frames = [frame_numbers[i] for i in idxs]
    fronts = [boxes[i]["x1"] for i in idxs]
    first = boxes[idxs[0]]
    return {
        "slug": sid,
        "segment": run["segment"],
        "first_idx": idxs[0],
        "last_idx": idxs[-1],
        "first_frame": frames[0],
        "last_frame": frames[-1],
        "frames_detected": len(idxs),
        "span_frames": frames[-1] - frames[0] + 1,
        "front_velocity_px_per_frame": _fit_velocity(frames, fronts, cfg.edge_margin),
        "max_extent_px": max(boxes[i]["x2"] - boxes[i]["x1"] for i in idxs),
        "mean_conf": sum(boxes[i]["conf"] for i in idxs) / len(idxs),
        "frames_with_merged_boxes": sum(1 for i in idxs if boxes[i]["merged"] > 1),
        "entered_from_right": first["x2"] >= width - cfg.edge_margin and first["x1"] > width * 0.5,
        "exited_left": any(x <= cfg.edge_margin for x in fronts),
        "cut_at_start": run["cut_at_start"],
        "cut_at_end": run["cut_at_end"],
        "complete": not run["cut_at_start"] and not run["cut_at_end"],
        "track": [{"idx": i, "frame": f, "front": x} for i, f, x in zip(idxs, frames, fronts)],
    }
