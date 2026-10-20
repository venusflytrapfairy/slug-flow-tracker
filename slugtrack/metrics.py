"""Flow metrics computed from tracked slugs.

Units: everything is measured in *frames* first, because the camera frame rate
of the sampled dataset is unknown. Lengths use the ruler calibration (px/cm).
If a frame rate is supplied (or read from an uploaded video), time-based
values (slugs per minute, cm/s, seconds) are added.
"""
from __future__ import annotations

from statistics import mean
from typing import Dict, List, Optional


def compute_metrics(frame_numbers: List[int], per_frame: List[List[dict]],
                    slugs: List[dict], segments: List[int],
                    px_per_cm: float, fps: Optional[float]) -> Dict:
    n_images = len(frame_numbers)
    # Observed time = the span of each segment in frame numbers (missing frame
    # numbers inside a segment are part of the observed interval).
    seg_spans = {}
    for f, s in zip(frame_numbers, segments):
        lo, hi = seg_spans.get(s, (f, f))
        seg_spans[s] = (min(lo, f), max(hi, f))
    observed_frames = sum(hi - lo + 1 for lo, hi in seg_spans.values())

    present = sum(1 for row in per_frame if row)
    n = len(slugs)
    complete = [s for s in slugs if s["complete"]]

    vels = [s["front_velocity_px_per_frame"] for s in slugs
            if s["front_velocity_px_per_frame"] is not None]
    v_px = mean(vels) if vels else None              # px/frame, negative = leftwards
    v_cm = abs(v_px) / px_per_cm if v_px is not None else None   # cm/frame

    extents = [s["max_extent_px"] / px_per_cm for s in slugs]
    durations = [s["span_frames"] for s in complete]

    # Frames between consecutive slug arrivals inside the same segment.
    periods = []
    for a, b in zip(slugs, slugs[1:]):
        if a["segment"] == b["segment"]:
            periods.append(b["first_frame"] - a["first_frame"])

    out = {
        "images": n_images,
        "observed_frames": observed_frames,
        "segments": len(seg_spans),
        "slug_count": n,
        "complete_slugs": len(complete),
        "slugs_per_1000_frames": 1000.0 * n / observed_frames if observed_frames else 0.0,
        "presence_fraction": present / n_images if n_images else 0.0,
        "front_velocity_px_per_frame": v_px,
        "front_velocity_cm_per_frame": v_cm,
        "max_visible_length_cm": max(extents) if extents else None,
        "mean_passage_frames": mean(durations) if durations else None,
        "mean_period_frames": mean(periods) if periods else None,
        "fps": fps,
    }
    if fps:
        seconds = observed_frames / fps
        out.update({
            "observed_seconds": seconds,
            "slugs_per_minute": 60.0 * n / seconds if seconds else 0.0,
            "front_velocity_cm_per_s": v_cm * fps if v_cm is not None else None,
            "mean_passage_s": out["mean_passage_frames"] / fps if durations else None,
            "mean_period_s": out["mean_period_frames"] / fps if periods else None,
        })
    return out
