"""Slug-front tracker and slug counting.

Physical idea
-------------
Flow runs right -> left in the camera view, so every slug enters on the right
and its *leading edge* (the "front": the left edge x1 of the detection box)
travels steadily leftwards. Tracking that front gives

* identity : a detection whose front sits where the previous slug's front was
             predicted to be is the same slug; a front that shows up somewhere
             else (typically back on the right) is a new slug;
* velocity : pixels per frame travelled by the front (translational velocity);
* extent   : the visible length of the slug body (box width).

Mechanics: an alpha-beta (constant-velocity) filter on the front position and
greedy, cost-ordered association. Frames carry *frame numbers*, which may jump
in sampled data; jumps up to `max_bridge` are bridged with the motion model,
longer jumps start an independent segment (all tracks are closed).

Counting rule (matches the annotation rule used for the dataset): consecutive
frames showing the same slug are one slug; a slug may go undetected for up to
`gap_tolerance` images (missed detections) and still be the same slug; a slug
must be seen in at least `min_frames` images to be counted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TrackerConfig:
    conf: float = 0.40          # minimum detection confidence
    gap_tolerance: int = 3      # images a slug may be missed before its track closes
    min_frames: int = 3         # detections needed before a track counts as a slug
    max_bridge: int = 30        # largest jump in frame numbers bridged by the motion model
    match_tol: float = 0.18     # allowed front-position error (fraction of frame width)
    alpha: float = 0.6          # position smoothing of the alpha-beta filter
    beta: float = 0.3           # velocity smoothing of the alpha-beta filter
    edge_margin: int = 6        # px; a box this close to a frame edge is cut by the edge


@dataclass
class Track:
    tid: int
    segment: int
    first_idx: int
    front: float                          # filtered front position (px)
    vel: float = 0.0                      # px per frame (negative = moving left)
    last_idx: int = 0
    last_frame: int = 0
    misses: int = 0
    frames: List[int] = field(default_factory=list)
    idxs: List[int] = field(default_factory=list)
    fronts: List[float] = field(default_factory=list)
    widths: List[float] = field(default_factory=list)
    confs: List[float] = field(default_factory=list)
    entered_from_right: bool = False
    exited_left: bool = False
    cut_at_start: bool = False
    cut_at_end: bool = False

    def add(self, i: int, fnum: int, d: dict, edge: int) -> None:
        self.last_idx, self.last_frame, self.misses = i, fnum, 0
        self.frames.append(fnum)
        self.idxs.append(i)
        self.fronts.append(d["x1"])
        self.widths.append(d["x2"] - d["x1"])
        self.confs.append(d["conf"])
        if d["x1"] <= edge:
            self.exited_left = True


def segment_ids(frame_numbers: List[int], max_bridge: int) -> List[int]:
    """Segment id per image; a new segment starts after a jump > max_bridge."""
    seg, out = 0, []
    for i, f in enumerate(frame_numbers):
        if i > 0 and f - frame_numbers[i - 1] > max_bridge:
            seg += 1
        out.append(seg)
    return out


def track_slugs(frame_numbers: List[int], detections: List[List[dict]],
                width: int, cfg: TrackerConfig) -> Dict:
    """Track slug fronts through a sequence.

    frame_numbers : time index of each image (ascending).
    detections    : per image, list of {"x1","y1","x2","y2","conf"} in pixels.
    width         : frame width in px.

    Returns {"per_frame": [[det + {"slug": id}]], "slugs": [summary], "segments": [id]}.
    """
    n = len(frame_numbers)
    segs = segment_ids(frame_numbers, cfg.max_bridge)
    tol = cfg.match_tol * width
    edge = cfg.edge_margin
    right_edge = width - edge

    all_tracks: List[Track] = []
    active: List[Track] = []
    raw_assign: List[List[dict]] = [[] for _ in range(n)]
    next_id = 1

    for i in range(n):
        fnum = frame_numbers[i]
        new_segment = i == 0 or segs[i] != segs[i - 1]
        if new_segment and active:
            for t in active:
                t.cut_at_end = True
            active = []

        dets = sorted((d for d in detections[i] if d["conf"] >= cfg.conf),
                      key=lambda d: d["x1"])

        # Candidate (track, detection) pairs ordered by front-position error.
        pairs = []
        for ti, t in enumerate(active):
            dt = max(fnum - t.last_frame, 1)
            pred = max(t.front + t.vel * dt, 0.0)
            allowed = tol + abs(t.vel) * (dt - 1)
            for di, d in enumerate(dets):
                err = abs(d["x1"] - pred)
                if err <= allowed:
                    pairs.append((err, ti, di))
        pairs.sort()

        used_t, used_d = set(), set()
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            t, d = active[ti], dets[di]
            dt = max(fnum - t.last_frame, 1)
            pred = max(t.front + t.vel * dt, 0.0)
            resid = d["x1"] - pred
            t.front = pred + cfg.alpha * resid
            if d["x1"] > edge:                     # front still visible: update velocity
                t.vel += cfg.beta * resid / dt
            t.add(i, fnum, d, edge)
            raw_assign[i].append({**d, "tid": t.tid})

        # Unmatched tracks accumulate misses.
        for ti, t in enumerate(active):
            if ti not in used_t:
                t.misses += 1

        # Unmatched detections start new tracks.
        for di, d in enumerate(dets):
            if di in used_d:
                continue
            t = Track(tid=next_id, segment=segs[i], first_idx=i, front=d["x1"])
            next_id += 1
            t.entered_from_right = d["x2"] >= right_edge and d["x1"] > width * 0.5
            t.cut_at_start = new_segment
            t.add(i, fnum, d, edge)
            all_tracks.append(t)
            active.append(t)
            raw_assign[i].append({**d, "tid": t.tid})

        # Close tracks missed for too long; flag tracks still open at segment end.
        last_of_segment = i == n - 1 or segs[i + 1] != segs[i]
        kept = []
        for t in active:
            if t.misses > cfg.gap_tolerance:
                continue
            if last_of_segment:
                t.cut_at_end = True
            kept.append(t)
        active = kept

    confirmed = sorted((t for t in all_tracks if len(t.frames) >= cfg.min_frames),
                       key=lambda t: (t.first_idx, t.fronts[0]))
    relabel = {t.tid: k + 1 for k, t in enumerate(confirmed)}

    per_frame = [[{**d, "slug": relabel[d["tid"]]} for d in row if d["tid"] in relabel]
                 for row in raw_assign]
    rejected = [[d for d in row if d["tid"] not in relabel] for row in raw_assign]
    slugs = [_summarise(t, relabel[t.tid], cfg) for t in confirmed]
    return {"per_frame": per_frame, "rejected": rejected, "slugs": slugs, "segments": segs}


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


def _summarise(t: Track, sid: int, cfg: TrackerConfig) -> dict:
    return {
        "slug": sid,
        "segment": t.segment,
        "first_frame": t.frames[0],
        "last_frame": t.frames[-1],
        "frames_detected": len(t.frames),
        "span_frames": t.frames[-1] - t.frames[0] + 1,
        "front_velocity_px_per_frame": _fit_velocity(t.frames, t.fronts, cfg.edge_margin),
        "max_extent_px": max(t.widths),
        "mean_conf": sum(t.confs) / len(t.confs),
        "entered_from_right": t.entered_from_right,
        "exited_left": t.exited_left,
        "cut_at_start": t.cut_at_start,
        "cut_at_end": t.cut_at_end,
        "complete": not t.cut_at_start and not t.cut_at_end,
        "first_idx": t.idxs[0],
        "last_idx": t.idxs[-1],
        "track": [{"idx": i, "frame": f, "front": x}
                  for i, f, x in zip(t.idxs, t.frames, t.fronts)],
    }
