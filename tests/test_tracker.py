"""Checks of the slug tracker. Run: python -m pytest -q"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from slugtrack.metrics import compute_metrics          # noqa: E402
from slugtrack.tracker import TrackerConfig, merge_frame, track_slugs  # noqa: E402

W = 960


def slug_box(front, tail=W, conf=0.9):
    return {"x1": float(max(front, 0)), "y1": 235.0, "x2": float(min(tail, W)), "y2": 300.0, "conf": conf}


def passage(n_frames, speed=6.0, start=W - 20, length=500):
    """Detections of one slug entering on the right and moving left."""
    out = []
    for k in range(n_frames):
        front = start - speed * k
        tail = front + length if front + length < W else W
        out.append([] if tail <= 0 else [slug_box(front, tail)])
    return out


def run(dets, frames=None, **kw):
    frames = frames or list(range(len(dets)))
    return track_slugs(frames, dets, W, TrackerConfig(**kw))


def max_slugs_in_any_frame(r):
    return max((len(row) for row in r["per_frame"]), default=0)


# ---------------------------------------------------------------- one slug per frame
def test_two_boxes_in_one_frame_become_one_slug_box():
    d = [[slug_box(300, 500, 0.8), slug_box(600, 900, 0.6)] for _ in range(20)]
    r = run(d)
    assert len(r["slugs"]) == 1
    assert max_slugs_in_any_frame(r) == 1
    box = r["per_frame"][0][0]
    assert (box["x1"], box["x2"], box["conf"], box["merged"]) == (300, 900, 0.8, 2)
    assert r["slugs"][0]["frames_with_merged_boxes"] == 20


def test_low_confidence_extra_box_is_not_merged():
    box = merge_frame([slug_box(300, 500, 0.9), slug_box(700, 900, 0.2)], conf=0.4)
    assert (box["x1"], box["x2"], box["merged"]) == (300, 500, 1)


def test_new_front_on_the_right_without_a_break_is_still_one_slug():
    # A second front entering while the first is in view used to count as a new
    # slug. Without frames free of slug in between, it is the same slug.
    first = passage(170, length=300)
    second = [[] for _ in range(110)] + passage(60, length=300)
    d = [a + b for a, b in zip(first, second)]
    r = run(d)
    assert len(r["slugs"]) == 1
    assert max_slugs_in_any_frame(r) == 1


def test_front_jump_without_break_is_same_slug():
    d = passage(80, speed=12, length=900) + [[slug_box(820)] for _ in range(20)]
    assert len(run(d)["slugs"]) == 1


# ---------------------------------------------------------------- interruptions
def test_empty_sequence_counts_nothing():
    assert run([[] for _ in range(125)])["slugs"] == []


def test_one_passage_is_one_slug_with_right_speed():
    r = run(passage(120))
    assert len(r["slugs"]) == 1
    assert abs(r["slugs"][0]["front_velocity_px_per_frame"] + 6.0) < 0.2


def test_two_slugs_separated_by_liquid_film():
    d = passage(60) + [[] for _ in range(30)] + passage(60)
    r = run(d)
    assert [s["slug"] for s in r["slugs"]] == [1, 2]
    assert max_slugs_in_any_frame(r) == 1


def test_short_dropout_is_a_missed_detection_not_a_new_slug():
    d = passage(100)
    for k in (30, 31, 60):
        d[k] = []
    r = run(d, gap_tolerance=3)
    assert len(r["slugs"]) == 1
    assert r["active"][30] == r["active"][31] == 1      # bridged frames still belong to slug 1


def test_dropout_longer_than_tolerance_is_an_interruption():
    d = passage(100)
    for k in range(40, 45):                              # 5 empty frames
        d[k] = []
    assert len(run(d, gap_tolerance=3)["slugs"]) == 2
    assert len(run(d, gap_tolerance=5)["slugs"]) == 1


def test_strict_mode_any_empty_frame_interrupts():
    d = passage(60)
    d[30] = []
    assert len(run(d, gap_tolerance=0)["slugs"]) == 2


def test_single_frame_false_alarm_is_ignored():
    d = [[] for _ in range(50)]
    d[20] = [slug_box(500, 700, conf=0.95)]
    r = run(d, min_frames=3)
    assert r["slugs"] == []
    assert r["rejected"][20]


def test_low_confidence_is_filtered():
    d = [[slug_box(900 - 6 * k, conf=0.2)] for k in range(40)]
    assert run(d, conf=0.4)["slugs"] == []
    assert len(run(d, conf=0.1)["slugs"]) == 1


# ---------------------------------------------------------------- frame-number gaps
def test_large_frame_jump_starts_a_new_interval():
    d = passage(40) + passage(40)
    frames = list(range(40)) + list(range(1000, 1040))
    r = run(d, frames=frames, max_bridge=30)
    assert max(r["segments"]) == 1
    assert len(r["slugs"]) == 2
    assert r["slugs"][0]["cut_at_end"] and r["slugs"][1]["cut_at_start"]


def test_missing_frame_numbers_are_not_an_interruption():
    full = passage(120)
    keep = [k for k in range(120) if not (40 <= k < 60)]   # 20 frame numbers never observed
    r = run([full[k] for k in keep], frames=keep, max_bridge=30)
    assert len(r["slugs"]) == 1


# ---------------------------------------------------------------- metrics
def test_metrics_frame_based_and_with_fps():
    d = passage(60) + [[] for _ in range(40)] + passage(60)
    r = run(d)
    frames = list(range(len(d)))
    m = compute_metrics(frames, r["per_frame"], r["slugs"], r["segments"], 11.0, None)
    assert m["slug_count"] == 2
    assert abs(m["slugs_per_1000_frames"] - 2 / 160 * 1000) < 1e-9
    assert m["mean_period_frames"] == 100
    m2 = compute_metrics(frames, r["per_frame"], r["slugs"], r["segments"], 11.0, 500.0)
    assert abs(m2["slugs_per_minute"] - 2 / (160 / 500) * 60) < 1e-9
    assert abs(m2["front_velocity_cm_per_s"] - 6 / 11 * 500) < 2
