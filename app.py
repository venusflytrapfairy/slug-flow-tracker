"""Slug flow tracker: Streamlit demo.

Run locally:   streamlit run app.py
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from slugtrack.pipeline import (WEIGHTS, analyse, inline_frames, load_demo, load_uploaded_images,
                                load_uploaded_video, player_payload, render_player_html)
from slugtrack.tracker import TrackerConfig

MIN_IMAGES = 125
DEFAULT_PX_PER_CM = 11.0

st.set_page_config(page_title="Slug flow tracker", page_icon="🌊", layout="wide")

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,400..800&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"], .stApp, .stMarkdown, button, input, select, textarea {
  font-family: "Archivo", system-ui, sans-serif !important; font-variant-numeric: tabular-nums; }
.block-container { padding-top: 2.2rem; max-width: 1240px; }
h1 { font-stretch: 118%; font-weight: 800 !important; letter-spacing: -0.015em; margin-bottom: .1rem !important; }
.lede { color: #4B5B64; font-size: 1.02rem; max-width: 78ch; margin: 0 0 1.1rem; }
.mgrid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: .2rem 0 .6rem; }
.mcard { background: #F6F8F7; border: 1px solid #C9D2D0; border-radius: 6px; padding: 10px 14px; }
.mcard .mk { color: #4B5B64; font-size: .88rem; margin: 0 0 2px; }
.mcard .mv { font-stretch: 115%; font-weight: 750; font-size: 2rem; line-height: 1.1; margin: 0; color: #1B2A33; }
.mcard .mv span { font-stretch: 100%; font-weight: 500; font-size: .9rem; color: #4B5B64; margin-left: 6px; }
[data-testid="stMetric"] { background: #F6F8F7; border: 1px solid #C9D2D0; border-radius: 6px; padding: 10px 14px; }
[data-testid="stMetricValue"] { font-stretch: 112%; font-weight: 700; }
[data-testid="stMetricLabel"] p { color: #4B5B64; }
.note { border-left: 3px solid #C98A12; background: #F6F8F7; padding: 10px 14px; border-radius: 0 6px 6px 0;
        color: #1B2A33; font-size: .95rem; max-width: 92ch; margin: .4rem 0 1rem; }
.note p { margin: 0 0 .35rem; } .note p:last-child { margin: 0; }
section[data-testid="stSidebar"] h3 { font-stretch: 110%; font-weight: 700; margin-top: .6rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_detector():
    from slugtrack.detector import SlugDetector
    return SlugDetector(WEIGHTS)


@st.cache_data(show_spinner=False)
def get_demo():
    return load_demo(get_detector)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### Footage")
    source = st.radio("Footage", ["Sampled lab frames (500)", "Upload images", "Upload video"],
                      label_visibility="collapsed")

    run = False
    files = video = None
    step = 1
    if source == "Upload images":
        files = st.file_uploader(
            f"Frames in time order (at least {MIN_IMAGES})", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
            accept_multiple_files=True,
            help="Files are ordered by the number in their name, e.g. Img000123.jpg. "
                 "Jumps in that number are treated as missing frames.")
        n_files = len(files) if files else 0
        if files and n_files < MIN_IMAGES:
            st.caption(f"{n_files} added. Add {MIN_IMAGES - n_files} more to run tracking.")
        run = st.button("Track slugs", type="primary", disabled=n_files < MIN_IMAGES,
                        width="stretch")
    elif source == "Upload video":
        video = st.file_uploader("Video file", type=["mp4", "avi", "mov", "mkv"])
        step = st.slider("Analyse every Nth frame", 1, 10, 1,
                         help="Higher values are faster. Keep 1 for fast flows so no slug is skipped.")
        st.caption("Up to 1,500 analysed frames per upload.")
        run = st.button("Track slugs", type="primary", disabled=video is None,
                        width="stretch")

    st.markdown("### Counting rules")
    conf = st.slider("Detector confidence threshold", 0.10, 0.90, 0.40, 0.05,
                     help="Detections below this score are ignored.")
    gap = st.slider("Missed frames allowed inside one slug", 0, 10, 3,
                    help="A slug can go undetected for this many frames and still count as the same slug.")
    min_frames = st.slider("Frames needed to count a slug", 1, 15, 3,
                           help="Shorter detections are treated as noise.")
    max_bridge = st.number_input("Largest jump in frame numbers to bridge", 1, 500, 30,
                                 help="Inside a jump this small the tracker predicts where the slug front moved. "
                                      "Larger jumps start a new, separate interval.")

    st.markdown("### Units")
    px_per_cm = st.number_input("Pixels per cm", 1.0, 200.0, DEFAULT_PX_PER_CM, 0.1,
                                help="11.0 px/cm was measured from the ruler in the 960×480 lab frames. "
                                     "Change it for footage from another camera setup.")
    src_state = st.session_state.get("src")
    video_fps = src_state.get("fps") if (src_state and source == "Upload video") else None
    know_fps = st.checkbox("I know the camera frame rate", value=bool(video_fps))
    fps = None
    if know_fps:
        fps = st.number_input("Frames per second", 1.0, 100000.0, float(video_fps or 1000.0), 1.0,
                              help="Read automatically from an uploaded video. For image sequences, "
                                   "use the camera's recording rate, not the playback rate.")

# ------------------------------------------------------------------ data
cfg = TrackerConfig(conf=conf, gap_tolerance=gap, min_frames=min_frames, max_bridge=int(max_bridge))

if run:
    bar = st.progress(0.0, text="Starting")
    cb = lambda frac, text: bar.progress(min(max(frac, 0.0), 1.0), text=text)
    try:
        if source == "Upload images":
            st.session_state["src"] = load_uploaded_images(files, get_detector(), cb)
        else:
            st.session_state["src"] = load_uploaded_video(video.getvalue(), Path(video.name).suffix,
                                                          get_detector(), step=step, progress=cb)
        st.session_state["src_kind"] = source
        bar.empty()
        st.rerun()
    except ValueError as e:
        bar.empty()
        st.error(str(e))

if source == "Sampled lab frames (500)":
    with st.spinner("Loading the sampled frames"):
        src = get_demo()
elif st.session_state.get("src_kind") == source:
    src = st.session_state["src"]
else:
    src = None

# ------------------------------------------------------------------ header
st.title("Slug flow tracker")
st.markdown('<p class="lede">Your YOLOv8 model finds the slug body in every frame. The tracker then follows '
            'each slug’s front as it travels from right to left and counts it once, '
            'even if a few frames are missed.</p>', unsafe_allow_html=True)

if src is None:
    if source == "Upload images":
        st.markdown(f"Add at least {MIN_IMAGES} frames in the sidebar, then choose **Track slugs**.")
    else:
        st.markdown("Add a video in the sidebar, then choose **Track slugs**.")
    st.stop()

result = analyse(src, cfg, px_per_cm, fps)
m, tr = result["metrics"], result["track"]

if src["kind"] == "video" and src.get("truncated"):
    st.warning("Only the first 1,500 analysed frames were used. Raise “Analyse every Nth frame” to cover the whole video.")

payload = player_payload(src, result, px_per_cm, fps)
if not st.get_option("server.enableStaticServing"):
    # .streamlit/config.toml missing: still works, just loads slower.
    payload = inline_frames(payload)
st.iframe(render_player_html(payload), height="content")

# ------------------------------------------------------------------ metrics
def f(v, d=2, unit=""):
    return "–" if v is None else f"{v:,.{d}f}{unit}"

def card(label, value, unit, tip):
    return (f'<div class="mcard" title="{tip}"><p class="mk">{label}</p>'
            f'<p class="mv">{value}<span>{unit}</span></p></div>')

cards = [card("Slugs counted", m["slug_count"], "",
              "Distinct slugs tracked, including slugs only partly in view.")]
if fps:
    cards.append(card("Slug frequency", f(m["slugs_per_minute"], 1), "per min",
                      f"Over {f(m['observed_seconds'], 2)} s of observed footage at {fps:g} fps."))
else:
    cards.append(card("Slug frequency", f(m["slugs_per_1000_frames"], 1), "per 1000 frames",
                      "Frame-based because the frame rate is unknown. Tick the frame-rate box in the sidebar for slugs per minute."))
cards.append(card("Frames with a slug", f(100 * m["presence_fraction"], 0), "%",
                  "Share of frames with a slug in view. For continuous footage this is the flow intermittency."))
if fps:
    cards.append(card("Front speed", f(m.get("front_velocity_cm_per_s"), 1), "cm/s",
                      "Mean leading-edge speed, fitted over each slug's track."))
else:
    cards.append(card("Front speed", f(m["front_velocity_cm_per_frame"], 2), "cm/frame",
                      "Mean leading-edge speed, fitted over each slug's track. Multiply by the frame rate for cm/s."))
cards.append(card("Longest visible slug", f(m["max_visible_length_cm"], 0), "cm",
                  "Largest box width seen. Slugs longer than the view are cut by the frame edge."))
st.markdown('<div class="mgrid">' + "".join(cards) + "</div>", unsafe_allow_html=True)

if src["kind"] == "demo":
    st.markdown(
        '<div class="note"><p><b>About these 500 frames.</b> They come from four separate stretches of a longer '
        'recording (frames 0–124, 251–375, 714–838 and 11055–11240), hatched in the diagram. The tracker handles '
        'each stretch on its own. Half the frames were picked because they show a slug, so the frequency above '
        'describes this sample, not the flow. Upload an unbroken recording to measure real slug frequency.</p>'
        '<p>The two slugs here move at nearly the same front speed, which is a useful sanity check. '
        '400 of the 500 frames were also used to train the model, so detection here is not a test of how well '
        'it generalises.</p></div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ tables
slug_rows = []
for s in tr["slugs"]:
    v = s["front_velocity_px_per_frame"]
    v_cm = abs(v) / px_per_cm if v is not None else None
    row = {
        "Slug": s["slug"],
        "First frame": s["first_frame"],
        "Last frame": s["last_frame"],
        "Frames detected": s["frames_detected"],
        "Front velocity (cm/frame)": round(v_cm, 3) if v_cm is not None else None,
    }
    if fps:
        row["Front velocity (cm/s)"] = round(v_cm * fps, 2) if v_cm is not None else None
        row["Passage time (s)"] = round(s["span_frames"] / fps, 4)
    row.update({
        "Longest visible length (cm)": round(s["max_extent_px"] / px_per_cm, 1),
        "Mean confidence": round(s["mean_conf"], 3),
        "Whole passage seen": "yes" if s["complete"] else
            ("no, already in view at start" if s["cut_at_start"] and not s["cut_at_end"] else
             "no, still in view at end" if s["cut_at_end"] and not s["cut_at_start"] else
             "no, cut at both ends"),
    })
    slug_rows.append(row)
slug_df = pd.DataFrame(slug_rows)

frame_rows = []
for i, name in enumerate(src["names"]):
    boxes = tr["per_frame"][i]
    b = boxes[0] if boxes else None
    frame_rows.append({
        "image": name, "frame": src["frame_numbers"][i], "interval": tr["segments"][i] + 1,
        "slug_present": bool(boxes), "slug": b["slug"] if b else None,
        "confidence": b["conf"] if b else None,
        "front_x_px": b["x1"] if b else None, "tail_x_px": b["x2"] if b else None,
        "visible_length_cm": round((b["x2"] - b["x1"]) / px_per_cm, 2) if b else None,
        "boxes": len(boxes),
    })
frame_df = pd.DataFrame(frame_rows)

tab1, tab2, tab3 = st.tabs(["Slugs", "Every frame", "How it works"])
with tab1:
    if slug_df.empty:
        st.markdown("No slugs were counted with the current settings. Try lowering the confidence threshold.")
    else:
        st.dataframe(slug_df, hide_index=True, width="stretch")
with tab2:
    st.dataframe(frame_df, hide_index=True, width="stretch", height=360)
with tab3:
    st.markdown(f"""
**Detection.** The YOLOv8s model trained on the 500 annotated frames (mAP50 0.995, mAP50-95 0.888 on its
validation split) draws a box around the slug body in each frame.

**Tracking.** Flow runs right to left, so the left edge of each box is the slug front. The tracker keeps a
constant-velocity estimate of every front and matches each new box to the front it best continues. A box whose
front appears somewhere unexpected, usually back at the right edge, starts a new slug. Gaps in frame numbering
of up to {int(max_bridge)} frames are crossed using the predicted front position; longer gaps split the footage
into separate intervals.

**Counting.** One track is one slug, the same rule used when annotating: consecutive frames showing a slug are
one slug, and a slug that reappears after frames without one is a new slug. Up to {gap} missed detections are
tolerated, and a track must last at least {min_frames} frames to count.

**Units without a frame rate.** Frequency is reported per 1000 frames and speed in cm per frame. Lengths use
the ruler: {px_per_cm:g} px per cm. Enter the camera frame rate to convert to slugs per minute and cm/s.
""")

d1, d2, d3 = st.columns([1, 1, 3])
d1.download_button("Download slug table (CSV)", slug_df.to_csv(index=False), "slugs.csv", "text/csv",
                   disabled=slug_df.empty)
d2.download_button("Download frame data (CSV)", frame_df.to_csv(index=False), "frames.csv", "text/csv")
