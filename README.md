# Slug flow tracker

A Streamlit app that detects slugs in two-phase pipe-flow footage with the trained YOLOv8s model, tracks each
slug's front from right to left, counts slugs and reports frequency, presence and front speed.

- **Demo:** the 500 annotated lab frames play like a video with live boxes, a running slug count and a
  space-time diagram.
- **Your own footage:** upload at least 125 images (ordered by the number in their file name) or a video.
  For a video, the frame rate is read from the file, so slug frequency comes out per minute.

## Deploy on Streamlit Community Cloud (free, gives a public link)

1. **Create a GitHub repository.** Go to github.com, choose **New repository**, name it `slug-flow-tracker`,
   keep it **Public**, and create it.
2. **Upload the files.** On the new repository page choose **uploading an existing file** and drag in
   everything from this folder: `app.py`, `requirements.txt`, `packages.txt`, `precompute_demo.py`,
   `README.md` and the folders `slugtrack`, `models`, `data`, `tests`. Choose **Commit changes**.
3. **Add the settings file.** The `.streamlit` folder is hidden on most computers, so add it by hand: choose
   **Add file → Create new file**, type the name `.streamlit/config.toml`, paste the contents of
   `.streamlit/config.toml` from this folder, and commit. (Without it the app still works, but loads the demo
   more slowly.)
4. **Deploy.** Go to share.streamlit.io, sign in with GitHub, choose **Create app → Deploy a public app from
   GitHub**, pick the repository, branch `main`, main file `app.py`. Under **Advanced settings** choose
   Python 3.12. Choose **Deploy**. The first build takes about 5–10 minutes.
5. Share the `…streamlit.app` link.

The demo opens instantly because detections for the 500 frames are pre-computed in `data/demo_detections.json`.
Uploads run the model live on the server's CPU, roughly 0.2–0.4 s per frame, so 125 frames take under a minute.

## Run on your own computer

```bash
pip install -r requirements.txt
streamlit run app.py
```

## What the numbers mean

| Shown as | Meaning |
| --- | --- |
| Slugs counted | Distinct slug tracks (one per slug passage), including slugs only partly in view |
| Slug frequency | Slugs per 1000 frames; slugs per minute when the frame rate is known |
| Frames with a slug | Share of frames with a slug in view (flow intermittency for continuous footage) |
| Front speed | Leading-edge velocity from a straight-line fit of front position against frame number |
| Longest visible slug | Largest box width in cm; slugs longer than the view are cut by the frame edge |

The original camera frame rate is not known for the image dataset, so time-based values are reported per frame.
Tick **I know the camera frame rate** in the sidebar to convert them. Lengths use **11.0 px per cm**, measured
from the ruler (centimetre scale) in the 960×480 lab frames; change it for other camera setups.

## How tracking and counting work

- Flow runs right to left, so the left edge of each detection box is the slug **front**.
- Each front is followed with a constant-velocity (alpha-beta) filter. A new box is matched to the front it
  best continues; a box whose front appears somewhere unexpected, usually back on the right, is a new slug.
- Counting follows the annotation rule: consecutive frames showing a slug are one slug; a slug that reappears
  after frames without one is a new slug. Up to 3 missed detections are tolerated (adjustable), and a track
  must last 3 frames to count.
- File names are read as frame numbers. Gaps of up to 30 missing frame numbers are crossed using the predicted
  front position; longer gaps split the footage into separate intervals.

## About the bundled 500 frames

They come from four stretches of one recording: frames 0–124 and 251–375 (no slug), 714–838 (one slug
passing) and 11055–11240 (one slug passing, some frames not sampled). Half were chosen because they show a slug,
so the frequency for this sample describes the sampling, not the flow. 400 of them were used to train the model.
On all 500 the model separates slug from no-slug frames without error at confidence 0.25–0.5, and the two
slugs have front speeds of 0.51 and 0.56 cm/frame.

## Files

| Path | Contents |
| --- | --- |
| `app.py` | Streamlit interface |
| `slugtrack/detector.py` | YOLOv8 wrapper |
| `slugtrack/tracker.py` | Front tracker and slug counting |
| `slugtrack/metrics.py` | Frequency, presence, speed, length |
| `slugtrack/pipeline.py` | Loading images and video, preparing the player |
| `slugtrack/player.html` | Video player and space-time diagram |
| `models/slug_yolov8s_500img.pt` | Trained weights (500-image YOLOv8s) |
| `data/demo_detections.json` | Pre-computed detections for the 500 demo frames |
| `data/demo_web_frames.zip` | The 500 demo frames, compressed for the browser |
| `precompute_demo.py` | Re-creates `demo_detections.json` after you change the weights |
| `tests/test_tracker.py` | Checks of the counting logic (`python -m pytest -q`) |
