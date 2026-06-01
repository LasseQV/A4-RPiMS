#!/usr/bin/env python3
"""
focus_tool.py — Interactive multispectral focus assistant.

Streams a live MJPEG feed from the stitched 4x OV9281 camera, overlays ArUco
marker detection and a Laplacian-variance focus score, and steps through each
of the four camera bands so you can dial in focus lens-by-lens.

The 5120×800 frame is laid out as four side-by-side 1280×800 bands.
Band layout (left → right):
    Band 0: cols   0–1279
    Band 1: cols 1280–2559
    Band 2: cols 2560–3839
    Band 3: cols 3840–5119

Usage:
    python3 focus_tool.py

Then open http://<raspberry-pi-ip>:5000 in a browser on the same network.

Requires:
    pip install flask picamera2 opencv-contrib-python numpy
"""

import io
import json
import os
import time
import threading

import cv2
import cv2.aruco as aruco
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request
from picamera2 import Picamera2

# ── Configuration ─────────────────────────────────────────────────────────────
FRAME_WIDTH  = 5120
FRAME_HEIGHT = 800
NUM_BANDS    = 4
BAND_WIDTH   = FRAME_WIDTH // NUM_BANDS   # 1280 px per band

# ArUco — use marker ID 0 from the chosen dictionary as the focus target.
# Must match aruco_field_config.py in your project.
ARUCO_DICT_ID = aruco.DICT_4X4_50

STREAM_QUALITY   = 80    # JPEG quality for the MJPEG stream
STREAM_MAX_FPS   = 15    # Cap stream rate to avoid saturating the Pi's CPU
DETECTION_SCALE  = 0.5   # Downscale factor for ArUco detection pass

# Laplacian focus score thresholds (tune for your optics / target distance)
FOCUS_POOR       = 50
FOCUS_ACCEPTABLE = 150
FOCUS_GOOD       = 300

# Alignment estimation
# How many frames of corner observations to average before producing a stable
# offset estimate. More = smoother but slower to update.
ALIGNMENT_AVERAGE_FRAMES = 30

# Flat-field / dark-frame calibration
# Number of frames to average when capturing a calibration frame.
CAL_AVERAGE_FRAMES = 20
# Directory to persist calibration arrays alongside this script.
CAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration")

# ── Global state (protected by a lock) ────────────────────────────────────────
state_lock = threading.Lock()
state = {
    "active_band": 0,          # Which band is currently being focused (0-3)
    "scores": [0.0] * NUM_BANDS,
    "marker_found": [False] * NUM_BANDS,
    "marker_corners": [None] * NUM_BANDS,
    # Alignment: running centroid accumulators, one per band [(x_sum, y_sum, n), ...]
    # Offset is relative to Band 0 centroid. Band 0 offset is always (0, 0).
    "centroid_acc": [(0.0, 0.0, 0) for _ in range(NUM_BANDS)],
    # Current smoothed offset estimates (dx, dy) relative to Band 0, in band pixels
    "offsets": [(0.0, 0.0)] * NUM_BANDS,
    # Whether the user has locked (saved) the alignment for export
    "alignment_locked": False,
    "locked_offsets": [(0.0, 0.0)] * NUM_BANDS,

    # Flat-field calibration arrays, per band (float32, band pixel dimensions).
    # dark_frame:  additive offset (dark current + stray light) — subtract first.
    # flat_gain:   multiplicative vignette gain map — multiply after dark subtraction.
    # Both are None until the respective calibration capture completes.
    "dark_frames": [None] * NUM_BANDS,
    "flat_gains":  [None] * NUM_BANDS,
    # Per-band calibration status strings for the UI
    "cal_status": ["uncal"] * NUM_BANDS,   # "uncal" | "dark_ok" | "cal_ok"
    # Calibration capture in progress: None or {"type": "dark"|"flat", "band": i,
    #   "acc": ndarray, "n": int}
    "cal_capture": None,
}

app = Flask(__name__)

# ── Camera ────────────────────────────────────────────────────────────────────

def init_camera() -> Picamera2:
    picam2 = Picamera2()
    config = picam2.create_still_configuration(
        main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "YUV420"},
        lores={"size": (FRAME_WIDTH // 2, FRAME_HEIGHT // 2), "format": "YUV420"},
        display="lores",
    )
    picam2.configure(config)
    picam2.set_controls({
        "AeEnable":           True,   # Keep AE on for focus tool — lighting varies
        "NoiseReductionMode": 0,      # Disable denoise to keep focus metric raw
    })
    picam2.start()
    time.sleep(1)
    return picam2


# ── Image processing helpers ───────────────────────────────────────────────────

def extract_band(y_full: np.ndarray, band_index: int) -> np.ndarray:
    """Slice the full-width grayscale Y plane to a single band."""
    x0 = band_index * BAND_WIDTH
    x1 = x0 + BAND_WIDTH
    return y_full[:, x0:x1]


def laplacian_variance(gray: np.ndarray) -> float:
    """Focus metric: variance of the Laplacian. Higher = sharper."""
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def detect_aruco(gray: np.ndarray, detector: aruco.ArucoDetector):
    """
    Run ArUco detection on a (possibly downscaled) band image.
    Returns (corners, ids) in original-band pixel coordinates.
    """
    h, w = gray.shape
    small = cv2.resize(gray, (int(w * DETECTION_SCALE), int(h * DETECTION_SCALE)))
    corners_s, ids, _ = detector.detectMarkers(small)

    if ids is None:
        return None, None

    # Scale corners back to full band resolution
    scale = 1.0 / DETECTION_SCALE
    corners_full = [c * scale for c in corners_s]
    return corners_full, ids


def score_label(score: float) -> str:
    if score >= FOCUS_GOOD:
        return "good"
    if score >= FOCUS_ACCEPTABLE:
        return "ok"
    return "poor"


def marker_centroid(corners) -> tuple[float, float] | None:
    """
    Return the centroid (cx, cy) of the first detected marker's four corners.
    corners is the list returned by detectMarkers, already scaled to band pixels.
    Returns None if corners is empty.
    """
    if not corners:
        return None
    # corners[0] shape: (1, 4, 2)
    pts = corners[0][0]   # shape (4, 2)
    cx = float(pts[:, 0].mean())
    cy = float(pts[:, 1].mean())
    return cx, cy


def update_alignment(band_centroids: list) -> None:
    """
    Given a list of (cx, cy)|None per band, accumulate centroid observations
    and recompute smoothed offsets relative to Band 0.
    Modifies state in-place (caller must hold state_lock).
    """
    acc = state["centroid_acc"]

    for i, pt in enumerate(band_centroids):
        if pt is None:
            continue
        cx, cy = pt
        xs, ys, n = acc[i]
        # Rolling average capped at ALIGNMENT_AVERAGE_FRAMES
        if n < ALIGNMENT_AVERAGE_FRAMES:
            acc[i] = (xs + cx, ys + cy, n + 1)
        else:
            # Discard oldest sample weight — simple exponential decay approximation
            alpha = 1.0 / ALIGNMENT_AVERAGE_FRAMES
            acc[i] = (
                xs * (1 - alpha) + cx * alpha * ALIGNMENT_AVERAGE_FRAMES,
                ys * (1 - alpha) + cy * alpha * ALIGNMENT_AVERAGE_FRAMES,
                n,
            )

    state["centroid_acc"] = acc

    # Recompute offsets relative to Band 0
    ref_n = acc[0][2]
    if ref_n == 0:
        return   # No Band 0 observation yet

    ref_cx = acc[0][0] / ref_n
    ref_cy = acc[0][1] / ref_n

    offsets = []
    for i in range(NUM_BANDS):
        xs, ys, n = acc[i]
        if n == 0:
            offsets.append((0.0, 0.0))
        else:
            offsets.append((xs / n - ref_cx, ys / n - ref_cy))

    state["offsets"] = offsets


def apply_calibration(band_gray: np.ndarray, band_index: int) -> np.ndarray:
    """
    Apply dark-frame subtraction and flat-field gain correction to a raw band.

    Pipeline (in order):
      1. Convert to float32.
      2. Subtract dark frame (removes additive bias: dark current, stray light,
         inter-lens reflections in the NIR band).
      3. Multiply by flat gain map (corrects multiplicative vignetting).
      4. Clip to [0, 255] and return as uint8.

    If either calibration array is absent the corresponding step is skipped,
    so partial calibration (dark only, or uncalibrated) works transparently.
    """
    img = band_gray.astype(np.float32)

    dark = state["dark_frames"][band_index]
    if dark is not None:
        img = img - dark
        np.clip(img, 0, None, out=img)   # prevent negative values before gain

    gain = state["flat_gains"][band_index]
    if gain is not None:
        img = img * gain

    np.clip(img, 0, 255, out=img)
    return img.astype(np.uint8)


def accumulate_cal_frame(band_gray: np.ndarray, band_index: int) -> bool:
    """
    Feed one frame into the active calibration capture for the given band.

    Returns True when enough frames have been averaged and the calibration
    array has been committed to state (caller should update cal_status).
    Modifies state["cal_capture"] in-place; caller must hold state_lock.
    """
    cap = state["cal_capture"]
    if cap is None or cap["band"] != band_index:
        return False

    frame_f = band_gray.astype(np.float32)

    if cap["acc"] is None:
        cap["acc"] = frame_f.copy()
        cap["n"] = 1
    else:
        cap["acc"] += frame_f
        cap["n"] += 1

    if cap["n"] < CAL_AVERAGE_FRAMES:
        return False

    # Enough frames — compute final array
    averaged = cap["acc"] / cap["n"]

    if cap["type"] == "dark":
        state["dark_frames"][band_index] = averaged
        state["cal_status"][band_index] = "dark_ok"
        _save_cal_npy("dark", band_index, averaged)

    elif cap["type"] == "flat":
        dark = state["dark_frames"][band_index]
        if dark is not None:
            flat_corrected = averaged - dark
            np.clip(flat_corrected, 1, None, out=flat_corrected)  # avoid div/0
        else:
            flat_corrected = averaged.copy()
            np.clip(flat_corrected, 1, None, out=flat_corrected)

        # Gain map: scale so mean gain = 1 (preserves overall brightness)
        mean_val = float(flat_corrected.mean())
        gain_map = mean_val / flat_corrected
        state["flat_gains"][band_index] = gain_map
        state["cal_status"][band_index] = "cal_ok"
        _save_cal_npy("flat_gain", band_index, gain_map)

    state["cal_capture"] = None
    return True


def _save_cal_npy(name: str, band_index: int, arr: np.ndarray) -> None:
    """Persist a calibration array to CAL_DIR as a .npy file."""
    try:
        os.makedirs(CAL_DIR, exist_ok=True)
        path = os.path.join(CAL_DIR, f"{name}_band{band_index}.npy")
        np.save(path, arr)
        print(f"[CAL] Saved {path}")
    except OSError as e:
        print(f"[CAL] Could not save {name} band {band_index}: {e}")


def load_saved_calibration() -> None:
    """
    On startup, reload any previously saved calibration arrays from CAL_DIR.
    Allows you to persist calibration across tool restarts.
    """
    if not os.path.isdir(CAL_DIR):
        return
    for i in range(NUM_BANDS):
        dark_path = os.path.join(CAL_DIR, f"dark_band{i}.npy")
        gain_path = os.path.join(CAL_DIR, f"flat_gain_band{i}.npy")
        dark_loaded = gain_loaded = False
        if os.path.exists(dark_path):
            state["dark_frames"][i] = np.load(dark_path)
            dark_loaded = True
        if os.path.exists(gain_path):
            state["flat_gains"][i] = np.load(gain_path)
            gain_loaded = True
        if gain_loaded:
            state["cal_status"][i] = "cal_ok"
        elif dark_loaded:
            state["cal_status"][i] = "dark_ok"
    print(f"[CAL] Loaded saved calibration from {CAL_DIR}")


def render_band_overlay(
    band_bgr: np.ndarray,
    score: float,
    corners,
    ids,
    is_active: bool,
    band_index: int,
) -> np.ndarray:
    """Draw focus score, ArUco overlay, and active-band highlight onto a band."""
    out = band_bgr.copy()
    h, w = out.shape[:2]

    # Active band border
    if is_active:
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 255, 128), 6)

    # ArUco overlay
    if corners and ids is not None:
        aruco.drawDetectedMarkers(out, corners, ids)

    # Focus score badge (top-left)
    label = score_label(score)
    color_map = {"good": (0, 220, 80), "ok": (0, 180, 255), "poor": (0, 60, 220)}
    badge_color = color_map[label]

    badge_text = f"Band {band_index}  |  {score:.0f}"
    (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(out, (8, 8), (tw + 24, th + 24), (0, 0, 0), -1)
    cv2.putText(out, badge_text, (16, th + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, badge_color, 2, cv2.LINE_AA)

    # Focus bar (right edge of band)
    bar_h = int(min(score / FOCUS_GOOD, 1.0) * (h - 20))
    bar_x = w - 18
    cv2.rectangle(out, (bar_x, h - 10 - bar_h), (bar_x + 10, h - 10), badge_color, -1)
    cv2.rectangle(out, (bar_x, 10), (bar_x + 10, h - 10), (80, 80, 80), 1)

    return out


# ── Frame generator ────────────────────────────────────────────────────────────

def generate_frames(picam2: Picamera2, detector: aruco.ArucoDetector):
    interval = 1.0 / STREAM_MAX_FPS
    last_frame_time = 0.0

    while True:
        now = time.time()
        elapsed = now - last_frame_time
        if elapsed < interval:
            time.sleep(interval - elapsed)
        last_frame_time = time.time()

        yuv = picam2.capture_array("main")
        y_full = yuv[0:FRAME_HEIGHT, :]   # grayscale Y plane

        with state_lock:
            active = state["active_band"]

        band_scores = []
        band_corners = []
        band_found = []
        band_bgrs = []

        for i in range(NUM_BANDS):
            band_gray_raw = extract_band(y_full, i)

            # Feed calibration accumulator while a capture is in progress
            with state_lock:
                cap = state["cal_capture"]
                if cap is not None and cap["band"] == i:
                    accumulate_cal_frame(band_gray_raw, i)

            # Apply dark + flat correction for display and focus scoring
            with state_lock:
                band_gray = apply_calibration(band_gray_raw, i)

            score = laplacian_variance(band_gray)
            corners, ids = detect_aruco(band_gray, detector)
            found = ids is not None and 0 in ids.flatten()

            band_scores.append(score)
            band_corners.append(corners)
            band_found.append(found)

            # Convert to BGR for overlay drawing
            band_bgr = cv2.cvtColor(band_gray, cv2.COLOR_GRAY2BGR)
            rendered = render_band_overlay(
                band_bgr, score, corners, ids, i == active, i
            )
            band_bgrs.append(rendered)

        with state_lock:
            state["scores"] = band_scores
            state["marker_found"] = band_found
            band_centroids = [marker_centroid(c) for c in band_corners]
            update_alignment(band_centroids)

        # Stitch bands back into a single wide frame, then scale to fit browser
        composite = np.concatenate(band_bgrs, axis=1)   # 5120 × 800
        # Scale down for streaming: 1280×200 per band displayed as 320×50 each
        display_w = 1280
        display_h = int(FRAME_HEIGHT * display_w / FRAME_WIDTH)
        display = cv2.resize(composite, (display_w, display_h))

        band_display_w = display_w // NUM_BANDS
        for i in range(1, NUM_BANDS):
            x = i * band_display_w
            cv2.line(display, (x, 0), (x, display_h), (60, 60, 60), 1)

        ok, buf = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, STREAM_QUALITY])
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        )


# ── Flask routes ───────────────────────────────────────────────────────────────

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MS Focus Tool</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;500;600&display=swap');

  :root {
    --bg:       #0a0d0f;
    --surface:  #111518;
    --border:   #1e2428;
    --dim:      #2a3038;
    --muted:    #4a5560;
    --text:     #c8d4dc;
    --bright:   #e8f0f4;
    --green:    #00e87a;
    --amber:    #ffb340;
    --red:      #ff4040;
    --blue:     #40b0ff;
    --accent:   #00e87a;
    --mono:     'Share Tech Mono', monospace;
    --sans:     'Barlow', sans-serif;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-weight: 300;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }

  header {
    border-bottom: 1px solid var(--border);
    padding: 14px 24px;
    display: flex;
    align-items: center;
    gap: 20px;
    background: var(--surface);
  }

  header h1 {
    font-family: var(--mono);
    font-size: 14px;
    letter-spacing: 0.12em;
    color: var(--accent);
    text-transform: uppercase;
  }

  header .subtitle {
    font-size: 12px;
    color: var(--muted);
    font-family: var(--mono);
  }

  header .spacer { flex: 1; }

  #status-pill {
    font-family: var(--mono);
    font-size: 11px;
    padding: 4px 12px;
    border-radius: 2px;
    background: var(--dim);
    color: var(--muted);
    border: 1px solid var(--border);
    letter-spacing: 0.08em;
  }
  #status-pill.live { color: var(--green); border-color: var(--green); }

  main {
    flex: 1;
    display: grid;
    grid-template-columns: 1fr 280px;
    gap: 0;
  }

  /* ── Camera feed ── */
  .feed-panel {
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  #feed-container {
    position: relative;
    background: #000;
    border: 1px solid var(--border);
    border-radius: 3px;
    overflow: hidden;
  }

  #feed {
    display: block;
    width: 100%;
    height: auto;
    image-rendering: pixelated;
  }

  /* Band selector tabs over the feed */
  .band-tabs {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 6px;
  }

  .band-tab {
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--muted);
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.1em;
    padding: 10px 6px;
    cursor: pointer;
    border-radius: 2px;
    text-align: center;
    transition: all 0.12s;
    position: relative;
  }

  .band-tab:hover { border-color: var(--dim); color: var(--text); }

  .band-tab.active {
    background: #0a1a10;
    border-color: var(--green);
    color: var(--green);
  }

  .band-tab .band-label {
    display: block;
    font-size: 13px;
    font-weight: 500;
    margin-bottom: 4px;
  }

  .band-tab .band-score {
    display: block;
    font-size: 10px;
    opacity: 0.7;
  }

  .band-tab .marker-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--muted);
    position: absolute;
    top: 8px;
    right: 8px;
    transition: background 0.2s;
  }
  .band-tab .marker-dot.found { background: var(--green); }

  /* ── Side panel ── */
  .side-panel {
    border-left: 1px solid var(--border);
    background: var(--surface);
    display: flex;
    flex-direction: column;
    overflow-y: auto;
  }

  .panel-section {
    border-bottom: 1px solid var(--border);
    padding: 18px 20px;
  }

  .panel-label {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.14em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 14px;
  }

  /* Active band focus score (big readout) */
  .score-readout {
    display: flex;
    flex-direction: column;
    gap: 4px;
    margin-bottom: 14px;
  }

  .score-value {
    font-family: var(--mono);
    font-size: 48px;
    line-height: 1;
    color: var(--bright);
    letter-spacing: -0.02em;
    transition: color 0.3s;
  }
  .score-value.good  { color: var(--green); }
  .score-value.ok    { color: var(--amber); }
  .score-value.poor  { color: var(--red); }

  .score-label {
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }
  .score-label.good  { color: var(--green); }
  .score-label.ok    { color: var(--amber); }
  .score-label.poor  { color: var(--red); }

  /* Focus bar */
  .focus-bar-wrap {
    height: 6px;
    background: var(--dim);
    border-radius: 1px;
    overflow: hidden;
    margin: 8px 0 16px;
  }
  .focus-bar {
    height: 100%;
    border-radius: 1px;
    background: var(--muted);
    transition: width 0.2s, background 0.3s;
  }
  .focus-bar.good  { background: var(--green); }
  .focus-bar.ok    { background: var(--amber); }
  .focus-bar.poor  { background: var(--red); }

  /* Threshold labels */
  .threshold-row {
    display: flex;
    justify-content: space-between;
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
    margin-top: -10px;
    margin-bottom: 6px;
  }

  /* Navigation buttons */
  .nav-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
  }

  .nav-btn {
    background: var(--dim);
    border: 1px solid var(--border);
    color: var(--text);
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.08em;
    padding: 10px;
    cursor: pointer;
    border-radius: 2px;
    text-align: center;
    transition: all 0.1s;
    text-transform: uppercase;
  }

  .nav-btn:hover { background: var(--border); color: var(--bright); }
  .nav-btn:active { transform: scale(0.97); }
  .nav-btn:disabled { opacity: 0.3; cursor: default; }

  /* Per-band score list */
  .band-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 7px 0;
    border-bottom: 1px solid var(--border);
    font-family: var(--mono);
    font-size: 12px;
  }
  .band-row:last-child { border-bottom: none; }

  .band-row .bname { color: var(--muted); width: 50px; }
  .band-row .bscore { color: var(--text); width: 52px; text-align: right; }
  .band-row .bbar-wrap {
    flex: 1;
    height: 4px;
    background: var(--dim);
    border-radius: 1px;
    overflow: hidden;
  }
  .band-row .bbar {
    height: 100%;
    border-radius: 1px;
    transition: width 0.25s, background 0.3s;
  }
  .band-row .bmarker {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--dim);
    flex-shrink: 0;
    transition: background 0.2s;
  }
  .band-row .bmarker.found { background: var(--green); }

  /* Marker status card */
  .marker-status {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 12px;
    border: 1px solid var(--border);
    border-radius: 2px;
    font-family: var(--mono);
    font-size: 11px;
  }
  .marker-icon {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: var(--muted);
    flex-shrink: 0;
    transition: background 0.3s, box-shadow 0.3s;
  }
  .marker-icon.found {
    background: var(--green);
    box-shadow: 0 0 6px var(--green);
  }

  /* Instructions */
  .instructions {
    font-size: 12px;
    line-height: 1.7;
    color: var(--muted);
  }
  .instructions strong { color: var(--text); font-weight: 500; }

  .tip {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.06em;
    color: var(--muted);
    border-left: 2px solid var(--dim);
    padding-left: 10px;
    margin-top: 12px;
    line-height: 1.6;
  }

  /* Alignment panel */
  .offset-grid {
    display: grid;
    grid-template-columns: auto 1fr 1fr;
    gap: 4px 12px;
    font-family: var(--mono);
    font-size: 11px;
    align-items: center;
  }
  .offset-grid .og-head {
    font-size: 9px;
    letter-spacing: 0.1em;
    color: var(--muted);
    text-transform: uppercase;
    padding-bottom: 4px;
    border-bottom: 1px solid var(--border);
  }
  .offset-grid .og-band { color: var(--muted); }
  .offset-grid .og-val  { color: var(--text); text-align: right; }
  .offset-grid .og-val.fresh { color: var(--green); }
  .offset-grid .og-val.ref   { color: var(--muted); font-style: italic; }

  .sample-count {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
    margin-top: 10px;
  }
  .sample-count span { color: var(--text); }

  .lock-btn {
    width: 100%;
    margin-top: 14px;
    padding: 10px;
    background: var(--dim);
    border: 1px solid var(--border);
    color: var(--text);
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    cursor: pointer;
    border-radius: 2px;
    transition: all 0.1s;
  }
  .lock-btn:hover  { background: var(--border); color: var(--bright); }
  .lock-btn:active { transform: scale(0.98); }
  .lock-btn.locked {
    border-color: var(--green);
    color: var(--green);
    background: #0a1a10;
  }

  .export-block {
    margin-top: 12px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 2px;
    padding: 10px 12px;
    font-family: var(--mono);
    font-size: 10px;
    color: var(--text);
    white-space: pre;
    overflow-x: auto;
    line-height: 1.6;
    display: none;
  }
  .export-block.visible { display: block; }

  .copy-btn {
    width: 100%;
    margin-top: 8px;
    padding: 7px;
    background: transparent;
    border: 1px solid var(--dim);
    color: var(--muted);
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    cursor: pointer;
    border-radius: 2px;
    transition: all 0.1s;
    display: none;
  }
  .copy-btn.visible { display: block; }
  .copy-btn:hover { color: var(--text); border-color: var(--border); }

  /* Calibration panel */
  .cal-band-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 0;
    border-bottom: 1px solid var(--border);
    font-family: var(--mono);
    font-size: 11px;
  }
  .cal-band-row:last-child { border-bottom: none; }
  .cal-band-name { color: var(--muted); width: 44px; flex-shrink: 0; }
  .cal-badge {
    font-size: 9px;
    letter-spacing: 0.08em;
    padding: 2px 7px;
    border-radius: 2px;
    text-transform: uppercase;
    flex-shrink: 0;
  }
  .cal-badge.uncal  { background: var(--dim);    color: var(--muted); }
  .cal-badge.dark   { background: #1a1500;        color: var(--amber); border: 1px solid #3a3000; }
  .cal-badge.ok     { background: #0a1a10;        color: var(--green); border: 1px solid #0e3018; }
  .cal-btns { display: flex; gap: 5px; margin-left: auto; }
  .cal-btn {
    padding: 3px 8px;
    background: var(--dim);
    border: 1px solid var(--border);
    color: var(--muted);
    font-family: var(--mono);
    font-size: 9px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    cursor: pointer;
    border-radius: 2px;
    transition: all 0.1s;
  }
  .cal-btn:hover { color: var(--text); border-color: var(--muted); }
  .cal-btn.dark-btn:hover { color: var(--amber); border-color: var(--amber); }
  .cal-btn.flat-btn:hover { color: var(--green); border-color: var(--green); }
  .cal-btn:disabled { opacity: 0.3; cursor: default; }

  .cal-progress {
    margin-top: 10px;
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
    min-height: 16px;
  }
  .cal-progress .prog-bar-wrap {
    height: 3px;
    background: var(--dim);
    border-radius: 1px;
    margin-top: 5px;
    overflow: hidden;
  }
  .cal-progress .prog-bar {
    height: 100%;
    background: var(--amber);
    border-radius: 1px;
    transition: width 0.15s;
  }
  .cal-progress .prog-bar.flat { background: var(--green); }

  /* Footer */
  footer {
    border-top: 1px solid var(--border);
    padding: 8px 24px;
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
    letter-spacing: 0.08em;
    display: flex;
    gap: 24px;
  }
  footer span { color: var(--dim); }
  footer .val { color: var(--muted); }
</style>
</head>
<body>

<header>
  <h1>MS // FOCUS</h1>
  <div class="subtitle">4× OV9281 · 5120×800 · ArUco 4×4</div>
  <div class="spacer"></div>
  <div id="status-pill">CONNECTING</div>
</header>

<main>
  <!-- Camera feed + band tabs -->
  <div class="feed-panel">
    <div id="feed-container">
      <img id="feed" src="/stream" alt="live feed"
           onload="document.getElementById('status-pill').textContent='LIVE';
                   document.getElementById('status-pill').className='live';">
    </div>

    <div class="band-tabs" id="band-tabs">
      <div class="band-tab active" data-band="0" onclick="selectBand(0)">
        <div class="marker-dot" id="dot-0"></div>
        <span class="band-label">BAND 0</span>
        <span class="band-score" id="tab-score-0">—</span>
      </div>
      <div class="band-tab" data-band="1" onclick="selectBand(1)">
        <div class="marker-dot" id="dot-1"></div>
        <span class="band-label">BAND 1</span>
        <span class="band-score" id="tab-score-1">—</span>
      </div>
      <div class="band-tab" data-band="2" onclick="selectBand(2)">
        <div class="marker-dot" id="dot-2"></div>
        <span class="band-label">BAND 2</span>
        <span class="band-score" id="tab-score-2">—</span>
      </div>
      <div class="band-tab" data-band="3" onclick="selectBand(3)">
        <div class="marker-dot" id="dot-3"></div>
        <span class="band-label">BAND 3</span>
        <span class="band-score" id="tab-score-3">—</span>
      </div>
    </div>
  </div>

  <!-- Side panel -->
  <div class="side-panel">

    <!-- Active band score -->
    <div class="panel-section">
      <div class="panel-label">Active band · focus score</div>
      <div class="score-readout">
        <div class="score-value poor" id="score-display">0</div>
        <div class="score-label poor" id="score-rating">POOR</div>
      </div>
      <div class="focus-bar-wrap">
        <div class="focus-bar poor" id="focus-bar" style="width:0%"></div>
      </div>
      <div class="threshold-row">
        <span>0</span>
        <span>ACCEPT {{ FOCUS_ACCEPTABLE }}</span>
        <span>GOOD {{ FOCUS_GOOD }}</span>
      </div>

      <div class="marker-status">
        <div class="marker-icon" id="marker-icon"></div>
        <span id="marker-text">Searching for ArUco ID 0…</span>
      </div>
    </div>

    <!-- Band navigation -->
    <div class="panel-section">
      <div class="panel-label">Band navigation</div>
      <div class="nav-row">
        <button class="nav-btn" id="btn-prev" onclick="stepBand(-1)">◀ Prev</button>
        <button class="nav-btn" id="btn-next" onclick="stepBand(1)">Next ▶</button>
      </div>
    </div>

    <!-- All-band overview -->
    <div class="panel-section">
      <div class="panel-label">All bands</div>
      <div id="band-list">
        <div class="band-row"><span class="bname">Band 0</span><span class="bscore" id="ls-0">—</span><div class="bbar-wrap"><div class="bbar poor" id="lb-0" style="width:0%"></div></div><div class="bmarker" id="lm-0"></div></div>
        <div class="band-row"><span class="bname">Band 1</span><span class="bscore" id="ls-1">—</span><div class="bbar-wrap"><div class="bbar poor" id="lb-1" style="width:0%"></div></div><div class="bmarker" id="lm-1"></div></div>
        <div class="band-row"><span class="bname">Band 2</span><span class="bscore" id="ls-2">—</span><div class="bbar-wrap"><div class="bbar poor" id="lb-2" style="width:0%"></div></div><div class="bmarker" id="lm-2"></div></div>
        <div class="band-row"><span class="bname">Band 3</span><span class="bscore" id="ls-3">—</span><div class="bbar-wrap"><div class="bbar poor" id="lb-3" style="width:0%"></div></div><div class="bmarker" id="lm-3"></div></div>
      </div>
    </div>

    <!-- Calibration panel -->
    <div class="panel-section">
      <div class="panel-label">Flat-field calibration</div>
      <div id="cal-band-list">
        <div class="cal-band-row">
          <span class="cal-band-name">Band 0</span>
          <span class="cal-badge uncal" id="cal-badge-0">uncal</span>
          <div class="cal-btns">
            <button class="cal-btn dark-btn" onclick="startCal('dark', 0)" id="cal-dark-0">Dark</button>
            <button class="cal-btn flat-btn" onclick="startCal('flat', 0)" id="cal-flat-0" disabled>Flat</button>
          </div>
        </div>
        <div class="cal-band-row">
          <span class="cal-band-name">Band 1</span>
          <span class="cal-badge uncal" id="cal-badge-1">uncal</span>
          <div class="cal-btns">
            <button class="cal-btn dark-btn" onclick="startCal('dark', 1)" id="cal-dark-1">Dark</button>
            <button class="cal-btn flat-btn" onclick="startCal('flat', 1)" id="cal-flat-1" disabled>Flat</button>
          </div>
        </div>
        <div class="cal-band-row">
          <span class="cal-band-name">Band 2</span>
          <span class="cal-badge uncal" id="cal-badge-2">uncal</span>
          <div class="cal-btns">
            <button class="cal-btn dark-btn" onclick="startCal('dark', 2)" id="cal-dark-2">Dark</button>
            <button class="cal-btn flat-btn" onclick="startCal('flat', 2)" id="cal-flat-2" disabled>Flat</button>
          </div>
        </div>
        <div class="cal-band-row">
          <span class="cal-band-name">Band 3</span>
          <span class="cal-badge uncal" id="cal-badge-3">uncal</span>
          <div class="cal-btns">
            <button class="cal-btn dark-btn" onclick="startCal('dark', 3)" id="cal-dark-3">Dark</button>
            <button class="cal-btn flat-btn" onclick="startCal('flat', 3)" id="cal-flat-3" disabled>Flat</button>
          </div>
        </div>
      </div>
      <div class="cal-progress" id="cal-progress">
        <span id="cal-progress-text"></span>
        <div class="prog-bar-wrap" id="cal-prog-wrap" style="display:none">
          <div class="prog-bar" id="cal-prog-bar" style="width:0%"></div>
        </div>
      </div>
    </div>

    <!-- Alignment estimation -->
    <div class="panel-section">
      <div class="panel-label">Parallax offset estimate</div>
      <div class="offset-grid">
        <span class="og-head">Band</span>
        <span class="og-head" style="text-align:right">dx (px)</span>
        <span class="og-head" style="text-align:right">dy (px)</span>
        <span class="og-band">0 (ref)</span>
        <span class="og-val ref" id="ox-0">0.0</span>
        <span class="og-val ref" id="oy-0">0.0</span>
        <span class="og-band">1</span>
        <span class="og-val" id="ox-1">—</span>
        <span class="og-val" id="oy-1">—</span>
        <span class="og-band">2</span>
        <span class="og-val" id="ox-2">—</span>
        <span class="og-val" id="oy-2">—</span>
        <span class="og-band">3</span>
        <span class="og-val" id="ox-3">—</span>
        <span class="og-val" id="oy-3">—</span>
      </div>
      <div class="sample-count">Samples: <span id="sample-count">0</span> / 30</div>

      <button class="lock-btn" id="lock-btn" onclick="lockAlignment()">
        Lock &amp; Export Offsets
      </button>
      <pre class="export-block" id="export-block"></pre>
      <button class="copy-btn" id="copy-btn" onclick="copyExport()">Copy to clipboard</button>
    </div>

    <!-- Instructions -->
    <div class="panel-section" style="flex:1">
      <div class="panel-label">How to use</div>
      <div class="instructions">
        <strong>1.</strong> Hold the ArUco marker (ID 0) at your working distance in front of the camera.<br><br>
        <strong>2.</strong> Use the band tabs or Prev/Next to select a band.<br><br>
        <strong>3.</strong> Adjust that lens until the focus score peaks and turns <span style="color:var(--green)">green</span>.<br><br>
        <strong>4.</strong> Repeat for each band.<br><br>
        <strong>5.</strong> Keep the marker visible in all bands until the offset estimate accumulates ~30 samples, then click <em>Lock &amp; Export</em>.
        <div class="tip">Offsets are centroid-based translations (dx, dy) in band-pixel coordinates, relative to Band 0. Apply them as a crop/shift in your processing pipeline before computing spectral indices.</div>
      </div>
    </div>

  </div>
</main>

<footer>
  <span>THRESHOLDS</span>
  <span>POOR &lt; {{ FOCUS_POOR }}</span>
  <span>ACCEPTABLE &gt; {{ FOCUS_ACCEPTABLE }}</span>
  <span>GOOD &gt; {{ FOCUS_GOOD }}</span>
</footer>

<script>
const FOCUS_POOR        = {{ FOCUS_POOR }};
const FOCUS_ACCEPTABLE  = {{ FOCUS_ACCEPTABLE }};
const FOCUS_GOOD        = {{ FOCUS_GOOD }};
const MAX_DISPLAY       = FOCUS_GOOD * 1.5;

let activeBand = 0;

function selectBand(b) {
  activeBand = b;
  fetch('/set_band', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({band: b})
  });
  document.querySelectorAll('.band-tab').forEach((el, i) => {
    el.classList.toggle('active', i === b);
  });
  document.getElementById('btn-prev').disabled = b === 0;
  document.getElementById('btn-next').disabled = b === 3;
}

function stepBand(dir) {
  const next = Math.max(0, Math.min(3, activeBand + dir));
  selectBand(next);
}

function ratingClass(score) {
  if (score >= FOCUS_GOOD) return 'good';
  if (score >= FOCUS_ACCEPTABLE) return 'ok';
  return 'poor';
}

function barWidth(score) {
  return Math.min(score / MAX_DISPLAY * 100, 100).toFixed(1) + '%';
}

function startCal(type, band) {
  fetch('/start_cal', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({type, band})
  }).then(r => r.json()).then(data => {
    if (!data.ok) return;
    const wrap = document.getElementById('cal-prog-wrap');
    const bar  = document.getElementById('cal-prog-bar');
    const text = document.getElementById('cal-progress-text');
    wrap.style.display = 'block';
    bar.style.width = '0%';
    bar.className = type === 'flat' ? 'prog-bar flat' : 'prog-bar';
    text.textContent = `Capturing ${type} — band ${band}…`;
  });
}

function lockAlignment() {
  fetch('/lock_alignment', {method: 'POST'})
    .then(r => r.json())
    .then(data => {
      const btn = document.getElementById('lock-btn');
      btn.textContent = 'Locked ✓';
      btn.className = 'lock-btn locked';

      const block = document.getElementById('export-block');
      block.textContent = data.config_json;
      block.className = 'export-block visible';

      document.getElementById('copy-btn').className = 'copy-btn visible';
    });
}

function copyExport() {
  const text = document.getElementById('export-block').textContent;
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.getElementById('copy-btn');
    btn.textContent = 'Copied!';
    setTimeout(() => { btn.textContent = 'Copy to clipboard'; }, 2000);
  });
}

function poll() {
  fetch('/state')
    .then(r => r.json())
    .then(data => {
      const scores  = data.scores;
      const found   = data.marker_found;
      const active  = data.active_band;
      const offsets = data.offsets;
      const samples = data.alignment_samples;

      // Active band big readout
      const aScore  = scores[active];
      const aClass  = ratingClass(aScore);
      const scoreEl = document.getElementById('score-display');
      scoreEl.textContent = Math.round(aScore);
      scoreEl.className = 'score-value ' + aClass;

      const ratingEl = document.getElementById('score-rating');
      ratingEl.textContent = aClass.toUpperCase();
      ratingEl.className = 'score-label ' + aClass;

      const barEl = document.getElementById('focus-bar');
      barEl.style.width = barWidth(aScore);
      barEl.className = 'focus-bar ' + aClass;

      const iconEl  = document.getElementById('marker-icon');
      const textEl  = document.getElementById('marker-text');
      if (found[active]) {
        iconEl.className = 'marker-icon found';
        textEl.textContent = 'ArUco ID 0 detected';
      } else {
        iconEl.className = 'marker-icon';
        textEl.textContent = 'Searching for ArUco ID 0…';
      }

      // Tab scores + dots
      for (let i = 0; i < 4; i++) {
        document.getElementById('tab-score-' + i).textContent = Math.round(scores[i]);
        document.getElementById('dot-' + i).className =
          'marker-dot' + (found[i] ? ' found' : '');

        document.getElementById('ls-' + i).textContent = Math.round(scores[i]);
        const lb = document.getElementById('lb-' + i);
        lb.style.width = barWidth(scores[i]);
        lb.className = 'bbar ' + ratingClass(scores[i]);

        const lm = document.getElementById('lm-' + i);
        lm.className = 'bmarker' + (found[i] ? ' found' : '');
      }

      // Alignment offsets
      document.getElementById('sample-count').textContent = samples;
      for (let i = 1; i < 4; i++) {
        const [dx, dy] = offsets[i];
        const isFresh = samples >= 5;
        const cls = isFresh ? 'og-val fresh' : 'og-val';
        const xEl = document.getElementById('ox-' + i);
        const yEl = document.getElementById('oy-' + i);
        xEl.textContent = samples > 0 ? dx.toFixed(2) : '—';
        yEl.textContent = samples > 0 ? dy.toFixed(2) : '—';
        xEl.className = samples > 0 ? cls : 'og-val';
        yEl.className = samples > 0 ? cls : 'og-val';
      }

      // Calibration status badges
      const calStatus = data.cal_status;
      const calProgress = data.cal_progress;  // {band, type, pct} or null
      for (let i = 0; i < 4; i++) {
        const badge    = document.getElementById('cal-badge-' + i);
        const darkBtn  = document.getElementById('cal-dark-' + i);
        const flatBtn  = document.getElementById('cal-flat-' + i);
        const status   = calStatus[i];
        const busy     = calProgress && calProgress.band === i;

        if (status === 'cal_ok') {
          badge.textContent = 'cal';
          badge.className = 'cal-badge ok';
          flatBtn.disabled = false;
        } else if (status === 'dark_ok') {
          badge.textContent = 'dark';
          badge.className = 'cal-badge dark';
          flatBtn.disabled = false;
        } else {
          badge.textContent = 'uncal';
          badge.className = 'cal-badge uncal';
          flatBtn.disabled = true;
        }
        darkBtn.disabled = busy;
        flatBtn.disabled = busy || status === 'uncal';
      }

      // Calibration progress bar
      if (calProgress) {
        const pct = (calProgress.pct * 100).toFixed(0);
        document.getElementById('cal-prog-wrap').style.display = 'block';
        document.getElementById('cal-prog-bar').style.width = pct + '%';
        document.getElementById('cal-prog-bar').className =
          calProgress.type === 'flat' ? 'prog-bar flat' : 'prog-bar';
        document.getElementById('cal-progress-text').textContent =
          `Capturing ${calProgress.type} — band ${calProgress.band} (${pct}%)`;
      } else {
        document.getElementById('cal-prog-wrap').style.display = 'none';
        document.getElementById('cal-progress-text').textContent = '';
      }
    })
    .catch(() => {});
}

selectBand(0);
setInterval(poll, 250);
</script>
</body>
</html>
""".replace("{{ FOCUS_POOR }}", str(FOCUS_POOR)) \
   .replace("{{ FOCUS_ACCEPTABLE }}", str(FOCUS_ACCEPTABLE)) \
   .replace("{{ FOCUS_GOOD }}", str(FOCUS_GOOD))


@app.route("/")
def index():
    return HTML_PAGE


@app.route("/stream")
def stream():
    picam2 = app.config["picam2"]
    detector = app.config["detector"]
    return Response(
        generate_frames(picam2, detector),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/state")
def get_state():
    with state_lock:
        samples = min(
            (acc[2] for acc in state["centroid_acc"] if acc[2] > 0),
            default=0,
        )
        cap = state["cal_capture"]
        cal_progress = None
        if cap is not None:
            pct = (cap["n"] / CAL_AVERAGE_FRAMES) if cap["n"] else 0.0
            cal_progress = {"band": cap["band"], "type": cap["type"], "pct": round(pct, 2)}
        return jsonify({
            "active_band":       state["active_band"],
            "scores":            state["scores"],
            "marker_found":      state["marker_found"],
            "offsets":           state["offsets"],
            "alignment_samples": min(samples, ALIGNMENT_AVERAGE_FRAMES),
            "alignment_locked":  state["alignment_locked"],
            "cal_status":        state["cal_status"],
            "cal_progress":      cal_progress,
        })


@app.route("/set_band", methods=["POST"])
def set_band():
    data = request.get_json(force=True)
    band = int(data.get("band", 0))
    band = max(0, min(NUM_BANDS - 1, band))
    with state_lock:
        state["active_band"] = band
    return jsonify({"ok": True, "active_band": band})


@app.route("/start_cal", methods=["POST"])
def start_cal():
    """
    Begin accumulating frames for a dark or flat calibration capture.

    Body: {"type": "dark"|"flat", "band": 0-3}

    For dark: cover the lens for this band (or all lenses, the others are ignored).
    For flat: point the camera at a uniformly-lit surface (grey card / overcast sky).
    The accumulator runs for CAL_AVERAGE_FRAMES frames then commits automatically.
    """
    data = request.get_json(force=True)
    cal_type = data.get("type")
    band = int(data.get("band", 0))

    if cal_type not in ("dark", "flat"):
        return jsonify({"ok": False, "error": "type must be 'dark' or 'flat'"}), 400
    if not 0 <= band < NUM_BANDS:
        return jsonify({"ok": False, "error": "invalid band"}), 400

    # Flat requires a dark frame first
    with state_lock:
        if cal_type == "flat" and state["dark_frames"][band] is None:
            return jsonify({"ok": False, "error": "capture dark frame first"}), 400
        if state["cal_capture"] is not None:
            return jsonify({"ok": False, "error": "capture already in progress"}), 409
        state["cal_capture"] = {"type": cal_type, "band": band, "acc": None, "n": 0}

    print(f"[CAL] Starting {cal_type} capture for band {band} ({CAL_AVERAGE_FRAMES} frames)")
    return jsonify({"ok": True})


@app.route("/lock_alignment", methods=["POST"])
def lock_alignment():
    """
    Snapshot the current smoothed offsets, write them to band_offsets.json
    next to this script, and return a formatted JSON config string for display.
    """
    with state_lock:
        offsets = list(state["offsets"])
        state["alignment_locked"] = True
        state["locked_offsets"] = offsets

    config = {
        "band_offsets_px": {
            f"band_{i}": {"dx": round(offsets[i][0], 3), "dy": round(offsets[i][1], 3)}
            for i in range(NUM_BANDS)
        },
        "reference_band": 0,
        "note": "dx/dy = shift to apply to band N so its pixels align with band 0",
    }
    config_json = json.dumps(config, indent=2)

    # Persist alongside this script
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "band_offsets.json")
    try:
        with open(out_path, "w") as f:
            f.write(config_json)
        print(f"[ALIGNMENT] Offsets saved → {out_path}")
    except OSError as e:
        print(f"[ALIGNMENT] Could not write {out_path}: {e}")

    return jsonify({"ok": True, "config_json": config_json})


@app.route("/reset_alignment", methods=["POST"])
def reset_alignment():
    """Clear accumulated centroid observations and start fresh."""
    with state_lock:
        state["centroid_acc"] = [(0.0, 0.0, 0) for _ in range(NUM_BANDS)]
        state["offsets"] = [(0.0, 0.0)] * NUM_BANDS
        state["alignment_locked"] = False
    return jsonify({"ok": True})


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, params)

    picam2 = init_camera()

    load_saved_calibration()

    app.config["picam2"]  = picam2
    app.config["detector"] = detector

    print("Focus tool running → http://0.0.0.0:5000")
    print("Open that URL in a browser on the same network as this Pi.")
    print("Ctrl-C to stop.")

    try:
        app.run(host="0.0.0.0", port=5000, threaded=True)
    finally:
        picam2.stop()
        print("Camera stopped.")