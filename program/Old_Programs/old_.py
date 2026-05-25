# Version 251002.2 (minimal cropping fix)
# ──────────────── Libraries Import ───────────────────────────────────
import time, threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pathlib import Path
import cv2

# --- SAM imports (replaces rembg) ---
import torch
from segment_anything import sam_model_registry, SamPredictor

camera_indices = [0, 1, 2]  # Try these camera indices in order

# ──────────────── Tk root window ─────────────────────────────────────
root = tk.Tk()
root.state('zoomed')
root.title("IC-Project  ·  Color Gradient Detection - Camera Mode")
root.geometry("1400x900")

# ──────────────── Global configuration ───────────────────────────────
SAMPLE_NUM      = 16    # Total number of samples
N_COLS          = 8

# Standardize camera frames to this size before drawing/cropping
CAM_W, CAM_H = 1280, 720

# Display size (your camera_container is already 800×500)
DISP_W, DISP_H = 800, 500

# How often to run full processing (SAM is heavy)
PROCESS_INTERVAL_SEC = 5.0

# ──────────────── SAM configuration ─────────────────────────────────
# Put checkpoint file next to this script (or change the path)
SAM_CHECKPOINT = Path("program/sam_vit_b_01ec64.pth")
SAM_MODEL_TYPE = "vit_b"  # "vit_h" | "vit_l" | "vit_b"
SAM_DEVICE     = "cpu"    # "cuda" if you have GPU + torch cuda build

sam_predictor = None  # will be initialized once

# ──────────────── ROI geometry (from .ipynb) ─────────────────────────
# ---- CALIBRATE THESE ONCE for your camera/layout ----
x_left_frac  = 0.2
x_right_frac = 0.81

y_top_frac = 0.07   # y-position (as fraction of H) near the liquid of the top row
y_bot_frac = 0.38   # y-position near the liquid of the bottom row

# Box size relative to spacing / image size
box_w_scale = 0.55   # box width = box_w_scale * tube spacing (dx)
box_up_frac = 0.04   # how far box extends above row y (fraction of H)
box_dn_frac = 0.10   # how far box extends below row y (fraction of H)


def tube_centers_and_boxes(H, W):
    x0, x1 = x_left_frac * W, x_right_frac * W
    xs = np.linspace(x0, x1, N_COLS)

    y_top = y_top_frac * H
    y_bot = y_bot_frac * H

    dx = (x1 - x0) / (N_COLS - 1)
    bw = box_w_scale * dx
    up = box_up_frac * H
    dn = box_dn_frac * H

    centers = []
    boxes = []
    for y in [y_top, y_bot]:
        for x in xs:
            x0b, y0b = x - bw / 2, y - up
            x1b, y1b = x + bw / 2, y + dn

            x0b = int(np.clip(x0b, 0, W - 1)); x1b = int(np.clip(x1b, 0, W - 1))
            y0b = int(np.clip(y0b, 0, H - 1)); y1b = int(np.clip(y1b, 0, H - 1))
            if x1b <= x0b: x1b = min(W - 1, x0b + 1)
            if y1b <= y0b: y1b = min(H - 1, y0b + 1)

            centers.append((x, y))
            boxes.append(np.array([x0b, y0b, x1b, y1b], dtype=np.float32))

    return np.array(centers, dtype=np.float32), boxes


# Precompute 16 ROIs in CAM space (because CAM_W/CAM_H are fixed)
CENTERS_CAM, BOXES_CAM = tube_centers_and_boxes(CAM_H, CAM_W)
TOP_BOXES_CAM = BOXES_CAM[:8]
BOT_BOXES_CAM = BOXES_CAM[8:]


def clip_rect(x1, y1, x2, y2, w, h):
    """Clip safely for numpy slicing; ensures non-empty crop."""
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))
    if x2 <= x1: x2 = min(w, x1 + 1)
    if y2 <= y1: y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


COLOUR_THRES    = 105.8             # Default: 105.8
INFO_PREFIX     = "*INFO: "         # Shown in console
ERROR_PREFIX    = "*ERROR: "        # Shown in console

# ──────────────── Program variables ──────────────────────────────────
# NOTE: keep name "erode_pixels", but now it truly means pixels (kernel radius)
erode_pixels        = 2          # was 20; 20px erosion is too aggressive for most ROIs
padding_ratio       = 0.05       # Tunable in GUI
frame_count         = 0
processing          = False
stop_processing     = False
camera              = None
results             = []

# Create output folders after GUI starts
output_folder = Path("output_folder_2")
output_folder.mkdir(exist_ok=True, parents=True)

# NEW: subfolders like the notebook
crops_dir = output_folder / "crops"
masks_dir = output_folder / "masks"
crops_dir.mkdir(exist_ok=True, parents=True)
masks_dir.mkdir(exist_ok=True, parents=True)

# Add a global reference for the processing thread (to join on closing)
processing_thread = None  # Added global thread reference


# ──────────────── SAM init ──────────────────────────────────────────
def init_sam_predictor():
    global sam_predictor

    if sam_predictor is not None:
        return True

    if not SAM_CHECKPOINT.exists():
        messagebox.showerror(
            "SAM Checkpoint Missing",
            f"Cannot find SAM checkpoint:\n\n{SAM_CHECKPOINT.resolve()}\n\n"
            "Download (e.g. sam_vit_b_01ec64.pth) and place it next to this script "
            "or update SAM_CHECKPOINT."
        )
        return False

    try:
        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=None)

        ckpt = torch.load(SAM_CHECKPOINT, map_location="cpu")
        if isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        sam.load_state_dict(state_dict, strict=True)
        sam.to(SAM_DEVICE)
        sam_predictor = SamPredictor(sam)

        print(f"{INFO_PREFIX}SAM loaded: {SAM_MODEL_TYPE} on {SAM_DEVICE}")
        return True
    except Exception as e:
        messagebox.showerror("SAM Init Error", f"Failed to initialize SAM:\n\n{str(e)}")
        return False


# ──────────────── Mask utilities (from .ipynb) ───────────────────────
def largest_component(mask_bool):
    m = (mask_bool.astype(np.uint8) * 255)
    num, labels = cv2.connectedComponents(m, connectivity=8)
    if num <= 1:
        return mask_bool
    areas = [(labels == i).sum() for i in range(1, num)]
    best = 1 + int(np.argmax(areas))
    return (labels == best)


def keep_lowest_component(mask_bool):
    m = mask_bool.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return mask_bool

    best = 1
    best_ymax = -1
    for i in range(1, n):
        top = stats[i, cv2.CC_STAT_TOP]
        h   = stats[i, cv2.CC_STAT_HEIGHT]
        ymax = top + h - 1
        if ymax > best_ymax:
            best_ymax = ymax
            best = i
    return labels == best


def inner_core_from_tube_mask(tube_mask, keep_core_frac=0.55):
    tube_u8 = tube_mask.astype(np.uint8)
    dist = cv2.distanceTransform(tube_u8, distanceType=cv2.DIST_L2, maskSize=5)
    m = dist.max()
    if m <= 1e-6:
        return tube_mask.copy()
    return tube_mask & (dist >= keep_core_frac * m)


def chemical_by_color(crop_rgb, tube_mask_crop):
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    S = hsv[..., 1]
    V = hsv[..., 2]

    s_vals = S[tube_mask_crop]
    if s_vals.size == 0:
        return np.zeros_like(tube_mask_crop, dtype=bool)

    s_thr = max(25, int(np.percentile(s_vals, 70)))
    v_thr = 245
    chem = tube_mask_crop & (S >= s_thr) & (V <= v_thr)

    chem = cv2.morphologyEx(
        chem.astype(np.uint8),
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1
    ).astype(bool)

    chem = cv2.morphologyEx(
        chem.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=2
    ).astype(bool)

    chem = largest_component(chem)
    return chem


def boundary_by_air_difference(crop_rgb, core_mask):
    H, W = core_mask.shape
    lab = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    y0 = int(0.05 * H)
    y1 = int(0.25 * H)
    top_band = np.zeros_like(core_mask)
    top_band[y0:y1] = True
    ref_mask = core_mask & top_band

    if ref_mask.sum() < 50:
        return int(0.65 * H)  # fallback

    ref = lab[ref_mask].mean(axis=0)  # [L,a,b]
    diff = np.linalg.norm(lab - ref, axis=2)

    prof = np.full(H, np.nan, dtype=np.float32)
    for y in range(H):
        row = core_mask[y]
        if row.any():
            prof[y] = diff[y, row].mean()

    idx = np.where(~np.isnan(prof))[0]
    if idx.size < 10:
        return int(0.65 * H)

    prof_f = prof.copy()
    prof_f[np.isnan(prof_f)] = np.interp(np.where(np.isnan(prof_f))[0], idx, prof_f[idx])
    prof_s = cv2.GaussianBlur(prof_f.reshape(-1, 1), (1, 21), 0).ravel()

    g = np.abs(np.gradient(prof_s))
    y_min = int(0.10 * H)
    y_max = int(0.95 * H)
    y_star = y_min + int(np.argmax(g[y_min:y_max]))

    if g[y_star] < 0.2:
        y_star = int(0.65 * H)

    return y_star


def chemical_from_crop_strict(
    crop_rgb,
    tube_mask_crop,
    min_area_frac=0.06,
    bottomness_frac=0.50,
    keep_core_frac=0.55
):
    H, W = tube_mask_crop.shape
    tube_area = int(tube_mask_crop.sum())
    if tube_area == 0:
        return np.zeros_like(tube_mask_crop, dtype=bool)

    core = inner_core_from_tube_mask(tube_mask_crop, keep_core_frac=keep_core_frac)

    chem_cue = chemical_by_color(crop_rgb, core)
    chem_cue = keep_lowest_component(chem_cue)

    cue_area = int(chem_cue.sum())
    cue_ymax = np.where(chem_cue)[0].max() if cue_area > 0 else -1
    cue_ok = (cue_area > 0) and (cue_ymax >= int(bottomness_frac * H))

    if cue_ok:
        y_star = int(np.where(chem_cue)[0].min())
    else:
        y_star = boundary_by_air_difference(crop_rgb, core)

    yy = np.arange(H)[:, None]
    chem = tube_mask_crop & (yy >= y_star)

    min_area = int(min_area_frac * tube_area)
    if chem.sum() < min_area:
        ys = np.where(tube_mask_crop)[0]
        y_sorted = np.sort(ys)
        q = max(0.0, 1.0 - (min_area / tube_area))
        y_q = int(np.quantile(y_sorted, q))
        chem = tube_mask_crop & (yy >= y_q)

    chem = keep_lowest_component(chem)
    return chem


def split_parts(crop_rgb, tube_mask_crop, chem_mask_crop, dilate_px=5):
    tube_wall = tube_mask_crop & (~chem_mask_crop)

    k = max(3, int(dilate_px) | 1)  # odd
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    chem_dil = cv2.dilate(chem_mask_crop.astype(np.uint8), ker, iterations=1).astype(bool)

    tube_overlap = tube_wall & chem_dil
    tube_nonoverlap = tube_wall & (~chem_dil)
    background = ~tube_mask_crop

    return background, tube_overlap, tube_nonoverlap, chem_mask_crop


# ──────────────── Cropping helpers ───────────────────────────────────
def erode_mask_bool(mask_bool, erode_px):
    """
    GUI says "Erode Pixels", so treat erode as pixel radius (kernel size),
    NOT iterations.
    """
    erode_px = int(erode_px)
    if erode_px <= 0:
        return mask_bool
    k = 2 * erode_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    eroded = cv2.erode(mask_bool.astype(np.uint8), kernel, iterations=1)
    return eroded.astype(bool)


def mask_to_rgba_and_trim(crop_rgb, mask_bool, erode_px, padding_ratio, trim_mask_bool=None):
    """
    Converts (crop_rgb + mask_bool) into RGBA, trims by trim_mask_bool bbox (default = mask_bool),
    and applies padding + optional erosion (radius in pixels).
    """
    H, W = mask_bool.shape

    # Erode the mask used for alpha (visual / HSV mask)
    mask_bool = erode_mask_bool(mask_bool, erode_px)

    # Use tube bbox (or other mask) for trimming if provided
    if trim_mask_bool is None:
        trim_mask_bool = mask_bool
    else:
        trim_mask_bool = trim_mask_bool.astype(bool)

    alpha = (mask_bool.astype(np.uint8) * 255)
    rgba = np.dstack([crop_rgb, alpha]).astype(np.uint8)

    coords = np.argwhere(trim_mask_bool)
    if coords.size == 0:
        rgba[..., 3] = 0
        return rgba

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    h, w = (y1 - y0), (x1 - x0)
    pad_y, pad_x = int(h * padding_ratio), int(w * padding_ratio)

    y0 = max(0, y0 - pad_y)
    y1 = min(H, y1 + pad_y)
    x0 = max(0, x0 - pad_x)
    x1 = min(W, x1 + pad_x)

    return rgba[y0:y1, x0:x1].copy()


def choose_best_sam_mask(masks_full, scores, crop_slice, prefer_not_full=True):
    """
    Pick a SAM mask that isn't "everything".
    crop_slice = (y0,y1,x0,x1)
    """
    y0, y1, x0, x1 = crop_slice
    best_idx = int(np.argmax(scores))
    best_val = -1e9

    for j in range(masks_full.shape[0]):
        m_crop = masks_full[j][y0:y1, x0:x1]
        area_frac = float(m_crop.mean()) if m_crop.size else 1.0  # [0..1]
        sc = float(scores[j])

        # penalize huge masks (often "whole ROI")
        penalty = 0.65 * area_frac if prefer_not_full else 0.0
        val = sc - penalty

        # optionally reject near-full masks unless all are bad
        if prefer_not_full and area_frac > 0.98:
            val -= 2.0

        if val > best_val:
            best_val = val
            best_idx = j

    return int(best_idx)


# ──────────────── Threading Functions ───────────────────────────────
def start_processing():
    """Start the processing schedule and its thread upon clicking start processing button"""
    global processing, stop_processing, camera, processing_thread

    if not init_sam_predictor():
        return

    camera = None

    for cam_index in camera_indices:
        try:
            camera = cv2.VideoCapture(cam_index)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
            if camera.isOpened():
                ret, test_frame = camera.read()
                if ret and test_frame is not None:
                    print(f"INFO: Using camera index {cam_index}")
                    break
            camera.release()
        except Exception as e:
            print(f"ERROR: Camera index {cam_index} failed: {str(e)}")

    if not camera or not camera.isOpened():
        messagebox.showerror("Camera Error", "Cannot access any camera. Please check camera connection.")
        return

    if processing:
        return
    processing = True
    stop_processing = False

    status_var.set("Starting the process... Please wait")

    process_btn.config(state=tk.DISABLED)
    stop_btn.config(state=tk.NORMAL)
    export_btn.config(state=tk.DISABLED)

    processing_thread = threading.Thread(target=main_process)
    processing_thread.daemon = False
    processing_thread.start()


def stop_processing_func():
    """Stop the processing schedule upon clicking stop processing button"""
    global stop_processing, camera
    stop_processing = True
    if camera:
        camera.release()
    status_var.set("Stopping the process...")


# ──────────────── Camera Processing ───────────────────────────────
def main_process():
    """The main camera processing function"""
    global processing, results, camera, frame_count

    results = []
    frame_count = 0
    last_process_time = time.time()

    while not stop_processing and camera.isOpened():
        ret, frame = camera.read()
        if not ret:
            status_var.set("Failed to capture frame from camera")
            break

        display_frame = frame.copy()
        update_camera_display(display_frame)

        current_time = time.time()
        if current_time - last_process_time >= PROCESS_INTERVAL_SEC:
            status_var.set(f"Processing: Frame {frame_count + 1}")

            current_erode = erode_var.get()
            current_padding = padding_var.get()

            result = process_camera_frame(frame, current_erode, current_padding)
            results.append(result)

            update_results_table(result)
            update_hsv_visualization(result)

            last_process_time = current_time
            frame_count += 1

        time.sleep(0.03)

    if camera:
        camera.release()
    processing = False
    process_btn.config(state=tk.NORMAL)
    stop_btn.config(state=tk.DISABLED)
    export_btn.config(state=tk.NORMAL)

    if stop_processing:
        status_var.set("Processing stopped successfully")
    else:
        status_var.set("Processing completed successfully")


def initialize_camera_display():
    root.update_idletasks()


def process_camera_frame(frame, erode_px, padding_ratio):
    """Process a single camera frame using SAM tube+chemical segmentation."""
    global sam_predictor

    samples = []

    frame_fixed = cv2.resize(frame, (CAM_W, CAM_H), interpolation=cv2.INTER_LINEAR)
    bgr_full = frame_fixed
    rgb_full = cv2.cvtColor(bgr_full, cv2.COLOR_BGR2RGB)
    h, w = rgb_full.shape[:2]

    sam_predictor.set_image(rgb_full)

    def handle_idx(center_xy, box_xyxy, tag):
        x0, y0, x1, y1 = box_xyxy.astype(int)

        # clip and reuse this SAME box for crop + SAM (fix)
        x0, y0, x1, y1 = clip_rect(x0, y0, x1, y1, w, h)
        box_clipped = np.array([x0, y0, x1, y1], dtype=np.float32)

        crop_rgb = rgb_full[y0:y1, x0:x1].copy()
        if crop_rgb.size == 0:
            return create_dummy_result(tag)

        try:
            # ensure point is inside the clipped box (more stable than precomputed center)
            cx = float((x0 + x1) / 2.0)
            cy = float((y0 + y1) / 2.0)
            point_coords = np.array([[cx, cy]], dtype=np.float32)
            point_labels = np.array([1], dtype=np.int32)

            masks, scores, _ = sam_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box_clipped,
                multimask_output=True
            )

            # choose a mask that isn't "everything"
            kbest = choose_best_sam_mask(masks, scores, crop_slice=(y0, y1, x0, x1), prefer_not_full=True)
            tube_mask_full = masks[kbest].astype(bool)
            tube_mask_crop = tube_mask_full[y0:y1, x0:x1]

            tube_mask_crop = largest_component(tube_mask_crop)

            # Chemical/liquid region inside tube
            chem_mask = chemical_from_crop_strict(crop_rgb, tube_mask_crop)

            dilate_px = max(3, int(0.03 * (x1 - x0)))
            bg, tube_ov, tube_non, chem = split_parts(crop_rgb, tube_mask_crop, chem_mask, dilate_px=dilate_px)

            label = np.zeros(bg.shape, dtype=np.uint8)
            label[tube_non] = 1
            label[tube_ov] = 2
            label[chem] = 3

            timestamp = int(time.time())

            # Save masks (unchanged)
            Image.fromarray((tube_mask_crop.astype(np.uint8) * 255)).save(masks_dir / f"tube_{timestamp}_{tag}_tube.png")
            Image.fromarray((chem.astype(np.uint8) * 255)).save(masks_dir / f"tube_{timestamp}_{tag}_chem.png")
            Image.fromarray((tube_ov.astype(np.uint8) * 255)).save(masks_dir / f"tube_{timestamp}_{tag}_tube_overlap.png")
            Image.fromarray((tube_non.astype(np.uint8) * 255)).save(masks_dir / f"tube_{timestamp}_{tag}_tube_nonoverlap.png")
            Image.fromarray(label).save(masks_dir / f"tube_{timestamp}_{tag}_label.png")

            # --- FIX: crops/tube_*.png should be a real cropped result (trimmed) ---
            tube_rgba_trim = mask_to_rgba_and_trim(
                crop_rgb=crop_rgb,
                mask_bool=tube_mask_crop,      # alpha = tube
                trim_mask_bool=tube_mask_crop, # trim bbox = tube
                erode_px=0,                    # keep tube edges for bbox; avoid eroding bbox away
                padding_ratio=padding_ratio
            )
            Image.fromarray(tube_rgba_trim).save(crops_dir / f"tube_{timestamp}_{tag}.png")

            # --- HSV image/mask ---
            # Use chemical pixels for HSV if available, else tube.
            use_mask_for_hsv = chem if chem.sum() > 0 else tube_mask_crop

            # --- FIX: camera_sample should be trimmed by tube bbox (even if alpha=chem) ---
            result_rgba = mask_to_rgba_and_trim(
                crop_rgb=crop_rgb,
                mask_bool=use_mask_for_hsv,    # alpha = chem (or tube)
                trim_mask_bool=tube_mask_crop, # trim bbox = tube (this makes it visibly cropped)
                erode_px=erode_px,
                padding_ratio=padding_ratio
            )

            sample_path = output_folder / f"camera_sample_{timestamp}_{tag}.png"
            Image.fromarray(result_rgba).save(sample_path)

            stats = calculate_hsv_stats(result_rgba, cv2)
            stats["image_path"] = str(sample_path)
            return stats

        except Exception as e:
            print(f"ERROR processing {tag}: {str(e)}")
            return create_dummy_result(tag)

    for idx in range(16):
        center = CENTERS_CAM[idx]
        box = BOXES_CAM[idx]
        tag = f"top_{idx+1}" if idx < 8 else f"bottom_{(idx-8)+1}"
        samples.append(handle_idx(center, box, tag))

    return {
        "original_path": f"camera_frame_{int(time.time())}",
        "samples": samples,
    }


def create_dummy_result(sample_type):
    return {
        "result": "ERROR",
        "h_avg": 0, "h_min": 0, "h_max": 0,
        "s_avg": 0, "s_min": 0, "s_max": 0,
        "v_avg": 0, "v_min": 0, "v_max": 0,
        "image_path": f"error_{sample_type}"
    }


def update_camera_display(frame):
    """Update camera display in the GUI (SAM-style ROIs in CAM space)."""
    frame_fixed = cv2.resize(frame, (CAM_W, CAM_H), interpolation=cv2.INTER_LINEAR)

    for i, box in enumerate(TOP_BOXES_CAM, start=1):
        x0, y0, x1, y1 = box.astype(int)
        cv2.rectangle(frame_fixed, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(frame_fixed, f"T{i}", (x0 + 5, y0 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    for i, box in enumerate(BOT_BOXES_CAM, start=1):
        x0, y0, x1, y1 = box.astype(int)
        cv2.rectangle(frame_fixed, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(frame_fixed, f"B{i}", (x0 + 5, y0 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    display_frame = cv2.resize(frame_fixed, (DISP_W, DISP_H), interpolation=cv2.INTER_LINEAR)

    rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb_frame)
    imgtk = ImageTk.PhotoImage(image=img)

    camera_label.imgtk = imgtk
    camera_label.configure(image=imgtk)


# ──────────────── HSV Analysis ──────────────────────────────────────
def color_decision(value):
    return "RED" if value < COLOUR_THRES else "PURPLE"


def calculate_hsv_stats(rgba_arr, cv2):
    mask = rgba_arr[:, :, 3] > 0

    bgr = cv2.cvtColor(rgba_arr, cv2.COLOR_RGBA2BGR)
    hsv_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)

    if np.any(mask):
        return {
            "result": color_decision(h[mask].mean()),
            "h_avg": float(h[mask].mean()),
            "h_min": int(h[mask].min()),
            "h_max": int(h[mask].max()),
            "s_avg": float(s[mask].mean()),
            "s_min": int(s[mask].min()),
            "s_max": int(s[mask].max()),
            "v_avg": float(v[mask].mean()),
            "v_min": int(v[mask].min()),
            "v_max": int(v[mask].max())
        }
    else:
        return {
            "result": "NONE",
            "h_avg": 0, "h_min": 0, "h_max": 0,
            "s_avg": 0, "s_min": 0, "s_max": 0,
            "v_avg": 0, "v_min": 0, "v_max": 0
        }


def update_results_table(result):
    for item in result_tree.get_children():
        result_tree.delete(item)

    for i, sample in enumerate(result["samples"]):
        result_tree.insert(
            "",
            tk.END,
            values=(
                i + 1,
                sample['result'],
                f"{sample['h_avg']:.1f}",
                sample["h_min"],
                sample["h_max"],
                f"{sample['s_avg']:.1f}",
                sample["s_min"],
                sample["s_max"],
                f"{sample['v_avg']:.1f}",
                sample["v_min"],
                sample["v_max"]
            ),
            tags=('red_row' if sample['result'] == "RED" else 'pur_row')
        )


def update_hsv_visualization(result):
    global frame_count
    ax.clear()

    sample_nums = range(1, SAMPLE_NUM + 1)
    h_avgs = [s["h_avg"] for s in result["samples"]]
    s_avgs = [s["s_avg"] for s in result["samples"]]
    v_avgs = [s["v_avg"] for s in result["samples"]]

    ax.plot(sample_nums, h_avgs, "o-", label="Hue", color="#3BCF00")
    ax.plot(sample_nums, s_avgs, "o:", label="Saturation", color="#A1A1A1")
    ax.plot(sample_nums, v_avgs, "o:", label="Value", color="#949494")

    ax.set_title(f"HSV Values of {SAMPLE_NUM} Samples (Current: Frame {frame_count})")
    ax.set_xlabel("Sample Number")
    ax.set_ylabel("Value")
    ax.set_xticks(range(1, SAMPLE_NUM + 1))
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend()

    canvas.draw()


def export_results():
    if not results:
        messagebox.showwarning("No Results", "No processing results to export")
        return

    data = []
    for result in results:
        for i, sample in enumerate(result["samples"]):
            data.append(
                {
                    "original_image": result["original_path"],
                    "sample_number": i + 1,
                    "result": sample['result'],
                    "h_avg": sample['h_avg'],
                    "h_min": sample['h_min'],
                    "h_max": sample['h_max'],
                    "s_avg": sample['s_avg'],
                    "s_min": sample['s_min'],
                    "s_max": sample['s_max'],
                    "v_avg": sample['v_avg'],
                    "v_min": sample['v_min'],
                    "v_max": sample['v_max'],
                    "sample_path": sample['image_path']
                }
            )
    df = pd.DataFrame(data)

    save_path = filedialog.asksaveasfilename(
        defaultextension=".xlsx",
        filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
    )

    if save_path:
        df.to_excel(save_path, index=False)
        status_var.set(f"Results exported to {Path(save_path).name}")


# ──────────────── GUI Variables ──────────────────────────────────────
global orig_label, proc_label, status_var, progress_var, process_btn, stop_btn
global export_btn, result_tree, camera_label, erode_var, padding_var

# ──────────────── GUI layout ────────────────────────────────────────
main_frame = ttk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

top_frame = ttk.Frame(main_frame)
top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=False)

left_frame = ttk.Frame(top_frame)
left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5, pady=5)

left_top_frame = ttk.Frame(left_frame)
left_top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=False)
left_bot_frame = ttk.Frame(left_frame)
left_bot_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=False)

right_frame = ttk.Frame(top_frame)
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

bottom_frame = ttk.Frame(main_frame)
bottom_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=False)

# ──────────────── LEFT FRAME: Controls and Results ──────────────────
control_frame = ttk.LabelFrame(left_top_frame, text="Controls")
control_frame.pack(fill=tk.X, padx=5, pady=(0, 10))

params_frame = ttk.Frame(control_frame)
params_frame.pack(fill=tk.X, padx=5, pady=5)

erode_frame = ttk.Frame(params_frame)
erode_frame.pack(fill=tk.X, pady=2)
ttk.Label(erode_frame, text="Erode Pixels:").pack(side=tk.LEFT, padx=5)
erode_var = tk.IntVar(value=erode_pixels)
ttk.Scale(erode_frame, from_=0, to=50, variable=erode_var,
          command=lambda v: erode_label.config(text=f"{int(float(v))} px"),
          orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
erode_label = ttk.Label(erode_frame, text=f"{erode_pixels} px")
erode_label.pack(side=tk.LEFT, padx=5)

padding_frame = ttk.Frame(params_frame)
padding_frame.pack(fill=tk.X, pady=2)
ttk.Label(padding_frame, text="Padding Ratio:").pack(side=tk.LEFT, padx=5)
padding_var = tk.DoubleVar(value=padding_ratio)
ttk.Scale(padding_frame, from_=0, to=0.2, variable=padding_var,
          command=lambda v: padding_label.config(text=f"{float(v):.2f}"),
          orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
padding_label = ttk.Label(padding_frame, text=f"{padding_ratio:.2f}")
padding_label.pack(side=tk.LEFT, padx=5)

btn_frame = ttk.Frame(control_frame)
btn_frame.pack(fill=tk.X, padx=5, pady=10)

process_btn = ttk.Button(btn_frame, text="Start", command=start_processing)
process_btn.pack(side=tk.LEFT, padx=5)

stop_btn = ttk.Button(btn_frame, text="Stop", command=stop_processing_func, state=tk.DISABLED)
stop_btn.pack(side=tk.LEFT, padx=5)

export_btn = ttk.Button(btn_frame, text="Export", command=export_results, state=tk.DISABLED)
export_btn.pack(side=tk.LEFT, padx=5)

status_var = tk.StringVar(value="Ready to process camera")
status_label = ttk.Label(btn_frame, textvariable=status_var)
status_label.pack(side=tk.LEFT, padx=5)

vis_frame = ttk.LabelFrame(left_bot_frame, text="HSV Visualization Chart")
vis_frame.pack(fill=tk.BOTH, expand=False, padx=5, pady=5)
fig, ax = plt.subplots(figsize=(7, 4))
canvas = FigureCanvasTkAgg(fig, master=vis_frame)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=False)

result_frame = ttk.LabelFrame(bottom_frame, text="HSV Results Table")
result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

tree_frame = ttk.Frame(result_frame)
tree_frame.pack(fill=tk.BOTH, expand=True)

columns = ("Sample", "Result", "H_avg", "H_min", "H_max", "S_avg", "S_min", "S_max", "V_avg", "V_min", "V_max")
result_tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=16)

column_widths = {
    "Sample": 60,
    "Result": 70,
    "H_avg": 70, "H_min": 70, "H_max": 70,
    "S_avg": 70, "S_min": 70, "S_max": 70,
    "V_avg": 70, "V_min": 70, "V_max": 70
}

for col in columns:
    result_tree.heading(col, text=col)
    result_tree.column(col, width=column_widths.get(col, 80), anchor=tk.CENTER)

result_tree.tag_configure('red_row', background="#FFB4B4")
result_tree.tag_configure('pur_row', background="#E8AEFF")

tree_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=result_tree.yview)
result_tree.configure(yscroll=tree_scrollbar.set)

result_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# ──────────────── RIGHT FRAME: Camera Feed ──────────────────────────
camera_frame = ttk.LabelFrame(right_frame, text="Camera Output")
camera_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

camera_container = tk.Frame(camera_frame, width=800, height=500, bg="black")
camera_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
camera_container.pack_propagate(False)

camera_label = ttk.Label(camera_container, background="black")
camera_label.pack(fill=tk.BOTH, expand=True)

# ──────────────── GUI window ────────────────────────────────────────
# Handle closing the application carefully
def on_closing():
    global stop_processing, camera, processing_thread
    stop_processing = True
    print("Process closing")
    
    # Release camera resources
    if camera:
        try:
            camera.release()
        except Exception as e:
            print(f"Error releasing camera: {e}")
    
    # Destroy any OpenCV windows
    try:
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error destroying CV2 windows: {e}")
    
    # Wait for processing thread to finish
    if processing_thread and processing_thread.is_alive():
        processing_thread.join(timeout=3.0)
        if processing_thread.is_alive():
            print("Warning: Processing thread did not finish in time")
    
    # Destroy the root window
    try:
        root.quit()
        root.destroy()
    except Exception as e:
        print(f"Error destroying root window: {e}")
    
    print("Process closed")

root.after(100, initialize_camera_display)
root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()