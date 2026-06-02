# Version 260601.1
# ──────────────── Libraries Import ───────────────────────────────────
from multiprocessing.dummy import Process
import time, threading
from pathlib import Path
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
from PIL import Image, ImageTk
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch
from segment_anything import sam_model_registry, SamPredictor
matplotlib.use("TkAgg")

# ──────────────── Tk root window ─────────────────────────────────────
root = tk.Tk()
root.state("normal")
root.title("IC-Project  ·  Chemical Color Gradient Detection")
root.geometry("1400x900")

# ──────────────── Global configuration ───────────────────────────────
SAMPLE_NUM      = 16    # Total number of samples
N_COLS          = 8     # Number of columns (tubes) per row

CAM_W, CAM_H = 1280, 720    # Standardize camera frames to this size before drawing/cropping
DISP_W, DISP_H = 800, 500   # Display size
Y_CUT_ENABLED      = False  # Hard-coded Y cleanup (post-SAM)
Y_CUT_FRAC         = 0.9    # Always remove mask pixels below this fraction of the *crop* height (0..1)
Y_TRIGGER_FRAC     = 0      # If the mask's bottom-most pixel y is beyond this fraction, trigger the "÷2" rule

# ──────────────── SAM configuration ─────────────────────────────────
SAM_CHECKPOINT = Path("program/sam_vit_b_01ec64.pth")   # Put checkpoint file next to this script (or change the path)
SAM_MODEL_TYPE = "vit_b"  # "vit_h" | "vit_l" | "vit_b"
SAM_DEVICE     = "cpu"    # "cuda" if you have GPU + torch cuda build

sam_predictor = None  # will be initialized once

# Optional: enable negative corner points if you still get "whole ROI" masks.
USE_NEGATIVE_POINTS = False     # Keep False for minimal changes / notebook-like prompting.

# ──────────────── ROI geometry ─────────────────────────
# ---- CALIBRATE THESE ONCE for your camera/layout ----
x_left_frac  = 0.2
x_right_frac = 0.8

y_top_frac = 0.08   # y-position (as fraction of H) near the liquid of the top row
y_bot_frac = 0.40   # y-position near the liquid of the bottom row

# Box size relative to spacing / image size
box_w_scale = 0.50   # box width = box_w_scale * tube spacing (dx)
box_up_frac = 0.00   # how far box extends above row y (fraction of H)
box_dn_frac = 0.04   # how far box extends below row y (fraction of H)


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


colour_threshold_h    = 105.0             # Default: 105.8
colour_margin_h       = 30.0              # Default: 10.0   (MIN) Not used
colour_threshold_s    = 20.0              # Default: 20.0   (MIN)
colour_threshold_v    = 220.0             # Default: 200.0  (MAX)

INFO_PREFIX     = "*INFO: "         # Shown in console
ERROR_PREFIX    = "*ERROR: "        # Shown in console

# ──────────────── Program variables ──────────────────────────────────
# NOTE: keep name "erode_pixels", but now it truly means pixels (kernel radius)
erode_pixels        = 0
padding_ratio       = 0.05
frame_count         = 0
processing          = False
previewing          = False
stop_preview_flag   = False
camera              = None
preview_camera_obj  = None
camera_indices      = [0, 1, 2]  # Try these camera indices in order
results             = []

output_folder = None
crops_dir = None
masks_dir = None
camera_output = None

def folder_creation():
    global output_folder, crops_dir, masks_dir
    # Create output folders after GUI starts
    output_folder = Path("output_folder")

    # Remove the output folders if they exist from previous runs
    if output_folder.exists():
        shutil.rmtree(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    # subfolders
    crops_dir = output_folder / "crops"
    crops_dir.mkdir(exist_ok=True, parents=True)
    masks_dir = output_folder / "masks"
    masks_dir.mkdir(exist_ok=True, parents=True)
    '''
    sample_dir = output_folder / "samples"
    sample_dir.mkdir(exist_ok=True, parents=True)
    '''

# processing thread reference (to join on closing)
processing_thread = None
preview_thread = None


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


# ──────────────── Mask utilities ───────────────────────
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
    min_area_frac=0.06,     # minimum chemical area as fraction of tube area
    bottomness_frac=0.50,   # minimum bottomness of chemical cue to be valid
    keep_core_frac=0.55     # fraction of distance transform to keep as core
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
    Pick a SAM mask that isn't "everything". (Kept for reference; notebook uses argmax(scores).)
    crop_slice = (y0,y1,x0,x1)
    """
    y0, y1, x0, x1 = crop_slice
    best_idx = int(np.argmax(scores))
    best_val = -1e9

    for j in range(masks_full.shape[0]):
        m_crop = masks_full[j][y0:y1, x0:x1]
        area_frac = float(m_crop.mean()) if m_crop.size else 1.0  # [0..1]
        sc = float(scores[j])

        penalty = 0.65 * area_frac if prefer_not_full else 0.0
        val = sc - penalty

        if prefer_not_full and area_frac > 0.98:
            val -= 2.0

        if val > best_val:
            best_val = val
            best_idx = j

    return int(best_idx)


def hard_y_cleanup(mask_bool, cut_frac=Y_CUT_FRAC, trigger_frac=Y_TRIGGER_FRAC):
    if not Y_CUT_ENABLED:
        return mask_bool

    H = mask_bool.shape[0]
    if H <= 0:
        return mask_bool

    out = mask_bool.copy()
    trigger_y = int(trigger_frac * H)

    if mask_bool.any():
        ymax = int(np.where(mask_bool)[0].max())

        if ymax >= trigger_y:
            ys = np.where(mask_bool)[0]
            ymin, ymax = int(ys.min()), int(ys.max())
            cut_at = ymin + (ymax - ymin) // 2
            out[:cut_at, :] = False   # keep bottom half of the mask

    cut_y = int(cut_frac * H)
    cut_y = int(np.clip(cut_y, 0, H))
    out[cut_y:, :] = False

    return out


# ──────────────── Threading Functions ───────────────────────────────
def start_processing():
    """Start the processing schedule and its thread upon clicking start processing button"""
    global processing, camera, processing_thread

    if not init_sam_predictor():
        return

    camera = None
    folder_creation()

    try:
        print("INFO: Trying camera index:", cam_index_var.get())
        
        camera = cv2.VideoCapture(cam_index_var.get())
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        if not camera.isOpened():
            raise Exception(f"Cannot open camera index {cam_index_var.get()}")
            
        ret, test_frame = camera.read()
        if not ret or test_frame is None:
            camera.release()
            raise Exception(f"Cannot read from camera index {cam_index_var.get()}")
            
        print(f"INFO: Successfully using camera index {cam_index_var.get()}")
        
    except Exception as e:
        print(f"ERROR: Camera index {cam_index_var.get()} failed: {str(e)}")
        if camera:
            camera.release()
        messagebox.showerror("Camera Error", f"Cannot access camera {cam_index_var.get()}. Please check camera connection.")
        return

    if processing:
        return
    processing = True

    status_var.set("Starting the process... Please wait")

    process_btn.config(state=tk.DISABLED)
    export_btn.config(state=tk.DISABLED)
    export_img_btn.config(state=tk.DISABLED)

    processing_thread = threading.Thread(target=main_process)
    processing_thread.daemon = False
    processing_thread.start()


def preview_camera():
    """Start camera preview as a thread"""
    global previewing, stop_preview_flag, preview_camera_obj, preview_thread
    
    if previewing:
        return
    
    preview_camera_obj = None
    status_var.set("Starting camera preview...")
    
    try:
        print("INFO: Starting camera preview, index:", cam_index_var.get())
        
        preview_camera_obj = cv2.VideoCapture(cam_index_var.get())
        preview_camera_obj.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        preview_camera_obj.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        
        if not preview_camera_obj.isOpened():
            raise Exception(f"Cannot open camera index {cam_index_var.get()}")
            
        ret, test_frame = preview_camera_obj.read()
        if not ret or test_frame is None:
            preview_camera_obj.release()
            raise Exception(f"Cannot read from camera index {cam_index_var.get()}")
            
        print(f"INFO: Preview using camera index {cam_index_var.get()}")
        
    except Exception as e:
        print(f"ERROR: Camera preview failed: {str(e)}")
        if preview_camera_obj:
            preview_camera_obj.release()
        messagebox.showerror("Camera Error", f"Cannot access camera {cam_index_var.get()}. Please check camera connection.")
        return
    
    previewing = True
    stop_preview_flag = False
    
    preview_btn.config(state=tk.DISABLED)
    stop_preview_btn.config(state=tk.NORMAL)
    process_btn.config(state=tk.DISABLED)
    cam_index_menu.config(state=tk.DISABLED)

    status_var.set("Started camera preview")
    
    preview_thread = threading.Thread(target=preview_loop)
    preview_thread.daemon = False
    preview_thread.start()


def stop_preview():
    """Stop camera preview"""
    global stop_preview_flag
    status_var.set("Stopping preview...")
    stop_preview_flag = True


def preview_loop():
    """Preview loop that shows camera feed with ROI boxes"""
    global previewing, preview_camera_obj
    
    while not stop_preview_flag and preview_camera_obj and preview_camera_obj.isOpened():
        ret, frame = preview_camera_obj.read()
        if not ret:
            break
        
        update_camera_display(frame)
        time.sleep(0.03)
    
    if preview_camera_obj:
        preview_camera_obj.release()
    
    previewing = False
    preview_btn.config(state=tk.NORMAL)
    stop_preview_btn.config(state=tk.DISABLED)
    process_btn.config(state=tk.NORMAL)
    cam_index_menu.config(state="readonly")
    status_var.set("Preview stopped. Ready to process")


# ──────────────── Camera Processing ───────────────────────────────
def main_process():
    """The main camera processing function"""
    global processing, results, camera, frame_count

    results = []

    if camera and camera.isOpened():
        ret, frame = camera.read()
        if not ret:
            status_var.set("Failed to capture frame from camera")
            print("ERROR: Failed to capture frame from camera")
        else:
            display_frame = frame.copy()
            update_camera_display(display_frame)

            status_var.set(f"Processing: Frame {frame_count+1}")
            print(f"INFO: Processing frame {frame_count+1}.")

            # FIX 5: ensure int/float types are correct
            current_erode = int(float(erode_pixels))
            current_padding = float(padding_ratio)

            result = process_camera_frame(frame, current_erode, current_padding)
            results.append(result)

            update_results_table(result)
            update_hsv_visualization(result)

            frame_count += 1
    else:  
        status_var.set("Camera is not available")
        print("ERROR: Camera is not available")

    if camera:
        camera.release()
    processing = False
    process_btn.config(state=tk.NORMAL)
    export_btn.config(state=tk.NORMAL)
    export_img_btn.config(state=tk.NORMAL)

    status_var.set(f"Frame {frame_count} completed successfully!")
    print(f"INFO: Completed processing frame {frame_count}.")


def initialize_camera_display():
    root.update_idletasks()


def process_camera_frame(frame, erode_px, padding_ratio):
    """Process a single camera frame using SAM tube+chemical segmentation."""
    global sam_predictor, crops_dir, masks_dir, frame_count

    samples = []

    frame_fixed = cv2.resize(frame, (CAM_W, CAM_H), interpolation=cv2.INTER_LINEAR)
    bgr_full = frame_fixed
    rgb_full = cv2.cvtColor(bgr_full, cv2.COLOR_BGR2RGB)
    h, w = rgb_full.shape[:2]

    sam_predictor.set_image(rgb_full)

    def handle_idx(center_xy, box_xyxy, tag):
        global sam_predictor, frame_count, crops_dir, masks_dir
        x0, y0, x1, y1 = box_xyxy.astype(int)

        # clip and reuse this SAME box for crop + SAM
        x0, y0, x1, y1 = clip_rect(x0, y0, x1, y1, w, h)
        box_clipped = np.array([x0, y0, x1, y1], dtype=np.float32)

        crop_rgb = rgb_full[y0:y1, x0:x1].copy()
        if crop_rgb.size == 0:
            return create_dummy_result(tag)

        try:
            # FIX 2: use notebook-style center prompt (near liquid), but clamp into box
            cx = float(np.clip(center_xy[0], x0 + 1, x1 - 2))
            cy = float(np.clip(center_xy[1], y0 + 1, y1 - 2))

            if USE_NEGATIVE_POINTS:
                point_coords = np.array([
                    [cx, cy],           # positive
                    [x0 + 2, y0 + 2],    # negatives (corners)
                    [x1 - 3, y0 + 2],
                    [x0 + 2, y1 - 3],
                    [x1 - 3, y1 - 3],
                ], dtype=np.float32)
                point_labels = np.array([1, 0, 0, 0, 0], dtype=np.int32)
            else:
                point_coords = np.array([[cx, cy]], dtype=np.float32)
                point_labels = np.array([1], dtype=np.int32)

            masks, scores, _ = sam_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box_clipped,
                multimask_output=True
            )

            # FIX 3: notebook behavior = choose argmax(scores)
            kbest = int(np.argmax(scores))

            tube_mask_full = masks[kbest].astype(bool)

            tube_mask_crop = tube_mask_full[y0:y1, x0:x1]
            tube_mask_crop = largest_component(tube_mask_crop)
            tube_mask_crop = hard_y_cleanup(tube_mask_crop) # hard-coded cleanup (removes pixels below certain y; triggers ÷2 rule if too low)

            # Chemical/liquid region inside tube
            chem_mask = chemical_from_crop_strict(crop_rgb, tube_mask_crop)
            chem_mask = hard_y_cleanup(chem_mask)  # hard-coded cleanup

            dilate_px = max(3, int(0.03 * (x1 - x0)))
            bg, tube_ov, tube_non, chem = split_parts(crop_rgb, tube_mask_crop, chem_mask, dilate_px=dilate_px)

            label = np.zeros(bg.shape, dtype=np.uint8)
            label[tube_non] = 1
            label[tube_ov] = 2
            label[chem] = 3

            # Save masks
            '''
            Image.fromarray((tube_mask_crop.astype(np.uint8) * 255)).save(masks_dir / f"{frame_count}_tube_{tag}_tube.png")
            Image.fromarray((chem.astype(np.uint8) * 255)).save(masks_dir / f"{frame_count}_tube_{tag}_chem.png")
            Image.fromarray((tube_ov.astype(np.uint8) * 255)).save(masks_dir / f"{frame_count}_tube_{tag}_tube_overlap.png")
            Image.fromarray((tube_non.astype(np.uint8) * 255)).save(masks_dir / f"{frame_count}_tube_{tag}_tube_nonoverlap.png")
            Image.fromarray(label).save(masks_dir / f"{frame_count}_tube_{tag}_label.png")
            Image.fromarray(crop_rgb).save(crops_dir / f"{frame_count}_tube_{tag}_raw.png")
            '''

            # Cropped tube RGBA (trimmed)
            tube_rgba_trim = mask_to_rgba_and_trim(
                crop_rgb=crop_rgb,
                mask_bool=tube_mask_crop,      # alpha = tube
                trim_mask_bool=tube_mask_crop, # trim bbox = tube
                erode_px=0,                    # keep tube edges for bbox
                padding_ratio=padding_ratio
            )
            crop_path = crops_dir / f"{frame_count}_cropped_{tag}.png"
            Image.fromarray(tube_rgba_trim).save(crop_path)

            stats = calculate_hsv_stats(tube_rgba_trim, cv2)    # Use cropped tube RGBA for HSV stats!
            stats["image_path"] = str(crop_path)

            # --------------Optional: save sample RGBA trimmed by tube bbox-----------------
            # Use chemical pixels for HSV if available, else tube
            use_mask_for_hsv = chem if chem.sum() > 0 else tube_mask_crop

            # Sample trimmed by tube bbox (alpha can be chem)
            result_rgba = mask_to_rgba_and_trim(
                crop_rgb=crop_rgb,
                mask_bool=use_mask_for_hsv,    # alpha = chem (or tube)
                trim_mask_bool=tube_mask_crop, # trim bbox = tube
                erode_px=erode_px,
                padding_ratio=padding_ratio
            )
            '''
            Image.fromarray(result_rgba).save(sample_dir / f"{frame_count}_sample_{tag}.png")
            '''
            # ------------------------------------------------------------------------------

            return stats

        except Exception as e:
            print(f"ERROR processing {tag}: {str(e)}")
            return create_dummy_result(tag)

    for idx in range(16):
        center = CENTERS_CAM[idx]
        box = BOXES_CAM[idx]
        tag = f"{idx+1}"
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
    global camera_output
    
    frame_fixed = cv2.resize(frame, (CAM_W, CAM_H), interpolation=cv2.INTER_LINEAR)
    camera_output = frame_fixed.copy()

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
def color_decision(h_avg, s_avg, v_avg):
    h_thres = float(colour_thres_var.get())
    s_thres = float(colour_s_thres_var.get())

    if s_avg < s_thres or v_avg > colour_threshold_v:
        return "NONE"

    return "RED" if h_avg < h_thres else "PURPLE"


def calculate_hsv_stats(rgba_arr, cv2):
    mask = rgba_arr[:, :, 3] > 0

    bgr = cv2.cvtColor(rgba_arr, cv2.COLOR_RGBA2BGR)
    hsv_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)

    if np.any(mask):
        return {
            "result": color_decision(h[mask].mean(), s[mask].mean(), v[mask].mean()),
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
                f"Top {i + 1}" if i < N_COLS else f"Bottom {i - 7}",
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
            tags=(
                'red_row' if sample['result'] == "RED" 
                else 'pur_row' if sample['result'] == "PURPLE"
                else 'none_row'
            )
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

    ax.set_title(f"HSV Values of {SAMPLE_NUM} Samples (Current: Frame {frame_count+1})")
    ax.set_xlabel("Sample Number")
    ax.set_ylabel("Value")
    ax.set_xticks(range(1, SAMPLE_NUM + 1))
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend()

    canvas.draw()


# ──────────────── Result Export ──────────────────────────────────────
def export_result():
    """Export the current processing results to a CSV file."""
    if not results:
        messagebox.showwarning("No Results", "No processing results to export")
        return

    data = []
    for result in results:
        for i, sample in enumerate(result["samples"]):
            data.append(
                {
                    # "original_image": result["original_path"],
                    "Running count": frame_count,
                    "Sample No.": f"Top {i + 1}" if i < N_COLS else f"Bottom {i - 7}",
                    "Result": sample['result'],
                    "H_avg": sample['h_avg'],
                    "H_min": sample['h_min'],
                    "H_max": sample['h_max'],
                    "S_avg": sample['s_avg'],
                    "S_min": sample['s_min'],
                    "S_max": sample['s_max'],
                    "V_avg": sample['v_avg'],
                    "V_min": sample['v_min'],
                    "V_max": sample['v_max'],
                    "Path": sample['image_path']
                }
            )
        data.append({})  # Append empty separation row after each frame's samples
    df = pd.DataFrame(data)

    save_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )

    if save_path:
        df.to_csv(save_path, index=False)
        status_var.set(f"Results exported to {Path(save_path).name}")


def export_image():
    """Export the current camera output to a PNG file."""
    global camera_output

    if not results:
        messagebox.showwarning("No Results", "No processing results to export")
        return
    
    save_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
    )

    if save_path:
        # Use camera_output from the last processed frame
        if camera_output is not None:
            bgr_output = cv2.cvtColor(camera_output, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, bgr_output)
            status_var.set(f"Camera image exported to {Path(save_path).name}")
        else:
            messagebox.showerror("No Image", "No camera image available to export")


# ──────────────── GUI Variables ──────────────────────────────────────
global status_var, process_btn
global export_btn, export_img_btn, result_tree, camera_label
global preview_btn, stop_preview_btn

# ──────────────── GUI layout ────────────────────────────────────────
main_frame = ttk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

top_frame = ttk.Frame(main_frame)
top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=False)

left_frame = ttk.Frame(top_frame)
left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=0, pady=5)

left_top_frame = ttk.Frame(left_frame)
left_top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=False)
left_bot_frame = ttk.Frame(left_frame)
left_bot_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=False)

right_frame = ttk.Frame(top_frame)
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=0, pady=5)

right_top_frame = ttk.Frame(right_frame)
right_top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=0, pady=0)
right_bot_frame = ttk.Frame(right_frame)
right_bot_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=0, pady=0)

bottom_frame = ttk.Frame(main_frame)
bottom_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=False)

# ──────────────── LEFT FRAME: Controls and Results ──────────────────

# ═════════════════ TOP LEFT FRAME: Controls ═══════════════════════
control_frame = ttk.LabelFrame(left_top_frame, text="Controls")
control_frame.pack(fill=tk.X, padx=5, pady=(0, 10))

btn_frame = ttk.Frame(control_frame)
btn_frame.pack(fill=tk.X, padx=5, pady=10)

process_btn = ttk.Button(btn_frame, text="Start", command=start_processing)
process_btn.pack(side=tk.LEFT, padx=5)

export_btn = ttk.Button(btn_frame, text="Export Result", command=export_result, state=tk.DISABLED)
export_btn.pack(side=tk.LEFT, padx=5)

export_img_btn = ttk.Button(btn_frame, text="Export Image", command=export_image, state=tk.DISABLED)
export_img_btn.pack(side=tk.LEFT, padx=5)

status_var = tk.StringVar(value="Ready to process")
status_label = ttk.Label(btn_frame, textvariable=status_var)
status_label.pack(side=tk.LEFT, padx=5)

# ═══════════════ BOTTOM LEFT FRAME: HSV Visualization and Results ═══════════════════════
vis_frame = ttk.LabelFrame(left_bot_frame, text="HSV Visualization Chart")
vis_frame.pack(fill=tk.BOTH, expand=False, padx=5, pady=5)
fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
canvas = FigureCanvasTkAgg(fig, master=vis_frame)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=False)

result_frame = ttk.LabelFrame(bottom_frame, text="HSV Results Table")
result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

tree_frame = ttk.Frame(result_frame)
tree_frame.pack(fill=tk.BOTH, expand=True)

columns = ("Sample No.", "Result", "H_avg", "H_min", "H_max", "S_avg", "S_min", "S_max", "V_avg", "V_min", "V_max")
result_tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=16)

column_widths = {
    "Sample No.": 60,
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
result_tree.tag_configure('none_row', background="#D9D9D9")

tree_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=result_tree.yview)
result_tree.configure(yscroll=tree_scrollbar.set)

result_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# ──────────────── RIGHT FRAME: Camera Feed ──────────────────────────
threshold_frame = ttk.LabelFrame(right_top_frame, text="Thresholds (H_avg / S_avg)")
threshold_frame.pack(side=tk.LEFT, fill=tk.X, padx=5, pady=(0, 10), expand=True)

thres_inner_frame = ttk.Frame(threshold_frame)
thres_inner_frame.pack(fill=tk.X, padx=5, pady=5)

# H_avg slider
h_row = ttk.Frame(thres_inner_frame)
h_row.pack(fill=tk.X, pady=(0, 6))

colour_thres_var = tk.DoubleVar(value=colour_threshold_h)
ttk.Label(h_row, text="H_avg (RED / PURPLE)").pack(side=tk.LEFT, padx=(0, 8))

ttk.Scale(
    h_row,
    from_=0, to=180,
    variable=colour_thres_var,
    command=lambda v: colour_thres_h_label.config(text=f"{float(v):.1f}"),
    orient=tk.HORIZONTAL
).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

colour_thres_h_label = ttk.Label(h_row, text=f"{colour_threshold_h:.1f}")
colour_thres_h_label.pack(side=tk.LEFT, padx=5)

# S_avg slider
s_row = ttk.Frame(thres_inner_frame)
s_row.pack(fill=tk.X)

colour_s_thres_var = tk.DoubleVar(value=colour_threshold_s)
ttk.Label(s_row, text="S_avg (NONE / COLOR)").pack(side=tk.LEFT, padx=(0, 8))

ttk.Scale(
    s_row,
    from_=0, to=255,
    variable=colour_s_thres_var,
    command=lambda v: colour_thres_s_label.config(text=f"{float(v):.1f}"),
    orient=tk.HORIZONTAL
).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

colour_thres_s_label = ttk.Label(s_row, text=f"{colour_threshold_s:.1f}")
colour_thres_s_label.pack(side=tk.LEFT, padx=5)

# ---
cam_menu_frame = ttk.LabelFrame(right_top_frame, text="Camera Selection")
cam_menu_frame.pack(side=tk.RIGHT, fill=tk.X, padx=5, pady=(0, 10), expand=True)

cam_controls_frame = ttk.Frame(cam_menu_frame)
cam_controls_frame.pack(padx=5, pady=5)

cam_index_var = tk.IntVar(value=camera_indices[0] if camera_indices else 0)
cam_index_menu = ttk.Combobox(
    cam_controls_frame,
    textvariable=cam_index_var,
    values=camera_indices,
    state="readonly",
    width=10
)
cam_index_menu.pack(side=tk.LEFT, padx=5)

preview_btn = ttk.Button(cam_controls_frame, text="Start Preview", command=preview_camera)
preview_btn.pack(side=tk.LEFT, padx=5)

stop_preview_btn = ttk.Button(cam_controls_frame, text="Stop Preview", command=stop_preview, state=tk.DISABLED)
stop_preview_btn.pack(side=tk.LEFT, padx=5)

# ---
camera_frame = ttk.LabelFrame(right_bot_frame, text="Camera Output")
camera_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, padx=5, pady=(0, 10), expand=True)

camera_container = tk.Frame(camera_frame, width=600, height=400, bg="black")
camera_container.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
camera_container.pack_propagate(False)

camera_label = ttk.Label(camera_container, background="black")
camera_label.pack(fill=tk.BOTH, expand=True)

# ──────────────── GUI window ────────────────────────────────────────
# Handle closing the application carefully
def on_closing():
    global camera, processing_thread, stop_preview_flag, preview_camera_obj, preview_thread
    stop_preview_flag = True
    print("Process closing")
    
    # Release camera resources
    if camera:
        try:
            camera.release()
        except Exception as e:
            print(f"Error releasing camera: {e}")
    
    # Release preview camera
    if preview_camera_obj:
        try:
            preview_camera_obj.release()
        except Exception as e:
            print(f"Error releasing preview camera: {e}")
    
    # Destroy any OpenCV windows
    try:
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error destroying CV2 windows: {e}")
    
    # Wait for processing thread to finish
    if processing_thread and processing_thread.is_alive():
        print("Waiting for processing thread to finish...")
        processing_thread.join(timeout=3.0)
        if processing_thread.is_alive():
            print("Warning: Processing thread did not finish in time")
    
    # Wait for preview thread to finish
    if preview_thread and preview_thread.is_alive():
        print("Waiting for preview thread to finish...")
        preview_thread.join(timeout=3.0)
        if preview_thread.is_alive():
            print("Warning: Preview thread did not finish in time")
    
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