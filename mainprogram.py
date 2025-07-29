"""
IC-Project : Color Gradient Detection - PC GUI
Version 250729.1
────────────────────────────────────────────────────────────────────────
Tested with Python 3.12

To-do:
1. Add 8 fixed positions (width, height)
"""

# ──────────────── Libraries Import ───────────────────────────────────
import os
import time
import threading
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
from rembg import remove

# ──────────────── Tk root window ─────────────────────────────────────
root = tk.Tk()
root.title("IC-Project  ·  Color Gradient Detection")
root.geometry("1200x800")

# ──────────────── Global configuration ───────────────────────────────
INFO_PREFIX     = "*INFO: "         # Shown in console
ERROR_PREFIX    = "*ERROR: "        # Shown in console

# ──────────────── Program variables ──────────────────────────────────
erode_pixels        = 20                                # Tunable
padding_ratio       = 0.05                              # Tunable
input_folder        = Path("image_to_be_processed")   
output_folder       = Path("image_processed")        
current_image_idx   = 0
image_paths         = []
processing          = False
stop_processing     = False
results             = []

# ──────────────── Threading Functions ───────────────────────────────
"""Start the processing schedule and its thread upon clicking start processing button"""
def start_processing():
    global processing, stop_processing, image_paths, current_image_idx
    
    # Load images from folder
    image_paths = [p for p in input_folder.glob("*") 
                   if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    
    if not image_paths:
        messagebox.showwarning("No Images", "Please select an input folder with images")
        print(f"{INFO_PREFIX}No image found at {input_folder}")
        return
    else:
        print(f"{INFO_PREFIX}Found {len(image_paths)} images")
        
    if processing:
        return
    processing = True
    stop_processing = False
    current_image_idx = 0
    
    # Button UI
    process_btn.config(state=tk.DISABLED)
    stop_btn.config(state=tk.NORMAL)
    export_btn.config(state=tk.DISABLED)

    # Threading
    processing_thread = threading.Thread(target=main_process)
    processing_thread.daemon = True
    processing_thread.start()

"""Stop the processing schedule upon clicking stop processing button"""
def stop_processing():
    global stop_processing
    stop_processing = True
    status_var.set("Processing stopped by user")

# ──────────────── Image(s) Processing ───────────────────────────────
"""The main image processing function"""
def main_process():
    global processing, current_image_idx, results

    results = []
    for idx, image_path in enumerate(image_paths[current_image_idx:], start=current_image_idx):
        print(f"{INFO_PREFIX}FUNCTION: Main process: idx = {idx}")
        if stop_processing:
            break

        status_var.set(f"Processing image {idx+1}/{len(image_paths)}: {image_path.name}")
        progress_var.set((idx + 1) / len(image_paths) * 100)
        root.update()

        try:
            # Process image
            result = image_process(image_path, cv2, remove)
            results.append(result)
            update_results_table(result)
            update_hsv_visualization(result)

            # Delay before next image
            for i in range(5):
                if stop_processing:
                    break
                status_var.set(f"Processing image {idx+1}/{len(image_paths)}: {image_path.name} - Waiting {5-i}s")
                time.sleep(1)
            current_image_idx = idx + 1

        except Exception as e:
            status_var.set(f"Error whilst processing!")
            print(f"{ERROR_PREFIX}Error processing {image_path.name}: {str(e)}")
            continue

    # After completing ALL images
    processing = False
    process_btn.config(state=tk.NORMAL)
    stop_btn.config(state=tk.DISABLED)
    export_btn.config(state=tk.NORMAL)

    if stop_processing:
        status_var.set("Processing stopped")
    else:
        status_var.set("Processing completed")

"""Parent function for processing ONE image"""
def image_process(image_path, cv2, remove):
    print(f"{INFO_PREFIX}FUNCTION: Image process")
    
    # Remove background and crop
    cropped_rgba = remove_background_and_crop(image_path)

    # Save combined image
    combined_path = output_folder / "combined" / f"{image_path.stem}_combined.png"
    Image.fromarray(cropped_rgba).save(combined_path)

    # Split into 8 horizontal samples
    height, width = cropped_rgba.shape[:2]
    segment_width = width // 8
    samples = []

    for i in range(8):
        left = i * segment_width
        right = left + segment_width
        sample_rgba = cropped_rgba[:, left:right]

        # Save the sample
        sample_path = output_folder / "samples" / f"{image_path.stem}_sample_{i+1}.png"
        Image.fromarray(sample_rgba).save(sample_path)

        # Calculate HSV values
        stats = calculate_hsv_stats(sample_rgba, cv2)
        stats["image_path"] = str(sample_path)
        samples.append(stats)

    return {
        "original_path": str(image_path),
        "combined_path": str(combined_path),
        "samples": samples,
    }

"""Parent function of removing background and cropping sections of ONE image"""
def remove_background_and_crop(image_path):
    print(f"{INFO_PREFIX}FUNCTION: Remove background and crop")
    
    # Background removal
    img = Image.open(image_path)
    img_rgba = remove(img)
    rgba_arr = np.array(img_rgba)

    # Find non-transparent pixels
    alpha = rgba_arr[:, :, 3]
    coords = np.argwhere(alpha > 0)
    if coords.size == 0:
        return rgba_arr

    # Initial crop with padding
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    h, w = y1 - y0, x1 - x0
    pad_y, pad_x = int(h * padding_var.get()), int(w * padding_var.get())
    y0 = max(0, y0 - pad_y)
    y1 = min(rgba_arr.shape[0], y1 + pad_y)
    x0 = max(0, x0 - pad_x)
    x1 = min(rgba_arr.shape[1], x1 + pad_x)
    cropped = rgba_arr[y0:y1, x0:x1].copy()

    # Erode mask to remove outer edges
    alpha_c = cropped[:, :, 3]
    inner_mask = erode_mask(alpha_c)
    cropped[~inner_mask, 3] = 0         # Set trimed edges to transparent

    # Recrop to remove transparent borders
    alpha_new = cropped[:, :, 3]
    coords2 = np.argwhere(alpha_new > 0)
    if coords2.size == 0:
        return cropped

    y0b, x0b = coords2.min(axis=0)
    y1b, x1b = coords2.max(axis=0) + 1
    return cropped[y0b:y1b, x0b:x1b]

"""Applying erode mask after removing the image's background"""
def erode_mask(alpha):
    print(f"{INFO_PREFIX}FUNCTION: Erode mask")
    
    mask = (alpha > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(mask, kernel, iterations=erode_var.get())
    return eroded > 0

# ──────────────── HSV Analysis ──────────────────────────────────────
"""Calculating HSV values of individual pixels of a processed image"""
def calculate_hsv_stats(rgba_arr, cv2):
    print(f"{INFO_PREFIX}FUNCTION: Calculate HSV Stats")
    
    mask = rgba_arr[:, :, 3] > 0        # Create mask of non-transparent pixels
    
    # Converting to HSV spec
    bgr = cv2.cvtColor(rgba_arr, cv2.COLOR_RGBA2BGR)
    hsv_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)

    if np.any(mask):
        return {
            "h_avg": h[mask].mean(),
            "h_min": h[mask].min(),
            "h_max": h[mask].max(),
            "s_avg": s[mask].mean(),
            "s_min": s[mask].min(),
            "s_max": s[mask].max(),
            "v_avg": v[mask].mean(),
            "v_min": v[mask].min(),
            "v_max": v[mask].max()
        }
    else:
        return {
            "h_avg": 0, "h_min": 0, "h_max": 0,
            "s_avg": 0, "s_min": 0, "s_max": 0,
            "v_avg": 0, "v_min": 0, "v_max": 0
        }

"""Updating the GUI result table"""
def update_results_table(result):
    print(f"{INFO_PREFIX}FUNCTION: Updating result table")
    
    # Clear existing data
    for item in result_tree.get_children():
        result_tree.delete(item)

    # Add new data
    for i, sample in enumerate(result["samples"]):
        result_tree.insert(
            "",
            tk.END,
            values=(
                i + 1,
                f"{sample['h_avg']:.1f}",
                sample["h_min"],
                sample["h_max"],
                f"{sample['s_avg']:.1f}",
                sample["s_min"],
                sample["s_max"],
                f"{sample['v_avg']:.1f}",
                sample["v_min"],
                sample["v_max"]
            )
        )

"""Updating the GUI table visualization with plt"""
def update_hsv_visualization(result):
    print(f"{INFO_PREFIX}FUNCTION: Updating HSV visualization")
    
    ax.clear()

    # Prepare data
    sample_nums = range(1, 9)
    h_avgs = [s["h_avg"] for s in result["samples"]]
    s_avgs = [s["s_avg"] for s in result["samples"]]
    v_avgs = [s["v_avg"] for s in result["samples"]]

    # Plot HSV averages
    ax.plot(sample_nums, h_avgs, "o-", label="Hue", color="red")
    ax.plot(sample_nums, s_avgs, "o-", label="Saturation", color="green")
    ax.plot(sample_nums, v_avgs, "o-", label="Value", color="blue")

    # Format plot
    ax.set_title("HSV Values by Sample")
    ax.set_xlabel("Sample Number")
    ax.set_ylabel("Value")
    ax.set_xticks(range(1, 9))
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend()

    canvas.draw()

"""With Excel format (.xlsx) exporting function"""
def export_results():
    if not results:
        messagebox.showwarning("No Results", "No processing results to export")
        return

    data = []
    for result in results:
        for i, sample in enumerate(result["samples"]):
            data.append(
                {
                    "original_image": Path(result["original_path"]).name,
                    "sample_number": i + 1,
                    "h_avg": sample["h_avg"],
                    "h_min": sample["h_min"],
                    "h_max": sample["h_max"],
                    "s_avg": sample["s_avg"],
                    "s_min": sample["s_min"],
                    "s_max": sample["s_max"],
                    "v_avg": sample["v_avg"],
                    "v_min": sample["v_min"],
                    "v_max": sample["v_max"],
                    "sample_path": sample["image_path"],
                }
            )
    df = pd.DataFrame(data)

    # Save to Excel
    save_path = filedialog.asksaveasfilename(
        defaultextension=".xlsx",
        filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
    )

    if save_path:
        df.to_excel(save_path, index=False)
        status_var.set(f"Results exported to {Path(save_path).name}")

# ──────────────── GUI Variable ──────────────────────────────────────
global orig_label, proc_label, status_var, progress_var, process_btn, stop_btn
global export_btn, result_tree, canvas, ax, erode_var, padding_var

# ──────────────── GUI layout ────────────────────────────────────────
main_frame = ttk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

control_frame = ttk.LabelFrame(main_frame, text="Controls")
control_frame.pack(fill=tk.X, padx=5, pady=5)

params_frame = ttk.Frame(control_frame)
params_frame.grid(row=2, column=0, columnspan=3, sticky=tk.EW, padx=5, pady=5)

ttk.Label(params_frame, text="Erode Pixels:").grid(row=0, column=0, padx=5, pady=5)
erode_var = tk.IntVar(value=erode_pixels)
ttk.Scale(params_frame, from_=0, to=50, variable=erode_var, command=lambda v: erode_label.config(text=f"{int(float(v))} px"),).grid(row=0, column=1, padx=5, pady=5)
erode_label = ttk.Label(params_frame, text=f"{erode_pixels} px")
erode_label.grid(row=0, column=2, padx=5, pady=5)

ttk.Label(params_frame, text="Padding Ratio:").grid(row=0, column=3, padx=5, pady=5)
padding_var = tk.DoubleVar(value=padding_ratio)
ttk.Scale(params_frame, from_=0, to=0.2, variable=padding_var, command=lambda v: padding_label.config(text=f"{float(v):.2f}"),).grid(row=0, column=4, padx=5, pady=5)
padding_label = ttk.Label(params_frame, text=f"{padding_ratio:.2f}")
padding_label.grid(row=0, column=5, padx=5, pady=5)

btn_frame = ttk.Frame(control_frame)
btn_frame.grid(row=3, column=0, columnspan=3, pady=10)

process_btn = ttk.Button(btn_frame, text="Start Processing", command=start_processing)
process_btn.pack(side=tk.LEFT, padx=5)

stop_btn = ttk.Button(btn_frame, text="Stop Processing", command=stop_processing, state=tk.DISABLED,)
stop_btn.pack(side=tk.LEFT, padx=5)

export_btn = ttk.Button(btn_frame, text="Export Results", command=export_results, state=tk.DISABLED)
export_btn.pack(side=tk.LEFT, padx=5)

status_frame = ttk.LabelFrame(main_frame,  text="Status")
status_frame.pack(fill=tk.X, padx=5, pady=5)
status_var = tk.StringVar(value="Ready to process images")
ttk.Label(status_frame, textvariable=status_var).pack(side=tk.TOP, padx=5, pady=5, anchor='nw')
    
progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(status_frame, variable=progress_var, mode="determinate")
progress_bar.pack(side=tk.BOTTOM, fill=tk.X, expand=True, padx=5, pady=5)

result_frame = ttk.LabelFrame(main_frame, text="HSV Results")
result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
columns = ("Sample", "H Avg", "H Min", "H Max", "S Avg", "S Min", "S Max", "V Avg", "V Min", "V Max",)
    
result_tree = ttk.Treeview(result_frame, columns=columns, show="headings")
for col in columns:
    result_tree.heading(col, text=col)
    result_tree.column(col, width=80, anchor=tk.CENTER)

result_tree.column("Sample", width=60)
scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=result_tree.yview)
result_tree.configure(yscroll=scrollbar.set)
result_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

vis_frame = ttk.LabelFrame(main_frame, text="HSV Visualization")
vis_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
fig, ax = plt.subplots(figsize=(10, 4))
canvas = FigureCanvasTkAgg(fig, master=vis_frame)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Create output folders after GUI starts
output_folder.mkdir(exist_ok=True, parents=True)
(output_folder / "samples").mkdir(exist_ok=True, parents=True)
(output_folder / "combined").mkdir(exist_ok=True, parents=True)

# ──────────────── GUI window ────────────────────────────────────────
def on_closing():
    stop_processing = True
    root.destroy()
    plt.close("all")  # Clean up matplotlib
root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()