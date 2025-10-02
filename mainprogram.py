# Version 251002.1
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
from rembg import remove

# ──────────────── Tk root window ─────────────────────────────────────
root = tk.Tk()
root.state('zoomed')
root.title("IC-Project  ·  Color Gradient Detection - Camera Mode")
root.geometry("1400x900")

# ──────────────── Global configuration ───────────────────────────────
SAMPLE_NUM      = 16    # Total number of samples

# For the top row of 8 samples: 
# First list of tuples corresponds to the top-left-hand coordinates
# Second list of tuples corresponds to the bottom-right-hand coordinates
TOP_ROW_COORD   = [
    [(345,35), (495,35), (665,35), (845,35), (1005,35), (1175,35), (1335,35), (1495,35)],
    [(445,150), (595,150), (765,150), (945,150), (1105,150), (1275,150), (1435,150), (1595,150)]
]

# For the bottom row of 8 samples: 
# First list of tuples corresponds to the top-left-hand coordinates
# Second list of tuples corresponds to the bottom-right-hand coordinates
BOT_ROW_COORD   = [
    [(345,375), (495,375), (665,375), (845,375), (1005,375), (1175,375), (1335,375), (1495,375)],
    [(445,500), (595,500), (765,500), (945,500), (1105,500), (1275,500), (1435,500), (1595,500)]
]

COLOUR_THRES    = 105.8             # Default: 105.8
INFO_PREFIX     = "*INFO: "         # Shown in console
ERROR_PREFIX    = "*ERROR: "        # Shown in console

# ──────────────── Program variables ──────────────────────────────────
erode_pixels        = 20                                # Tunable in GUI
padding_ratio       = 0.05                              # Tunable in GUI
frame_count         = 0
processing          = False
stop_processing     = False
camera              = None
results             = []

# ──────────────── Threading Functions ───────────────────────────────
def start_processing():
    """Start the processing schedule and its thread upon clicking start processing button"""
    global processing, stop_processing, camera
    
    # Initialize camera
    try:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            messagebox.showerror("Camera Error", "Cannot access camera")
            return
    except Exception as e:
        messagebox.showerror("Camera Error", f"Failed to initialize camera: {str(e)}")
        return
        
    if processing:
        return
    processing = True
    stop_processing = False

    status_var.set(f"Starting the process... Please wait")
    
    # Button UI
    process_btn.config(state=tk.DISABLED)
    stop_btn.config(state=tk.NORMAL)
    export_btn.config(state=tk.DISABLED)

    # Threading
    processing_thread = threading.Thread(target=main_process)
    processing_thread.daemon = True
    processing_thread.start()

def stop_processing_func():
    """Stop the processing schedule upon clicking stop processing button"""
    global stop_processing, camera
    stop_processing = True
    if camera:
        camera.release()
    status_var.set(f"Stopping the process...")

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
            
        # Display camera feed with bounding boxes
        display_frame = frame.copy()
        
        # Draw bounding boxes for all 16 samples
        for i in range(8):  # Top row samples
            cv2.rectangle(display_frame, 
                         TOP_ROW_COORD[0][i], TOP_ROW_COORD[1][i], 
                         (0, 255, 0), 2)
            cv2.putText(display_frame, f"T{i+1}", 
                       (TOP_ROW_COORD[0][i][0] + 5, TOP_ROW_COORD[0][i][1] + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        for i in range(8):  # Bottom row samples
            cv2.rectangle(display_frame, 
                         BOT_ROW_COORD[0][i], BOT_ROW_COORD[1][i], 
                         (0, 255, 0), 2)
            cv2.putText(display_frame, f"B{i+1}", 
                       (BOT_ROW_COORD[0][i][0] + 5, BOT_ROW_COORD[0][i][1] + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Update camera display
        update_camera_display(display_frame)
        
        # Process frame every 10 seconds
        current_time = time.time()
        if current_time - last_process_time >= 5:
            status_var.set(f"Processing: Frame {frame_count + 1}")
            # Get current parameter values from GUI
            current_erode = erode_var.get()
            current_padding = padding_var.get()
            result = process_camera_frame(frame, current_erode, current_padding)
            results.append(result)
            update_results_table(result)
            update_hsv_visualization(result)
            last_process_time = current_time
            frame_count += 1
            
        # Small delay to prevent high CPU usage
        time.sleep(0.03)
        
    # Cleanup
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

def process_camera_frame(frame, erode_iterations, padding_ratio):
    """Process a single camera frame with current parameters"""
    samples = []
    
    # Process top row samples (8 samples)
    for i in range(8):
        x1, y1 = TOP_ROW_COORD[0][i]
        x2, y2 = TOP_ROW_COORD[1][i]
        sample_frame = frame[y1:y2, x1:x2]
        
        # Convert to PIL Image for processing
        sample_pil = Image.fromarray(cv2.cvtColor(sample_frame, cv2.COLOR_BGR2RGB))
        result_image = remove_background_and_trim(sample_pil, erode_iterations, padding_ratio)
        
        # Save the sample
        timestamp = int(time.time())
        sample_path = output_folder / f"camera_sample_{timestamp}_top_{i+1}.png"
        Image.fromarray(result_image).save(sample_path)

        # Calculate HSV values
        stats = calculate_hsv_stats(result_image, cv2)
        stats["image_path"] = str(sample_path)
        samples.append(stats)
    
    # Process bottom row samples (8 samples)
    for i in range(8):
        x1, y1 = BOT_ROW_COORD[0][i]
        x2, y2 = BOT_ROW_COORD[1][i]
        sample_frame = frame[y1:y2, x1:x2]
        
        # Convert to PIL Image for processing
        sample_pil = Image.fromarray(cv2.cvtColor(sample_frame, cv2.COLOR_BGR2RGB))
        result_image = remove_background_and_trim(sample_pil, erode_iterations, padding_ratio)
        
        # Save the sample
        timestamp = int(time.time())
        sample_path = output_folder / f"camera_sample_{timestamp}_bottom_{i+1}.png"
        Image.fromarray(result_image).save(sample_path)

        # Calculate HSV values
        stats = calculate_hsv_stats(result_image, cv2)
        stats["image_path"] = str(sample_path)
        samples.append(stats)
    
    return {
        "original_path": f"camera_frame_{int(time.time())}",
        "samples": samples,
    }

def update_camera_display(frame):
    """Update camera display in the GUI"""
    # Resize frame to fit display
    display_height = 400
    h, w = frame.shape[:2]
    aspect_ratio = w / h
    display_width = int(display_height * aspect_ratio)
    
    resized_frame = cv2.resize(frame, (display_width, display_height))
    
    # Convert to PhotoImage
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb_frame)
    imgtk = ImageTk.PhotoImage(image=img)
    
    # Update camera label
    camera_label.imgtk = imgtk
    camera_label.configure(image=imgtk)

def remove_background_and_trim(image, erode_iterations, padding_ratio):
    """Parent function of removing background and trim the edge of ONE sample"""
    # Background removal
    img_rgba = remove(image)
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
    pad_y, pad_x = int(h * padding_ratio), int(w * padding_ratio)
    y0 = max(0, y0 - pad_y)
    y1 = min(rgba_arr.shape[0], y1 + pad_y)
    x0 = max(0, x0 - pad_x)
    x1 = min(rgba_arr.shape[1], x1 + pad_x)
    cropped = rgba_arr[y0:y1, x0:x1].copy()

    # Erode mask to remove outer edges
    alpha_c = cropped[:, :, 3]
    inner_mask = erode_mask(alpha_c, erode_iterations)
    cropped[~inner_mask, 3] = 0         # Set trimed edges to transparent

    # Recrop to remove transparent borders
    alpha_new = cropped[:, :, 3]
    coords2 = np.argwhere(alpha_new > 0)
    if coords2.size == 0:
        return cropped

    y0b, x0b = coords2.min(axis=0)
    y1b, x1b = coords2.max(axis=0) + 1
    return cropped[y0b:y1b, x0b:x1b]

def erode_mask(alpha, erode_iterations):
    """Applying erode mask after removing the image's background"""
    mask = (alpha > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(mask, kernel, iterations=erode_iterations)
    return eroded > 0

# ──────────────── HSV Analysis ──────────────────────────────────────
def color_decision(value):
    """Decision making based on average HSV"""
    return "RED" if value < COLOUR_THRES else "PURPLE"

def calculate_hsv_stats(rgba_arr, cv2):
    """Calculating HSV values of individual pixels of a processed image"""
    mask = rgba_arr[:, :, 3] > 0        # Create mask of all non-transparent pixels
    
    # Converting to HSV spec
    bgr = cv2.cvtColor(rgba_arr, cv2.COLOR_RGBA2BGR)
    hsv_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)

    if np.any(mask):
        return {
            "result": color_decision(h[mask].mean()),
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
            "result": "NONE",
            "h_avg": 0, "h_min": 0, "h_max": 0,
            "s_avg": 0, "s_min": 0, "s_max": 0,
            "v_avg": 0, "v_min": 0, "v_max": 0
        }

def update_results_table(result):
    """Updating the GUI result table"""
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
    """Updating the GUI table visualization with plt"""
    global frame_count
    ax.clear()

    # Prepare data
    sample_nums = range(1, SAMPLE_NUM + 1)
    h_avgs = [s["h_avg"] for s in result["samples"]]
    s_avgs = [s["s_avg"] for s in result["samples"]]
    v_avgs = [s["v_avg"] for s in result["samples"]]

    # Plot HSV averages
    ax.plot(sample_nums, h_avgs, "o-", label="Hue", color="#3BCF00")
    ax.plot(sample_nums, s_avgs, "o:", label="Saturation", color="#A1A1A1")
    ax.plot(sample_nums, v_avgs, "o:", label="Value", color="#949494")

    # Format plot
    ax.set_title(f"HSV Values of {SAMPLE_NUM} Samples (Current: Frame {frame_count})")
    ax.set_xlabel("Sample Number")
    ax.set_ylabel("Value")
    ax.set_xticks(range(1, SAMPLE_NUM + 1))
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend()

    canvas.draw()

def export_results():
    """With Excel format (.xlsx) exporting function"""
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
global export_btn, result_tree, camera_label, erode_var, padding_var

# ──────────────── GUI layout ────────────────────────────────────────
main_frame = ttk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Create top pane for the left pane
top_frame = ttk.Frame(main_frame)
top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=False)

# Create left and right panes
left_frame = ttk.Frame(top_frame)
left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5, pady=5)

left_top_frame = ttk.Frame(left_frame)
left_top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=False)
left_bot_frame = ttk.Frame(left_frame)
left_bot_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=False)

right_frame = ttk.Frame(top_frame)
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Crrate bottom pane
bottom_frame = ttk.Frame(main_frame)
bottom_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=False)

# ──────────────── LEFT FRAME: Controls and Results ──────────────────
# Controls section
control_frame = ttk.LabelFrame(left_top_frame, text="Controls")
control_frame.pack(fill=tk.X, padx=5, pady=(0, 10))

params_frame = ttk.Frame(control_frame)
params_frame.pack(fill=tk.X, padx=5, pady=5)

# Erode parameter
erode_frame = ttk.Frame(params_frame)
erode_frame.pack(fill=tk.X, pady=2)
ttk.Label(erode_frame, text="Erode Pixels:").pack(side=tk.LEFT, padx=5)
erode_var = tk.IntVar(value=erode_pixels)
ttk.Scale(erode_frame, from_=0, to=50, variable=erode_var, 
          command=lambda v: erode_label.config(text=f"{int(float(v))} px"),
          orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
erode_label = ttk.Label(erode_frame, text=f"{erode_pixels} px")
erode_label.pack(side=tk.LEFT, padx=5)

# Padding parameter
padding_frame = ttk.Frame(params_frame)
padding_frame.pack(fill=tk.X, pady=2)
ttk.Label(padding_frame, text="Padding Ratio:").pack(side=tk.LEFT, padx=5)
padding_var = tk.DoubleVar(value=padding_ratio)
ttk.Scale(padding_frame, from_=0, to=0.2, variable=padding_var, 
          command=lambda v: padding_label.config(text=f"{float(v):.2f}"),
          orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
padding_label = ttk.Label(padding_frame, text=f"{padding_ratio:.2f}")
padding_label.pack(side=tk.LEFT, padx=5)

# Buttons and status
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

# HSV Visualization
vis_frame = ttk.LabelFrame(left_bot_frame, text="HSV Visualization Chart")
vis_frame.pack(fill=tk.BOTH, expand=False, padx=5, pady=5)
fig, ax = plt.subplots(figsize=(7, 4))
canvas = FigureCanvasTkAgg(fig, master=vis_frame)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=False)

# Results table section
result_frame = ttk.LabelFrame(bottom_frame, text="HSV Results Table")
result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

# Create treeview with scrollbar
tree_frame = ttk.Frame(result_frame)
tree_frame.pack(fill=tk.BOTH, expand=True)

columns = ("Sample", "Result", "H_avg", "H_min", "H_max", "S_avg", "S_min", "S_max", "V_avg", "V_min", "V_max")
result_tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=16)

# Configure columns
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

# Scrollbar for treeview
tree_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=result_tree.yview)
result_tree.configure(yscroll=tree_scrollbar.set)

result_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# ──────────────── RIGHT FRAME: Camera Feed ──────────────────────────
camera_frame = ttk.LabelFrame(right_frame, text="Camera Output")
camera_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

camera_label = ttk.Label(camera_frame, background="black")
camera_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

# Create output folders after GUI starts
output_folder = Path("output")
output_folder.mkdir(exist_ok=True, parents=True)

# ──────────────── GUI window ────────────────────────────────────────
def on_closing():
    global stop_processing, camera
    stop_processing = True
    print("Process closing")
    if camera:
        camera.release()
    root.destroy()
    cv2.destroyAllWindows()
    print("Process closed")
    
root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()