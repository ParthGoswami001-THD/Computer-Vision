#-------------------------------------------------------------------------------
# Name:        app.py
# Purpose:     Graphical main application for perspective transformation assignment with 8 Degrees of freedom (8DOF)
# Features:
# - Reads an input image and applies perspective transformations based on predefined test cases.
# - Saves the resulting images and computes mean and maximum reprojection errors.
# - Graphically displays results in side-by-side comparison windows.
# - Gives Mathematical insights into homography computation and error analysis.
# - Provides animated visualizations of the transformation process.
# Author:      Parth Goswami
# Created:     2026-04-25
#-------------------------------------------------------------------------------


import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2 as cv
import numpy as np
from PIL import Image, ImageTk



from homography import compute_homography_8dof, project_points, warp_image_numpy
from visualization_tools import draw_points, side_by_side_result


def make_fallback_image(width=640, height=460):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    block = 40

    for y in range(0, height, block):
        for x in range(0, width, block):
            value = 220 if ((x // block + y // block) % 2 == 0) else 90
            image[y:y + block, x:x + block] = (value, value, value)

    cv.putText(
        image,
        "Fallback checkerboard",
        (130, height // 2),
        cv.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2,
        cv.LINE_AA,
    )
    return image


def make_point(width, height, x_ratio, y_ratio):
    return [float(width * x_ratio), float(height * y_ratio)]


def build_demo_cases(width, height):
    # Same point order as the sample GUI code:
    # P1 = top-left, P2 = top-right, P3 = bottom-left, P4 = bottom-right.
    src_points = [
        make_point(width, height, 0.20, 0.20),
        make_point(width, height, 0.80, 0.20),
        make_point(width, height, 0.20, 0.80),
        make_point(width, height, 0.80, 0.80),
    ]

    case_list = [
        {
            "name": "case01_scaling",
            "src_points": src_points,
            "dst_points": [
                make_point(width, height, 0.15, 0.15),
                make_point(width, height, 0.85, 0.15),
                make_point(width, height, 0.15, 0.85),
                make_point(width, height, 0.85, 0.85),
            ],
        },
        {
            "name": "case02_rotation_like",
            "src_points": src_points,
            "dst_points": [
                make_point(width, height, 0.28, 0.10),
                make_point(width, height, 0.88, 0.28),
                make_point(width, height, 0.12, 0.72),
                make_point(width, height, 0.72, 0.90),
            ],
        },
        {
            "name": "case03_trapezoid",
            "src_points": src_points,
            "dst_points": [
                make_point(width, height, 0.35, 0.18),
                make_point(width, height, 0.65, 0.18),
                make_point(width, height, 0.10, 0.85),
                make_point(width, height, 0.90, 0.85),
            ],
        },
        {
            "name": "case04_shear_like",
            "src_points": src_points,
            "dst_points": [
                make_point(width, height, 0.22, 0.15),
                make_point(width, height, 0.78, 0.22),
                make_point(width, height, 0.32, 0.75),
                make_point(width, height, 0.88, 0.82),
            ],
        },
        {
            "name": "case05_strong_perspective",
            "src_points": src_points,
            "dst_points": [
                make_point(width, height, 0.42, 0.10),
                make_point(width, height, 0.62, 0.10),
                make_point(width, height, 0.08, 0.90),
                make_point(width, height, 0.92, 0.90),
            ],
        },
        {
            "name": "case06_extreme_skew",
            "src_points": src_points,
            "dst_points": [
                make_point(width, height, 0.40, 0.12),
                make_point(width, height, 0.58, 0.08),
                make_point(width, height, 0.06, 0.80),
                make_point(width, height, 0.96, 0.88),
            ],
        },
    ]

    return case_list


def image_to_tk(image_bgr, max_width, max_height):
    image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    pil_image.thumbnail((max_width, max_height), get_resize_filter())
    return ImageTk.PhotoImage(pil_image)


def get_resize_filter():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.LANCZOS
    return Image.LANCZOS


def format_matrix(matrix):
    lines = []
    for row in matrix:
        line = "  ".join(f"{value:12.6f}" for value in row)
        lines.append(line)
    return "\n".join(lines)


def format_points(points):
    lines = []
    for idx, point in enumerate(points):
        lines.append(f"P{idx + 1}: ({point[0]:.1f}, {point[1]:.1f})")
    return "\n".join(lines)


def order_points_tl_tr_bl_br(points):
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (4, 2):
        return points

    # Natural document/card order: top-left, top-right, bottom-left, bottom-right.
    points_by_y = points[np.argsort(points[:, 1])]
    top_points = points_by_y[:2]
    bottom_points = points_by_y[2:]

    top_points = top_points[np.argsort(top_points[:, 0])]
    bottom_points = bottom_points[np.argsort(bottom_points[:, 0])]

    return np.array(
        [
            top_points[0],
            top_points[1],
            bottom_points[0],
            bottom_points[1],
        ],
        dtype=np.float64,
    )


class HomographyDemoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Group 8 Perspective Transformation Demo")
        self.root.geometry("1320x860")
        self.root.minsize(1120, 720)

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.results_dir = os.path.join(self.base_dir, "results")
        os.makedirs(self.results_dir, exist_ok=True)

        self.source_image = self.load_default_image()
        self.source_path = "default image"
        self.current_result = None
        self.display_image = None
        self.display_photo = None
        self.gallery_image = None
        self.gallery_photo = None
        self.gallery_files = []
        self.animation_frames = []
        self.animation_after_id = None
        self.is_busy = False

        self.configure_style()
        self.build_ui()
        self.reset_points_for_current_image()
        self.show_start_image()
        self.refresh_gallery()

    def load_default_image(self):
        image_path = os.path.join(self.base_dir, "image", "checkerboard_final.jpg")
        image = cv.imread(image_path, cv.IMREAD_COLOR)
        if image is None:
            return make_fallback_image()
        return image

    def configure_style(self):
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")

        style.configure("App.TFrame", background="#eef1f5")
        style.configure("Card.TFrame", background="#ffffff")
        style.configure(
            "Title.TLabel",
            background="#eef1f5",
            foreground="#162033",
            font=("Segoe UI", 18, "bold"),
        )
        style.configure(
            "Section.TLabel",
            background="#ffffff",
            foreground="#162033",
            font=("Segoe UI", 11, "bold"),
        )
        style.configure(
            "Body.TLabel",
            background="#ffffff",
            foreground="#334155",
            font=("Segoe UI", 10),
        )
        style.configure("App.TNotebook.Tab", padding=(16, 8), font=("Segoe UI", 10, "bold"))

    def build_ui(self):
        outer = ttk.Frame(self.root, style="App.TFrame")
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer, style="App.TFrame")
        header.pack(fill="x", padx=16, pady=(12, 6))

        ttk.Label(
            header,
            text="Group 8 Homography Demo App",
            style="Title.TLabel",
        ).pack(anchor="w")

        notebook = ttk.Notebook(outer)
        notebook.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        self.demo_tab = ttk.Frame(notebook, style="App.TFrame")
        self.math_tab = ttk.Frame(notebook, style="App.TFrame")
        self.gallery_tab = ttk.Frame(notebook, style="App.TFrame")

        notebook.add(self.demo_tab, text="Upload and Transform")
        notebook.add(self.math_tab, text="Math Calculation")
        notebook.add(self.gallery_tab, text="Saved Results")

        self.build_demo_tab()
        self.build_math_tab()
        self.build_gallery_tab()

    def build_demo_tab(self):
        frame = ttk.Frame(self.demo_tab, style="App.TFrame")
        frame.pack(fill="both", expand=True, padx=14, pady=14)

        paned = ttk.Panedwindow(frame, orient="horizontal")
        paned.pack(fill="both", expand=True)

        controls = ttk.Frame(paned, style="Card.TFrame", width=360)
        controls.pack_propagate(False)
        preview = ttk.Frame(paned, style="Card.TFrame")
        paned.add(controls, weight=0)
        paned.add(preview, weight=1)

        ttk.Label(controls, text="Controls", style="Section.TLabel").pack(anchor="w", padx=12, pady=(12, 8))

        ttk.Button(controls, text="Upload Image", command=self.upload_image).pack(fill="x", padx=12, pady=3)
        ttk.Button(controls, text="Pick Source Points", command=self.open_source_picker).pack(fill="x", padx=12, pady=3)

        self.run_button = ttk.Button(controls, text="Run Transformation", command=self.run_transformation)
        self.run_button.pack(fill="x", padx=12, pady=3)

        self.animate_button = ttk.Button(controls, text="Animate Transformation", command=self.start_animation)
        self.animate_button.pack(fill="x", padx=12, pady=3)

        ttk.Button(controls, text="Save Current Result", command=self.save_current_result).pack(fill="x", padx=12, pady=3)

        ttk.Separator(controls).pack(fill="x", padx=12, pady=10)

        ttk.Label(controls, text="Preset Case", style="Body.TLabel").pack(anchor="w", padx=12)
        self.preset_var = tk.StringVar()
        self.preset_combo = ttk.Combobox(controls, textvariable=self.preset_var, state="readonly")
        self.preset_combo.pack(fill="x", padx=12, pady=3)
        ttk.Button(controls, text="Apply Preset", command=self.apply_selected_preset).pack(fill="x", padx=12, pady=3)

        self.relative_dst_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            controls,
            text="Move destination with picked source points",
            variable=self.relative_dst_var,
        ).pack(anchor="w", padx=12, pady=(4, 2))
        ttk.Label(
            controls,
            text="Points are auto-sorted as TL, TR, BL, BR before calculation.",
            style="Body.TLabel",
            wraplength=315,
        ).pack(anchor="w", padx=12, pady=(0, 2))

        ttk.Separator(controls).pack(fill="x", padx=12, pady=10)

        self.src_vars = self.create_point_editor(controls, "Source Points S1..S4")
        self.dst_vars = self.create_point_editor(controls, "Destination Points D1..D4")

        ttk.Separator(controls).pack(fill="x", padx=12, pady=10)

        ttk.Label(controls, text="Status", style="Body.TLabel").pack(anchor="w", padx=12)
        self.status_text = tk.Text(
            controls,
            height=7,
            wrap="word",
            font=("Consolas", 10),
            bg="#f8fafc",
            relief="solid",
            bd=1,
        )
        self.status_text.pack(fill="x", padx=12, pady=(3, 12))
        self.set_status("Upload an image or use the default checkerboard.")

        ttk.Label(preview, text="Result Preview", style="Section.TLabel").pack(anchor="w", padx=12, pady=(12, 4))
        holder = ttk.Frame(preview, style="Card.TFrame")
        holder.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        self.preview_label = tk.Label(holder, bg="#111827")
        self.preview_label.pack(fill="both", expand=True)
        self.preview_label.bind("<Configure>", lambda _event: self.render_preview_image())

    def build_math_tab(self):
        frame = ttk.Frame(self.math_tab, style="Card.TFrame")
        frame.pack(fill="both", expand=True, padx=16, pady=16)

        ttk.Label(frame, text="Homography Calculation", style="Section.TLabel").pack(anchor="w", padx=12, pady=(12, 6))
        self.math_text = tk.Text(
            frame,
            wrap="none",
            font=("Consolas", 11),
            bg="#ffffff",
            relief="solid",
            bd=1,
        )
        y_scroll = ttk.Scrollbar(frame, orient="vertical", command=self.math_text.yview)
        x_scroll = ttk.Scrollbar(frame, orient="horizontal", command=self.math_text.xview)
        self.math_text.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        self.math_text.pack(side="left", fill="both", expand=True, padx=(12, 0), pady=(0, 12))
        y_scroll.pack(side="right", fill="y", pady=(0, 12))
        x_scroll.pack(side="bottom", fill="x", padx=12)

        self.write_math_intro()

    def build_gallery_tab(self):
        frame = ttk.Frame(self.gallery_tab, style="App.TFrame")
        frame.pack(fill="both", expand=True, padx=14, pady=14)

        paned = ttk.Panedwindow(frame, orient="horizontal")
        paned.pack(fill="both", expand=True)

        left = ttk.Frame(paned, style="Card.TFrame", width=330)
        left.pack_propagate(False)
        right = ttk.Frame(paned, style="Card.TFrame")
        paned.add(left, weight=0)
        paned.add(right, weight=1)

        ttk.Label(left, text="Saved Images", style="Section.TLabel").pack(anchor="w", padx=12, pady=(12, 8))

        list_frame = ttk.Frame(left, style="Card.TFrame")
        list_frame.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        self.gallery_list = tk.Listbox(list_frame, bg="#f8fafc", relief="solid", bd=1)
        list_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.gallery_list.yview)
        self.gallery_list.configure(yscrollcommand=list_scroll.set)
        self.gallery_list.pack(side="left", fill="both", expand=True)
        list_scroll.pack(side="right", fill="y")
        self.gallery_list.bind("<<ListboxSelect>>", self.on_gallery_select)

        ttk.Button(left, text="Refresh Gallery", command=self.refresh_gallery).pack(fill="x", padx=12, pady=(0, 12))

        self.gallery_info_var = tk.StringVar(value="Select a saved result.")
        ttk.Label(right, textvariable=self.gallery_info_var, style="Body.TLabel").pack(anchor="w", padx=12, pady=(12, 4))

        holder = ttk.Frame(right, style="Card.TFrame")
        holder.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.gallery_label = tk.Label(holder, bg="#111827")
        self.gallery_label.pack(fill="both", expand=True)
        self.gallery_label.bind("<Configure>", lambda _event: self.render_gallery_image())

    def create_point_editor(self, parent, title):
        box = ttk.LabelFrame(parent, text=title)
        box.pack(fill="x", padx=12, pady=4)

        point_labels = ["TL", "TR", "BL", "BR"]
        point_vars = []
        for idx in range(4):
            row = ttk.Frame(box)
            row.pack(fill="x", padx=6, pady=2)

            ttk.Label(row, text=f"P{idx + 1} {point_labels[idx]}:", width=7).pack(side="left")
            var = tk.StringVar(value="0, 0")
            ttk.Entry(row, textvariable=var, width=22).pack(side="left", fill="x", expand=True)
            point_vars.append(var)

        return point_vars

    def reset_points_for_current_image(self):
        height, width = self.source_image.shape[:2]
        cases = build_demo_cases(width, height)
        self.cases = cases

        names = [case["name"] for case in cases]
        self.preset_combo.configure(values=names)
        self.preset_var.set(names[0])

        src_points = np.asarray(cases[0]["src_points"], dtype=np.float64)
        dst_points = np.asarray(cases[0]["dst_points"], dtype=np.float64)
        self.set_point_vars(self.src_vars, src_points)
        self.set_point_vars(self.dst_vars, dst_points)

    def set_point_vars(self, point_vars, points):
        for idx, point in enumerate(points):
            point_vars[idx].set(f"{point[0]:.1f}, {point[1]:.1f}")

    def parse_points(self, point_vars):
        points = []
        for var in point_vars:
            text = var.get().strip()
            pieces = text.split(",")
            if len(pieces) != 2:
                raise ValueError(f"Invalid point format: {text}. Use x, y")

            x = float(pieces[0].strip())
            y = float(pieces[1].strip())
            points.append([x, y])

        return np.asarray(points, dtype=np.float64)

    def apply_selected_preset(self):
        selected_name = self.preset_var.get()
        for case in self.cases:
            if case["name"] == selected_name:
                self.set_point_vars(self.src_vars, np.asarray(case["src_points"], dtype=np.float64))
                self.set_point_vars(self.dst_vars, np.asarray(case["dst_points"], dtype=np.float64))
                self.set_status(f"Applied preset: {selected_name}")
                return

    def upload_image(self):
        path = filedialog.askopenfilename(
            title="Select source image",
            initialdir=self.base_dir,
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        image = cv.imread(path, cv.IMREAD_COLOR)
        if image is None:
            messagebox.showerror("Image error", f"Could not read image:\n{path}")
            return

        self.stop_animation()
        self.source_image = image
        self.source_path = path
        self.current_result = None
        self.reset_points_for_current_image()
        self.show_start_image()
        self.write_math_intro()
        self.set_status(f"Loaded: {os.path.basename(path)}\nSize: {image.shape[1]} x {image.shape[0]}")

    def show_start_image(self):
        src_points = self.parse_points(self.src_vars)
        preview = draw_points(self.source_image, src_points, "S")
        cv.putText(
            preview,
            "Upload image, adjust points, then run transformation",
            (18, 35),
            cv.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
            cv.LINE_AA,
        )
        self.set_preview_image(preview)

    def run_transformation(self):
        if self.is_busy:
            return

        try:
            src_points = order_points_tl_tr_bl_br(self.parse_points(self.src_vars))
            dst_points = order_points_tl_tr_bl_br(self.parse_points(self.dst_vars))
            self.set_point_vars(self.src_vars, src_points)
            self.set_point_vars(self.dst_vars, dst_points)
        except Exception as exc:
            messagebox.showerror("Point error", str(exc))
            return

        self.set_busy("Running NumPy homography transform...")
        self.stop_animation()

        worker = threading.Thread(
            target=self.run_transformation_worker,
            args=(src_points, dst_points),
            daemon=True,
        )
        worker.start()

    def run_transformation_worker(self, src_points, dst_points):
        try:
            H, A = compute_homography_8dof(src_points, dst_points)
            output_size = (self.source_image.shape[0], self.source_image.shape[1])
            warped = warp_image_numpy(self.source_image, H, output_size)

            projected = project_points(H, src_points)
            errors = np.linalg.norm(projected - dst_points, axis=1)
            cond_number = float(np.linalg.cond(A))

            info = (
                f"interactive demo | cond(A)={cond_number:.2e} | "
                f"mean={np.mean(errors):.4f}px | max={np.max(errors):.4f}px"
            )
            result_image = side_by_side_result(
                self.source_image,
                warped,
                src_points,
                dst_points,
                info,
            )

            math_report = self.create_math_report(H, A, src_points, dst_points, projected, errors, cond_number)
            status = (
                f"Condition number: {cond_number:.3e}\n"
                f"Mean reprojection error: {np.mean(errors):.6f} px\n"
                f"Max reprojection error: {np.max(errors):.6f} px"
            )

            self.root.after(0, lambda: self.finish_transformation(result_image, math_report, status))
        except Exception as exc:
            self.root.after(0, lambda: self.show_error(str(exc)))

    def finish_transformation(self, result_image, math_report, status):
        self.current_result = result_image
        self.set_preview_image(result_image)
        self.write_math_text(math_report)
        self.set_status(status)
        self.clear_busy()

    def create_math_report(self, H, A, src_points, dst_points, projected, errors, cond_number):
        lines = []
        lines.append("Group 8 Homography Calculation")
        lines.append("=" * 72)
        lines.append("")
        lines.append("Source points:")
        lines.append(format_points(src_points))
        lines.append("")
        lines.append("Destination points:")
        lines.append(format_points(dst_points))
        lines.append("")
        lines.append("Equation form for each pair:")
        lines.append("h11*x + h12*y + h13 - h31*u*x - h32*u*y = u")
        lines.append("h21*x + h22*y + h23 - h31*v*x - h32*v*y = v")
        lines.append("")
        lines.append("A matrix from normalized DLT in homography_tools.py:")
        lines.append(format_matrix(A))
        lines.append("")
        lines.append("Calculated H matrix:")
        lines.append(format_matrix(H))
        lines.append("")
        lines.append(f"Condition number of A: {cond_number:.6e}")
        lines.append("")
        lines.append("Reprojection check:")
        for idx in range(4):
            lines.append(
                f"P{idx + 1}: projected=({projected[idx, 0]:.6f}, {projected[idx, 1]:.6f}) "
                f"target=({dst_points[idx, 0]:.6f}, {dst_points[idx, 1]:.6f}) "
                f"error={errors[idx]:.8f} px"
            )

        lines.append("")
        lines.append(f"Mean error: {np.mean(errors):.8f} px")
        lines.append(f"Max error:  {np.max(errors):.8f} px")
        lines.append("")
        lines.append("Warping method:")
        lines.append("For each destination pixel, use inverse H to find the source coordinate.")
        lines.append("Then use bilinear interpolation with NumPy arrays.")
        lines.append("No OpenCV homography or warp helper is used.")
        return "\n".join(lines)

    def start_animation(self):
        if self.is_busy:
            return

        try:
            src_points = order_points_tl_tr_bl_br(self.parse_points(self.src_vars))
            dst_points = order_points_tl_tr_bl_br(self.parse_points(self.dst_vars))
            self.set_point_vars(self.src_vars, src_points)
            self.set_point_vars(self.dst_vars, dst_points)
        except Exception as exc:
            messagebox.showerror("Point error", str(exc))
            return

        self.set_busy("Preparing animation frames...")
        self.stop_animation()

        worker = threading.Thread(
            target=self.animation_worker,
            args=(src_points, dst_points),
            daemon=True,
        )
        worker.start()

    def animation_worker(self, src_points, dst_points):
        try:
            preview_image, scale = self.make_animation_source()
            src_small = src_points * scale
            dst_small = dst_points * scale
            source_marked = draw_points(preview_image, src_small, "S")

            frames = []
            frame_count = 12
            for idx in range(frame_count):
                t = idx / (frame_count - 1)
                dst_t = src_small + t * (dst_small - src_small)
                H_t, _ = compute_homography_8dof(src_small, dst_t)
                warped_t = warp_image_numpy(preview_image, H_t, preview_image.shape[:2])
                target_marked = draw_points(warped_t, dst_t, "T")
                frame = self.combine_for_animation(source_marked, target_marked, t)
                frames.append(frame)

            self.root.after(0, lambda: self.animation_ready(frames))
        except Exception as exc:
            self.root.after(0, lambda: self.show_error(str(exc)))

    def make_animation_source(self):
        height, width = self.source_image.shape[:2]
        max_side = max(width, height)
        scale = min(1.0, 520.0 / max_side)

        if scale < 1.0:
            new_width = max(2, int(round(width * scale)))
            new_height = max(2, int(round(height * scale)))
            image = cv.resize(self.source_image, (new_width, new_height), interpolation=cv.INTER_AREA)
            return image, scale

        return self.source_image.copy(), 1.0

    def combine_for_animation(self, left, right, t):
        info = f"animation frame | t={t:.2f} | dst_t = src + t(dst-src)"
        empty_points = np.empty((0, 2), dtype=np.float64)
        return side_by_side_result(left, right, empty_points, empty_points, info)

    def animation_ready(self, frames):
        self.animation_frames = frames
        self.clear_busy()
        self.set_status("Animation ready. Showing intermediate homography frames.")
        self.play_animation_frame(0)

    def play_animation_frame(self, index):
        if not self.animation_frames:
            return

        if index >= len(self.animation_frames):
            self.set_status("Animation finished.")
            self.animation_after_id = None
            return

        self.set_preview_image(self.animation_frames[index])
        self.animation_after_id = self.root.after(130, lambda: self.play_animation_frame(index + 1))

    def stop_animation(self):
        if self.animation_after_id is not None:
            self.root.after_cancel(self.animation_after_id)
            self.animation_after_id = None

    def save_current_result(self):
        if self.current_result is None:
            messagebox.showinfo("No result", "Run the transformation first.")
            return

        path = filedialog.asksaveasfilename(
            title="Save current result",
            initialdir=self.results_dir,
            defaultextension=".png",
            filetypes=[
                ("PNG image", "*.png"),
                ("JPEG image", "*.jpg"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        ok = cv.imwrite(path, self.current_result)
        if not ok:
            messagebox.showerror("Save error", f"Could not save image:\n{path}")
            return

        self.set_status(f"Saved result:\n{path}")
        self.refresh_gallery()

    def open_source_picker(self):
        picker = tk.Toplevel(self.root)
        picker.title("Pick 4 Source Points")
        picker.geometry("980x740")
        picker.transient(self.root)
        picker.grab_set()

        ttk.Label(
            picker,
            text="Click four source points in this order: top-left, top-right, bottom-left, bottom-right",
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w", padx=10, pady=(10, 4))

        status_var = tk.StringVar(value="Clicked: 0 / 4")
        ttk.Label(picker, textvariable=status_var).pack(anchor="w", padx=10, pady=(0, 6))

        max_width = 940
        max_height = 610
        height, width = self.source_image.shape[:2]
        scale = min(max_width / width, max_height / height)
        display_width = int(width * scale)
        display_height = int(height * scale)

        image_rgb = cv.cvtColor(self.source_image, cv.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb).resize((display_width, display_height), get_resize_filter())
        picker_photo = ImageTk.PhotoImage(pil_image)

        canvas = tk.Canvas(
            picker,
            width=display_width,
            height=display_height,
            bg="black",
            highlightthickness=1,
            highlightbackground="#888888",
        )
        canvas.pack(anchor="center", padx=10, pady=8)
        canvas.create_image(0, 0, anchor="nw", image=picker_photo)
        canvas.image = picker_photo

        picked = []
        colors = ["#ff4040", "#25b85a", "#3478f6", "#d09a00"]

        def redraw_marks():
            canvas.delete("mark")
            for idx, point in enumerate(picked):
                x = point[0] * scale
                y = point[1] * scale
                canvas.create_oval(
                    x - 7,
                    y - 7,
                    x + 7,
                    y + 7,
                    fill=colors[idx],
                    outline="white",
                    width=2,
                    tags="mark",
                )
                canvas.create_text(
                    x + 16,
                    y - 12,
                    text=f"S{idx + 1}",
                    fill=colors[idx],
                    font=("Segoe UI", 10, "bold"),
                    tags="mark",
                )
            status_var.set(f"Clicked: {len(picked)} / 4")

        def on_click(event):
            if len(picked) >= 4:
                return
            x_image = event.x / scale
            y_image = event.y / scale
            if 0 <= x_image < width and 0 <= y_image < height:
                picked.append((x_image, y_image))
                redraw_marks()

        def reset_clicked_points():
            picked.clear()
            redraw_marks()

        def apply_clicked_points():
            if len(picked) != 4:
                messagebox.showwarning("Need four points", "Please click exactly four points.")
                return

            try:
                old_src = self.parse_points(self.src_vars)
                new_src = order_points_tl_tr_bl_br(np.asarray(picked, dtype=np.float64))
                self.set_point_vars(self.src_vars, new_src)

                if self.relative_dst_var.get():
                    self.move_destination_relative_to_source(old_src, new_src)

                self.show_start_image()
                self.set_status("Source points updated from mouse picker.")
                picker.destroy()
            except Exception as exc:
                messagebox.showerror("Point error", str(exc))

        canvas.bind("<Button-1>", on_click)

        button_row = ttk.Frame(picker)
        button_row.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Button(button_row, text="Reset", command=reset_clicked_points).pack(side="left")
        ttk.Button(button_row, text="Apply Points", command=apply_clicked_points).pack(side="right")

    def move_destination_relative_to_source(self, old_src, new_src):
        try:
            old_dst = self.parse_points(self.dst_vars)
            new_dst = new_src + (old_dst - old_src)
            height, width = self.source_image.shape[:2]
            new_dst[:, 0] = np.clip(new_dst[:, 0], 0, width - 1)
            new_dst[:, 1] = np.clip(new_dst[:, 1], 0, height - 1)
            self.set_point_vars(self.dst_vars, new_dst)
        except Exception:
            pass

    def refresh_gallery(self):
        self.gallery_list.delete(0, tk.END)
        files = []

        if os.path.isdir(self.results_dir):
            for name in os.listdir(self.results_dir):
                lower = name.lower()
                if lower.endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    files.append(os.path.join(self.results_dir, name))

        files.sort()
        self.gallery_files = files

        if not files:
            self.gallery_list.insert(tk.END, "No result images found.")
            self.gallery_info_var.set("Save a result to preview it here.")
            self.gallery_image = None
            self.gallery_label.configure(image="")
            return

        for path in files:
            self.gallery_list.insert(tk.END, os.path.basename(path))

        self.gallery_info_var.set("Select a result image.")

    def on_gallery_select(self, _event):
        selected = self.gallery_list.curselection()
        if not selected:
            return

        index = selected[0]
        if index >= len(self.gallery_files):
            return

        path = self.gallery_files[index]
        image = cv.imread(path, cv.IMREAD_COLOR)
        if image is None:
            return

        self.gallery_image = image
        self.gallery_info_var.set(f"{os.path.basename(path)} | {image.shape[1]} x {image.shape[0]}")
        self.render_gallery_image()

    def set_preview_image(self, image):
        self.display_image = image.copy()
        self.render_preview_image()

    def render_preview_image(self):
        if self.display_image is None:
            return

        max_width = max(self.preview_label.winfo_width() - 8, 320)
        max_height = max(self.preview_label.winfo_height() - 8, 240)
        self.display_photo = image_to_tk(self.display_image, max_width, max_height)
        self.preview_label.configure(image=self.display_photo)

    def render_gallery_image(self):
        if self.gallery_image is None:
            return

        max_width = max(self.gallery_label.winfo_width() - 8, 320)
        max_height = max(self.gallery_label.winfo_height() - 8, 240)
        self.gallery_photo = image_to_tk(self.gallery_image, max_width, max_height)
        self.gallery_label.configure(image=self.gallery_photo)

    def set_status(self, text):
        self.status_text.configure(state="normal")
        self.status_text.delete("1.0", tk.END)
        self.status_text.insert("1.0", text)
        self.status_text.configure(state="disabled")

    def set_busy(self, text):
        self.is_busy = True
        self.run_button.configure(state="disabled")
        self.animate_button.configure(state="disabled")
        self.root.configure(cursor="watch")
        self.set_status(text)

    def clear_busy(self):
        self.is_busy = False
        self.run_button.configure(state="normal")
        self.animate_button.configure(state="normal")
        self.root.configure(cursor="")

    def show_error(self, error_text):
        self.clear_busy()
        messagebox.showerror("Demo error", error_text)

    def write_math_intro(self):
        text = (
            "Run the transformation to show the calculation.\n\n"
            "The app calculates H from four source points and four destination points.\n"
            "Core math uses NumPy only:\n\n"
            "  [u, v, 1]^T ~ H [x, y, 1]^T\n\n"
            "For each point pair, two equations are added to the linear system.\n"
            "The image is transformed by inverse mapping and bilinear interpolation.\n"
        )
        self.write_math_text(text)

    def write_math_text(self, text):
        self.math_text.configure(state="normal")
        self.math_text.delete("1.0", tk.END)
        self.math_text.insert("1.0", text)
        self.math_text.configure(state="disabled")


def main():
    root = tk.Tk()
    HomographyDemoApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
