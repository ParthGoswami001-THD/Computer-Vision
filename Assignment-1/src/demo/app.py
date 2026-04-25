from __future__ import annotations

import base64
import datetime as dt
import os
import re
import threading
import uuid
from pathlib import Path

import cv2 as cv
import numpy as np
from flask import Flask, jsonify, make_response, render_template, request, send_from_directory

# Add src directory to import local modules.
CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent.parent
import sys

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from homography import compute_homography_8points, project_points, warp_image_numpy
from visualization_tools import draw_points, side_by_side_result


SESSION_COOKIE = "homography_demo_session"
STORE_LOCK = threading.Lock()
STATE_STORE: dict[str, dict] = {}

APP = Flask(
    __name__,
    template_folder=str(CURRENT_FILE.parent / "templates"),
    static_folder=str(CURRENT_FILE.parent / "static"),
)
APP.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024


def make_fallback_image(width: int = 640, height: int = 460) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    block = 40

    for y in range(0, height, block):
        for x in range(0, width, block):
            value = 220 if ((x // block + y // block) % 2 == 0) else 90
            image[y : y + block, x : x + block] = (value, value, value)

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


def make_point(width: int, height: int, x_ratio: float, y_ratio: float) -> list[float]:
    return [float(width * x_ratio), float(height * y_ratio)]


def build_demo_cases(width: int, height: int) -> list[dict]:
    src_points = [
        make_point(width, height, 0.20, 0.20),
        make_point(width, height, 0.80, 0.20),
        make_point(width, height, 0.20, 0.80),
        make_point(width, height, 0.80, 0.80),
    ]

    return [
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


def order_points_tl_tr_bl_br(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (4, 2):
        return points

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


def format_matrix(matrix: np.ndarray) -> str:
    lines = []
    for row in matrix:
        line = "  ".join(f"{value:12.6f}" for value in row)
        lines.append(line)
    return "\n".join(lines)


def format_points(points: np.ndarray) -> str:
    lines = []
    for idx, point in enumerate(points):
        lines.append(f"P{idx + 1}: ({point[0]:.1f}, {point[1]:.1f})")
    return "\n".join(lines)


def create_math_report(
    H: np.ndarray,
    A: np.ndarray,
    src_points: np.ndarray,
    dst_points: np.ndarray,
    projected: np.ndarray,
    errors: np.ndarray,
    cond_number: float,
) -> str:
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
    lines.append("A matrix from normalized DLT in homography.py:")
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


def make_animation_source(image: np.ndarray) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    max_side = max(width, height)
    scale = min(1.0, 520.0 / max_side)

    if scale < 1.0:
        new_width = max(2, int(round(width * scale)))
        new_height = max(2, int(round(height * scale)))
        small = cv.resize(image, (new_width, new_height), interpolation=cv.INTER_AREA)
        return small, scale

    return image.copy(), 1.0


def combine_for_animation(left: np.ndarray, right: np.ndarray, t: float) -> np.ndarray:
    info = f"animation frame | t={t:.2f} | dst_t = src + t(dst-src)"
    empty_points = np.empty((0, 2), dtype=np.float64)
    return side_by_side_result(left, right, empty_points, empty_points, info)


def image_to_data_url(image: np.ndarray, ext: str = ".png") -> str:
    ok, encoded = cv.imencode(ext, image)
    if not ok:
        raise ValueError("Could not encode image.")

    b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    mime = "image/png" if ext.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def load_default_image() -> np.ndarray:
    candidate_paths = [
        SRC_DIR / "images" / "checkerboard_final.jpg",
        CURRENT_FILE.parent / "image" / "checkerboard_final.jpg",
    ]

    for path in candidate_paths:
        if path.is_file():
            image = cv.imread(str(path), cv.IMREAD_COLOR)
            if image is not None:
                return image

    return make_fallback_image()


def intro_text() -> str:
    return (
        "Run the transformation to show the calculation.\n\n"
        "The app calculates H from four source points and four destination points.\n"
        "Core math uses NumPy only:\n\n"
        "  [u, v, 1]^T ~ H [x, y, 1]^T\n\n"
        "For each point pair, two equations are added to the linear system.\n"
        "The image is transformed by inverse mapping and bilinear interpolation.\n"
    )


def init_state() -> dict:
    image = load_default_image()
    h, w = image.shape[:2]
    cases = build_demo_cases(w, h)
    first = cases[0]

    return {
        "source_image": image,
        "source_path": "default image",
        "results_dir": str(SRC_DIR / "results"),
        "cases": cases,
        "src_points": np.asarray(first["src_points"], dtype=np.float64),
        "dst_points": np.asarray(first["dst_points"], dtype=np.float64),
        "current_result": None,
        "status": "Upload an image or use the default checkerboard.",
        "math_text": intro_text(),
    }


def get_state(create: bool = True) -> tuple[str | None, dict | None]:
    session_id = request.cookies.get(SESSION_COOKIE)

    with STORE_LOCK:
        if session_id and session_id in STATE_STORE:
            return session_id, STATE_STORE[session_id]

        if not create:
            return None, None

        session_id = uuid.uuid4().hex
        STATE_STORE[session_id] = init_state()
        return session_id, STATE_STORE[session_id]


def parse_points(points_raw: list[list[float]]) -> np.ndarray:
    if not isinstance(points_raw, list) or len(points_raw) != 4:
        raise ValueError("Points must contain exactly 4 entries.")

    points = []
    for p in points_raw:
        if not isinstance(p, list) or len(p) != 2:
            raise ValueError("Each point must be [x, y].")
        points.append([float(p[0]), float(p[1])])

    return np.asarray(points, dtype=np.float64)


def move_destination_relative_to_source(state: dict, old_src: np.ndarray, new_src: np.ndarray) -> None:
    old_dst = state["dst_points"]
    new_dst = new_src + (old_dst - old_src)

    height, width = state["source_image"].shape[:2]
    new_dst[:, 0] = np.clip(new_dst[:, 0], 0, width - 1)
    new_dst[:, 1] = np.clip(new_dst[:, 1], 0, height - 1)
    state["dst_points"] = new_dst


def make_start_preview(state: dict) -> np.ndarray:
    preview = draw_points(state["source_image"], state["src_points"], "S")
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
    return preview


def serialize_state(state: dict, include_source: bool = True) -> dict:
    source = state["source_image"]
    payload = {
        "source_path": state["source_path"],
        "source_size": [int(source.shape[1]), int(source.shape[0])],
        "presets": [c["name"] for c in state["cases"]],
        "src_points": state["src_points"].tolist(),
        "dst_points": state["dst_points"].tolist(),
        "status": state["status"],
        "math_text": state["math_text"],
        "preview_image": image_to_data_url(state["current_result"] if state["current_result"] is not None else make_start_preview(state)),
        "has_result": state["current_result"] is not None,
    }

    if include_source:
        payload["source_image"] = image_to_data_url(state["source_image"])

    return payload


def json_error(message: str, code: int = 400):
    return jsonify({"ok": False, "error": message}), code


def save_result_image(state: dict, filename: str | None = None) -> str:
    if state["current_result"] is None:
        raise ValueError("Run the transformation first.")

    results_dir = Path(state["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    if filename:
        cleaned = re.sub(r"[^a-zA-Z0-9._-]", "_", filename).strip("._")
        if not cleaned:
            cleaned = "result"
        if not cleaned.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            cleaned += ".png"
        path = results_dir / cleaned
    else:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = results_dir / f"result_{stamp}.png"

    ok = cv.imwrite(str(path), state["current_result"])
    if not ok:
        raise ValueError("Could not save image.")

    return path.name


@APP.route("/")
def index():
    session_id, _state = get_state(create=True)
    response = make_response(render_template("index.html"))
    response.set_cookie(SESSION_COOKIE, session_id, httponly=True, samesite="Lax")
    return response


@APP.get("/api/state")
def api_state():
    session_id, state = get_state(create=True)
    response = jsonify({"ok": True, **serialize_state(state)})
    response.set_cookie(SESSION_COOKIE, session_id, httponly=True, samesite="Lax")
    return response


@APP.post("/api/upload")
def api_upload():
    _session_id, state = get_state(create=True)
    if "image" not in request.files:
        return json_error("Missing uploaded image.")

    uploaded = request.files["image"]
    raw = uploaded.read()
    if not raw:
        return json_error("Uploaded file is empty.")

    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv.imdecode(arr, cv.IMREAD_COLOR)
    if image is None:
        return json_error("Could not decode uploaded image.")

    state["source_image"] = image
    state["source_path"] = uploaded.filename or "uploaded image"
    state["current_result"] = None

    h, w = image.shape[:2]
    state["cases"] = build_demo_cases(w, h)
    state["src_points"] = np.asarray(state["cases"][0]["src_points"], dtype=np.float64)
    state["dst_points"] = np.asarray(state["cases"][0]["dst_points"], dtype=np.float64)
    state["status"] = f"Loaded: {state['source_path']}\nSize: {w} x {h}"
    state["math_text"] = intro_text()

    return jsonify({"ok": True, **serialize_state(state)})


@APP.post("/api/apply-preset")
def api_apply_preset():
    _session_id, state = get_state(create=True)
    body = request.get_json(silent=True) or {}
    name = str(body.get("name", ""))

    for case in state["cases"]:
        if case["name"] == name:
            state["src_points"] = np.asarray(case["src_points"], dtype=np.float64)
            state["dst_points"] = np.asarray(case["dst_points"], dtype=np.float64)
            state["current_result"] = None
            state["status"] = f"Applied preset: {name}"
            return jsonify({"ok": True, **serialize_state(state, include_source=False)})

    return json_error("Preset not found.")


@APP.post("/api/set-points")
def api_set_points():
    _session_id, state = get_state(create=True)
    body = request.get_json(silent=True) or {}

    try:
        src_points = order_points_tl_tr_bl_br(parse_points(body.get("src_points", [])))
        dst_points = order_points_tl_tr_bl_br(parse_points(body.get("dst_points", [])))
    except Exception as exc:
        return json_error(str(exc))

    state["src_points"] = src_points
    state["dst_points"] = dst_points
    state["current_result"] = None
    state["status"] = "Points updated."

    return jsonify({"ok": True, **serialize_state(state, include_source=False)})


@APP.post("/api/set-source-points")
def api_set_source_points():
    _session_id, state = get_state(create=True)
    body = request.get_json(silent=True) or {}

    try:
        new_src = order_points_tl_tr_bl_br(parse_points(body.get("src_points", [])))
    except Exception as exc:
        return json_error(str(exc))

    move_destination = bool(body.get("move_destination", True))
    old_src = state["src_points"].copy()

    state["src_points"] = new_src
    if move_destination:
        move_destination_relative_to_source(state, old_src, new_src)

    state["current_result"] = None
    state["status"] = "Source points updated from picker."

    return jsonify({"ok": True, **serialize_state(state)})


@APP.post("/api/run")
def api_run():
    _session_id, state = get_state(create=True)

    try:
        src_points = order_points_tl_tr_bl_br(state["src_points"])
        dst_points = order_points_tl_tr_bl_br(state["dst_points"])

        H, A = compute_homography_8points(src_points, dst_points)
        output_size = (state["source_image"].shape[0], state["source_image"].shape[1])
        warped = warp_image_numpy(state["source_image"], H, output_size)

        projected = project_points(H, src_points)
        errors = np.linalg.norm(projected - dst_points, axis=1)
        cond_number = float(np.linalg.cond(A))

        info = (
            f"interactive demo | cond(A)={cond_number:.2e} | "
            f"mean={np.mean(errors):.4f}px | max={np.max(errors):.4f}px"
        )

        result_image = side_by_side_result(
            state["source_image"],
            warped,
            src_points,
            dst_points,
            info,
        )

        state["src_points"] = src_points
        state["dst_points"] = dst_points
        state["current_result"] = result_image
        state["status"] = (
            f"Condition number: {cond_number:.3e}\n"
            f"Mean reprojection error: {np.mean(errors):.6f} px\n"
            f"Max reprojection error: {np.max(errors):.6f} px"
        )
        state["math_text"] = create_math_report(H, A, src_points, dst_points, projected, errors, cond_number)

        return jsonify({"ok": True, **serialize_state(state, include_source=False)})
    except Exception as exc:
        return json_error(str(exc), code=500)


@APP.post("/api/animate")
def api_animate():
    _session_id, state = get_state(create=True)

    try:
        src_points = order_points_tl_tr_bl_br(state["src_points"])
        dst_points = order_points_tl_tr_bl_br(state["dst_points"])

        preview_image, scale = make_animation_source(state["source_image"])
        src_small = src_points * scale
        dst_small = dst_points * scale
        source_marked = draw_points(preview_image, src_small, "S")

        frames = []
        frame_count = 12
        for idx in range(frame_count):
            t = idx / (frame_count - 1)
            dst_t = src_small + t * (dst_small - src_small)
            H_t, _ = compute_homography_8points(src_small, dst_t)
            warped_t = warp_image_numpy(preview_image, H_t, preview_image.shape[:2])
            target_marked = draw_points(warped_t, dst_t, "T")
            frame = combine_for_animation(source_marked, target_marked, t)
            frames.append(image_to_data_url(frame))

        state["status"] = "Animation ready. Showing intermediate homography frames."

        return jsonify(
            {
                "ok": True,
                "frames": frames,
                "status": state["status"],
                "src_points": state["src_points"].tolist(),
                "dst_points": state["dst_points"].tolist(),
            }
        )
    except Exception as exc:
        return json_error(str(exc), code=500)


@APP.post("/api/save")
def api_save():
    _session_id, state = get_state(create=True)
    body = request.get_json(silent=True) or {}

    try:
        filename = body.get("filename")
        saved_name = save_result_image(state, filename=filename)
        state["status"] = f"Saved result:\n{saved_name}"
        return jsonify({"ok": True, "saved": saved_name, "status": state["status"]})
    except Exception as exc:
        return json_error(str(exc))


@APP.get("/api/gallery")
def api_gallery():
    _session_id, state = get_state(create=True)
    results_dir = Path(state["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for path in sorted(results_dir.glob("*")):
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
            files.append(path.name)

    return jsonify({"ok": True, "files": files})


@APP.get("/gallery/<path:filename>")
def gallery_file(filename: str):
    _session_id, state = get_state(create=True)
    return send_from_directory(state["results_dir"], filename)


def main() -> None:
    APP.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
