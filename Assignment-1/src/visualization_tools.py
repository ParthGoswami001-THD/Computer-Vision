"""
visualization_tools.py

Functions for drawing the source/destination points and building the final side-by-side result images.

Author: Dhruvi Vaishnav
"""

import cv2 as cv
import numpy as np


HEADER_HEIGHT = 54

# BGR colours matched to S1/D1 … S4/D4 point labels
POINT_COLORS = [
    (0,   0,   255),   # P1 – red
    (0,   255,  0 ),   # P2 – green
    (255,  0,   0 ),   # P3 – blue
    (0,   255, 255),   # P4 – cyan
]


def ensure_uint8_color(image):
    """Convert a grey or float image to a three-channel uint8 BGR image."""
    if image.ndim == 2:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image.copy()


def draw_points(image, points, prefix):
    """
    Draw labelled circles on the image at each point location.

    Parameters
    ----------
    image  : ndarray  BGR source image (will not be modified in place)
    points : ndarray  (N, 2) pixel coordinates
    prefix : str      label prefix, e.g. "S" produces "S1", "S2", …

    Returns
    -------
    out : ndarray  annotated copy of the image
    """
    out = ensure_uint8_color(image)
    for idx, point in enumerate(points):
        color = POINT_COLORS[idx % len(POINT_COLORS)]
        cx = int(round(point[0]))
        cy = int(round(point[1]))
        cv.circle(out, (cx, cy), 8, color, 2, cv.LINE_AA)
        cv.putText(
            out,
            f"{prefix}{idx + 1}",
            (cx + 10, cy - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv.LINE_AA,
        )
    return out


def side_by_side_result(source_image, warped_image, src_points, dst_points_shifted, info_text):
    """
    Build the final output image: source on the left, warped on the right,
    with correspondence points labelled and a text header.

    Parameters
    ----------
    source_image       : ndarray  original BGR image
    warped_image       : ndarray  transformed BGR image
    src_points         : ndarray  (4, 2) source coordinates
    dst_points_shifted : ndarray  (4, 2) destination coords in the warped canvas
    info_text          : str      one-line summary printed in the header

    Returns
    -------
    combined : ndarray  BGR image (header + side-by-side panels)
    """
    left = draw_points(source_image, src_points, "S")
    right = draw_points(warped_image, dst_points_shifted, "D")

    # Pad the shorter panel to match height
    left_h, left_w  = left.shape[:2]
    right_h, right_w = right.shape[:2]
    panel_h = max(left_h, right_h)

    def pad_height(img, target_h):
        diff = target_h - img.shape[0]
        if diff > 0:
            pad = np.zeros((diff, img.shape[1], 3), dtype=np.uint8)
            return np.vstack([img, pad])
        return img

    body = np.hstack([pad_height(left, panel_h), pad_height(right, panel_h)])

    # Header bar with summary text
    header = np.zeros((HEADER_HEIGHT, body.shape[1], 3), dtype=np.uint8)
    cv.putText(
        header,
        info_text,
        (10, 34),
        cv.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv.LINE_AA,
    )

    return np.vstack([header, body])
