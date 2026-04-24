"""
config.py
"""

import argparse
from pathlib import Path


MIN_CASES = 6


def parse_args():
    """
    Parse command-line options.

    Defaults allow the script to run with no arguments by pointing at the
    project-local image and results directories.
    """
    base = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Group 8 – perspective transformation (8DOF)"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=str(base / "image" / "checkerboard_final.jpg"),
        help="Path to the input image.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(base / "results"),
        help="Directory where result images are saved.",
    )
    parser.add_argument(
        "--cases",
        type=int,
        default=MIN_CASES,
        help=f"Number of test cases to run (minimum {MIN_CASES}).",
    )
    parser.add_argument(
        "--show",
        type=int,
        choices=[0, 1],
        default=1,
        help="1 = display each result in a window, 0 = save only.",
    )
    return parser.parse_args()


def _pt(width, height, xr, yr):
    """Return a pixel coordinate from relative (fractional) ratios."""
    return [float(width * xr), float(height * yr)]


def build_default_cases(width, height):
    """
    Return a list of six test-case dictionaries.

    Each case defines four source and four destination pixel coordinates.
    Point order: P1=top-left, P2=top-right, P3=bottom-right, P4=bottom-left.

    Parameters
    ----------
    width, height : int   source image dimensions

    Returns
    -------
    list of dict, each with keys "name", "src_points", "dst_points"
    """
    # Fixed source quadrilateral – centred inner rectangle
    src = [
        _pt(width, height, 0.20, 0.20),
        _pt(width, height, 0.80, 0.20),
        _pt(width, height, 0.80, 0.80),
        _pt(width, height, 0.20, 0.80),
    ]

    return [
        {
            "name": "case01_scaling",
            "src_points": src,
            "dst_points": [
                _pt(width, height, 0.15, 0.15),
                _pt(width, height, 0.85, 0.15),
                _pt(width, height, 0.85, 0.85),
                _pt(width, height, 0.15, 0.85),
            ],
        },
        {
            "name": "case02_rotation_like",
            "src_points": src,
            "dst_points": [
                _pt(width, height, 0.28, 0.10),
                _pt(width, height, 0.88, 0.28),
                _pt(width, height, 0.72, 0.90),
                _pt(width, height, 0.12, 0.72),
            ],
        },
        {
            "name": "case03_trapezoid",
            "src_points": src,
            "dst_points": [
                _pt(width, height, 0.35, 0.18),
                _pt(width, height, 0.65, 0.18),
                _pt(width, height, 0.90, 0.85),
                _pt(width, height, 0.10, 0.85),
            ],
        },
        {
            "name": "case04_shear_like",
            "src_points": src,
            "dst_points": [
                _pt(width, height, 0.22, 0.15),
                _pt(width, height, 0.78, 0.22),
                _pt(width, height, 0.88, 0.82),
                _pt(width, height, 0.32, 0.75),
            ],
        },
        {
            "name": "case05_strong_perspective",
            "src_points": src,
            "dst_points": [
                _pt(width, height, 0.42, 0.10),
                _pt(width, height, 0.62, 0.10),
                _pt(width, height, 0.92, 0.90),
                _pt(width, height, 0.08, 0.90),
            ],
        },
        {
            "name": "case06_extreme_skew",
            "src_points": src,
            "dst_points": [
                _pt(width, height, 0.40, 0.12),
                _pt(width, height, 0.58, 0.08),
                _pt(width, height, 0.96, 0.88),
                _pt(width, height, 0.06, 0.80),
            ],
        },
    ]
