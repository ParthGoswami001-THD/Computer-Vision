"""
main.py

Entry point. Loads the input image, runs each predefined test case, prints
a numerical summary, and saves side-by-side comparison images to results/.

Author: Dhruvi Vaishnav

"""

"""
Usage
-----
    python main.py
    python main.py --image path/to/image.jpg --cases 6 --show 0
"""

import os
import sys

import cv2 as cv
import numpy as np

from config_io import MIN_CASES, build_default_cases, parse_args
from homography_tools import compute_homography, project_points, warp_image_with_bounds
from visualization_tools import side_by_side_result


def run_case(source_image, case, output_dir, show_window):
    """
    Execute one test case end-to-end and save the result image.

    Parameters
    ----------
    source_image : ndarray  BGR source image
    case         : dict     name, src_points, dst_points
    output_dir   : str      where to write the output file
    show_window  : bool     whether to open an OpenCV display window

    Returns
    -------
    mean_error : float   mean reprojection error in pixels
    """
    name = case.get("name", "case")
    src_pts = np.asarray(case["src_points"], dtype=np.float64)
    dst_pts = np.asarray(case["dst_points"], dtype=np.float64)

    H, method, cond = compute_homography(src_pts, dst_pts)
    warped, min_x, min_y = warp_image_with_bounds(source_image, H, dst_pts)

    projected  = project_points(H, src_pts)
    errors     = np.linalg.norm(projected - dst_pts, axis=1)
    mean_err   = float(errors.mean())
    max_err    = float(errors.max())

    dst_shifted = dst_pts - np.array([min_x, min_y], dtype=np.float64)
    header_text = (
        f"{name} | {method} | cond(A)={cond:.2e} | "
        f"mean={mean_err:.4f} px | max={max_err:.4f} px"
    )

    result = side_by_side_result(source_image, warped, src_pts, dst_shifted, header_text)

    out_path = os.path.join(output_dir, f"{name}.png")
    cv.imwrite(out_path, result)
    print(f"  Saved  {os.path.basename(out_path)}")

    if show_window:
        cv.imshow("Group 8 – Perspective Transformation", result)
        key = cv.waitKey(0) & 0xFF
        if key == 27:  # Esc to stop early
            return mean_err, True

    return mean_err, False


def main():
    args = parse_args()

    if args.cases < MIN_CASES:
        print(f"Minimum number of cases is {MIN_CASES}; using {MIN_CASES}.")
        args.cases = MIN_CASES

    os.makedirs(args.output_dir, exist_ok=True)

    source_image = cv.imread(args.image, cv.IMREAD_COLOR)
    if source_image is None:
        sys.exit(f"Error: could not read image at '{args.image}'")

    height, width = source_image.shape[:2]
    cases = build_default_cases(width, height)

    if len(cases) < args.cases:
        sys.exit(
            f"Error: only {len(cases)} predefined cases available, "
            f"but --cases {args.cases} was requested."
        )

    print(f"\nGroup 8 – Perspective Transformation")
    print(f"Image: {args.image}  ({width} x {height})")
    print(f"Running {args.cases} case(s)...\n")

    results = []
    for case in cases[: args.cases]:
        mean_err, aborted = run_case(source_image, case, args.output_dir, args.show == 1)
        results.append((case["name"], mean_err))
        if aborted:
            break

    cv.destroyAllWindows()

    # Summary table
    print(f"\n{'Name':<35}  {'Mean error (px)':>16}")
    print("-" * 54)
    for name, err in results:
        print(f"{name:<35}  {err:>16.6f}")
    print(f"\nAll results saved to: {args.output_dir}\n")


if __name__ == "__main__":
    main()
