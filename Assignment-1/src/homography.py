"""
homography.py
This module contains functions for computing homography matrices and applying them to warp images.
All perspective transformation math lives here.
Only NumPy is used; OpenCV is intentionally excluded from this module.
Authors: Parth Goswami
"""

import math
import numpy as np

#-----------------------------------------
# Coordinate helpers functions
#-----------------------------------------  

def to_homogeneous(points):
    """ Appends a column of ones to convert (N,2) coordinates to homogeneous (N,3) coordinates. """
    points = np.asarray(points, dtype=np.float64)
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    homogeneous_points = np.hstack((points, ones))
    return homogeneous_points

#-----------------------------------------
# Point Validation
#-----------------------------------------
def points_are_unique(points, eps=1e-6):
    """ Checks if all points are unique within a small epsilon. """
    points = np.asarray(points, dtype=np.float64)
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if np.linalg.norm(points[i] - points[j]) < eps:
                return False
    return True

def has_non_collinear_triple(points, eps=1e-6):
    """
    Checks if there exists at least one triple of points that are not collinear.
    This is necessary for a valid homography, as three collinear points cannot define a plane
    """
    points = np.asarray(points, dtype=np.float64)
    for i, j, k in [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]:
        v1 = points[j] - points[i]
        v2 = points[k] - points[i]
        if abs(v1[0] * v2[1] - v1[1] * v2[0]) > eps:
            return True
    return False

def validate_correspondences(src_points, dst_points):
    """
    Verify shape, uniqueness, and non-collinearity for both point sets.
    Returns float64 arrays or raises ValueError.
    """
    src_points = np.asarray(src_points, dtype=np.float64)
    dst_points = np.asarray(dst_points, dtype=np.float64)

    if src_points.shape != (4, 2) or dst_points.shape != (4, 2):
        raise ValueError("Both point arrays must have shape (4, 2).")
    if not points_are_unique(src_points):
        raise ValueError("Source points contain duplicates.")
    if not points_are_unique(dst_points):
        raise ValueError("Destination points contain duplicates.")
    if not has_non_collinear_triple(src_points):
        raise ValueError("Source points are collinear – homography is undefined.")
    if not has_non_collinear_triple(dst_points):
        raise ValueError("Destination points are collinear – homography is undefined.")

    return src_points, dst_points

#-----------------------------------------
# Normalization 
#-----------------------------------------

def normalize_points(points):
    """
    Isotropic normalisation (Hartley 1997):
      - shift centroid to origin
      - scale so that mean distance from origin equals sqrt(2)

    Returns normalised points and the 3x3 similarity transform T such that
    normalised = T @ homogeneous(points).
    """
    points = np.asarray(points, dtype=np.float64)
    center = points.mean(axis=0)
    shifted = points - center

    mean_dist = np.sqrt((shifted ** 2).sum(axis=1)).mean()
    if mean_dist < 1e-12:
        raise ValueError("Points are degenerate; normalisation failed.")

    scale = math.sqrt(2.0) / mean_dist
    T = np.array([
        [scale, 0.0,  -scale * center[0]],
        [0.0,  scale, -scale * center[1]],
        [0.0,  0.0,   1.0               ],
    ], dtype=np.float64)

    pts_h = to_homogeneous(points)
    pts_norm_h = (T @ pts_h.T).T
    pts_norm = pts_norm_h[:, :2] / pts_norm_h[:, 2:3]
    return pts_norm, T

#-----------------------------------------
# DLT (Direct Linear Transformation) Homography Computation
#-----------------------------------------

def compute_homography_8points(src_points, dst_points):
    """
    Compute the 3x3 homography matrix from four point correspondences using
    the normalised Direct Linear Transform (DLT).

    Steps
    -----
    1. Validate inputs.
    2. Normalise both point sets (Hartley normalisation).
    3. Build the 8x9 DLT system Ah = 0.
    4. Solve via SVD – the last right singular vector is the solution.
    5. Denormalise back to pixel coordinates.
    6. Normalise so H[2,2] = 1.

    Returns
    -------
    H : ndarray (3, 3)
    A : ndarray (8, 9)   equation matrix (for condition number reporting)
    """
    src_points, dst_points = validate_correspondences(src_points, dst_points)

    src_norm, T_src = normalize_points(src_points)
    dst_norm, T_dst = normalize_points(dst_points)

    # Build A: two rows per point correspondence
    A = np.zeros((8, 9), dtype=np.float64)
    for i in range(4):
        x, y = src_norm[i]
        u, v = dst_norm[i]
        A[2 * i    ] = [ x,  y,  1,  0,  0,  0, -u*x, -u*y, -u]
        A[2 * i + 1] = [ 0,  0,  0,  x,  y,  1, -v*x, -v*y, -v]

    # Solve Ah = 0: take last row of V^T from SVD
    _, _, vh = np.linalg.svd(A)
    H_norm = vh[-1].reshape(3, 3)

    # Denormalise
    H_raw = np.linalg.inv(T_dst) @ H_norm @ T_src

    # Scale so H[2,2] == 1
    denom = H_raw[2, 2]
    H = H_raw / denom if abs(denom) > 1e-15 else H_raw / np.linalg.norm(H_raw)

    return H, A

def compute_homography(src_points, dst_points):
    """
    Public wrapper around compute_homography_8points.

    Returns
    -------
    H            : ndarray (3, 3)
    method_label : str
    cond_number  : float   condition number of the DLT equation matrix
    """
    H, A = compute_homography_8points(src_points, dst_points)
    return H, "svd_norm_dlt", float(np.linalg.cond(A))

# ---------------------------------------------------------------------------
# Point projection
# ---------------------------------------------------------------------------

def project_points(H, src_points):
    """
    Apply homography H to an array of (N, 2) source points.
    Performs perspective division to return Cartesian (N, 2) coordinates.
    """
    src_h = to_homogeneous(src_points)
    dst_h = (H @ src_h.T).T
    return dst_h[:, :2] / dst_h[:, 2:3]


# ---------------------------------------------------------------------------
# Output bounds computation
# ---------------------------------------------------------------------------

def get_output_bounds(dst_points):
    """
    Find the axis-aligned bounding box of the four destination points.

    Returns
    -------
    min_x, min_y : int   top-left corner of the bounding box
    out_w, out_h : int   output canvas width and height
    """
    dst_points = np.asarray(dst_points, dtype=np.float64)
    min_x = int(math.floor(dst_points[:, 0].min()))
    min_y = int(math.floor(dst_points[:, 1].min()))
    max_x = int(math.ceil(dst_points[:, 0].max()))
    max_y = int(math.ceil(dst_points[:, 1].max()))
    out_w = max(1, max_x - min_x + 1)
    out_h = max(1, max_y - min_y + 1)
    return min_x, min_y, out_w, out_h

#----------------------------------------------------------------------------
# Image warping
#----------------------------------------------------------------------------

def warp_image_numpy(source_image, H, output_size):
    """
    Warp source_image using H via vectorised backward mapping.

    For every output pixel the inverse homography locates the source
    coordinate. Bilinear interpolation fills sub-pixel positions.
    Pixels that map outside the source bounds are left black.

    Parameters
    ----------
    source_image : ndarray (H_src, W_src, C) or (H_src, W_src)
    H            : ndarray (3, 3)
    output_size  : tuple (out_h, out_w)

    Returns
    -------
    output : ndarray same dtype as source_image
    """
    out_h, out_w = output_size
    src_h, src_w = source_image.shape[:2]
    is_color = source_image.ndim == 3

    channels = source_image.shape[2] if is_color else 1
    output = np.zeros((out_h, out_w, channels), dtype=np.float64)

    H_inv = np.linalg.inv(H)

    # Build a (3, out_h * out_w) matrix of all destination pixel coordinates
    yy, xx = np.indices((out_h, out_w), dtype=np.float64)
    dst_coords = np.stack((xx.ravel(), yy.ravel(), np.ones(out_h * out_w)), axis=0)

    # Map destination → source
    src_coords = H_inv @ dst_coords
    w = src_coords[2]
    valid_w = np.abs(w) > 1e-12

    src_x = np.where(valid_w, src_coords[0] / np.where(valid_w, w, 1.0), -1.0)
    src_y = np.where(valid_w, src_coords[1] / np.where(valid_w, w, 1.0), -1.0)

    in_bounds = (
        valid_w
        & (src_x >= 0.0) & (src_x <= src_w - 1)
        & (src_y >= 0.0) & (src_y <= src_h - 1)
    )
    if not in_bounds.any():
        return output.reshape(out_h, out_w) if not is_color else output[:, :, :].astype(np.uint8)

    x = src_x[in_bounds]
    y = src_y[in_bounds]

    # Bilinear interpolation
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, src_w - 1)
    y1 = np.clip(y0 + 1, 0, src_h - 1)
    dx = (x - x0)[:, None]
    dy = (y - y0)[:, None]

    src = source_image.reshape(src_h, src_w, channels).astype(np.float64)
    q00 = src[y0, x0]
    q10 = src[y0, x1]
    q01 = src[y1, x0]
    q11 = src[y1, x1]

    samples = (1 - dx) * (1 - dy) * q00 \
            +      dx  * (1 - dy) * q10 \
            + (1 - dx) *      dy  * q01 \
            +      dx  *      dy  * q11

    out_flat = output.reshape(-1, channels)
    out_flat[np.where(in_bounds)[0]] = samples

    result = np.clip(output, 0, 255).astype(np.uint8)
    return result if is_color else result[:, :, 0]


def warp_image_with_bounds(source_image, H, dst_points):
    """
    Warp source_image and fit the output canvas to the destination points.

    A translation is prepended to H so that the top-left corner of the
    bounding box of dst_points maps to (0, 0) in the output image.

    Returns
    -------
    output_image : ndarray
    min_x        : int   x-offset of the canvas origin in destination space
    min_y        : int   y-offset
    """
    min_x, min_y, out_w, out_h = get_output_bounds(dst_points)

    # Shift canvas so that (min_x, min_y) maps to (0, 0)
    T_shift = np.array([
        [1.0, 0.0, -float(min_x)],
        [0.0, 1.0, -float(min_y)],
        [0.0, 0.0,  1.0         ],
    ], dtype=np.float64)

    H_local = T_shift @ H
    output_image = warp_image_numpy(source_image, H_local, (out_h, out_w))
    return output_image, min_x, min_y
