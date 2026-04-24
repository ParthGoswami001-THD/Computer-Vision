# Computer Vision Assignment 1: Perspective Transformation (8DOF)

![Python](https://img.shields.io/badge/Python-3.7%2B-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-only%20core%20math-013243?logo=numpy)
![OpenCV](https://img.shields.io/badge/OpenCV-image%20I%2FO-5C3EE8?logo=opencv)
![License](https://img.shields.io/badge/License-Academic%20%E2%80%94%20Group%208-lightgrey)

> Implements **perspective transformation with 8 degrees of freedom (8DOF)** using **homography matrices** — computed via Direct Linear Transform (DLT) with point normalization, fully in NumPy.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements & Installation](#requirements--installation)
- [Usage](#usage)
  - [CLI](#command-line-interface-cli)
  - [GUI](#interactive-gui-application)
- [Core Modules](#core-modules)
- [Test Cases](#test-cases)
- [Output & Metrics](#output--metrics)
- [Mathematical Background](#mathematical-background)
- [References](#references)
- [Authors](#authors)

---

## Overview

**Key Features:**

| Feature | Details |
|---|---|
| Core math | Pure NumPy — no OpenCV in homography computation |
| Algorithm | Direct Linear Transform (DLT) with Hartley normalization |
| Error analysis | Reprojection error (mean & max) + condition number |
| Visualization | Side-by-side annotated image comparison |
| Interface | CLI batch runner + interactive Tkinter GUI |
| Test cases | 6 predefined: scaling, rotation-like, trapezoid, shear, strong perspective, extreme skew |

---

## Project Structure

```
Assignment-1/
├── src/
│   ├── homography.py              # Core homography computation (NumPy only)
│   ├── visualization_tools.py     # Point drawing and result image generation
│   ├── config.py                  # CLI argument parsing and test case definitions
│   ├── main.py                    # Command-line interface entry point
│   ├── demo/
│   │   └── app.py                 # Interactive GUI application (Tkinter)
│   ├── images/                    # Input test images
│   └── results/                   # Output transformation results
├── requirements.txt
└── README.md
```

---

## Requirements & Installation

- Python 3.7+
- `numpy` — numerical computations
- `opencv-python` — image I/O and display
- `pillow` — GUI image handling

```bash
pip install -r requirements.txt
```

---

## Usage

### Command-Line Interface (CLI)

```bash
# Default: checkerboard image, all 6 test cases
python src/main.py

# Custom image, 3 cases, show results
python src/main.py --image path/to/image.jpg --cases 3 --show 1

# Save results without opening display windows
python src/main.py --show 0
```

**Options:**

| Flag | Type | Default | Description |
|---|---|---|---|
| `--image` | path | `src/images/checkerboard_final.jpg` | Input image path |
| `--output-dir` | path | `src/results` | Output directory |
| `--cases` | int (min 6) | `6` | Number of test cases to run |
| `--show` | `0` or `1` | `1` | Display result windows |

### Interactive GUI Application

```bash
python src/demo/app.py
```

<details>
<summary>GUI Features (click to expand)</summary>

- **Load Image** — browse and select any input image
- **Run Transform** — execute homography with the selected test case
- **Animate** — frame-by-frame transformation playback
- **Display Results** — side-by-side view with labeled correspondence points

**Real-time metrics panel:**
- Homography matrix **H**
- Normalization matrix **T**
- **A** matrix (from DLT construction)
- Mean reprojection error (pixels)
- Max reprojection error (pixels)
- Condition number of A matrix

</details>

---

## Core Modules

<details>
<summary><strong>homography.py</strong> — core math (click to expand)</summary>

| Function | Description |
|---|---|
| `compute_homography_8points(src, dst)` | Computes H using 8-point DLT |
| `compute_homography(src, dst)` | Wrapper with method selection |
| `project_points(H, src)` | Projects source points through H |
| `warp_image_numpy(img, H, size)` | Backward-mapped image warp |
| `normalize_points(points)` | Isotropic normalization (Hartley 1997) |
| `validate_correspondences(src, dst)` | Validates point pair inputs |

**DLT Pipeline:**
1. Normalize source and destination point sets
2. Build A matrix from correspondences (2 equations per point pair)
3. SVD → extract null space (smallest singular vector)
4. Reshape to 3×3 H
5. Denormalize using T matrices
6. Validate via reprojection error

</details>

<details>
<summary><strong>visualization_tools.py</strong> — annotations (click to expand)</summary>

| Function | Description |
|---|---|
| `draw_points(image, points, prefix)` | Labels points with circles (S1–S4 / D1–D4) |
| `side_by_side_result(src, warped, src_pts, dst_pts, info)` | Builds comparison image with header |

</details>

<details>
<summary><strong>config.py</strong> — CLI & test case definitions (click to expand)</summary>

Manages `argparse` configuration and defines all 6 test cases as named point-pair sets.

</details>

<details>
<summary><strong>main.py</strong> — CLI entry point (click to expand)</summary>

- Loads input image
- Iterates through test cases
- Computes statistics
- Saves annotated result images to `src/results/`

</details>

<details>
<summary><strong>demo/app.py</strong> — Tkinter GUI (click to expand)</summary>

Interactive explorer with image loading, test case selection, animation, and live metrics.

</details>

---

## Test Cases

Each case is defined by 4 source + 4 destination point pairs (normalized to image dimensions).

| # | Name | Transformation |
|---|---|---|
| 1 | `case01_scaling` | Uniform scale — 15% outward expansion |
| 2 | `case02_rotation_like` | Rotational-style point shift |
| 3 | `case03_trapezoid` | Trapezoidal perspective distortion |
| 4 | `case04_shear_like` | Horizontal shear |
| 5 | `case05_strong_perspective` | Strong one-sided perspective |
| 6 | `case06_extreme_skew` | Extreme skew / projective distortion |

<details>
<summary>Example — case01_scaling point mapping</summary>

| Point | Source | Destination |
|---|---|---|
| P1 | (20%, 20%) | (15%, 15%) |
| P2 | (80%, 20%) | (85%, 15%) |
| P3 | (80%, 80%) | (85%, 85%) |
| P4 | (20%, 80%) | (15%, 85%) |

</details>

---

## Output & Metrics

Results are saved to `src/results/`. Each output image contains:

- **Left panel** — original image with source points labeled S1–S4
- **Right panel** — warped image with destination points labeled D1–D4
- **Header** — test case name, method, and error metrics

| Metric | Meaning |
|---|---|
| `cond(A)` | Condition number of A matrix — lower is numerically better |
| `mean` | Mean reprojection error across all point pairs (pixels) |
| `max` | Maximum reprojection error across all point pairs (pixels) |

---

## Mathematical Background

<details>
<summary>Homography Matrix (click to expand)</summary>

A 3×3 matrix **H** mapping projective points between planes:

$$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} \sim H \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

8 DOF (scale is fixed by the constraint $h_9 = 1$), so 4 point correspondences (8 equations) are the minimum required.

</details>

<details>
<summary>Direct Linear Transform (DLT) (click to expand)</summary>

Each point pair $(x_i, y_i) \leftrightarrow (x_i', y_i')$ contributes two linear equations to the system **Ah = 0**:

$$A = \begin{bmatrix} -x & -y & -1 & 0 & 0 & 0 & x'x & x'y & x' \\ 0 & 0 & 0 & -x & -y & -1 & y'x & y'y & y' \end{bmatrix}$$

The solution **h** is the right singular vector of A corresponding to the smallest singular value (via SVD), then reshaped to 3×3.

</details>

<details>
<summary>Point Normalization — Hartley 1997 (click to expand)</summary>

Before constructing A, points are isotropically normalized:
1. Translate centroid to origin
2. Scale so mean distance from origin = $\sqrt{2}$

This produces a transformation matrix **T** (and **T'**), and the final homography is denormalized as:

$$H = T'^{-1} \hat{H} T$$

Normalization dramatically improves the condition number of A and the numerical accuracy of the solution.

</details>

---

## References

<details>
<summary>Key Papers & Books (click to expand)</summary>

1. **Hartley & Zisserman** (2003). *Multiple View Geometry in Computer Vision* (2nd ed.). Cambridge University Press.
   - Chapter 4: Estimating the Homography

2. **Hartley, R. I.** (1997). "In Defense of the Eight-Point Algorithm." *IEEE TPAMI*, 19(6), 580–593.
   - Normalization technique and numerical stability analysis

3. **Zhang, Z.** (1998). "A Flexible New Technique for Camera Calibration." *IEEE TPAMI*, 22(11), 1330–1334.
   - Homography in camera calibration context

**Methods used:** DLT · SVD · Isotropic point normalization · Backward mapping · Bilinear interpolation

</details>

---

## Authors

| Role | Name |
|---|---|
| Homography implementation & GUI | Parth Goswami |
| Visualization & configuration | Dhruvi Vaishnav |

**Group 8 — Academic Assignment**

> Core mathematical algorithms (DLT, point normalization, SVD-based homography, reprojection error) were independently implemented, reviewed, and validated by the team. GitHub Copilot was used for documentation templates only.