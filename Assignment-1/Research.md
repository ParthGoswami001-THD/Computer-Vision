# Perspective Transformation Research: 8DOF Homography
## Comprehensive Research Document for Group Project

---

## 0. RESEARCH DEVELOPMENT

This section explains how the final solution was developed from the assignment requirement into a working implementation. It is divided into the main stages followed during the project.

### 0.1 Problem Understanding

The assignment asks for a configurable perspective transformation using four source points and four destination points. The required transformation is a homography, because a homography maps points from one plane to another plane using a 3×3 matrix in homogeneous coordinates.

The important restrictions were:
- The homography matrix must be calculated manually.
- The image transformation must be implemented manually.
- NumPy is allowed for mathematical computation.
- OpenCV is allowed only for image reading, drawing, displaying, and saving.
- OpenCV homography and warp helper functions are not allowed.

### 0.2 Literature Study

The research started from the standard projective geometry explanation in Hartley and Zisserman. Their work explains why a planar projective transformation can be represented with a 3×3 matrix and why the matrix has 8 degrees of freedom after scale normalization.

Szeliski was used to understand image alignment and image warping from a computer vision perspective. This helped confirm that backward mapping is the better practical choice, because direct forward mapping can leave empty holes in the destination image.

The Direct Linear Transformation idea was studied because it gives a clean way to build equations from point correspondences. Gonzalez and Woods was used for the image sampling and interpolation part, especially for understanding why bilinear interpolation gives smoother results than nearest-neighbor sampling.

### 0.3 Mathematical Development

The first mathematical decision was to use homogeneous coordinates:

```
(x, y)  ->  (x, y, 1)
```

This makes the perspective transformation possible using matrix multiplication. The homography matrix has nine entries, but it is scale-invariant, so one entry can be fixed by normalization. This leaves 8 unknown values.

Each point correspondence gives two equations:

```
source point (x, y)  ->  destination point (u, v)
```

Four correspondences therefore provide eight equations, exactly enough to solve the eight independent homography parameters.

### 0.4 Algorithm Development

The implementation uses these main steps:

1. Validate that the source and destination points are usable.
2. Reject duplicate points.
3. Reject nearly collinear point sets.
4. Normalize the points to improve numerical stability.
5. Build the DLT equation matrix.
6. Solve the system using SVD.
7. Denormalize the matrix back to image coordinates.
8. Normalize the final homography matrix.

SVD was selected because it is more stable than directly inverting the equation matrix, especially when the point positions create a poorly conditioned system.

### 0.5 Transformation Development

Two image transformation approaches were considered:

| Approach | Problem / Benefit |
|----------|-------------------|
| Forward mapping | Simple idea, but can leave holes because not every destination pixel receives a value |
| Backward mapping | Preferred method because every destination pixel is mapped back to a source coordinate |

The final solution uses backward mapping:

```
destination pixel -> inverse homography -> source coordinate
```

The source coordinate is often fractional, so bilinear interpolation is used to estimate the pixel value. If the mapped source coordinate is outside the image, the output pixel is left black.

### 0.6 Validation Development

The validation was divided into mathematical and visual checks.

**Mathematical checks:**
- Reproject the four source points through the calculated homography.
- Compare the projected points with the destination points.
- Report mean and maximum reprojection error.
- Report the condition number of the equation matrix.

**Visual checks:**
- Save side-by-side source and transformed images.
- Mark the source points and destination points.
- Test several different cases: scaling, rotation-like mapping, trapezoid perspective, shear-like mapping, strong perspective, and extreme skew.

### 0.7 Final Implementation Direction

The final implementation keeps the assignment solution simple and separated:

- `homography.py` contains the NumPy-only homography and warping math.
- `visualization_tools.py` contains OpenCV drawing and side-by-side output.
- `config.py` contains the predefined point cases and command-line configuration.
- `main.py` runs all cases and saves the results.
- `app.py` and `Presentation` are presentation/demo helpers and are separate from the main submission path.

---

## 1. MATHEMATICAL FOUNDATIONS

### 1.1 Homography Definition

A **homography** (projective transformation) is a linear transformation that maps points from one plane to another. In computer vision, it represents a 3×3 matrix that relates corresponding points across two images of the same planar surface.

**Homogeneous Coordinates:**
- 2D point (x, y) → homogeneous form (x, y, 1)
- Allows affine and projective transformations to be represented as matrix multiplication
- Scale invariance: (x, y, w) ≡ (λx, λy, λw) for any λ ≠ 0

**Homography Matrix:**
```
H = [h₁₁  h₁₂  h₁₃]
    [h₂₁  h₂₂  h₂₃]
    [h₃₁  h₃₂  h₃₃]
```

**Degrees of Freedom (8DOF):**
- 9 elements in 3×3 matrix
- 1 DOF lost due to scale invariance (we normalize h₃₃ = 1)
- Result: 8 independent parameters

### 1.2 Point Mapping Equation

For a point p = (x, y) mapping to p' = (x', y'):

```
[x']   [h₁₁  h₁₂  h₁₃] [x]
[y'] = [h₂₁  h₂₂  h₂₃] [y]
[w']   [h₃₁  h₃₂  h₃₃] [1]
```

The projected point is obtained via perspective division:
```
x' = (h₁₁x + h₁₂y + h₁₃) / (h₃₁x + h₃₂y + h₃₃)
y' = (h₂₁x + h₂₂y + h₂₃) / (h₃₁x + h₃₂y + h₃₃)
```

This is **non-linear** due to the division. To solve it, we reformulate as a **linear system**.

### 1.3 Linear System Formulation

For each correspondence point (x, y) → (x', y'), we derive 2 linear equations:

**Equation 1 (for x'):**
```
h₁₁x + h₁₂y + h₁₃ - h₃₁x'x - h₃₂x'y - h₃₃x' = 0
```

**Equation 2 (for y'):**
```
h₂₁x + h₂₂y + h₂₃ - h₃₁y'x - h₃₂y'y - h₃₃y' = 0
```

Rearranging in matrix form:
```
[x   y   1   0   0   0  -x'x  -x'y] [h₁₁]   [x']
[0   0   0   x   y   1  -y'x  -y'y] [h₁₂]   [y']
                                    [h₁₃] = [  ]
                                    [h₂₁]   [  ]
                                    [h₂₂]   [  ]
                                    [h₂₃]   [  ]
                                    [h₃₁]   [  ]
                                    [h₃₂]   [  ]
```

**4 points → 8 equations → 8 unknowns (exactly determined system)**

### 1.4 Complete Linear System (Ah = b)

For 4 correspondence points:

```
A = [x₁  y₁  1   0   0   0  -x₁'x₁  -x₁'y₁]
    [0   0   0   x₁  y₁  1  -y₁'x₁  -y₁'y₁]
    [x₂  y₂  1   0   0   0  -x₂'x₂  -x₂'y₂]
    [0   0   0   x₂  y₂  1  -y₂'x₂  -y₂'y₂]
    [x₃  y₃  1   0   0   0  -x₃'x₃  -x₃'y₃]
    [0   0   0   x₃  y₃  1  -y₃'x₃  -y₃'y₃]
    [x₄  y₄  1   0   0   0  -x₄'x₄  -x₄'y₄]
    [0   0   0   x₄  y₄  1  -y₄'x₄  -y₄'y₄]

b = [x₁', y₁', x₂', y₂', x₃', y₃', x₄', y₄']ᵀ

h = [h₁₁, h₁₂, h₁₃, h₂₁, h₂₂, h₂₃, h₃₁, h₃₂]ᵀ
```

**Solving:**
- For exact data: h = A⁻¹b
- For noisy data: h = (AᵀA)⁻¹Aᵀb (least squares)
- Most robust: h = right singular vector of A (SVD method)

---

## 2. ALGORITHM DESIGN

### 2.1 Step-by-Step Homography Calculation

**Input:** 4 source points (x₁, y₁), (x₂, y₂), (x₃, y₃), (x₄, y₄)
         4 destination points (x₁', y₁'), (x₂', y₂'), (x₃', y₃'), (x₄', y₄')

**Step 1: Construct Matrix A and Vector b**
```
for each point pair (xi, yi) → (xi', yi'):
    row_even = [xi, yi, 1, 0, 0, 0, -xi'*xi, -xi'*yi, -xi']
    row_odd  = [0, 0, 0, xi, yi, 1, -yi'*xi, -yi'*yi, -yi']
    append rows to A and b
```

**Step 2: Solve Linear System**
```
Method 1: Direct inversion
    h = np.linalg.solve(A, b)

Method 2: Least squares (more robust)
    h, residuals, rank, s = np.linalg.lstsq(A, b)

Method 3: SVD (most robust)
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1, :] / Vt[-1, -1]  # last row, normalized
```

**Step 3: Reshape and Normalize**
```
H = np.array([
    [h[0], h[1], h[2]],
    [h[3], h[4], h[5]],
    [h[6], h[7], 1.0]
])
```

**Output:** 3×3 Homography matrix H

### 2.2 Image Transformation

**Challenge:** Direct forward mapping (source → destination) leaves holes.
**Solution:** Backward mapping (destination ← source)

**Algorithm:**

1. For each output pixel (x', y'):
   - Compute source coordinates: (x, y) = H⁻¹(x', y')
   - Use bilinear interpolation to sample pixel at (x, y)
   - Place sampled value at (x', y')

2. **Perspective Division:**
```
[x*w]        [x']
[y*w]  = H⁻¹[y']
[w]          [1]

x = x*w / w
y = y*w / w
```

3. **Bilinear Interpolation:**
```
For fractional coordinates (x_float, y_float):
x_int = floor(x_float)
y_int = floor(y_float)
dx = x_float - x_int
dy = y_float - y_int

I(x_float, y_float) = 
    (1-dx)(1-dy)*I[y_int,   x_int  ] +
    dx*(1-dy)*I[y_int,   x_int+1] +
    (1-dx)*dy*I[y_int+1, x_int  ] +
    dx*dy*I[y_int+1, x_int+1]
```

**Boundary Handling:**
- Check if (x, y) is within image bounds
- If outside: use black pixel (0) or repeat edge

### 2.3 Stability Considerations

**Numerical Stability:**
- **Data Normalization:** Scale coordinates to [-1, 1] range
  - Improves condition number of matrix A
  - Implementation: translate centroid to origin, scale by mean distance
  
- **Use SVD over direct inversion**
  - More robust to ill-conditioned matrices
  - Handles near-singular systems gracefully

- **Avoid small matrix determinants**
  - Check det(A) before solving
  - Use pseudo-inverse if det(A) ≈ 0

**Point Configuration Stability:**
- **Non-collinear points:** Critical requirement
  - Collinear points make A rank-deficient
  - Algorithm should detect and reject

- **Well-distributed points:** Better numerical stability
  - Avoid points clustered in one region
  - Optimal: points at corners of image

- **Non-degenerate configurations:**
  - Avoid small areas (< 10×10 pixels)
  - Avoid extreme aspect ratios

---

## 3. TEST CASES FOR STABILITY

### 3.1 Test Case 1: Simple Rectangle Scaling

**Purpose:** Verify basic transformation works
**Source points:** Corners of a rectangle
**Destination points:** Scaled rectangle (same orientation)

```
Source:      Destination:
(50, 50)     (0, 0)
(150, 50)    (200, 0)
(50, 150)    (0, 200)
(150, 150)   (200, 200)
```

**Expected:** Clean scaling, no distortion

### 3.2 Test Case 2: Rotation

**Purpose:** Verify perspective handles rotation
**Transformation:** Rotate by 15° and scale by 1.5

**Expected:** Image rotated and scaled uniformly

### 3.3 Test Case 3: Perspective Distortion (Trapezoid)

**Purpose:** True perspective transformation
**Source points:** Rectangle
**Destination points:** Trapezoid (simulating viewing angle)

```
Source:      Destination (simulating tilt):
(50, 50)     (80, 20)      ← top-left moves right
(150, 50)    (120, 20)     ← top-right moves left
(50, 150)    (40, 180)     ← bottom stays
(150, 150)   (190, 180)
```

**Expected:** Perspective distortion, edges converge

### 3.4 Test Case 4: Document Straightening

**Purpose:** Real-world use case
**Source:** Tilted document photo
**Destination:** Straight rectangular output

**Expected:** Document appears photographed from directly above

### 3.5 Test Case 5: Extreme Perspective

**Purpose:** Test near-degenerate cases
**Transformation:** Very steep viewing angle

**Expected:** Stable computation, reasonable result (or warning)

### 3.6 Test Case 6: Large Scale Change

**Purpose:** Test numerical stability with different scales
**Transformation:** 10× zoom in one axis

**Expected:** Stable computation, no numerical artifacts

---

## 4. ERROR ANALYSIS

### 4.1 Sources of Error

**1. Point Localization Error**
- User clicks offset by a few pixels
- Effect: Homography matrix slightly wrong
- Solution: Use corner detection algorithms (Harris, SIFT)

**2. Rounding Errors**
- Floating-point arithmetic
- Perspective division denominator near zero
- Solution: Use double precision (float64)

**3. Interpolation Artifacts**
- Bilinear interpolation is linear, not ideal
- Causes slight blur near strong edges
- Solution: Use higher-order interpolation (bicubic) or leave as-is

**4. Boundary Effects**
- Warped image extends outside source bounds
- Missing data at edges
- Solution: Use transparent/black fill, or crop output

### 4.2 Numerical Stability Metrics

**Condition Number of A:**
```
κ(A) = σ_max / σ_min  (ratio of largest to smallest singular values)
κ(A) > 100: ill-conditioned (prone to errors)
κ(A) < 10: well-conditioned (stable)
```

**Reprojection Error:**
```
For each point (xi, yi) → (xi', yi'):
    (xi_proj, yi_proj) = H(xi, yi)
    error_i = ||(xi_proj, yi_proj) - (xi', yi')||
    
RMSE = sqrt(sum(error_i²) / n_points)
```

Should be near zero (< 1 pixel) for correct H.

---

## 5. IMPLEMENTATION CONSIDERATIONS

### 5.1 Key Functions Implemented

**homography.py**

1. **to_homogeneous(points)**
   - Input: (N, 2) point array
   - Output: (N, 3) homogeneous coordinates

2. **points_are_unique(points)**
   - Input: point array
   - Output: bool — False if any two points are within epsilon distance

3. **has_non_collinear_triple(points)**
   - Input: 4-point array
   - Output: bool — True if at least one non-collinear triple exists

4. **validate_correspondences(src_points, dst_points)**
   - Input: two (4, 2) arrays
   - Output: validated float64 arrays, or raises ValueError

5. **normalize_points(points)**
   - Input: (N, 2) point array
   - Output: normalised points and 3×3 similarity transform T (Hartley normalisation)

6. **compute_homography_8points(src_points, dst_points)**
   - Input: 4 source and 4 destination points
   - Output: 3×3 homography H, 8×9 DLT equation matrix A

7. **compute_homography(src_points, dst_points)**
   - Public wrapper around `compute_homography_8points`
   - Input: source and destination points
   - Output: H (3×3), method label (str), condition number (float)

8. **project_points(H, src_points)**
   - Input: homography H and (N, 2) source points
   - Output: (N, 2) projected destination points (with perspective division)

9. **get_output_bounds(dst_points)**
   - Input: (4, 2) destination points
   - Output: min_x, min_y, out_w, out_h (axis-aligned bounding box)

10. **warp_image_numpy(source_image, H, output_size)**
    - Input: image, homography, (out_h, out_w) tuple
    - Output: warped image via vectorised backward mapping with bilinear interpolation

11. **warp_image_with_bounds(source_image, H, dst_points)**
    - Input: image, homography, destination points
    - Output: warped image fitted to destination bounding box, canvas offsets min_x, min_y

**visualization_tools.py**

12. **ensure_uint8_color(image)**
    - Input: any image (grey or float)
    - Output: three-channel uint8 BGR image

13. **draw_points(image, points, prefix)**
    - Input: image, (N, 2) points, label prefix string (e.g. "S" → "S1", "S2", …)
    - Output: annotated image copy with labelled, colour-coded circles

14. **side_by_side_result(source_image, warped_image, src_points, dst_points_shifted, info_text)**
    - Input: source image, warped image, correspondence points, one-line header text
    - Output: combined BGR image with header bar and side-by-side panels

**config.py**

15. **parse_args()**
    - Output: parsed command-line arguments (image path, output dir, case count, show flag)

16. **build_default_cases(width, height)**
    - Input: source image dimensions
    - Output: list of 6 test-case dicts (name, src_points, dst_points)

**main.py**

17. **run_case(source_image, case, output_dir, show_window)**
    - Input: source image, case dict, output directory, display flag
    - Output: mean reprojection error (float), abort flag (bool)

### 5.2 Data Types and Precision

```python
# Use float64 for all calculations
src_points = np.array([...], dtype=np.float64)
dst_points = np.array([...], dtype=np.float64)

# Homography matrix
H = np.zeros((3, 3), dtype=np.float64)

# Image may be uint8, but convert to float for calculations
image_float = image.astype(np.float64) / 255.0
```

### 5.3 Handling Edge Cases

**Edge Case 1: Collinear Points**
```python
def has_non_collinear_triple(points, eps=1e-6):
    points = np.asarray(points, dtype=np.float64)
    for i, j, k in [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]:
        v1 = points[j] - points[i]
        v2 = points[k] - points[i]
        if abs(v1[0] * v2[1] - v1[1] * v2[0]) > eps:
            return True   # at least one non-collinear triple found
    return False          # all triples are collinear
```

**Edge Case 2: Very Small Determinant**
```python
def compute_homography(src_points, dst_points):
    H, A = compute_homography_8points(src_points, dst_points)

    # Check condition number of the DLT equation matrix
    cond = np.linalg.cond(A)
    if cond > 1e10:
        print(f"Warning: Ill-conditioned matrix (κ={cond})")

    return H, "svd_norm_dlt", float(cond)
```

**Edge Case 3: Output Image Bounds**
```python
# Project destination points to find the output canvas size
min_x, min_y, out_w, out_h = get_output_bounds(dst_points)

# Or project arbitrary corners through the homography
corners_src = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float64)
corners_dst = project_points(H, corners_src)

x_min, y_min = corners_dst.min(axis=0)
x_max, y_max = corners_dst.max(axis=0)
output_size = (int(y_max - y_min), int(x_max - x_min))
```

---

## 6. VALIDATION STRATEGY

### 6.1 Forward Verification

Reprojection is computed with `project_points` and compared against the destination:
```python
# In main.py — run_case()
H, method, cond = compute_homography(src_pts, dst_pts)

projected = project_points(H, src_pts)          # (4, 2) projected coords
errors    = np.linalg.norm(projected - dst_pts, axis=1)
mean_err  = float(errors.mean())
max_err   = float(errors.max())

for i in range(4):
    print(f"Point {i+1}: error = {errors[i]:.6f} pixels")
```

All errors should be < 1 pixel (ideally < 0.1 pixel).

### 6.2 Image Quality Assessment

1. **Visual inspection:** No obvious distortions or artifacts
2. **Boundary check:** No black lines at edges (unless expected)
3. **Continuity:** No jumps or discontinuities in intensity
4. **Reprojection:** Test points visible in output match positions

---

## 7. RESEARCH REFERENCES

### Key Papers and Resources

1. **Multiple View Geometry in Computer Vision** (Hartley & Zisserman, 2003)
   - Chapter 4: Perspective Transformations
   - Section 4.1.2: Computing the homography from point correspondences

2. **Computer Vision: Algorithms and Applications** (Szeliski, 2010)
   - Chapter 3.1: 2D transformations
   - Chapter 8: Image alignment and stitching

3. **Direct Linear Transformation (DLT)**
   - Abdel-Aziz & Karara (1971): Original paper on DLT algorithm
   - Used for both 2D homography and 3D camera calibration

4. **Digital Image Processing** (Gonzalez & Woods)
   - Chapter 2.5: Geometric transformations
   - Bilinear interpolation techniques

### Additional Concepts

- **SIFT/SURF:** Robust feature detection for finding correspondences
- **RANSAC:** For outlier rejection in automatic point matching
- **Bundle Adjustment:** For refining multiple homographies
- **Affine Transformation:** Special case of homography (h₃₁=h₃₂=0)

---

## 8. EXPECTED OUTCOMES

### Deliverables

1. **Python implementation** (using only numpy)
   - Homography computation from 4 points
   - Image transformation via backward mapping
   - No OpenCV in core algorithm

2. **Test cases** (at least 6 different scenarios)
   - Simple scaling
   - Rotation
   - Perspective distortion
   - Real-world document case
   - Extreme perspective
   - Large scale change

3. **Visualization**
   - Side-by-side source and destination images
   - Marked correspondence points
   - Clear, labeled output

4. **Documentation**
   - Mathematical derivation
   - Algorithm explanation
   - Test case results
   - Error analysis

### Quality Criteria

- ✓ Homography matrix correctly computed (verified against test points)
- ✓ Image transformation stable for all point configurations
- ✓ No artifacts or numerical errors
- ✓ Handles edge cases gracefully
- ✓ Well-documented code with clear variable names
- ✓ Professional visualization of results

---

## 9. COMMON PITFALLS AND SOLUTIONS

| Pitfall | Cause | Solution |
|---------|-------|----------|
| Black holes in output | Direct forward mapping | Use backward mapping (inverse H) |
| Numerical instability | Ill-conditioned A matrix | Use SVD or normalize data |
| Wrong transformation | Mismatched point correspondence | Carefully verify point order |
| Collinear points error | Points on same line | Check and reject collinear configs |
| Boundary artifacts | Edge pixels poorly defined | Implement proper boundary handling |
| Perspective division by zero | H₃₁x + H₃₂y + H₃₃ ≈ 0 | Check for degenerate cases |
| Blurry output | Nearest-neighbor interpolation | Use bilinear or higher-order |
| Memory issues | Large output image | Limit output size or use sparse methods |



---

**Document Version:** 1.0  
**Last Updated:** April 2026  
**For:** Groups 8 and 20 - Computer Vision Project
