"""
Checkerboard Photo Evaluator & Filter
======================================
Scans a directory of checkerboard photos, evaluates each one across
several quality criteria, then copies the suitable ones to a new
directory ready for use by the calibration script.

USAGE:
    python evaluate_calib_images.py

    Edit the CONFIGURATION block at the bottom before running.

DEPENDENCIES:
    pip install opencv-python numpy Pillow scipy

EVALUATION CRITERIA:
    1. Corner detection     — can OpenCV find all board corners?
    2. Sharpness            — Laplacian variance; blurry images give
                              poor sub-pixel corner accuracy
    3. Exposure             — mean brightness; over/underexposed images
                              reduce checkerboard contrast
    4. Board coverage       — what fraction of the image the board fills;
                              a tiny board in the corner contributes little
    5. Board tilt (angle)   — some tilt is essential; flat-on shots all
                              look the same to the solver and add redundancy
                              without improving calibration
    6. Board position       — where the board sits in the frame; good sets
                              have boards spread across the image including
                              near the edges, where distortion is strongest
    7. Per-image reprojection error — a preliminary calibration flags images
                              that are individually inconsistent with the
                              rest of the set (likely due to motion blur,
                              partial occlusion, or board warp)
    8. Set diversity        — after filtering individuals, check that the
                              surviving set covers diverse positions/angles
                              so the solver is well-constrained

OUTPUT:
    - A directory of suitable images ready for the calibration script
    - A CSV report: one row per image with all metric values and pass/fail
    - A contact sheet image showing thumbnails with pass/fail annotations
"""

import cv2
import numpy as np
import os
import glob
import shutil
import csv
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Union


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: THRESHOLDS
# Adjust these if your camera, lighting, or board differs from the defaults.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Thresholds:
    # Sharpness (Laplacian variance). Below this = too blurry.
    # Typical sharp images: >200. Blurry: <80.
    # Lower this if your camera produces inherently soft images.
    sharpness_min: float = 100.0

    # Exposure (mean pixel brightness, 0–255).
    # Below min = underexposed / too dark to see the squares.
    # Above max = overexposed / blown-out contrast.
    brightness_min: float = 40.0
    brightness_max: float = 220.0

    # Board coverage: fraction of image area the board bounding box occupies.
    # Below min = board is too small / far away; contributes little.
    # Above max = board fills too much of the frame; corners may be cut off.
    coverage_min: float = 0.05   # at least 5 % of image area
    coverage_max: float = 0.90   # not more than 90 %

    # Board tilt angle in degrees (angle of the board's long axis vs. horizontal).
    # Images where the board is perfectly flat-on contribute less than tilted ones.
    # Require at least this many degrees of tilt for the image to count as "angled".
    tilt_min_deg: float = 5.0

    # Per-image reprojection error threshold (pixels).
    # Images whose individual RMS error exceeds this after a preliminary
    # calibration run are flagged as outliers and excluded.
    reproj_error_max: float = 1.5

    # Minimum number of valid images required to run the preliminary calibration.
    min_images_for_reproj: int = 6

    # Diversity: the selected set should span at least this many distinct
    # board positions (grid cells) across the image. Image is divided into
    # a grid_size × grid_size grid; each selected image must fall in a
    # different cell for it to count toward coverage.
    diversity_grid_size: int = 3   # 3×3 = 9 cells

    # Minimum number of grid cells that must be covered by the final set.
    diversity_min_cells: int = 4


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: PER-IMAGE RESULT DATA STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ImageResult:
    filename: str
    # Raw metrics
    sharpness: float           = 0.0
    brightness: float          = 0.0
    corners_found: bool        = False
    coverage: float            = 0.0   # fraction of image area
    tilt_deg: float            = 0.0   # board tilt angle
    board_center_x: float      = 0.0   # board center as fraction of image width
    board_center_y: float      = 0.0   # board center as fraction of image height
    reproj_error: float        = 0.0   # set after preliminary calibration
    # Per-criterion pass/fail
    pass_sharpness: bool       = False
    pass_brightness: bool      = False
    pass_corners: bool         = False
    pass_coverage: bool        = False
    pass_tilt: bool            = False
    pass_reproj: bool          = True  # True until proven otherwise
    # Overall verdict
    selected: bool             = False
    reject_reasons: List[str]  = field(default_factory=list)
    # Internal — not written to CSV
    corners: Optional[np.ndarray] = field(default=None, repr=False)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: INDIVIDUAL IMAGE METRICS
# ─────────────────────────────────────────────────────────────────────────────

def measure_sharpness(gray: np.ndarray) -> float:
    """
    Sharpness via Laplacian variance.

    The Laplacian is a second-order derivative filter that responds strongly
    to edges and texture. A sharp image has large, high-contrast gradients
    at the checkerboard edges → high variance of the Laplacian output.
    A blurry image smears those edges → low variance.

    This is the standard focus measure used in autofocus systems.
    It is robust to overall brightness because it measures *change*,
    not absolute level.
    """
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def measure_brightness(gray: np.ndarray) -> float:
    """
    Mean pixel intensity as a simple exposure check.
    Values near 0 = very dark; near 255 = very bright / blown out.
    For a well-exposed checkerboard you want both white and black squares
    clearly distinguishable, typically falling in the range 40–220.
    """
    return float(gray.mean())


def load_calibration_gray(path: Union[str, os.PathLike[str]]) -> Optional[np.ndarray]:
    """
    Load an image as single-channel uint8 for checkerboard detection.

    Matches the historical ``cv2.imread`` + ``BGR2GRAY`` path for ordinary
    8-bit colour JPEGs, but also supports:

    - Single-channel (already gray) inputs
    - BGRA (e.g. PNG with alpha)
    - 16-bit integer images (scaled to 8-bit before grayscale conversion)
    - Paths where ``cv2.imread`` returns None (``np.fromfile`` +
      ``cv2.imdecode`` fallback)
    """
    p = os.fspath(path)
    img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if img is None:
        try:
            buf = np.fromfile(p, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        except OSError:
            return None
    if img is None or img.size == 0:
        return None

    if img.dtype != np.uint8:
        if np.issubdtype(img.dtype, np.integer):
            imax = float(img.max()) if img.max() > 0 else 1.0
            img = np.clip(
                img.astype(np.float64) * (255.0 / imax), 0, 255
            ).astype(np.uint8)
        else:
            img = cv2.normalize(
                img, None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)

    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] == 1:
        return img[:, :, 0]
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return None


def detect_corners(gray: np.ndarray, board_size: tuple) -> Optional[np.ndarray]:
    """
    Attempt to detect the interior corners of the checkerboard.

    Uses cv2.findChessboardCorners followed by sub-pixel refinement
    (cornerSubPix) for the best possible accuracy.

    Returns:
        Refined corner array (N×1×2 float32) if found, else None.

    If detection fails, it may be due to:
        - Blurry image (sharpness check will also catch this)
        - Poor contrast / bad exposure
        - Board partially out of frame
        - Wrong BOARD_W / BOARD_H settings — double-check these first
        - Board warped or creased
    """
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
             cv2.CALIB_CB_NORMALIZE_IMAGE |
             cv2.CALIB_CB_FAST_CHECK)

    found, corners = cv2.findChessboardCorners(gray, board_size, flags)

    if not found:
        return None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners  = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners


def measure_board_geometry(corners: np.ndarray,
                            img_w: int, img_h: int) -> tuple:
    """
    From the detected corners, compute:
        coverage   : bounding box area as a fraction of image area
        tilt_deg   : angle of the board's long axis vs. horizontal
        center_x/y : board centroid as fractions of image dimensions

    The tilt is estimated by fitting a line through the first row of corners
    (the top edge of the board). An angle close to 0° or 90° means the board
    was held flat-on to the camera — acceptable in small numbers, but a set
    consisting entirely of flat-on shots is poorly constrained for lens
    distortion estimation.
    """
    pts = corners.reshape(-1, 2)

    # Bounding box
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    bbox_area     = (x_max - x_min) * (y_max - y_min)
    coverage      = bbox_area / (img_w * img_h)

    # Board centroid
    cx = float(pts[:, 0].mean()) / img_w
    cy = float(pts[:, 1].mean()) / img_h

    # Tilt: angle of vector from first corner to last corner in first row
    # corners are ordered row-by-row from findChessboardCorners
    p1 = pts[0]
    p2 = pts[-1]
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = float(np.degrees(np.arctan2(abs(dy), abs(dx))))
    # Normalize to [0, 45]: 0=horizontal, 45=diagonal, then flat-on maps near 0
    tilt = min(angle, 90.0 - angle)

    return float(coverage), float(tilt), cx, cy


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: PRELIMINARY CALIBRATION FOR REPROJECTION ERROR
# ─────────────────────────────────────────────────────────────────────────────

def run_preliminary_calibration(results: List[ImageResult],
                                 board_size: tuple,
                                 square_size_m: float,
                                 img_size: tuple) -> None:
    """
    Run a full OpenCV calibration on all images that passed the individual
    checks so far. Then compute per-image reprojection error and store it
    back in each ImageResult.

    WHY THIS MATTERS:
        A single image can pass all individual checks (sharp, well-exposed,
        corners found) yet still hurt calibration accuracy if:
        - The board was slightly warped or creased
        - There was micro-motion blur not caught by the Laplacian check
        - The board was partially in shadow on one side
        These images show up as outliers with high per-image reprojection error.

    PER-IMAGE REPROJECTION ERROR:
        After calibration, we project the 3D board points back into the image
        using the computed K, D, R, t and measure the RMS distance between
        the projected points and the detected corners. A well-behaved image
        should have < 0.5 px RMS error; anything above ~1.5 px is suspect.
    """
    # Gather candidates
    candidates = [r for r in results
                  if r.pass_corners and r.pass_sharpness
                  and r.pass_brightness and r.pass_coverage]

    if len(candidates) < 6:
        print(f"[REPROJ] Only {len(candidates)} candidate images — "
              f"skipping preliminary calibration (need ≥ 6).")
        return

    # Build object points (flat board, Z=0)
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0],
                            0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size_m

    objpoints = [objp] * len(candidates)
    imgpoints = [r.corners for r in candidates]

    print(f"[REPROJ] Running preliminary calibration on "
          f"{len(candidates)} images...")

    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None
    )
    print(f"[REPROJ] Overall RMS: {rms:.4f} px")

    # Per-image error
    for i, r in enumerate(candidates):
        proj_pts, _ = cv2.projectPoints(objp, rvecs[i], tvecs[i], K, D)
        diff        = r.corners.reshape(-1, 2) - proj_pts.reshape(-1, 2)
        per_img_rms = float(np.sqrt((diff ** 2).sum(axis=1).mean()))
        r.reproj_error = per_img_rms


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: SET DIVERSITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_set_diversity(results: List[ImageResult],
                            grid_size: int,
                            min_cells: int) -> List[str]:
    """
    Check whether the selected images cover diverse board positions
    across the image frame.

    WHY IT MATTERS:
        Lens distortion is strongest near the edges and corners of the frame.
        If all your calibration images have the board in the center, the solver
        has no data about edge distortion and will extrapolate poorly.

    METHOD:
        Divide the image into a grid_size × grid_size grid.
        Map each selected image to the grid cell its board centroid falls in.
        Count how many distinct cells are occupied.

    Returns a list of warning strings (empty if diversity is acceptable).
    """
    selected = [r for r in results if r.selected]
    if not selected:
        return ["No images selected."]

    occupied = set()
    for r in selected:
        cell_x = int(r.board_center_x * grid_size)
        cell_y = int(r.board_center_y * grid_size)
        # Clamp to grid bounds
        cell_x = min(cell_x, grid_size - 1)
        cell_y = min(cell_y, grid_size - 1)
        occupied.add((cell_x, cell_y))

    warnings = []
    n_cells = len(occupied)
    total   = grid_size * grid_size

    if n_cells < min_cells:
        warnings.append(
            f"Board positions cover only {n_cells}/{total} grid cells "
            f"(minimum recommended: {min_cells}). "
            f"Retake images with the board in different areas of the frame, "
            f"especially near the edges and corners."
        )

    # Warn if no images have the board near the frame edges
    edge_cells = {(x, y)
                  for x in range(grid_size)
                  for y in range(grid_size)
                  if x in (0, grid_size-1) or y in (0, grid_size-1)}
    if not occupied & edge_cells:
        warnings.append(
            "No images have the board near the frame edges. "
            "Edge coverage is important for distortion estimation — "
            "retake some shots with the board near the corners of the frame."
        )

    # Warn if tilt variety is low
    tilts = [r.tilt_deg for r in selected]
    if max(tilts) - min(tilts) < 10.0:
        warnings.append(
            f"Board tilt range is narrow ({min(tilts):.1f}°–{max(tilts):.1f}°). "
            f"Include images with varied board rotation for a well-conditioned solve."
        )

    return warnings


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: CONTACT SHEET (visual summary)
# ─────────────────────────────────────────────────────────────────────────────

def make_contact_sheet(image_paths: List[str],
                        results_map: dict,
                        output_path: str,
                        thumb_size: tuple = (320, 240),
                        cols: int = 4) -> None:
    """
    Produce a contact sheet image: a grid of thumbnails with each image's
    verdict (PASS/FAIL) and key metrics annotated.
    Useful for quickly spotting patterns in what was rejected and why.
    """
    thumbs = []
    for path in image_paths:
        r   = results_map[os.path.basename(path)]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((thumb_size[1], thumb_size[0], 3), np.uint8)
        else:
            img = cv2.resize(img, thumb_size)

        # Colour border: green = selected, red = rejected
        colour = (0, 200, 0) if r.selected else (0, 0, 220)
        cv2.rectangle(img, (0, 0),
                      (thumb_size[0]-1, thumb_size[1]-1), colour, 6)

        # Verdict label
        verdict = "PASS" if r.selected else "FAIL"
        cv2.putText(img, verdict, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)

        # Key metrics in small text
        lines = [
            f"Sharp:{r.sharpness:.0f}",
            f"Bright:{r.brightness:.0f}",
            f"Cover:{r.coverage*100:.1f}%",
            f"Tilt:{r.tilt_deg:.1f}d",
        ]
        if r.reproj_error > 0:
            lines.append(f"Reproj:{r.reproj_error:.2f}px")
        if r.reject_reasons:
            lines.append(r.reject_reasons[0][:28])

        for j, line in enumerate(lines):
            cv2.putText(img, line, (6, thumb_size[1] - 12 - j * 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (230, 230, 230), 1)

        thumbs.append(img)

    # Pad to full grid
    rows_needed = (len(thumbs) + cols - 1) // cols
    while len(thumbs) < rows_needed * cols:
        thumbs.append(np.zeros((thumb_size[1], thumb_size[0], 3), np.uint8))

    rows_imgs = []
    for i in range(rows_needed):
        row = np.hstack(thumbs[i * cols:(i + 1) * cols])
        rows_imgs.append(row)

    sheet = np.vstack(rows_imgs)
    cv2.imwrite(output_path, sheet)
    print(f"[SHEET] Contact sheet saved to {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: REPORT
# ─────────────────────────────────────────────────────────────────────────────

def write_csv_report(results: List[ImageResult], output_path: str) -> None:
    """Write per-image metrics and verdicts to a CSV file."""
    fields = [
        "filename", "selected",
        "sharpness",    "pass_sharpness",
        "brightness",   "pass_brightness",
        "corners_found","pass_corners",
        "coverage",     "pass_coverage",
        "tilt_deg",     "pass_tilt",
        "reproj_error", "pass_reproj",
        "board_center_x", "board_center_y",
        "reject_reasons",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            row = {k: getattr(r, k) for k in fields if k != "reject_reasons"}
            row["reject_reasons"] = "; ".join(r.reject_reasons)
            row["coverage"]       = f"{r.coverage:.4f}"
            row["sharpness"]      = f"{r.sharpness:.1f}"
            row["brightness"]     = f"{r.brightness:.1f}"
            row["tilt_deg"]       = f"{r.tilt_deg:.2f}"
            row["reproj_error"]   = f"{r.reproj_error:.4f}"
            writer.writerow(row)
    print(f"[REPORT] CSV report saved to {output_path}")


def print_summary(results: List[ImageResult],
                  diversity_warnings: List[str],
                  thresholds: Thresholds) -> None:
    """Print a human-readable summary to the terminal."""
    total    = len(results)
    selected = sum(1 for r in results if r.selected)

    print("\n" + "=" * 60)
    print(f"  EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Total images evaluated : {total}")
    print(f"  Selected (suitable)    : {selected}")
    print(f"  Rejected               : {total - selected}")
    print()

    # Rejection breakdown
    reasons = {}
    for r in results:
        for reason in r.reject_reasons:
            reasons[reason] = reasons.get(reason, 0) + 1
    if reasons:
        print("  Rejection reasons:")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"    {count:3d}×  {reason}")
        print()

    if diversity_warnings:
        print("  ⚠  Set diversity warnings:")
        for w in diversity_warnings:
            print(f"     • {w}")
        print()

    if selected < 6:
        print("  ⚠  FEWER THAN 6 IMAGES SELECTED.")
        print("     Calibration requires at least 6 images.")
        print("     Retake more checkerboard photos and re-run this script.")
    elif selected < 15:
        print(f"  ⚠  {selected} images selected. 15–25 is recommended for")
        print("     robust calibration. Consider retaking more shots.")
    else:
        print(f"  ✓  {selected} images selected — sufficient for calibration.")
    print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_directory(input_dir: str,
                        output_dir: str,
                        board_w: int,
                        board_h: int,
                        square_size_m: float,
                        thresholds: Thresholds = None,
                        report_dir: str = None) -> List[ImageResult]:
    """
    Main pipeline. Evaluates all images in input_dir and copies suitable
    ones to output_dir.

    PARAMETERS:
        input_dir     : folder containing checkerboard .jpg / .png images
        output_dir    : folder to copy suitable images into (created if needed)
        board_w       : interior corners horizontally (squares - 1)
        board_h       : interior corners vertically   (squares - 1)
        square_size_m : physical size of one square in meters
        thresholds    : Thresholds instance; uses defaults if None
        report_dir    : where to write CSV report and contact sheet;
                        defaults to input_dir if None

    RETURNS:
        List of ImageResult, one per image processed.
    """
    if thresholds is None:
        thresholds = Thresholds()

    board_size  = (board_w, board_h)
    report_dir  = report_dir or input_dir
    os.makedirs(output_dir, exist_ok=True)

    # ── Find images ───────────────────────────────────────────────────────────
    extensions  = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif"]
    image_paths = []
    for ext in extensions:
        image_paths += glob.glob(os.path.join(input_dir, ext))
        image_paths += glob.glob(os.path.join(input_dir, ext.upper()))
    image_paths = sorted(set(image_paths))

    if not image_paths:
        raise FileNotFoundError(f"No images found in {input_dir}")

    print(f"[EVAL] Found {len(image_paths)} images in {input_dir}")
    print(f"[EVAL] Board size: {board_w}×{board_h} interior corners, "
          f"square={square_size_m*100:.1f} cm\n")

    results    = []
    results_map = {}
    img_size   = None

    # ── Evaluate each image individually ─────────────────────────────────────
    for path in image_paths:
        fname = os.path.basename(path)
        r     = ImageResult(filename=fname)
        gray  = load_calibration_gray(path)

        if gray is None:
            r.reject_reasons.append("Could not read image file")
            results.append(r)
            results_map[fname] = r
            print(f"  ✗ {fname} — unreadable")
            continue

        h_px, w_px = gray.shape
        if img_size is None:
            img_size = (w_px, h_px)

        # 1. Sharpness
        r.sharpness     = measure_sharpness(gray)
        r.pass_sharpness = r.sharpness >= thresholds.sharpness_min
        if not r.pass_sharpness:
            r.reject_reasons.append(
                f"Too blurry (sharpness={r.sharpness:.0f} < "
                f"{thresholds.sharpness_min:.0f})"
            )

        # 2. Exposure
        r.brightness     = measure_brightness(gray)
        r.pass_brightness = (thresholds.brightness_min
                              <= r.brightness
                              <= thresholds.brightness_max)
        if not r.pass_brightness:
            side = ("underexposed" if r.brightness < thresholds.brightness_min
                    else "overexposed")
            r.reject_reasons.append(
                f"Poor exposure — {side} "
                f"(brightness={r.brightness:.0f}, "
                f"range={thresholds.brightness_min:.0f}–"
                f"{thresholds.brightness_max:.0f})"
            )

        # 3. Corner detection
        corners          = detect_corners(gray, board_size)
        r.corners_found  = corners is not None
        r.pass_corners   = r.corners_found
        if not r.pass_corners:
            r.reject_reasons.append(
                "Checkerboard corners not detected "
                "(check board_w/board_h settings, lighting, or occlusion)"
            )

        # 4–5. Board geometry (only if corners were found)
        if corners is not None:
            r.corners = corners
            (r.coverage,
             r.tilt_deg,
             r.board_center_x,
             r.board_center_y) = measure_board_geometry(corners, w_px, h_px)

            r.pass_coverage = (thresholds.coverage_min
                               <= r.coverage
                               <= thresholds.coverage_max)
            if not r.pass_coverage:
                side = ("too small" if r.coverage < thresholds.coverage_min
                        else "too large / partially cropped")
                r.reject_reasons.append(
                    f"Board coverage {side} "
                    f"({r.coverage*100:.1f}%, "
                    f"range={thresholds.coverage_min*100:.0f}–"
                    f"{thresholds.coverage_max*100:.0f}%)"
                )

            r.pass_tilt = r.tilt_deg >= thresholds.tilt_min_deg
            if not r.pass_tilt:
                r.reject_reasons.append(
                    f"Board held too flat-on (tilt={r.tilt_deg:.1f}° < "
                    f"{thresholds.tilt_min_deg:.1f}°)"
                )

        # Status line
        status = ("✓" if not r.reject_reasons else "✗")
        details = (f"sharp={r.sharpness:.0f} bright={r.brightness:.0f} "
                   f"cover={r.coverage*100:.1f}% tilt={r.tilt_deg:.1f}°")
        print(f"  {status} {fname:40s}  {details}")
        if r.reject_reasons:
            for reason in r.reject_reasons:
                print(f"      → {reason}")

        results.append(r)
        results_map[fname] = r

    # ── Preliminary calibration → per-image reprojection error ────────────────
    print()
    run_preliminary_calibration(results, board_size, square_size_m,
                                 img_size or (2592, 1944))

    # Apply reprojection error threshold
    for r in results:
        if r.reproj_error > thresholds.reproj_error_max:
            r.pass_reproj = False
            r.reject_reasons.append(
                f"High reprojection error ({r.reproj_error:.2f} px > "
                f"{thresholds.reproj_error_max:.2f} px threshold) — "
                f"possible board warp, subtle motion blur, or partial shadow"
            )

    # ── Final selection ───────────────────────────────────────────────────────
    for r in results:
        r.selected = (r.pass_corners and r.pass_sharpness and
                      r.pass_brightness and r.pass_coverage and
                      r.pass_tilt and r.pass_reproj)

    # ── Diversity check on selected set ───────────────────────────────────────
    diversity_warnings = evaluate_set_diversity(
        results,
        thresholds.diversity_grid_size,
        thresholds.diversity_min_cells
    )

    # ── Copy selected images to output dir ────────────────────────────────────
    selected = [r for r in results if r.selected]
    for r in selected:
        src = os.path.join(input_dir, r.filename)
        dst = os.path.join(output_dir, r.filename)
        shutil.copy2(src, dst)
    print(f"\n[COPY] {len(selected)} images copied to {output_dir}")

    # ── Write report and contact sheet ───────────────────────────────────────
    report_csv = os.path.join(report_dir, "calibration_image_report.csv")
    write_csv_report(results, report_csv)

    sheet_path = os.path.join(report_dir, "contact_sheet.jpg")
    make_contact_sheet(image_paths, results_map, sheet_path)

    print_summary(results, diversity_warnings, thresholds)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: CONFIGURATION & ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

# Default checkerboard layout (interior corner counts; metres per square).
# Shared with ``calib_filter_sensor_tree`` — change here only.
DEFAULT_BOARD_W = 24
DEFAULT_BOARD_H = 17
DEFAULT_SQUARE_SIZE_M = 0.03

if __name__ == "__main__":

    # ── REQUIRED: set these to match your setup ───────────────────────────────

    INPUT_DIR     = "./calib_images_raw"   # folder with all candidate images
    OUTPUT_DIR    = "./calib_images"       # folder for selected images
                                           # (this is what the calibration
                                           #  script points image_dir at)

    # Match these to your printed checkerboard (defaults below; override if needed):
    # INTERIOR corners = number of squares - 1 in each direction
    BOARD_W       = DEFAULT_BOARD_W
    BOARD_H       = DEFAULT_BOARD_H
    SQUARE_SIZE_M = DEFAULT_SQUARE_SIZE_M

    # ── OPTIONAL: adjust quality thresholds ──────────────────────────────────
    # Defaults work well for the OV5647/Dorhea camera in normal indoor lighting.
    # Loosen sharpness_min if your camera is inherently soft (e.g. wide FOV lens).
    # Tighten reproj_error_max if you want a stricter set.

    thresholds = Thresholds(
        sharpness_min     = 100.0,
        brightness_min    = 40.0,
        brightness_max    = 220.0,
        coverage_min      = 0.05,
        coverage_max      = 0.90,
        tilt_min_deg      = 5.0,
        reproj_error_max  = 1.5,
        diversity_grid_size = 3,
        diversity_min_cells = 4,
    )

    evaluate_directory(
        input_dir     = INPUT_DIR,
        output_dir    = OUTPUT_DIR,
        board_w       = BOARD_W,
        board_h       = BOARD_H,
        square_size_m = SQUARE_SIZE_M,
        thresholds    = thresholds,
    )
