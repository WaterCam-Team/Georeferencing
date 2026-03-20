"""
Lens Distortion Correction Tool
=================================
Applies calibration data (K, D) from calibration.json to remove lens
distortion from images taken with the same camera.

USAGE MODES:
    1. Single image  — correct one photo and save it
    2. Batch         — correct an entire directory of images
    3. Interactive   — side-by-side comparison with sliders to visualise
                       the effect of distortion correction

DEPENDENCIES:
    pip install opencv-python numpy Pillow

WHAT THIS CORRECTS:
    Radial distortion  — straight lines near frame edges appear curved
                         (barrel distortion is common in short focal-length
                         lenses like the OV5647's 3.6 mm lens)
    Tangential distortion — slight skew from the lens not being perfectly
                            parallel to the sensor plane

WHAT THIS DOES NOT CORRECT:
    Perspective — objects closer to the camera appear larger; this is
                  a property of projective geometry, not a lens defect
    Vignetting  — darkening toward frame edges; a photometric effect,
                  not a geometric one
    Chromatic aberration — colour fringing; requires per-channel correction
"""

import cv2
import numpy as np
import json
import os
import glob
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: LOAD CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

def load_calibration(calib_path: str) -> tuple:
    """
    Load K and D from the calibration.json produced by the calibration script.

    K — 3×3 camera matrix:
        [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]
        fx, fy : focal lengths in pixels
        cx, cy : principal point (optical centre)

    D — distortion coefficients [k1, k2, p1, p2, k3]:
        k1, k2, k3 : radial distortion (k1 dominates for wide-angle lenses)
        p1, p2     : tangential distortion

    Positive k1 → barrel distortion (edges bow outward) — typical for
    wide-angle lenses like the OV5647 at 3.6 mm focal length.
    Negative k1 → pincushion distortion (edges bow inward).
    """
    with open(calib_path) as f:
        d = json.load(f)
    K = np.array(d["K"], dtype=np.float64)
    D = np.array(d["D"], dtype=np.float64)
    rms = d.get("rms", "N/A")
    print(f"[CALIB] Loaded from {calib_path}")
    print(f"  Reprojection error (RMS): {rms} px")
    print(f"  fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")
    print(f"  D = {D.ravel().round(6)}")
    return K, D


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: CORE UNDISTORTION
# ─────────────────────────────────────────────────────────────────────────────

def undistort_image(image: np.ndarray, K: np.ndarray, D: np.ndarray,
                    alpha: float = 0.0) -> tuple:
    """
    Remove lens distortion from an image.

    HOW IT WORKS:
        1. cv2.getOptimalNewCameraMatrix() computes a new camera matrix K_new
           that defines the output image's field of view after correction.
        2. cv2.undistort() builds a remap — for every pixel in the output,
           it computes which pixel in the distorted input it corresponds to,
           then samples that location (with bilinear interpolation).
        3. The result is cropped to the valid region (no black borders).

    THE ALPHA PARAMETER controls the trade-off between:
        alpha = 0.0  (default) — crop all black border pixels introduced
                      by the correction. The output image contains only
                      valid pixels but is slightly smaller and has a
                      narrower field of view than the original.
        alpha = 1.0  — retain all original pixels. The full field of view
                      is preserved but black borders appear in the corners
                      where the remapping has no source data.
        alpha = 0.5  — a middle ground; some border pixels retained,
                      minimal black corners.

    RECOMMENDATION for UFO-Net flood mapping:
        Use alpha=0.0. You want clean rectangular images for the edge-AI
        classifier and georeferencing pipeline. The slight FoV reduction
        (typically a few percent for the OV5647) is negligible.

    RETURNS:
        undistorted image (cropped), new camera matrix K_new
        NOTE: K_new should be used instead of K for any subsequent
        geometric operations (georeferencing, pixel→GPS mapping) because
        the undistortion changes the effective focal length and principal point.
    """
    h, w = image.shape[:2]

    # Compute optimal camera matrix for the output image
    # roi is the valid pixel region after undistortion
    K_new, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=alpha)

    # Remap: correct the image
    undistorted = cv2.undistort(image, K, D, None, K_new)

    # Crop to the valid region (removes black borders when alpha < 1)
    x, y, cw, ch = roi
    undistorted = undistorted[y:y+ch, x:x+cw]

    # Adjust K_new to reflect the crop offset
    K_cropped       = K_new.copy()
    K_cropped[0, 2] -= x   # shift principal point x by crop offset
    K_cropped[1, 2] -= y   # shift principal point y by crop offset

    return undistorted, K_cropped


def build_undistort_maps(K: np.ndarray, D: np.ndarray,
                          img_size: tuple, alpha: float = 0.0) -> tuple:
    """
    Pre-compute undistortion maps for fast repeated application.

    When correcting many images from the same camera (e.g. a flood monitoring
    node capturing one frame per minute), computing the remap arrays once and
    reusing them is significantly faster than calling undistort() each time.

    RETURNS:
        map1, map2  : remap arrays to pass to cv2.remap()
        K_new       : new camera matrix (use this for georeferencing)
        roi         : valid pixel region (x, y, w, h)

    USAGE:
        map1, map2, K_new, roi = build_undistort_maps(K, D, (2592, 1944))
        for frame in video_frames:
            corrected = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
    """
    w, h   = img_size
    K_new, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=alpha)
    map1, map2 = cv2.initUndistortRectifyMap(
        K, D, None, K_new, (w, h), cv2.CV_32FC1
    )
    return map1, map2, K_new, roi


def undistort_with_maps(image: np.ndarray,
                         map1: np.ndarray, map2: np.ndarray,
                         roi: tuple) -> np.ndarray:
    """Apply pre-computed remap arrays. Much faster for batch processing."""
    corrected    = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)
    x, y, cw, ch = roi
    return corrected[y:y+ch, x:x+cw]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: VISUALISE DISTORTION
# ─────────────────────────────────────────────────────────────────────────────

def visualise_distortion(image: np.ndarray, K: np.ndarray,
                          D: np.ndarray) -> None:
    """
    Draw a grid of straight lines over the image in both distorted and
    undistorted form, to make the correction visible.

    WHY A GRID:
        The human eye is poor at noticing gradual barrel distortion in
        natural images — a slight bow in a fence or road edge looks normal.
        A regular grid of lines that should be perfectly straight makes
        the distortion immediately visible.

    USE THIS to verify that:
        1. Your calibration is correct (grid lines are straight after correction)
        2. The distortion model fits your lens
           (if lines are overcorrected or wavy after undistortion,
            your calibration images were insufficient or misconfigured)
    """
    undist, K_new = undistort_image(image, K, D, alpha=0.0)
    h_orig, w_orig = image.shape[:2]
    h_new,  w_new  = undist.shape[:2]

    def draw_grid(img, step=80, colour=(0, 255, 0), thickness=1):
        out = img.copy()
        h, w = out.shape[:2]
        for x in range(0, w, step):
            cv2.line(out, (x, 0), (x, h), colour, thickness)
        for y in range(0, h, step):
            cv2.line(out, (0, y), (w, y), colour, thickness)
        return out

    orig_grid  = draw_grid(image)
    undist_grid = draw_grid(undist)

    # Resize both to same height for side-by-side display
    target_h = 600
    scale_o  = target_h / h_orig
    scale_u  = target_h / h_new
    orig_disp  = cv2.resize(orig_grid,
                             (int(w_orig * scale_o), target_h))
    undist_disp = cv2.resize(undist_grid,
                              (int(w_new  * scale_u), target_h))

    cv2.putText(orig_disp,  "ORIGINAL (distorted)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(undist_disp, "CORRECTED",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    combined = np.hstack([orig_disp, undist_disp])
    cv2.imshow("Distortion Correction — Q to close", combined)
    print("[VIS] Press Q to close the visualisation window.")
    while True:
        if cv2.waitKey(50) & 0xFF in (ord('q'), 27):
            break
    cv2.destroyAllWindows()


def visualise_distortion_field(K: np.ndarray, D: np.ndarray,
                                img_size: tuple,
                                output_path: str = None) -> None:
    """
    Draw a vector field showing how much each pixel moves during undistortion.

    Each arrow shows the displacement of a pixel from its distorted position
    to its corrected position. Arrow length and direction indicate the
    magnitude and direction of the correction.

    Key things to look for:
        - Arrows should point outward from the image centre (barrel distortion)
          or inward (pincushion distortion)
        - Arrows at the very centre should be near-zero (centre is undistorted)
        - Pattern should be radially symmetric for pure radial distortion
        - Asymmetric or irregular arrows suggest tangential distortion or
          a poor calibration
    """
    w, h    = img_size
    step    = max(w, h) // 20    # sample every ~5% of image width
    field   = np.ones((h, w, 3), dtype=np.uint8) * 40   # dark background

    # Sample grid of points
    src_pts = []
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            src_pts.append([[x, y]], )

    src_pts = np.array(src_pts, dtype=np.float32)

    # Undistort the points (no image, just coordinates)
    dst_pts = cv2.undistortPoints(src_pts, K, D, P=K)

    max_disp = 0.0
    arrows   = []
    for src, dst in zip(src_pts, dst_pts):
        x0, y0 = int(src[0][0]), int(src[0][1])
        x1, y1 = int(dst[0][0]), int(dst[0][1])
        disp   = np.sqrt((x1-x0)**2 + (y1-y0)**2)
        max_disp = max(max_disp, disp)
        arrows.append((x0, y0, x1, y1, disp))

    print(f"[FIELD] Max pixel displacement: {max_disp:.1f} px")

    for (x0, y0, x1, y1, disp) in arrows:
        # Colour by magnitude: blue=small, red=large
        ratio = disp / max_disp if max_disp > 0 else 0
        colour = (int(255*(1-ratio)), 50, int(255*ratio))
        cv2.arrowedLine(field, (x0, y0), (x1, y1), colour,
                        1, tipLength=0.3)

    cv2.putText(field, f"Max displacement: {max_disp:.1f} px",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220,220,220), 1)
    cv2.putText(field, f"Image size: {w}x{h}",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220,220,220), 1)

    if output_path:
        cv2.imwrite(output_path, field)
        print(f"[FIELD] Displacement field saved to {output_path}")

    cv2.imshow("Distortion Displacement Field — Q to close", field)
    while True:
        if cv2.waitKey(50) & 0xFF in (ord('q'), 27):
            break
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: SINGLE IMAGE CORRECTION
# ─────────────────────────────────────────────────────────────────────────────

def correct_single(image_path: str, K: np.ndarray, D: np.ndarray,
                   output_path: str = None, alpha: float = 0.0,
                   save_K_new: bool = True) -> tuple:
    """
    Correct distortion in a single image.

    PARAMETERS:
        image_path  : path to the distorted input image
        K, D        : calibration parameters
        output_path : where to save the corrected image; if None, derived
                      from input path (e.g. photo.jpg → photo_undist.jpg)
        alpha       : 0.0 = crop black borders (recommended)
                      1.0 = keep full field of view with black corners
        save_K_new  : if True, save the updated camera matrix alongside
                      the image as a JSON sidecar file. This K_new must
                      be used (instead of K) in the georeferencing script.

    RETURNS:
        corrected image (numpy array), K_new (updated camera matrix)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read {image_path}")

    undist, K_new = undistort_image(img, K, D, alpha=alpha)

    if output_path is None:
        p = Path(image_path)
        output_path = str(p.parent / f"{p.stem}_undist{p.suffix}")

    cv2.imwrite(output_path, undist)
    print(f"[SINGLE] Saved corrected image → {output_path}")
    print(f"  Original size : {img.shape[1]}×{img.shape[0]} px")
    print(f"  Corrected size: {undist.shape[1]}×{undist.shape[0]} px")

    if save_K_new:
        sidecar = output_path.rsplit(".", 1)[0] + "_K_new.json"
        with open(sidecar, "w") as f:
            json.dump({"K_new": K_new.tolist(),
                       "note": ("Use K_new instead of K for georeferencing "
                                "undistorted images.")}, f, indent=2)
        print(f"  Updated camera matrix saved → {sidecar}")

    return undist, K_new


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: BATCH CORRECTION
# ─────────────────────────────────────────────────────────────────────────────

def correct_batch(input_dir: str, output_dir: str,
                  K: np.ndarray, D: np.ndarray,
                  alpha: float = 0.0,
                  extensions: list = None) -> None:
    """
    Correct all images in a directory.

    Pre-computes the undistortion maps once, then applies them to every
    image — much faster than calling undistort() per image.

    Saves a single K_new.json in output_dir for use with the georeferencing
    script. All corrected images share the same K_new because the correction
    depends only on the camera, not the scene.

    PARAMETERS:
        input_dir   : folder of distorted images
        output_dir  : folder to write corrected images (created if needed)
        K, D        : calibration parameters
        alpha       : 0.0 = crop black borders (default)
        extensions  : list of file extensions to process;
                      defaults to [".jpg",".jpeg",".png",".tif",".tiff"]
    """
    os.makedirs(output_dir, exist_ok=True)

    if extensions is None:
        extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff",
                      ".JPG", ".JPEG", ".PNG", ".TIF", ".TIFF"]

    paths = [p for p in glob.glob(os.path.join(input_dir, "*"))
             if Path(p).suffix in extensions]

    if not paths:
        print(f"[BATCH] No images found in {input_dir}")
        return

    print(f"[BATCH] Processing {len(paths)} images...")

    # Read first image to get size for map pre-computation
    first = cv2.imread(paths[0])
    if first is None:
        raise IOError(f"Cannot read first image: {paths[0]}")
    h, w = first.shape[:2]

    map1, map2, K_new, roi = build_undistort_maps(K, D, (w, h), alpha=alpha)

    # Save K_new once for the whole batch
    sidecar = os.path.join(output_dir, "K_new.json")
    with open(sidecar, "w") as f:
        json.dump({"K_new": K_new.tolist(),
                   "roi":   list(roi),
                   "note":  ("Use K_new (not original K) for georeferencing "
                             "undistorted images from this batch.")},
                  f, indent=2)
    print(f"[BATCH] Updated camera matrix → {sidecar}")

    for i, path in enumerate(paths):
        img = cv2.imread(path)
        if img is None:
            print(f"  ✗ {Path(path).name} — could not read")
            continue

        corrected = undistort_with_maps(img, map1, map2, roi)
        out_path  = os.path.join(output_dir, Path(path).name)
        cv2.imwrite(out_path, corrected)
        print(f"  [{i+1}/{len(paths)}] {Path(path).name} → {out_path}")

    print(f"[BATCH] Done. {len(paths)} images corrected.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: INTERACTIVE COMPARISON VIEWER
# ─────────────────────────────────────────────────────────────────────────────

def interactive_viewer(image_path: str, K: np.ndarray, D: np.ndarray) -> None:
    """
    Show original and corrected images side by side.
    Uses keyboard controls instead of a trackbar — works with any
    OpenCV backend (Qt, GTK, or headless fallback).

    CONTROLS:
        +  or  =      increase alpha by 0.05 (show more of the original FoV)
        -             decrease alpha by 0.05 (crop more black border)
        0             reset alpha to 0.0 (maximum crop, recommended)
        1             set alpha to 1.0 (full FoV, black corners visible)
        S             save current corrected image to disk
        Q or ESC      quit

    WHAT ALPHA DOES:
        alpha=0.0  — all black border pixels are cropped out. The output
                     image is slightly smaller but fully valid. Use this
                     for georeferencing and the edge-AI pipeline.
        alpha=1.0  — the full original field of view is preserved but
                     black triangles appear in the corners where the
                     remapping has no source data.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read {image_path}")

    window  = "Distortion Correction | +/- adjust alpha | S=save | Q=quit"
    alpha   = 0.0
    step    = 0.05
    target_h = 560

    def render(alpha: float) -> tuple:
        """Build the side-by-side display frame and return it with undist."""
        undist, _ = undistort_image(img, K, D, alpha=alpha)

        h_o, w_o = img.shape[:2]
        h_u, w_u = undist.shape[:2]

        orig_disp   = cv2.resize(img,   (int(w_o * target_h / h_o), target_h))
        undist_disp = cv2.resize(undist, (int(w_u * target_h / h_u), target_h))

        cv2.putText(orig_disp, "ORIGINAL",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 80, 255), 2)
        cv2.putText(undist_disp,
                    f"CORRECTED  alpha={alpha:.2f}  "
                    f"{undist.shape[1]}x{undist.shape[0]}px",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)

        # Instructions line at the bottom
        info = "+/- : adjust alpha    S : save    Q : quit"
        for disp in (orig_disp, undist_disp):
            cv2.putText(disp, info,
                        (10, target_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # Pad both panels to the same width before hstacking
        max_w = max(orig_disp.shape[1], undist_disp.shape[1])
        def pad_to(im, w):
            gap = w - im.shape[1]
            if gap > 0:
                return np.hstack([im,
                                  np.zeros((im.shape[0], gap, 3), np.uint8)])
            return im

        combined = np.hstack([pad_to(orig_disp, max_w),
                               pad_to(undist_disp, max_w)])
        return combined, undist

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1400, 600)

    frame, current_undist = render(alpha)
    cv2.imshow(window, frame)

    print("[VIEWER] Controls: +/- adjust alpha | 0=min | 1=max | S=save | Q=quit")
    print(f"[VIEWER] Current alpha: {alpha:.2f}")

    while True:
        key = cv2.waitKey(50) & 0xFF

        if key in (ord('q'), 27):                           # Q / ESC — quit
            break
        elif key in (ord('+'), ord('=')):                   # + — more FoV
            alpha = min(1.0, round(alpha + step, 2))
        elif key == ord('-'):                               # - — crop more
            alpha = max(0.0, round(alpha - step, 2))
        elif key == ord('0'):                               # 0 — reset
            alpha = 0.0
        elif key == ord('1'):                               # 1 — full FoV
            alpha = 1.0
        elif key == ord('s'):                               # S — save
            p        = Path(image_path)
            out_path = str(p.parent /
                           f"{p.stem}_undist_a{int(alpha*100):03d}{p.suffix}")
            cv2.imwrite(out_path, current_undist)
            print(f"[VIEWER] Saved → {out_path}")
            continue
        else:
            continue

        print(f"[VIEWER] alpha={alpha:.2f}")
        frame, current_undist = render(alpha)
        cv2.imshow(window, frame)

    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: MAIN — CHOOSE YOUR MODE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── REQUIRED ──────────────────────────────────────────────────────────────
    CALIB_PATH = "./calibration.json"   # from the calibration script

    # ── CHOOSE A MODE ─────────────────────────────────────────────────────────
    MODE = "interactive"
    # Options:
    #   "single"       — correct one image
    #   "batch"        — correct all images in a directory
    #   "visualise"    — side-by-side grid comparison
    #   "field"        — show pixel displacement vector field
    #   "interactive"  — real-time alpha slider comparison

    K, D = load_calibration(CALIB_PATH)

    if MODE == "single":
        IMAGE_PATH  = "./field_image.jpg"
        OUTPUT_PATH = "./field_image_undist.jpg"
        # alpha=0.0: crop black borders (use for georeferencing)
        # alpha=1.0: keep full frame with black corners (use for archiving)
        correct_single(IMAGE_PATH, K, D, output_path=OUTPUT_PATH, alpha=0.0)

    elif MODE == "batch":
        INPUT_DIR  = "./field_images"
        OUTPUT_DIR = "./field_images_undist"
        correct_batch(INPUT_DIR, OUTPUT_DIR, K, D, alpha=0.0)

    elif MODE == "visualise":
        IMAGE_PATH = "./field_image.jpg"
        img = cv2.imread(IMAGE_PATH)
        visualise_distortion(img, K, D)

    elif MODE == "field":
        # Show pixel displacement field for a 2592×1944 image
        IMG_W, IMG_H = 2592, 1944
        visualise_distortion_field(K, D, (IMG_W, IMG_H),
                                   output_path="./distortion_field.jpg")

    elif MODE == "interactive":
        IMAGE_PATH = "./field_image.jpg"
        interactive_viewer(IMAGE_PATH, K, D)
