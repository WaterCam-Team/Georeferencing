"""
Calibration photo filter — sensor session tree layout
======================================================
Scans a root directory whose *immediate subdirectories* each hold several
images from one capture (e.g. two RGB/optical frames + one LWIR). Only
optical images are evaluated for checkerboard quality (LWIR is skipped).

Suitable optical frames are copied into a new folder under the root
(default: ``<root>/calib_filtered_suitable/``) with names prefixed by the
source subdirectory to avoid collisions.

Depends on the same helpers/thresholds as ``calib_filter.py``.

Example::

    python calib_filter_sensor_tree.py "Calibration Photos April 22 2026"

    # Board defaults match ``calib_filter.py`` (DEFAULT_BOARD_*). Override if needed:
    python calib_filter_sensor_tree.py ROOT --board-w 8 --board-h 6 --square-m 0.025

If LWIR files are not matched by the default patterns, add e.g.::

    --lwir-pattern lepton --lwir-pattern _thermal_
"""

from __future__ import annotations

import argparse
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from calib_filter import (
    DEFAULT_BOARD_H,
    DEFAULT_BOARD_W,
    DEFAULT_SQUARE_SIZE_M,
    ImageResult,
    Thresholds,
    detect_corners,
    evaluate_set_diversity,
    load_calibration_gray,
    measure_board_geometry,
    measure_brightness,
    measure_sharpness,
    print_summary,
    run_preliminary_calibration,
    write_csv_report,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Default substrings (case-insensitive) on the *stem* to treat as LWIR / thermal.
# Avoid generic "_ir_" / "-ir-" — they match many RGB filenames (e.g. rgb_ir_vis).
DEFAULT_LWIR_SUBSTRINGS = (
    "lwir",
    "thermal",
    "flir",
    "lepton",
)


def _is_lwir(path: Path, patterns: Sequence[str]) -> bool:
    stem = path.stem.lower()
    return any(p.lower() in stem for p in patterns)


def _collect_optical_paths(
    root: Path, lwir_patterns: Sequence[str]
) -> List[Tuple[Path, str, str]]:
    """
    Returns list of (absolute_path, subdir_name, unique_id).

    unique_id is ``subdir_name/filename`` for reporting and maps.
    """
    out: List[Tuple[Path, str, str]] = []
    if not root.is_dir():
        raise NotADirectoryError(str(root))

    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("."):
            continue
        sub = child.name
        imgs = sorted(
            p
            for p in child.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not imgs:
            continue
        optical = [p for p in imgs if not _is_lwir(p, lwir_patterns)]
        for p in optical:
            uid = f"{sub}/{p.name}"
            out.append((p, sub, uid))
    return out


def _evaluate_one_image(
    path: Path,
    image_id: str,
    board_size: Tuple[int, int],
    thresholds: Thresholds,
) -> Tuple[ImageResult, Tuple[int, int] | None]:
    """Run per-image checks; attach ``_src_path`` and ``_image_id`` for downstream."""
    r = ImageResult(filename=image_id)
    setattr(r, "_src_path", str(path))

    gray = load_calibration_gray(path)
    if gray is None:
        r.reject_reasons.append("Could not read image file")
        return r, None
    h_px, w_px = gray.shape
    setattr(r, "_shape", (w_px, h_px))

    r.sharpness = measure_sharpness(gray)
    r.pass_sharpness = r.sharpness >= thresholds.sharpness_min
    if not r.pass_sharpness:
        r.reject_reasons.append(
            f"Too blurry (sharpness={r.sharpness:.0f} < "
            f"{thresholds.sharpness_min:.0f})"
        )

    r.brightness = measure_brightness(gray)
    r.pass_brightness = (
        thresholds.brightness_min <= r.brightness <= thresholds.brightness_max
    )
    if not r.pass_brightness:
        side = (
            "underexposed"
            if r.brightness < thresholds.brightness_min
            else "overexposed"
        )
        r.reject_reasons.append(
            f"Poor exposure — {side} "
            f"(brightness={r.brightness:.0f}, "
            f"range={thresholds.brightness_min:.0f}–"
            f"{thresholds.brightness_max:.0f})"
        )

    corners = detect_corners(gray, board_size)
    r.corners_found = corners is not None
    r.pass_corners = r.corners_found
    if not r.pass_corners:
        r.reject_reasons.append(
            "Checkerboard corners not detected "
            "(check board_w/board_h settings, lighting, or occlusion)"
        )

    if corners is not None:
        r.corners = corners
        (
            r.coverage,
            r.tilt_deg,
            r.board_center_x,
            r.board_center_y,
        ) = measure_board_geometry(corners, w_px, h_px)

        r.pass_coverage = (
            thresholds.coverage_min <= r.coverage <= thresholds.coverage_max
        )
        if not r.pass_coverage:
            side = (
                "too small"
                if r.coverage < thresholds.coverage_min
                else "too large / partially cropped"
            )
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

    status = "✓" if not r.reject_reasons else "✗"
    details = (
        f"sharp={r.sharpness:.0f} bright={r.brightness:.0f} "
        f"cover={r.coverage*100:.1f}% tilt={r.tilt_deg:.1f}°"
    )
    print(f"  {status} {image_id:50s}  {details}")
    for reason in r.reject_reasons:
        print(f"      → {reason}")

    return r, (w_px, h_px)


def _apply_reproj_by_resolution(
    results: List[ImageResult],
    board_size: Tuple[int, int],
    square_size_m: float,
    thresholds: Thresholds,
) -> None:
    """Run preliminary calibration separately per image resolution."""
    by_shape: Dict[Tuple[int, int], List[ImageResult]] = defaultdict(list)
    for r in results:
        shape = getattr(r, "_shape", None)
        if shape is not None:
            by_shape[shape].append(r)

    for shape, group in sorted(by_shape.items()):
        w, h = shape
        print(
            f"\n[REPROJ] Resolution {w}×{h}: {len(group)} image(s) "
            f"— preliminary calibration on this group only."
        )
        run_preliminary_calibration(
            group, board_size, square_size_m, (w, h)
        )

    for r in results:
        if r.reproj_error > thresholds.reproj_error_max:
            r.pass_reproj = False
            r.reject_reasons.append(
                f"High reprojection error ({r.reproj_error:.2f} px > "
                f"{thresholds.reproj_error_max:.2f} px threshold) — "
                f"possible board warp, subtle motion blur, or partial shadow"
            )


def _make_contact_sheet_paths(
    image_paths: List[str],
    results_map: Dict[str, ImageResult],
    output_path: str,
    thumb_size: Tuple[int, int] = (320, 240),
    cols: int = 4,
) -> None:
    """Like ``calib_filter.make_contact_sheet`` but keys results by full path."""
    thumbs = []
    for path in image_paths:
        r = results_map[path]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((thumb_size[1], thumb_size[0], 3), np.uint8)
        else:
            img = cv2.resize(img, thumb_size)

        colour = (0, 200, 0) if r.selected else (0, 0, 220)
        cv2.rectangle(
            img, (0, 0), (thumb_size[0] - 1, thumb_size[1] - 1), colour, 6
        )

        verdict = "PASS" if r.selected else "FAIL"
        cv2.putText(
            img, verdict, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2
        )

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
            cv2.putText(
                img,
                line,
                (6, thumb_size[1] - 12 - j * 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (230, 230, 230),
                1,
            )

        thumbs.append(img)

    rows_needed = (len(thumbs) + cols - 1) // cols
    while len(thumbs) < rows_needed * cols:
        thumbs.append(np.zeros((thumb_size[1], thumb_size[0], 3), np.uint8))

    rows_imgs = []
    for i in range(rows_needed):
        row = np.hstack(thumbs[i * cols : (i + 1) * cols])
        rows_imgs.append(row)

    sheet = np.vstack(rows_imgs)
    cv2.imwrite(output_path, sheet)
    print(f"[SHEET] Contact sheet saved to {output_path}")


def _safe_copy_name(subdir: str, original_name: str) -> str:
    """Flatten subdir into a single filename-safe prefix."""
    safe_sub = subdir.replace(os.sep, "__").replace("/", "__").replace("\\", "__")
    return f"{safe_sub}__{original_name}"


def evaluate_sensor_tree(
    root_dir: str | Path,
    board_w: int = DEFAULT_BOARD_W,
    board_h: int = DEFAULT_BOARD_H,
    square_size_m: float = DEFAULT_SQUARE_SIZE_M,
    output_subdir: str = "calib_filtered_suitable",
    thresholds: Thresholds | None = None,
    lwir_patterns: Sequence[str] | None = None,
) -> List[ImageResult]:
    root = Path(root_dir).expanduser().resolve()
    if thresholds is None:
        thresholds = Thresholds()
    patterns = tuple(lwir_patterns) if lwir_patterns is not None else DEFAULT_LWIR_SUBSTRINGS

    board_size = (board_w, board_h)
    entries = _collect_optical_paths(root, patterns)
    if not entries:
        raise FileNotFoundError(
            f"No subdirectories with optical images under {root} "
            f"(after excluding LWIR patterns: {patterns})"
        )

    print(f"[EVAL] Root: {root}")
    print(f"[EVAL] Optical candidates: {len(entries)} (LWIR skipped by filename)")
    print(
        f"[EVAL] Board size: {board_w}×{board_h} interior corners, "
        f"square={square_size_m * 100:.1f} cm\n"
    )

    results: List[ImageResult] = []
    path_by_id: Dict[str, Path] = {}

    for path, sub, uid in entries:
        path_by_id[uid] = path
        r, _ = _evaluate_one_image(path, uid, board_size, thresholds)
        results.append(r)

    _apply_reproj_by_resolution(results, board_size, square_size_m, thresholds)

    for r in results:
        r.selected = (
            r.pass_corners
            and r.pass_sharpness
            and r.pass_brightness
            and r.pass_coverage
            and r.pass_tilt
            and r.pass_reproj
        )

    diversity_warnings = evaluate_set_diversity(
        results,
        thresholds.diversity_grid_size,
        thresholds.diversity_min_cells,
    )

    out_dir = root / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = [r for r in results if r.selected]
    for r in selected:
        src = Path(getattr(r, "_src_path"))
        sub = src.parent.name
        dst_name = _safe_copy_name(sub, src.name)
        dst = out_dir / dst_name
        shutil.copy2(src, dst)

    print(f"\n[COPY] {len(selected)} image(s) → {out_dir}")

    report_csv = out_dir / "calibration_image_report.csv"
    write_csv_report(results, str(report_csv))

    # Contact sheet: map full source path → result (ids are unique)
    sheet_paths = [str(path_by_id[r.filename]) for r in results]
    results_map = {str(path_by_id[r.filename]): r for r in results}
    _make_contact_sheet_paths(
        sheet_paths, results_map, str(out_dir / "contact_sheet.jpg")
    )

    print_summary(results, diversity_warnings, thresholds)
    return results


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Filter checkerboard calibration photos from a tree of sensor "
            "sessions (one subfolder per capture: optical + LWIR)."
        )
    )
    p.add_argument(
        "root_dir",
        type=str,
        help='Root folder, e.g. "Calibration Photos April 22 2026"',
    )
    p.add_argument(
        "--output-subdir",
        default="calib_filtered_suitable",
        help="Name of folder created under root_dir for copies + reports",
    )
    p.add_argument(
        "--board-w",
        type=int,
        default=DEFAULT_BOARD_W,
        help=f"Interior corners wide (default: {DEFAULT_BOARD_W}, same as calib_filter.py)",
    )
    p.add_argument(
        "--board-h",
        type=int,
        default=DEFAULT_BOARD_H,
        help=f"Interior corners tall (default: {DEFAULT_BOARD_H}, same as calib_filter.py)",
    )
    p.add_argument(
        "--square-m",
        type=float,
        default=DEFAULT_SQUARE_SIZE_M,
        help=(
            "Checker square size in metres "
            f"(default: {DEFAULT_SQUARE_SIZE_M}, same as calib_filter.py)"
        ),
    )
    p.add_argument(
        "--lwir-pattern",
        action="append",
        default=None,
        metavar="SUBSTRING",
        help=(
            "Extra case-insensitive substring matched against the *filename stem* "
            "to skip as LWIR (can be repeated). Defaults: lwir, thermal, flir, "
            "lepton. Add e.g. _ir_ only if your LWIR names need it (optical "
            "names like rgb_ir_vis can false-match _ir_)."
        ),
    )
    args = p.parse_args()

    lwir_patterns = tuple(DEFAULT_LWIR_SUBSTRINGS) + tuple(
        args.lwir_pattern or ()
    )

    evaluate_sensor_tree(
        root_dir=args.root_dir,
        board_w=args.board_w,
        board_h=args.board_h,
        square_size_m=args.square_m,
        output_subdir=args.output_subdir,
        lwir_patterns=lwir_patterns,
    )


if __name__ == "__main__":
    main()
