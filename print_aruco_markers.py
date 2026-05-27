"""
Generate printable ArUco marker sheets (DICT_4X4_50) for UFONet GCP workflow.

Output: one PDF per marker, letter-size (8.5×11 in).
The marker is placed in PDF point-space at exactly the requested physical size
(1 pt = 1/72 in), so the printed dimensions are correct regardless of DPI
interpretation — print at Page Scaling = None / Actual Size.
Marker: 15×15 cm centred on page with ID label below.

Usage:
    uv run print_aruco_markers.py                    # IDs 0-7, output ./markers/
    uv run print_aruco_markers.py --ids 0 2 4 6      # specific IDs
    uv run print_aruco_markers.py --size-cm 20       # different physical size
    uv run print_aruco_markers.py --out /tmp/markers
"""

from __future__ import annotations

import argparse
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ── Constants ─────────────────────────────────────────────────────────────────

DICT_NAME   = "DICT_4X4_50"
ARUCO_DICT  = getattr(cv2.aruco, DICT_NAME)
BORDER_BITS = 1
MARKER_CM   = 15.0
DPI         = 300
PAGE_W_IN   = 8.5
PAGE_H_IN   = 11.0


# ── Unit helpers ──────────────────────────────────────────────────────────────

def cm_to_px(cm: float, dpi: int = DPI) -> int:
    return round(cm / 2.54 * dpi)

def in_to_px(inches: float, dpi: int = DPI) -> int:
    return round(inches * dpi)

def cm_to_pt(cm: float) -> float:
    """Centimetres → PDF points (1 pt = 1/72 in)."""
    return cm / 2.54 * 72.0

def in_to_pt(inches: float) -> float:
    return inches * 72.0


# ── Font loader ───────────────────────────────────────────────────────────────

def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "LiberationSans-Regular.ttf",
        "DejaVuSans.ttf",
        "Arial.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


# ── Image generators ──────────────────────────────────────────────────────────

def _marker_image(marker_id: int, dpi: int, marker_cm: float) -> np.ndarray:
    """Return a square grayscale marker bitmap.

    Pixel count controls print resolution only; physical dimensions are
    encoded separately in PDF point-space by _write_pdf.
    """
    size_px = cm_to_px(marker_cm, dpi)
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    out = np.zeros((size_px, size_px), dtype=np.uint8)
    cv2.aruco.generateImageMarker(aruco_dict, marker_id, size_px, out, BORDER_BITS)
    return out


def _label_image(marker_id: int, marker_cm: float, dpi: int) -> np.ndarray:
    """Return a white grayscale bitmap with a centred ID/dict/size label."""
    width_px  = cm_to_px(marker_cm, dpi)
    height_px = cm_to_px(1.2, dpi)
    font_size = round(height_px * 0.45)

    img  = Image.new("L", (width_px, height_px), 255)
    draw = ImageDraw.Draw(img)
    font = _load_font(size=font_size)
    text = f"ArUco ID {marker_id}   {DICT_NAME}   {marker_cm:.0f} × {marker_cm:.0f} cm"
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((width_px - tw) // 2, (height_px - th) // 2), text, fill=0, font=font)
    return np.array(img)


# ── PDF writer ────────────────────────────────────────────────────────────────

def _write_pdf(
    marker_np: np.ndarray,
    label_np: np.ndarray,
    marker_cm: float,
    output_path: str,
    page_w_in: float = PAGE_W_IN,
    page_h_in: float = PAGE_H_IN,
) -> None:
    """Embed marker and label as two image XObjects placed at exact PDF coordinates.

    Physical size is encoded in PDF points (1 pt = 1/72 in), not derived from
    pixel count, so the printed marker is exactly marker_cm wide regardless of
    how the OS or printer driver interprets DPI metadata.
    """
    pt_w = in_to_pt(page_w_in)   # 612.0
    pt_h = in_to_pt(page_h_in)   # 792.0

    marker_pt  = cm_to_pt(marker_cm)   # exact physical width/height of marker
    label_pt_h = cm_to_pt(1.2)        # label strip height
    gap_pt     = cm_to_pt(0.5)        # gap between marker bottom and label top

    # centre the marker+label block vertically; align marker left edge with label
    block_h    = marker_pt + gap_pt + label_pt_h
    block_y    = (pt_h - block_h) / 2        # bottom of block (PDF Y-up)
    marker_x   = (pt_w - marker_pt) / 2
    marker_y   = block_y + label_pt_h + gap_pt
    label_x    = marker_x

    content = (
        # marker: scaled to exact marker_pt × marker_pt at (marker_x, marker_y)
        f"q {marker_pt:.4f} 0 0 {marker_pt:.4f} "
        f"{marker_x:.4f} {marker_y:.4f} cm /Im0 Do Q\n"
        # label: same width, label_pt_h tall, just below marker
        f"q {marker_pt:.4f} 0 0 {label_pt_h:.4f} "
        f"{label_x:.4f} {block_y:.4f} cm /Im1 Do Q\n"
    ).encode()

    def _xobj(arr: np.ndarray) -> bytes:
        h, w = arr.shape
        data = zlib.compress(arr.tobytes(), level=6)
        return (
            f"<< /Type /XObject /Subtype /Image "
            f"/Width {w} /Height {h} "
            f"/ColorSpace /DeviceGray /BitsPerComponent 8 "
            f"/Filter /FlateDecode /Length {len(data)} >>\nstream\n"
        ).encode() + data + b"\nendstream"

    objs: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R "
            f"/MediaBox [0 0 {pt_w:.4f} {pt_h:.4f}] "
            f"/Contents 4 0 R "
            f"/Resources << /XObject << /Im0 5 0 R /Im1 6 0 R >> >> >>"
        ).encode(),
        f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"endstream",
        _xobj(marker_np),
        _xobj(label_np),
    ]

    body = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    xrefs: list[int] = []
    for i, obj_body in enumerate(objs, start=1):
        xrefs.append(len(body))
        body += f"{i} 0 obj\n".encode() + obj_body + b"\nendobj\n"

    xref_pos = len(body)
    n = len(objs) + 1
    xref_lines = f"xref\n0 {n}\n0000000000 65535 f \n"
    for off in xrefs:
        xref_lines += f"{off:010d} 00000 n \n"

    body += xref_lines.encode()
    body += (
        f"trailer\n<< /Size {n} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n"
    ).encode()

    Path(output_path).write_bytes(body)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Generate ArUco marker print sheets (PDF)")
    parser.add_argument(
        "--ids", nargs="+", type=int, default=list(range(8)),
        metavar="N", help="Marker IDs to generate (default: 0-7)",
    )
    parser.add_argument(
        "--size-cm", type=float, default=MARKER_CM,
        metavar="CM", help=f"Physical marker size in cm (default: {MARKER_CM})",
    )
    parser.add_argument(
        "--dpi", type=int, default=DPI,
        help=f"Raster resolution for marker/label bitmaps (default: {DPI})",
    )
    parser.add_argument(
        "--out", type=str, default="./markers",
        metavar="DIR", help="Output directory (default: ./markers)",
    )
    args = parser.parse_args()

    max_cm = min(PAGE_W_IN, PAGE_H_IN) * 2.54 - 2.0  # leave ~1 cm margin each side
    if not (0 < args.size_cm <= max_cm):
        print(
            f"ERROR: --size-cm must be > 0 and <= {max_cm:.1f} cm "
            f"(letter page minus margins)",
            file=sys.stderr,
        )
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for mid in args.ids:
        marker = _marker_image(mid, args.dpi, args.size_cm)
        label  = _label_image(mid, args.size_cm, args.dpi)
        fname  = str(out_dir / f"aruco_{mid:02d}_{DICT_NAME}.pdf")
        _write_pdf(marker, label, args.size_cm, fname)
        print(f"  wrote {fname}")

    print(f"\n{len(args.ids)} marker(s) → {out_dir.resolve()}")
    print(
        f"Print: Page Scaling = None / Actual Size. "
        f"Verify marker = {args.size_cm:.0f} cm with ruler before laminating."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
