"""
Generate printable ArUco marker sheets (DICT_4X4_50) for UFONet GCP workflow.

Output: one PDF per marker, letter-size (8.5×11 in).
Physical dimensions are encoded directly in PDF points (1 pt = 1/72 in),
so printer scale is always 100% regardless of DPI interpretation.
Marker: 15×15 cm centred on page with ID label below.

Usage:
    uv run print_aruco_markers.py                    # IDs 0-7, output ./markers/
    uv run print_aruco_markers.py --ids 0 2 4 6      # specific IDs
    uv run print_aruco_markers.py --size-cm 20       # different physical size
    uv run print_aruco_markers.py --out /tmp/markers
"""

from __future__ import annotations

import argparse
import zlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ── Constants ─────────────────────────────────────────────────────────────────

DPI         = 300
PAGE_W_IN   = 8.5
PAGE_H_IN   = 11.0
MARKER_CM   = 15.0
ARUCO_DICT  = cv2.aruco.DICT_4X4_50
BORDER_BITS = 1


# ── Helpers ───────────────────────────────────────────────────────────────────

def cm_to_px(cm: float, dpi: int = DPI) -> int:
    return round(cm / 2.54 * dpi)


def in_to_px(inches: float, dpi: int = DPI) -> int:
    return round(inches * dpi)


def cm_to_pt(cm: float) -> float:
    """Centimetres → PDF points (1 pt = 1/72 in)."""
    return cm / 2.54 * 72.0


def in_to_pt(inches: float) -> float:
    return inches * 72.0


# ── PDF writer ────────────────────────────────────────────────────────────────

def _write_pdf(
    page_np: np.ndarray,
    output_path: str,
    page_w_in: float = PAGE_W_IN,
    page_h_in: float = PAGE_H_IN,
) -> None:
    """
    Embed a full-page grayscale numpy image into a minimal PDF.

    The MediaBox is set in PDF points so the physical page size is exact.
    The image is scaled to fill the MediaBox exactly, which means the marker's
    physical size = (marker_px / page_px) × physical_page_size.
    """
    img_h, img_w = page_np.shape
    img_bytes = zlib.compress(page_np.tobytes(), level=6)

    pt_w = in_to_pt(page_w_in)   # 612.0 pt
    pt_h = in_to_pt(page_h_in)   # 792.0 pt

    # PDF Y-axis is up; image Y-axis is down → flip Y via negative scale + translate
    content = f"q {pt_w:.4f} 0 0 {-pt_h:.4f} 0 {pt_h:.4f} cm /Im0 Do Q\n".encode()

    objs: list[bytes] = [
        # 1 Catalog
        b"<< /Type /Catalog /Pages 2 0 R >>",
        # 2 Pages
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        # 3 Page
        (
            f"<< /Type /Page /Parent 2 0 R "
            f"/MediaBox [0 0 {pt_w:.4f} {pt_h:.4f}] "
            f"/Contents 4 0 R "
            f"/Resources << /XObject << /Im0 5 0 R >> >> >>"
        ).encode(),
        # 4 Content stream
        f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"endstream",
        # 5 Image XObject
        (
            f"<< /Type /XObject /Subtype /Image "
            f"/Width {img_w} /Height {img_h} "
            f"/ColorSpace /DeviceGray /BitsPerComponent 8 "
            f"/Filter /FlateDecode /Length {len(img_bytes)} >>\nstream\n"
        ).encode()
        + img_bytes
        + b"\nendstream",
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


# ── Marker sheet renderer ─────────────────────────────────────────────────────

def generate_marker_sheet(
    marker_id: int,
    marker_cm: float = MARKER_CM,
    dpi: int = DPI,
) -> np.ndarray:
    """Return a letter-size grayscale numpy array with one ArUco marker centred."""
    page_w = in_to_px(PAGE_W_IN, dpi)
    page_h = in_to_px(PAGE_H_IN, dpi)
    marker_px = cm_to_px(marker_cm, dpi)

    # Generate marker via OpenCV
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    marker_np = np.zeros((marker_px, marker_px), dtype=np.uint8)
    cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_px, marker_np, BORDER_BITS)

    # White letter-size page
    page_pil = Image.new("L", (page_w, page_h), 255)
    marker_pil = Image.fromarray(marker_np, mode="L")

    # Centre marker, shift up slightly to leave room for label
    x0 = (page_w - marker_px) // 2
    y0 = (page_h - marker_px) // 2 - cm_to_px(0.6, dpi)
    page_pil.paste(marker_pil, (x0, y0))

    # Label below marker
    draw = ImageDraw.Draw(page_pil)
    label = f"ArUco ID {marker_id}   DICT_4X4_50   {marker_cm:.0f} × {marker_cm:.0f} cm"
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/liberation/LiberationSans-Regular.ttf", size=36
        )
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    tx = (page_w - tw) // 2
    ty = y0 + marker_px + cm_to_px(0.5, dpi)
    draw.text((tx, ty), label, fill=0, font=font)

    return np.array(page_pil)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
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
        help=f"Raster resolution for text/label rendering (default: {DPI})",
    )
    parser.add_argument(
        "--out", type=str, default="./markers",
        metavar="DIR", help="Output directory (default: ./markers)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for mid in args.ids:
        page_np = generate_marker_sheet(mid, marker_cm=args.size_cm, dpi=args.dpi)
        fname = str(out_dir / f"aruco_{mid:02d}_DICT_4X4_50.pdf")
        _write_pdf(page_np, fname)
        print(f"  wrote {fname}")

    print(f"\n{len(args.ids)} marker(s) → {out_dir.resolve()}")
    print("Print: Page Scaling = None / Actual Size. Verify marker = "
          f"{args.size_cm:.0f} cm with ruler before laminating.")


if __name__ == "__main__":
    main()
