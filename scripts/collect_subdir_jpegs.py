#!/usr/bin/env python3
"""
Copy JPEG files from under a root folder into a new subdirectory (flat copies
with unique names).

By default, only looks at *immediate* child folders of ROOT (not JPEGs sitting
directly in ROOT, and not JPEGs nested deeper than one level). Use flags to
widen the search.

Recognizes ``.jpg`` and ``.jpeg`` (any case).

Example::

    python scripts/collect_subdir_jpegs.py "/path/to/Calibration Photos"
    python scripts/collect_subdir_jpegs.py ./data --output-subdir my_jpgs
    python scripts/collect_subdir_jpegs.py ./data --include-root --recursive
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

JPEG_SUFFIXES = frozenset({".jpg", ".jpeg"})


def _is_jpeg(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in JPEG_SUFFIXES


def _is_under_output(path: Path, out_resolved: Path) -> bool:
    pr = path.resolve()
    return pr == out_resolved or out_resolved in pr.parents


def _diagnose(root: Path, output_subdir: str) -> None:
    """Explain why a run might copy 0 files."""
    root = root.resolve()
    out = root / output_subdir
    print(f"Resolved ROOT: {root}")
    if not root.is_dir():
        print("ROOT is not a directory.")
        return

    children = sorted(
        d
        for d in root.iterdir()
        if d.is_dir() and d.resolve() != out.resolve()
    )
    root_jpegs = [p for p in root.iterdir() if _is_jpeg(p)]
    print(f"Immediate subdirectories: {len(children)}")
    if children:
        for d in children[:12]:
            here = sum(1 for p in d.iterdir() if _is_jpeg(p))
            deep = sum(1 for p in d.rglob("*") if _is_jpeg(p) and p.parent != d)
            print(
                f"  • {d.name}/  — {here} JPEG(s) at top level, "
                f"{deep} more below (any depth)"
            )
        if len(children) > 12:
            print(f"  … ({len(children) - 12} more subdirs)")
    print(f"JPEG files directly in ROOT (not in a subfolder): {len(root_jpegs)}")
    if root_jpegs and len(root_jpegs) <= 8:
        for p in root_jpegs:
            print(f"    {p.name}")
    elif root_jpegs:
        print(f"    (e.g. {root_jpegs[0].name}, …)")

    hints = []
    if root_jpegs:
        hints.append("--include-root copies JPEGs sitting directly in ROOT")
    any_deep = False
    for d in children:
        for p in d.rglob("*"):
            if _is_jpeg(p) and p.parent != d:
                any_deep = True
                break
        if any_deep:
            break
    if any_deep:
        hints.append("--recursive copies JPEGs inside nested folders")
    if not children and not root_jpegs:
        hints.append(
            "no JPEGs (.jpg/.jpeg) found under ROOT — confirm path and extensions"
        )
    if hints:
        print("Try: " + "; ".join(hints) + ".")


def collect_jpegs(
    root: Path,
    output_subdir: str,
    *,
    include_root: bool,
    recursive: bool,
) -> int:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(str(root))

    out = root / output_subdir
    if out.exists():
        raise FileExistsError(
            f"Output already exists: {out}\n"
            "Remove it or choose a different --output-subdir."
        )
    out.mkdir(parents=True)
    out_resolved = out.resolve()

    sources: list[tuple[Path, str]] = []

    if recursive:
        for p in sorted(root.rglob("*")):
            if not _is_jpeg(p):
                continue
            if _is_under_output(p, out_resolved):
                continue
            rel = p.relative_to(root)
            if rel.parts and rel.parts[0] == output_subdir:
                continue
            if len(rel.parts) == 1:
                prefix = "_root"
            else:
                prefix = "__".join(rel.parts[:-1])
            sources.append((p, prefix))
    else:
        if include_root:
            for f in sorted(root.iterdir()):
                if _is_jpeg(f):
                    sources.append((f, "_root"))
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            if child.resolve() == out_resolved:
                continue
            sub = child.name
            for f in sorted(child.iterdir()):
                if _is_jpeg(f):
                    sources.append((f, sub))

    n = 0
    for src, prefix in sources:
        safe = prefix.replace("/", "__").replace("\\", "__")
        dest = out / f"{safe}__{src.name}"
        if dest.exists():
            stem, suf = src.stem, src.suffix
            k = 1
            while dest.exists():
                dest = out / f"{safe}__{stem}_{k}{suf}"
                k += 1
        shutil.copy2(src, dest)
        n += 1

    if n == 0:
        try:
            out.rmdir()
        except OSError:
            pass

    return n


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Copy JPEG files (.jpg / .jpeg) from under ROOT into ROOT/OUTPUT_SUBDIR/. "
            "Default: only files placed directly inside each immediate subfolder of ROOT."
        )
    )
    p.add_argument(
        "root_dir",
        type=str,
        help="Root directory to scan",
    )
    p.add_argument(
        "--output-subdir",
        default="collected_jpgs",
        help="New folder under root_dir (default: collected_jpgs)",
    )
    p.add_argument(
        "--include-root",
        action="store_true",
        help="Also copy JPEGs that sit directly in ROOT (ignored if --recursive)",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Copy every .jpg/.jpeg under ROOT at any depth (skips OUTPUT_SUBDIR)",
    )
    args = p.parse_args()

    root = Path(args.root_dir)
    out = root.expanduser().resolve() / args.output_subdir

    count = collect_jpegs(
        root,
        args.output_subdir,
        include_root=args.include_root,
        recursive=args.recursive,
    )
    if count == 0:
        print("Copied 0 JPEG file(s); empty output folder was removed if possible.")
        print()
        _diagnose(root, args.output_subdir)
    else:
        print(f"Copied {count} JPEG file(s) → {out}")


if __name__ == "__main__":
    main()
