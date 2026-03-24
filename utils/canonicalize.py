#!/usr/bin/env python3
"""
Canonicalize raw images into a deterministic processed dataset.

Option A policy:
  1) Convert to grayscale
  2) Resize so shortest side == target_size
  3) Center-crop to target_size x target_size

Outputs:
  - Processed images in --output-dir
  - Metadata CSV in --metadata-csv
  - Deterministic filenames based on SHA256 of final image bytes

Example:
  python utils/canonicalize.py \
    --input-root data/raw \
    --output-dir data/processed/images \
    --metadata-csv data/processed/metadata.csv \
    --target-size 100 \
    --exts jpg,jpeg,png,webp,bmp \
    --max-count 20000 \
    --dataset-report data/processed/dataset_report.txt \
    --run-notes "COCO+FFHQ smoke mix; licenses checked in docs/datasets.md"
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from PIL import Image, ImageOps, UnidentifiedImageError

# Pillow safety guard for decompression bombs (optional tuning)
Image.MAX_IMAGE_PIXELS = 150_000_000


DEFAULT_EXTS: tuple[str, ...] = ("jpg", "jpeg", "png", "webp", "bmp")


@dataclass
class Stats:
    scanned: int = 0
    decoded_ok: int = 0
    written: int = 0
    skipped_ext: int = 0
    skipped_unreadable: int = 0
    skipped_too_small: int = 0
    deduped_exact: int = 0
    errors: int = 0


def iter_files(root: Path) -> Iterator[Path]:
    # Deterministic traversal
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        filenames.sort()
        base = Path(dirpath)
        for name in filenames:
            yield base / name


def has_allowed_ext(path: Path, exts: Sequence[str]) -> bool:
    ext = path.suffix.lower().lstrip(".")
    return ext in exts


def canonicalize_image(img: Image.Image, target_size: int) -> Image.Image:
    """
    Option A:
      - grayscale conversion
      - shortest-side resize to target
      - center crop square target x target
    """
    # Respect EXIF orientation first
    img = ImageOps.exif_transpose(img)

    # Convert to grayscale
    img = img.convert("L")

    w, h = img.size
    if w < 2 or h < 2:
        raise ValueError(f"Image too small before processing: {w}x{h}")

    # Scale so shortest side = target_size
    short = min(w, h)
    scale = target_size / short
    new_w = max(target_size, int(round(w * scale)))
    new_h = max(target_size, int(round(h * scale)))

    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Center crop target_size x target_size
    left = (new_w - target_size) // 2
    top = (new_h - target_size) // 2
    right = left + target_size
    bottom = top + target_size
    img = img.crop((left, top, right, bottom))

    if img.size != (target_size, target_size):
        raise RuntimeError(f"Unexpected output size: {img.size}")

    return img


def image_to_png_bytes(img: Image.Image, optimize: bool = True) -> bytes:
    from io import BytesIO

    buf = BytesIO()
    img = img.convert("L")

    # Defensive metadata cleanup: some source files carry malformed/irrelevant
    # ICC/PNG metadata that can cause save/load warnings or errors downstream.
    if hasattr(img, "info") and isinstance(img.info, dict):
        img.info.pop("icc_profile", None)
        img.info.pop("icc", None)

    # Keep save args minimal/portable across Pillow versions.
    img.save(
        buf,
        format="PNG",
        optimize=optimize,
        compress_level=6,
    )
    return buf.getvalue()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def infer_dataset_name(rel_src: str) -> str:
    """Infer top-level dataset bucket from relative source path."""
    parts = Path(rel_src).parts
    return parts[0] if parts else "unknown"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Canonicalize raw images into processed grayscale square images.")
    p.add_argument("--input-root", default="data/raw", help="Root directory to scan for raw images.")
    p.add_argument("--output-dir", default="data/processed/images", help="Directory to write processed images.")
    p.add_argument("--metadata-csv", default="data/processed/metadata.csv", help="Metadata CSV output path.")
    p.add_argument(
        "--dataset-report",
        default="data/processed/dataset_report.txt",
        help="Human-readable due-diligence report path.",
    )
    p.add_argument(
        "--run-notes",
        default="",
        help="Optional free-text notes to include in the report (licenses/sources/mix rationale).",
    )
    p.add_argument("--target-size", type=int, default=100, help="Square target size (default: 100).")
    p.add_argument("--exts", default=",".join(DEFAULT_EXTS), help="Comma-separated allowed input extensions.")
    p.add_argument("--max-count", type=int, default=None, help="Stop after writing this many images.")
    p.add_argument(
        "--min-side",
        type=int,
        default=32,
        help="Skip raw images with min(width,height) < min-side before resize (default: 32).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, overwrite existing output files when hash collision path exists (normally unnecessary).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    input_root = Path(args.input_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    metadata_csv = Path(args.metadata_csv).resolve()
    dataset_report = Path(args.dataset_report).resolve()

    exts = tuple(e.strip().lower().lstrip(".") for e in args.exts.split(",") if e.strip())
    if not input_root.exists():
        raise SystemExit(f"Input root does not exist: {input_root}")
    if args.target_size < 8:
        raise SystemExit("target-size must be >= 8")
    if args.min_side < 1:
        raise SystemExit("min-side must be >= 1")

    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_parent(metadata_csv)
    ensure_parent(dataset_report)

    # Track exact duplicates by output hash
    seen_hashes: set[str] = set()

    stats = Stats()
    dataset_scanned: Counter[str] = Counter()
    dataset_written: Counter[str] = Counter()
    dataset_status: Counter[tuple[str, str]] = Counter()

    # Metadata schema
    fieldnames = [
        "source_path",
        "dataset_name",
        "output_path",
        "output_sha256",
        "status",
        "src_width",
        "src_height",
        "out_width",
        "out_height",
        "message",
    ]

    with metadata_csv.open("w", encoding="utf-8", newline="") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
        writer.writeheader()

        sample_errors: list[str] = []

        for src in iter_files(input_root):
            stats.scanned += 1

            rel_src = src.relative_to(input_root).as_posix()
            dataset_name = infer_dataset_name(rel_src)
            dataset_scanned[dataset_name] += 1

            if not has_allowed_ext(src, exts):
                stats.skipped_ext += 1
                continue

            if args.max_count is not None and stats.written >= args.max_count:
                break

            try:
                with Image.open(src) as im:
                    src_w, src_h = im.size

                    if min(src_w, src_h) < args.min_side:
                        stats.skipped_too_small += 1
                        dataset_status[(dataset_name, "skipped_too_small")] += 1
                        writer.writerow(
                            {
                                "source_path": rel_src,
                                "dataset_name": dataset_name,
                                "output_path": "",
                                "output_sha256": "",
                                "status": "skipped_too_small",
                                "src_width": src_w,
                                "src_height": src_h,
                                "out_width": "",
                                "out_height": "",
                                "message": f"min_side={args.min_side}",
                            }
                        )
                        continue

                    canon = canonicalize_image(im, args.target_size)
                    out_bytes = image_to_png_bytes(canon, optimize=True)
                    out_hash = sha256_hex(out_bytes)

                stats.decoded_ok += 1

                if out_hash in seen_hashes:
                    stats.deduped_exact += 1
                    dataset_status[(dataset_name, "deduped_exact")] += 1
                    writer.writerow(
                        {
                            "source_path": rel_src,
                            "dataset_name": dataset_name,
                            "output_path": "",
                            "output_sha256": out_hash,
                            "status": "deduped_exact",
                            "src_width": src_w,
                            "src_height": src_h,
                            "out_width": args.target_size,
                            "out_height": args.target_size,
                            "message": "same canonical output already written",
                        }
                    )
                    continue

                seen_hashes.add(out_hash)

                out_name = f"{out_hash}.png"
                out_path = output_dir / out_name
                rel_out = out_path.relative_to(output_dir.parent).as_posix() if output_dir.parent in out_path.parents else out_path.name

                if out_path.exists() and not args.overwrite:
                    # Very unlikely unless rerun with same hash; treat as already-present success.
                    status = "exists"
                else:
                    with out_path.open("wb") as f:
                        f.write(out_bytes)
                    status = "written"

                dataset_status[(dataset_name, status)] += 1
                if status == "written":
                    dataset_written[dataset_name] += 1

                stats.written += 1

                writer.writerow(
                    {
                        "source_path": rel_src,
                        "dataset_name": dataset_name,
                        "output_path": rel_out,
                        "output_sha256": out_hash,
                        "status": status,
                        "src_width": src_w,
                        "src_height": src_h,
                        "out_width": args.target_size,
                        "out_height": args.target_size,
                        "message": "",
                    }
                )

            except (UnidentifiedImageError, OSError, ValueError) as e:
                stats.skipped_unreadable += 1
                dataset_status[(dataset_name, "skipped_unreadable")] += 1
                writer.writerow(
                    {
                        "source_path": rel_src,
                        "dataset_name": dataset_name,
                        "output_path": "",
                        "output_sha256": "",
                        "status": "skipped_unreadable",
                        "src_width": "",
                        "src_height": "",
                        "out_width": "",
                        "out_height": "",
                        "message": str(e),
                    }
                )
            except Exception as e:
                stats.errors += 1
                dataset_status[(dataset_name, "error")] += 1
                if len(sample_errors) < 10:
                    sample_errors.append(f"{rel_src}: {repr(e)}")
                writer.writerow(
                    {
                        "source_path": rel_src,
                        "dataset_name": dataset_name,
                        "output_path": "",
                        "output_sha256": "",
                        "status": "error",
                        "src_width": "",
                        "src_height": "",
                        "out_width": "",
                        "out_height": "",
                        "message": repr(e),
                    }
                )

    print("=== Canonicalization Summary ===")
    print(f"Input root:        {input_root}")
    print(f"Output dir:        {output_dir}")
    print(f"Metadata CSV:      {metadata_csv}")
    print(f"Target size:       {args.target_size}x{args.target_size} grayscale")
    print(f"Allowed exts:      {', '.join(exts)}")
    print(f"Scanned files:     {stats.scanned}")
    print(f"Decoded OK:        {stats.decoded_ok}")
    print(f"Written:           {stats.written}")
    print(f"Skipped ext:       {stats.skipped_ext}")
    print(f"Skipped unreadable:{stats.skipped_unreadable}")
    print(f"Skipped too small: {stats.skipped_too_small}")
    print(f"Deduped exact:     {stats.deduped_exact}")
    print(f"Errors:            {stats.errors}")

    if sample_errors:
        print("Sample errors (first up to 10):")
        for msg in sample_errors:
            print(f"  - {msg}")

    # Human-readable due-diligence report
    timestamp = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    status_keys = sorted(set(k for _, k in dataset_status.keys()))

    lines: list[str] = []
    lines.append("MicahNet Dataset Report")
    lines.append("=" * 80)
    lines.append(f"Generated (UTC): {timestamp}")
    lines.append(f"Input root: {input_root}")
    lines.append(f"Output dir: {output_dir}")
    lines.append(f"Metadata CSV: {metadata_csv}")
    lines.append(f"Target format: grayscale PNG {args.target_size}x{args.target_size}")
    lines.append(f"Allowed extensions: {', '.join(exts)}")
    lines.append(f"Min side filter: {args.min_side}")
    lines.append("")
    lines.append("Global Summary")
    lines.append("-" * 80)
    lines.append(f"Scanned files: {stats.scanned}")
    lines.append(f"Decoded OK: {stats.decoded_ok}")
    lines.append(f"Written: {stats.written}")
    lines.append(f"Skipped ext: {stats.skipped_ext}")
    lines.append(f"Skipped unreadable: {stats.skipped_unreadable}")
    lines.append(f"Skipped too small: {stats.skipped_too_small}")
    lines.append(f"Deduped exact: {stats.deduped_exact}")
    lines.append(f"Errors: {stats.errors}")
    lines.append("")
    lines.append("Per-dataset Summary")
    lines.append("-" * 80)

    dataset_names = sorted(set(dataset_scanned.keys()) | set(dataset_written.keys()) | {d for d, _ in dataset_status.keys()})
    if not dataset_names:
        lines.append("(no datasets discovered)")
    else:
        for ds in dataset_names:
            lines.append(f"Dataset: {ds}")
            lines.append(f"  scanned: {dataset_scanned.get(ds, 0)}")
            lines.append(f"  written: {dataset_written.get(ds, 0)}")
            for sk in status_keys:
                cnt = dataset_status.get((ds, sk), 0)
                if cnt:
                    lines.append(f"  {sk}: {cnt}")
            lines.append("")

    lines.append("Due Diligence Notes")
    lines.append("-" * 80)
    lines.append("1) Verify each source dataset license/terms at original source before redistribution.")
    lines.append("2) This report documents preprocessing and dataset mix counts, not legal clearance.")
    lines.append("3) Keep dataset URLs, versions, and retrieval dates in project docs.")
    if args.run_notes.strip():
        lines.append("")
        lines.append("Run Notes")
        lines.append("-" * 80)
        lines.append(args.run_notes.strip())

    dataset_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Dataset report:    {dataset_report}")


if __name__ == "__main__":
    main()