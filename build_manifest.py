#!/usr/bin/env python3
"""
Build an image manifest file from a root directory.

Writes one path per line for files matching extensions (jpg/jpeg/png by default).
By default, paths are written relative to the root for portability.
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Iterable, List


DEFAULT_EXTS = ("jpg", "jpeg", "png")


def iter_image_paths(root: Path, exts: Iterable[str]) -> Iterable[Path]:
    exts = tuple(e.lower().lstrip(".") for e in exts)
    for dirpath, dirnames, filenames in os.walk(root):
        # Deterministic traversal order across filesystems/platforms.
        dirnames.sort()
        filenames.sort()
        for name in filenames:
            ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
            if ext in exts:
                yield Path(dirpath) / name


def write_manifest(
    paths: Iterable[Path],
    output_path: Path,
    root: Path,
    absolute: bool,
    max_count: int | None,
    shuffle: bool,
    seed: int | None,
) -> int:
    if shuffle:
        all_paths: List[Path] = list(paths)
        if seed is not None:
            random.seed(seed)
        random.shuffle(all_paths)
        if max_count is not None:
            all_paths = all_paths[:max_count]
        paths_to_write = all_paths
    else:
        paths_to_write = paths

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for p in paths_to_write:
            if not shuffle and max_count is not None and count >= max_count:
                break
            line = str(p if absolute else p.relative_to(root))
            f.write(line + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an image manifest file.")
    parser.add_argument(
        "--root",
        default="data",
        help="Root directory to scan (default: data).",
    )
    parser.add_argument(
        "--output",
        default="data/manifest.txt",
        help="Output manifest path (default: data/manifest.txt).",
    )
    parser.add_argument(
        "--exts",
        default=",".join(DEFAULT_EXTS),
        help="Comma-separated extensions to include (default: jpg,jpeg,png).",
    )
    parser.add_argument(
        "--absolute",
        action="store_true",
        help="Write absolute paths (default: relative to root).",
    )
    parser.add_argument(
        "--max-count",
        type=int,
        default=None,
        help="Limit number of entries written (default: all).",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle before writing (requires loading paths into memory).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for shuffle.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = Path(args.output).resolve()
    exts = [e.strip() for e in args.exts.split(",") if e.strip()]

    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")

    count = write_manifest(
        paths=iter_image_paths(root, exts),
        output_path=output,
        root=root,
        absolute=args.absolute,
        max_count=args.max_count,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    print(f"Wrote {count} paths to {output}")


if __name__ == "__main__":
    main()
