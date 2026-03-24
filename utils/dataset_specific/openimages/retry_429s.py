#!/usr/bin/env python3
"""
Retry OpenImages downloads that failed with HTTP 429 / Too Many Requests.

Uses:
  - metadata CSV to find 429 rows
  - download_log CSV to recover URLs + image_id

Writes:
  - retry log CSV
  - updated metadata CSV (in-place by default)
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils.dataset_specific.openimages.download_openimages import (
    DEFAULT_UA,
    Row,
    build_ssl_context,
    download_one,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retry OpenImages 429s using metadata + download log.")
    p.add_argument(
        "--metadata-csv",
        default="data/cooked/openimages_metadata.csv",
        help="Metadata CSV from initial download.",
    )
    p.add_argument(
        "--metadata-out",
        default="data/cooked/openimages_metadata_retry.csv",
        help="Output metadata CSV with updated rows (used with --no-in-place).",
    )
    p.add_argument(
        "--log-csv",
        default="data/raw/openimages/images/download_log.csv",
        help="Download log CSV from initial download.",
    )
    p.add_argument(
        "--retry-log",
        default="data/raw/openimages/images/retry_429_log.csv",
        help="Retry log CSV output path.",
    )
    p.add_argument(
        "--output-dir",
        default="data/raw/openimages/images",
        help="Directory where images are stored.",
    )
    p.add_argument("--raw-root", default="data/raw", help="Root for source_path relative paths.")
    p.add_argument("--workers", type=int, default=4, help="Number of download threads.")
    p.add_argument("--timeout", type=float, default=15.0, help="Per-request timeout seconds.")
    p.add_argument("--retries", type=int, default=5, help="Retries per URL on failure.")
    p.add_argument("--user-agent", default=DEFAULT_UA, help="User-Agent header.")
    p.add_argument("--max-count", type=int, default=None, help="Stop after this many retries.")
    p.add_argument("--start", type=int, default=0, help="Skip first N retry candidates.")
    p.add_argument("--backoff-base", type=float, default=1.0, help="Base seconds for backoff.")
    p.add_argument("--backoff-max", type=float, default=60.0, help="Max seconds for backoff.")
    p.add_argument("--backoff-jitter", type=float, default=0.2, help="Jitter fraction for backoff.")
    p.add_argument(
        "--respect-retry-after",
        dest="respect_retry_after",
        action="store_true",
        help="Respect Retry-After header (default).",
    )
    p.add_argument(
        "--no-respect-retry-after",
        dest="respect_retry_after",
        action="store_false",
        help="Ignore Retry-After header.",
    )
    p.add_argument(
        "--in-place",
        dest="in_place",
        action="store_true",
        help="Overwrite metadata CSV in place (default).",
    )
    p.add_argument(
        "--no-in-place",
        dest="in_place",
        action="store_false",
        help="Write updated metadata to --metadata-out instead.",
    )
    p.set_defaults(in_place=True, respect_retry_after=True)
    return p.parse_args()


def normalize_path(path_str: str, raw_root: Path) -> str:
    p = Path(path_str)
    raw_root_res = raw_root.resolve()

    if p.is_absolute():
        try:
            return p.resolve().relative_to(raw_root_res).as_posix()
        except Exception:
            return p.as_posix()

    parts = p.parts
    raw_parts = raw_root.parts
    if len(parts) >= len(raw_parts) and parts[: len(raw_parts)] == raw_parts:
        return Path(*parts[len(raw_parts) :]).as_posix()

    return p.as_posix()


def is_429(row: Dict[str, str]) -> bool:
    msg = (row.get("message") or "").lower()
    return ("too many requests" in msg) or ("http 429" in msg) or (" 429" in msg)


def load_log_map(log_csv: Path, raw_root: Path) -> Tuple[Dict[str, Tuple[str, str]], Dict[str, Tuple[str, str]]]:
    by_rel: Dict[str, Tuple[str, str]] = {}
    by_name: Dict[str, Tuple[str, str]] = {}

    if not log_csv.exists():
        return by_rel, by_name

    with log_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            url, image_id, path = row[0], row[1], row[2]
            key = normalize_path(path, raw_root)
            if key not in by_rel:
                by_rel[key] = (url, image_id)

            name = Path(path).name
            if name and name not in by_name:
                by_name[name] = (url, image_id)

    return by_rel, by_name


def main() -> int:
    args = parse_args()
    metadata_csv = Path(args.metadata_csv)
    metadata_out = Path(args.metadata_out)
    log_csv = Path(args.log_csv)
    retry_log = Path(args.retry_log)
    output_dir = Path(args.output_dir)
    raw_root = Path(args.raw_root)

    if args.in_place:
        metadata_out = metadata_csv

    if not metadata_csv.exists():
        raise SystemExit(f"Metadata CSV not found: {metadata_csv}")

    by_rel, by_name = load_log_map(log_csv, raw_root)
    if not by_rel:
        raise SystemExit(f"No URLs found in log: {log_csv}")

    rows: List[Dict[str, str]] = []
    fieldnames: Optional[List[str]] = None
    with metadata_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for r in reader:
            rows.append(r)

    if not fieldnames:
        raise SystemExit("Metadata CSV missing headers.")

    # Build retry items
    retry_items: List[Tuple[int, str, str]] = []
    for idx, row in enumerate(rows):
        if not is_429(row):
            continue
        key = normalize_path(row.get("source_path") or row.get("output_path") or "", raw_root)
        if not key:
            continue
        url_image = by_rel.get(key)
        if url_image is None:
            # fallback by basename
            name = Path(key).name
            url_image = by_name.get(name)
        if url_image is None:
            continue
        url, image_id = url_image
        retry_items.append((idx, url, image_id))

    if args.start:
        retry_items = retry_items[args.start :]
    if args.max_count is not None:
        retry_items = retry_items[: args.max_count]

    if not retry_items:
        print("No 429 rows found to retry.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    retry_log.parent.mkdir(parents=True, exist_ok=True)

    ssl_context = build_ssl_context()

    results_by_index: Dict[int, Tuple[str, str, str, str, Optional[int], Optional[int], str]] = {}

    def submit_task(item: Tuple[int, str, str]):
        idx, url, image_id = item
        row = Row(url=url, size=None, image_id=image_id)
        res = download_one(
            row=row,
            output_dir=output_dir,
            timeout=args.timeout,
            retries=args.retries,
            user_agent=args.user_agent,
            compute_meta=True,
            ssl_context=ssl_context,
            backoff_base=args.backoff_base,
            backoff_max=args.backoff_max,
            backoff_jitter=args.backoff_jitter,
            respect_retry_after=args.respect_retry_after,
        )
        return idx, res

    ok = 0
    err = 0
    with retry_log.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "image_id", "path", "status", "message"])
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(submit_task, item) for item in retry_items]
            for fut in as_completed(futures):
                idx, res = fut.result()
                writer.writerow([res.url, res.image_id, res.path, res.status, res.message])
                msg = res.message
                if res.meta_message:
                    msg = f"{msg}; {res.meta_message}" if msg else res.meta_message
                results_by_index[idx] = (
                    res.status,
                    res.sha256,
                    str(res.src_width) if res.src_width is not None else "",
                    str(res.src_height) if res.src_height is not None else "",
                    msg,
                    str(res.src_width) if res.src_width is not None else "",
                    str(res.src_height) if res.src_height is not None else "",
                )
                if res.status in ("ok", "exists"):
                    ok += 1
                else:
                    err += 1

    # Write updated metadata
    temp_out = metadata_out.with_suffix(metadata_out.suffix + ".tmp") if args.in_place else metadata_out
    with temp_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rows):
            if idx in results_by_index:
                status, sha, sw, sh, msg, ow, oh = results_by_index[idx]
                row["status"] = "written" if status == "ok" else "exists" if status == "exists" else "error"
                row["output_sha256"] = sha
                row["src_width"] = sw
                row["src_height"] = sh
                row["out_width"] = ow
                row["out_height"] = oh
                row["message"] = msg
            writer.writerow(row)

    if args.in_place:
        temp_out.replace(metadata_out)

    print(f"Retried: {len(retry_items)}  ok/exists: {ok}  errors: {err}")
    print(f"Retry log: {retry_log}")
    print(f"Updated metadata: {metadata_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
