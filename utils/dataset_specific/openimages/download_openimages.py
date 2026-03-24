#!/usr/bin/env python3
"""
Download OpenImages-style TSV URL lists into a local folder.

Expected TSV format (no header):
  url <TAB> size_bytes <TAB> id

Some files start with a single header line like "TsvHttpData-1.0".
We skip that line automatically.

Example:
  python3 utils/download_openimages.py \
    --tsv data/raw/openimages/open-images-dataset-train0.tsv \
    --output-dir data/raw/openimages/images \
    --metadata-csv data/cooked/openimages_metadata.csv \
    --workers 16 \
    --max-count 100000
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import re
import ssl
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

try:
    import certifi  # type: ignore
except Exception:
    certifi = None

DEFAULT_UA = "Mozilla/5.0 (compatible; OpenImagesDownloader/1.0)"


@dataclass
class Row:
    url: str
    size: Optional[int]
    image_id: str


@dataclass
class Result:
    url: str
    image_id: str
    path: str
    status: str
    message: str
    sha256: str = ""
    src_width: Optional[int] = None
    src_height: Optional[int] = None
    meta_message: str = ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download OpenImages TSV URLs.")
    p.add_argument(
        "--tsv",
        default="data/raw/openimages/open-images-dataset-train0.tsv",
        help="Path to the TSV file of URLs.",
    )
    p.add_argument(
        "--output-dir",
        default="data/raw/openimages/images",
        help="Directory to write downloaded images.",
    )
    p.add_argument("--start", type=int, default=0, help="Skip first N rows (after header).")
    p.add_argument("--max-count", type=int, default=None, help="Stop after this many downloads.")
    p.add_argument("--workers", type=int, default=16, help="Number of download threads.")
    p.add_argument("--timeout", type=float, default=15.0, help="Per-request timeout seconds.")
    p.add_argument("--retries", type=int, default=2, help="Retries per URL on failure.")
    p.add_argument("--user-agent", default=DEFAULT_UA, help="User-Agent header.")
    p.add_argument(
        "--metadata-csv",
        default="data/cooked/openimages_metadata.csv",
        help="Metadata CSV path (canonicalize-style).",
    )
    p.add_argument(
        "--raw-root",
        default="data/raw",
        help="Root for source_path relative paths in metadata.",
    )
    p.add_argument(
        "--no-metadata",
        action="store_true",
        help="Disable metadata CSV generation.",
    )
    p.add_argument(
        "--log-csv",
        default=None,
        help="Optional CSV path for download log (default: <output-dir>/download_log.csv).",
    )
    p.add_argument(
        "--backoff-base",
        type=float,
        default=0.75,
        help="Base seconds for exponential backoff (default: 0.75).",
    )
    p.add_argument(
        "--backoff-max",
        type=float,
        default=20.0,
        help="Max seconds for exponential backoff (default: 20).",
    )
    p.add_argument(
        "--backoff-jitter",
        type=float,
        default=0.2,
        help="Jitter fraction added to backoff (default: 0.2).",
    )
    p.add_argument(
        "--respect-retry-after",
        action="store_true",
        help="If set, respect Retry-After header for HTTP 429/503.",
    )
    return p.parse_args()


def iter_rows(tsv_path: Path) -> Iterator[Row]:
    with tsv_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        first = True
        for cols in reader:
            if not cols:
                continue
            if first:
                first = False
                if cols[0].startswith("TsvHttpData-"):
                    continue
                if cols[0].lower().startswith("url"):
                    continue
            if len(cols) < 2:
                continue
            url = cols[0].strip()
            if not url:
                continue
            size = None
            if len(cols) >= 2:
                try:
                    size = int(cols[1])
                except ValueError:
                    size = None
            image_id = cols[2].strip() if len(cols) >= 3 else ""
            if not image_id:
                image_id = hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]
            yield Row(url=url, size=size, image_id=image_id)


def safe_id(image_id: str) -> str:
    # OpenImages IDs are often base64 with +/=; make filename-safe.
    if re.fullmatch(r"[A-Za-z0-9+/=]+", image_id):
        image_id = image_id.replace("/", "_").replace("+", "-").rstrip("=")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", image_id)


def guess_ext(url: str) -> str:
    path = urlparse(url).path
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    if not ext or len(ext) > 5:
        return ".jpg"
    return ext


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 256), b""):
            h.update(chunk)
    return h.hexdigest()


def image_size(path: Path) -> Tuple[Optional[int], Optional[int], str]:
    try:
        from PIL import Image  # type: ignore
    except Exception as e:  # noqa: BLE001
        return None, None, f"PIL not available: {e}"
    try:
        with Image.open(path) as im:
            w, h = im.size
        return w, h, ""
    except Exception as e:  # noqa: BLE001
        return None, None, f"unreadable: {e}"


def build_ssl_context() -> Optional[ssl.SSLContext]:
    if certifi is None:
        return None
    try:
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def backoff_delay(attempt: int, base: float, max_delay: float, jitter: float) -> float:
    delay = min(max_delay, base * (2 ** attempt))
    if jitter > 0:
        delay *= 1 + (random.random() * jitter)
    return delay


def download_one(
    row: Row,
    output_dir: Path,
    timeout: float,
    retries: int,
    user_agent: str,
    compute_meta: bool,
    ssl_context: Optional[ssl.SSLContext],
    backoff_base: float,
    backoff_max: float,
    backoff_jitter: float,
    respect_retry_after: bool,
) -> Result:
    out_name = safe_id(row.image_id) + guess_ext(row.url)
    out_path = output_dir / out_name

    if out_path.exists():
        sha = ""
        w = h = None
        meta_msg = ""
        if compute_meta:
            try:
                sha = sha256_file(out_path)
                w, h, meta_msg = image_size(out_path)
            except Exception as e:  # noqa: BLE001
                meta_msg = f"meta_error: {e}"
        if row.size is not None:
            try:
                if out_path.stat().st_size == row.size:
                    return Result(
                        row.url,
                        row.image_id,
                        str(out_path),
                        "exists",
                        "size match",
                        sha256=sha,
                        src_width=w,
                        src_height=h,
                        meta_message=meta_msg,
                    )
            except OSError:
                pass
        else:
            return Result(
                row.url,
                row.image_id,
                str(out_path),
                "exists",
                "file exists",
                sha256=sha,
                src_width=w,
                src_height=h,
                meta_message=meta_msg,
            )

    tmp_path = out_path.with_suffix(out_path.suffix + ".part")

    last_err = ""
    for attempt in range(retries + 1):
        hasher = hashlib.sha256()
        try:
            req = Request(row.url, headers={"User-Agent": user_agent})
            if ssl_context is None:
                resp_ctx = urlopen(req, timeout=timeout)
            else:
                resp_ctx = urlopen(req, timeout=timeout, context=ssl_context)
            with resp_ctx as resp, tmp_path.open("wb") as out_f:
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    hasher.update(chunk)
            if row.size is not None and tmp_path.stat().st_size != row.size:
                last_err = f"size mismatch {tmp_path.stat().st_size} != {row.size}"
                tmp_path.unlink(missing_ok=True)
                continue
            tmp_path.replace(out_path)
            sha = hasher.hexdigest() if compute_meta else ""
            w = h = None
            meta_msg = ""
            if compute_meta:
                w, h, meta_msg = image_size(out_path)
            return Result(
                row.url,
                row.image_id,
                str(out_path),
                "ok",
                "downloaded",
                sha256=sha,
                src_width=w,
                src_height=h,
                meta_message=meta_msg,
            )
        except HTTPError as e:
            last_err = f"HTTP {e.code}: {e.reason}"
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            if e.code in (429, 502, 503, 504) and attempt < retries:
                delay = backoff_delay(attempt, backoff_base, backoff_max, backoff_jitter)
                if respect_retry_after:
                    try:
                        ra = e.headers.get("Retry-After")
                        if ra is not None:
                            delay = max(delay, float(ra))
                    except Exception:
                        pass
                time.sleep(delay)
                continue
            return Result(row.url, row.image_id, str(out_path), "error", last_err)
        except URLError as e:
            last_err = str(e)
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            if attempt < retries:
                delay = backoff_delay(attempt, backoff_base, backoff_max, backoff_jitter)
                time.sleep(delay)
                continue
            return Result(row.url, row.image_id, str(out_path), "error", last_err)
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            if attempt < retries:
                delay = backoff_delay(attempt, backoff_base, backoff_max, backoff_jitter)
                time.sleep(delay)
                continue
            return Result(row.url, row.image_id, str(out_path), "error", last_err)

    return Result(row.url, row.image_id, str(out_path), "error", last_err)


def run_pool(rows: Iterable[Row], workers: int, fn) -> Iterator[Result]:
    with ThreadPoolExecutor(max_workers=workers) as ex:
        pending = set()
        for row in rows:
            while len(pending) >= workers * 2:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    yield fut.result()
            pending.add(ex.submit(fn, row))
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                yield fut.result()


def main() -> int:
    args = parse_args()
    tsv_path = Path(args.tsv)
    output_dir = Path(args.output_dir)
    log_csv = Path(args.log_csv) if args.log_csv else output_dir / "download_log.csv"
    metadata_enabled = not args.no_metadata
    metadata_csv = Path(args.metadata_csv)
    raw_root = Path(args.raw_root)
    ssl_context = build_ssl_context()

    if not tsv_path.exists():
        print(f"TSV not found: {tsv_path}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    log_csv.parent.mkdir(parents=True, exist_ok=True)
    if metadata_enabled:
        metadata_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = iter_rows(tsv_path)
    if args.start:
        for _ in range(args.start):
            try:
                next(rows)
            except StopIteration:
                return 0

    if args.max_count is not None:
        def limited_rows() -> Iterator[Row]:
            count = 0
            for r in rows:
                if count >= args.max_count:
                    break
                count += 1
                yield r
        rows_iter: Iterable[Row] = limited_rows()
    else:
        rows_iter = rows

    def task(row: Row) -> Result:
        return download_one(
            row=row,
            output_dir=output_dir,
            timeout=args.timeout,
            retries=args.retries,
            user_agent=args.user_agent,
            compute_meta=metadata_enabled,
            ssl_context=ssl_context,
            backoff_base=args.backoff_base,
            backoff_max=args.backoff_max,
            backoff_jitter=args.backoff_jitter,
            respect_retry_after=args.respect_retry_after,
        )

    ok = 0
    err = 0
    with log_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "image_id", "path", "status", "message"])
        if metadata_enabled:
            meta_f = metadata_csv.open("w", encoding="utf-8", newline="")
            meta_writer = csv.writer(meta_f)
            meta_writer.writerow(
                [
                    "source_path",
                    "output_path",
                    "output_sha256",
                    "status",
                    "src_width",
                    "src_height",
                    "out_width",
                    "out_height",
                    "message",
                ]
            )
        else:
            meta_f = None
            meta_writer = None

        try:
            for res in run_pool(rows_iter, args.workers, task):
                writer.writerow([res.url, res.image_id, res.path, res.status, res.message])

                if metadata_enabled and meta_writer is not None:
                    try:
                        src_rel = str(Path(res.path).resolve().relative_to(raw_root.resolve()).as_posix())
                    except Exception:
                        src_rel = Path(res.path).as_posix()

                    if res.status == "ok":
                        meta_status = "written"
                    elif res.status == "exists":
                        meta_status = "exists"
                    else:
                        meta_status = "error"

                    msg = res.message
                    if res.meta_message:
                        msg = f"{msg}; {res.meta_message}" if msg else res.meta_message

                    meta_writer.writerow(
                        [
                            src_rel,
                            src_rel,
                            res.sha256,
                            meta_status,
                            res.src_width if res.src_width is not None else "",
                            res.src_height if res.src_height is not None else "",
                            res.src_width if res.src_width is not None else "",
                            res.src_height if res.src_height is not None else "",
                            msg,
                        ]
                    )

                if res.status == "ok" or res.status == "exists":
                    ok += 1
                else:
                    err += 1
        finally:
            if meta_f is not None:
                meta_f.close()

    print(f"Done. ok/exists: {ok}  errors: {err}")
    print(f"Log: {log_csv}")
    if metadata_enabled:
        print(f"Metadata: {metadata_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
