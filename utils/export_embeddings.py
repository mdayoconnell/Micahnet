#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import tensorflow as tf

# Allow `python evaluation/export_embeddings.py ...` from project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.simclr_model import MicahNetConfig, build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export embeddings for a trained MicahNet run.")
    p.add_argument("--run-dir", required=True, help="Run directory containing run_config.json and weights.")
    p.add_argument("--weights", default=None, help="Weights file name/path (default: final.weights.h5 in run-dir).")
    p.add_argument("--images-dir", default=None, help="Images directory (default: inferred from run config).")
    p.add_argument("--manifest", default=None, help="Manifest path (default: inferred from run config).")
    p.add_argument("--max-images", type=int, default=0, help="Max images to embed (<=0 = all).")
    p.add_argument("--batch-size", type=int, default=256, help="Batch size for embedding extraction.")
    p.add_argument("--use-projection", action="store_true", help="Export projection embeddings z instead of encoder h.")
    p.add_argument("--dtype", choices=("float32", "float16"), default="float32", help="Output embedding dtype.")
    p.add_argument("--log-every", type=int, default=50, help="Progress log frequency in steps (0 disables).")
    p.add_argument(
        "--out-prefix",
        default=None,
        help="Output prefix without extension (default: <run-dir>/embeddings_h or embeddings_z).",
    )
    p.add_argument("--require-gpu", action="store_true", help="Fail fast if TensorFlow cannot see a GPU.")
    return p.parse_args()


def load_run_config(run_dir: Path) -> dict[str, Any]:
    cfg_path = run_dir / "run_config.json"
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_project_root(run_dir: Path, cfg: dict[str, Any]) -> Path:
    cfg_root = cfg.get("project_root")
    if cfg_root:
        root = Path(str(cfg_root)).expanduser()
        if root.is_absolute():
            return root
        return (run_dir.parent.parent / root).resolve()
    return run_dir.parent.parent.resolve()


def resolve_path(raw: Optional[str], bases: list[Path]) -> Optional[Path]:
    if raw is None:
        return None
    p = Path(str(raw)).expanduser()
    if p.is_absolute():
        return p.resolve()
    for base in bases:
        cand = (base / p).resolve()
        if cand.exists():
            return cand
    return (bases[0] / p).resolve()


def runtime_device_report(require_gpu: bool) -> dict[str, Any]:
    physical_cpus = tf.config.list_physical_devices("CPU")
    physical_gpus = tf.config.list_physical_devices("GPU")

    for gpu in physical_gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    logical_gpus = tf.config.list_logical_devices("GPU")
    gpu_rows: list[dict[str, Any]] = []
    for idx, gpu in enumerate(physical_gpus):
        details: dict[str, Any] = {}
        try:
            details = dict(tf.config.experimental.get_device_details(gpu))
        except Exception:
            details = {}

        cc = details.get("compute_capability")
        if isinstance(cc, tuple):
            cc = list(cc)

        gpu_rows.append(
            {
                "index": idx,
                "name": details.get("device_name", gpu.name),
                "physical_name": gpu.name,
                "compute_capability": cc,
            }
        )

    if require_gpu and not logical_gpus:
        raise RuntimeError(
            "--require-gpu was set, but TensorFlow did not detect a usable GPU. "
            "Aborting to avoid an accidental CPU-only run."
        )

    report = {
        "tensorflow_version": tf.__version__,
        "physical_cpu_count": len(physical_cpus),
        "physical_gpu_count": len(physical_gpus),
        "logical_gpu_count": len(logical_gpus),
        "gpus": gpu_rows,
    }

    print(
        f"[runtime] tf={report['tensorflow_version']} "
        f"cpus={report['physical_cpu_count']} "
        f"gpus(physical/logical)={report['physical_gpu_count']}/{report['logical_gpu_count']}"
    )
    if gpu_rows:
        for row in gpu_rows:
            cc_txt = f" cc={row['compute_capability']}" if row["compute_capability"] else ""
            print(f"[runtime] gpu[{row['index']}] {row['name']}{cc_txt}")
    else:
        print("[runtime] No GPU detected.")

    return report


def _resolve_manifest_path(raw_path: str, images_dir: Path, project_root: Path) -> Optional[Path]:
    raw_path = (raw_path or "").strip()
    if not raw_path:
        return None

    p = Path(raw_path)
    if p.is_absolute():
        return p

    candidates = [
        (project_root / p).resolve(),
        (images_dir / p).resolve(),
        (images_dir / p.name).resolve(),
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[-1]


def read_paths_from_manifest(
    manifest_path: Optional[Path],
    images_dir: Path,
    project_root: Path,
    max_images: int,
) -> list[str]:
    paths: list[str] = []
    exts = {".png", ".jpg", ".jpeg"}
    limit = None if max_images <= 0 else max_images

    if manifest_path and manifest_path.exists():
        parsed_as_csv = False
        with manifest_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            headers = [h for h in (reader.fieldnames or []) if h]
            header_map = {h.lower(): h for h in headers}

            path_key = None
            for k in ("output_path", "source_path", "path", "image_path", "file_path", "filepath"):
                if k in header_map:
                    path_key = header_map[k]
                    break
            status_key = header_map.get("status")

            if path_key:
                parsed_as_csv = True
                for r in reader:
                    if status_key:
                        status = (r.get(status_key) or "").strip().lower()
                        if status not in {"written", "exists", "ok", "success", ""}:
                            continue

                    resolved = _resolve_manifest_path(r.get(path_key, ""), images_dir, project_root)
                    if resolved is None:
                        continue

                    if resolved.exists() and resolved.suffix.lower() in exts:
                        paths.append(str(resolved))

                    if limit is not None and len(paths) >= limit:
                        break

        if not parsed_as_csv:
            with manifest_path.open("r", encoding="utf-8") as f:
                for line in f:
                    raw = line.strip()
                    if not raw or raw.startswith("#"):
                        continue

                    resolved = _resolve_manifest_path(raw, images_dir, project_root)
                    if resolved is None:
                        continue

                    if resolved.exists() and resolved.suffix.lower() in exts:
                        paths.append(str(resolved))

                    if limit is not None and len(paths) >= limit:
                        break

    if not paths:
        files = sorted([str(p) for p in images_dir.rglob("*") if p.suffix.lower() in exts])
        paths = files if limit is None else files[:limit]

    if limit is not None and len(paths) > limit:
        paths = paths[:limit]

    return paths


def decode_gray(path: tf.Tensor, img_size: int) -> tf.Tensor:
    b = tf.io.read_file(path)
    img = tf.io.decode_image(b, channels=1, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [img_size, img_size], method=tf.image.ResizeMethod.BILINEAR)
    img.set_shape([img_size, img_size, 1])
    return img


def build_model_from_run(
    run_dir: Path,
    project_root: Path,
    run_cfg: dict[str, Any],
    weights_arg: Optional[str],
) -> tuple[tf.keras.Model, int, Path]:
    img_size = int(run_cfg.get("img_size", 100))

    cfg = MicahNetConfig(
        input_shape=(img_size, img_size, 1),
        width_mult=float(run_cfg.get("width_mult", 1.0)),
        embedding_dim=int(run_cfg.get("embedding_dim", 256)),
        proj_dim=int(run_cfg.get("proj_dim", 128)),
        proj_hidden_dim=int(run_cfg.get("proj_hidden_dim", 512)),
    )

    model = build_model(cfg)

    if weights_arg:
        w_path = resolve_path(weights_arg, [run_dir, project_root, Path.cwd().resolve()])
        if w_path is None:
            raise FileNotFoundError(f"Unable to resolve weights path: {weights_arg}")
    else:
        w_path = (run_dir / "final.weights.h5").resolve()

    if w_path.is_dir():
        w_path = (w_path / "final.weights.h5").resolve()

    if not w_path.exists():
        raise FileNotFoundError(f"Weights not found: {w_path}")

    model.load_weights(str(w_path))
    return model, img_size, w_path


def main() -> int:
    args = parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    run_cfg = load_run_config(run_dir)
    project_root = resolve_project_root(run_dir, run_cfg)

    images_raw = args.images_dir or run_cfg.get("resolved_images_dir") or run_cfg.get("images_dir")
    if not images_raw:
        raise RuntimeError("Could not infer images dir. Pass --images-dir explicitly.")
    images_dir = resolve_path(images_raw, [project_root, run_dir, Path.cwd().resolve()])
    if images_dir is None or not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")

    if args.manifest is not None:
        manifest_raw = args.manifest
    else:
        manifest_raw = run_cfg.get("resolved_manifest") or run_cfg.get("manifest")
    manifest_path = resolve_path(manifest_raw, [project_root, run_dir, Path.cwd().resolve()]) if manifest_raw else None

    runtime_info = runtime_device_report(args.require_gpu)

    model, img_size, weights_path = build_model_from_run(
        run_dir=run_dir,
        project_root=project_root,
        run_cfg=run_cfg,
        weights_arg=args.weights,
    )

    paths = read_paths_from_manifest(manifest_path, images_dir, project_root, args.max_images)
    if not paths:
        raise RuntimeError("No usable images found for embedding export.")

    space = "z" if args.use_projection else "h"
    embed_dim = int(run_cfg.get("proj_dim", 128) if args.use_projection else run_cfg.get("embedding_dim", 256))
    out_prefix = Path(args.out_prefix).expanduser() if args.out_prefix else (run_dir / f"embeddings_{space}")
    if not out_prefix.is_absolute():
        out_prefix = (project_root / out_prefix).resolve()
    else:
        out_prefix = out_prefix.resolve()

    out_npy = Path(f"{out_prefix}.npy")
    out_paths = Path(f"{out_prefix}.paths.txt")
    out_meta = Path(f"{out_prefix}.meta.json")
    out_npy.parent.mkdir(parents=True, exist_ok=True)

    np_dtype = np.float16 if args.dtype == "float16" else np.float32
    print(
        f"[export] images={len(paths)} batch_size={args.batch_size} "
        f"space={space} dim={embed_dim} dtype={args.dtype}"
    )
    print(f"[export] writing embeddings to: {out_npy}")

    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(lambda p: decode_gray(p, img_size), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(args.batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    @tf.function
    def embed_step(batch: tf.Tensor) -> tf.Tensor:
        h, z = model(batch, training=False)
        return z if args.use_projection else h

    embeddings = np.lib.format.open_memmap(
        out_npy,
        mode="w+",
        dtype=np_dtype,
        shape=(len(paths), embed_dim),
    )

    start = time.perf_counter()
    offset = 0
    for step, batch in enumerate(ds, start=1):
        emb = embed_step(batch).numpy()
        if emb.ndim != 2:
            raise RuntimeError(f"Unexpected embedding rank: {emb.ndim}")
        if emb.shape[1] != embed_dim:
            raise RuntimeError(f"Embedding dim mismatch: got {emb.shape[1]}, expected {embed_dim}")

        n = emb.shape[0]
        embeddings[offset : offset + n] = emb.astype(np_dtype, copy=False)
        offset += n

        if args.log_every > 0 and step % args.log_every == 0:
            elapsed = time.perf_counter() - start
            ex_per_sec = (offset / elapsed) if elapsed > 0 else 0.0
            print(f"[embed] step={step} done={offset}/{len(paths)} ex_per_sec={ex_per_sec:.1f}")

    if offset != len(paths):
        raise RuntimeError(f"Wrote {offset} embeddings, expected {len(paths)}")

    del embeddings
    total_seconds = time.perf_counter() - start
    ex_per_sec = (len(paths) / total_seconds) if total_seconds > 0 else 0.0

    with out_paths.open("w", encoding="utf-8") as f:
        for p in paths:
            f.write(p + "\n")

    meta = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_dir": str(run_dir),
        "weights_path": str(weights_path),
        "images_dir": str(images_dir),
        "manifest_path": str(manifest_path) if manifest_path else None,
        "count": len(paths),
        "space": space,
        "embed_dim": embed_dim,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "total_seconds": float(total_seconds),
        "examples_per_sec": float(ex_per_sec),
        "runtime": runtime_info,
        "npy_path": str(out_npy),
        "paths_path": str(out_paths),
    }
    with out_meta.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[export] completed in {total_seconds:.2f}s ({ex_per_sec:.1f} ex/s)")
    print(f"[export] paths -> {out_paths}")
    print(f"[export] metadata -> {out_meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
