#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
from datetime import datetime
import time
from pathlib import Path
from typing import Any, Iterable, List, Optional

import numpy as np
import tensorflow as tf

from models.simclr_model import MicahNetConfig, build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MicahNet SimCLR smoke test on coco10k.")

    p.add_argument("--project-root", default=".", help="Project root directory.")

    # Data / manifest
    p.add_argument("--images-dir", default="data/cooked/coco10k_100/images", help="Directory of canonicalized images.")
    p.add_argument("--manifest", default="data/cooked/coco10k_100/manifest.csv", help="Manifest CSV path.")
    p.add_argument("--build-manifest-script", default="build_manifest.py", help="Path to build_manifest.py.")    
    p.add_argument("--rebuild-manifest", action="store_true", help="Force rebuild manifest before training.")
    p.add_argument("--max-images", type=int, default=10000, help="Max images to use from manifest (<=0 = all).")
    p.add_argument("--val-fraction", type=float, default=0.05, help="Validation fraction.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")

    # Training
    p.add_argument("--epochs", type=int, default=2, help="Smoke-test epochs.")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    p.add_argument("--temperature", type=float, default=0.2, help="NT-Xent temperature.")
    p.add_argument("--width-mult", type=float, default=1.0, help="Backbone width multiplier.")
    p.add_argument("--embedding-dim", type=int, default=256, help="Backbone embedding dim.")
    p.add_argument("--proj-dim", type=int, default=128, help="Projection dim.")
    p.add_argument("--proj-hidden-dim", type=int, default=512, help="Projection hidden dim.")
    p.add_argument("--log-every", type=int, default=50, help="Print running train loss every N train steps.")
    p.add_argument("--ckpt-every", type=int, default=200, help="Save mid-epoch checkpoint every N train steps.")
    p.add_argument("--max-train-steps", type=int, default=0, help="Optional cap on train steps per epoch (0 = full epoch).")

    # Augment / image
    p.add_argument("--img-size", type=int, default=100, help="Input image size.")
    p.add_argument("--jitter-brightness", type=float, default=0.2)
    p.add_argument("--jitter-contrast", type=float, default=0.2)
    p.add_argument(
        "--rotation-prob",
        type=float,
        default=0.1,
        help="Probability of applying random 90-degree rotation per view.",
    )
    p.add_argument(
        "--random-rotation",
        dest="random_rotation",
        action="store_true",
        help="Enable random rotation augmentation (default: enabled).",
    )
    p.add_argument(
        "--no-random-rotation",
        dest="random_rotation",
        action="store_false",
        help="Disable random rotation augmentation.",
    )
    p.set_defaults(random_rotation=True)

    # IO
    p.add_argument("--weights-dir", default="weights", help="Directory to save weights/checkpoints.")
    p.add_argument("--run-name", default=None, help="Optional run name.")
    p.add_argument("--require-gpu", action="store_true", help="Fail fast if TensorFlow cannot see a GPU.")

    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def runtime_device_report(require_gpu: bool) -> dict[str, Any]:
    physical_cpus = tf.config.list_physical_devices("CPU")
    physical_gpus = tf.config.list_physical_devices("GPU")

    for gpu in physical_gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            # Safe best-effort only; this can fail after runtime init.
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


def maybe_build_manifest(project_root: Path, images_dir: Path, manifest_path: Path, script_path: Path, force: bool) -> None:
    if manifest_path.exists() and not force:
        return

    if not script_path.exists():
        raise FileNotFoundError(
            f"build_manifest script not found: {script_path}. "
            "Either create manifest manually or pass correct --build-manifest-script"
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Try common CLI forms in case your build_manifest CLI changed
    candidate_cmds = [
    [
        "python", str(script_path),
        "--root", str(images_dir),
        "--output", str(manifest_path),
        "--exts", "png,jpg,jpeg",
        "--max-count", str(10_000_000),
        "--absolute",
    ],
    ["python", str(script_path), "--input-dir", str(images_dir), "--output-csv", str(manifest_path)],
    ["python", str(script_path), "--images-dir", str(images_dir), "--metadata-csv", str(manifest_path)],
    ["python", str(script_path), "--input-root", str(images_dir), "--metadata-csv", str(manifest_path)],
    ]

    last_err = ""
    for cmd in candidate_cmds:
        proc = subprocess.run(cmd, cwd=str(project_root), capture_output=True, text=True)
        if proc.returncode == 0 and manifest_path.exists():
            return
        last_err = (
            f"cmd: {' '.join(cmd)}\n"
            f"returncode: {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}\n"
        )

    raise RuntimeError(
        "Failed to build manifest with known CLI patterns. "
        "Run build_manifest manually, then rerun main without --rebuild-manifest.\n\n"
        f"Last attempt:\n{last_err}"
    )


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


def read_paths_from_manifest(manifest_path: Path, images_dir: Path, project_root: Path, max_images: int) -> List[str]:
    paths: List[str] = []
    exts = {".png", ".jpg", ".jpeg"}
    limit = None if max_images <= 0 else max_images

    if manifest_path.exists():
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


def split_train_val(paths: List[str], val_fraction: float, seed: int) -> tuple[List[str], List[str]]:
    rng = random.Random(seed)
    idx = list(range(len(paths)))
    rng.shuffle(idx)

    n_val = int(len(paths) * val_fraction)
    n_val = max(1, n_val) if len(paths) > 1 else 0
    val_idx = set(idx[:n_val])

    train = [p for i, p in enumerate(paths) if i not in val_idx]
    val = [p for i, p in enumerate(paths) if i in val_idx]
    return train, val


def decode_gray(path: tf.Tensor, img_size: int) -> tf.Tensor:
    b = tf.io.read_file(path)
    img = tf.io.decode_image(b, channels=1, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [img_size, img_size], method=tf.image.ResizeMethod.BILINEAR)
    img.set_shape([img_size, img_size, 1])
    return img


def _gaussian_kernel2d(ksize: int, sigma: tf.Tensor) -> tf.Tensor:
    """Return [ksize, ksize, 1, 1] gaussian kernel for depthwise conv."""
    radius = ksize // 2
    xs = tf.cast(tf.range(-radius, radius + 1), tf.float32)
    sigma = tf.maximum(tf.cast(sigma, tf.float32), 1e-2)
    g = tf.exp(-0.5 * tf.square(xs / sigma))
    g = g / tf.reduce_sum(g)
    k2 = tf.tensordot(g, g, axes=0)  # [ksize, ksize]
    return tf.reshape(k2, [ksize, ksize, 1, 1])

def _maybe_gaussian_blur(x: tf.Tensor, p: float = 0.5, ksize: int = 9,
                        sigma_min: float = 0.1, sigma_max: float = 1.5) -> tf.Tensor:
    apply_blur = tf.random.uniform([]) < p

    def _do_blur() -> tf.Tensor:
        sigma = tf.random.uniform([], sigma_min, sigma_max)
        kernel = _gaussian_kernel2d(ksize, sigma)
        x4 = tf.expand_dims(x, axis=0)  # [1,H,W,C]
        x4 = tf.nn.depthwise_conv2d(x4, kernel, strides=[1, 1, 1, 1], padding="SAME")
        return tf.squeeze(x4, axis=0)

    return tf.cond(apply_blur, _do_blur, lambda: x)

def _maybe_add_noise(x: tf.Tensor, p: float = 0.2,
                     std_min: float = 0.01, std_max: float = 0.05) -> tf.Tensor:
    apply_noise = tf.random.uniform([]) < p

    def _do_noise() -> tf.Tensor:
        std = tf.random.uniform([], std_min, std_max)
        return x + tf.random.normal(tf.shape(x), mean=0.0, stddev=std, dtype=x.dtype)

    return tf.cond(apply_noise, _do_noise, lambda: x)

def _maybe_cutout(x: tf.Tensor, p: float = 0.25,
                  frac_min: float = 0.08, frac_max: float = 0.25,
                  fill: float = 0.0) -> tf.Tensor:
    apply_cutout = tf.random.uniform([]) < p

    def _do_cutout() -> tf.Tensor:
        h = tf.shape(x)[0]
        w = tf.shape(x)[1]
        c = tf.shape(x)[2]

        frac = tf.random.uniform([], frac_min, frac_max)
        cut_h = tf.maximum(1, tf.cast(tf.cast(h, tf.float32) * frac, tf.int32))
        cut_w = tf.maximum(1, tf.cast(tf.cast(w, tf.float32) * frac, tf.int32))

        top = tf.random.uniform([], 0, tf.maximum(1, h - cut_h + 1), dtype=tf.int32)
        left = tf.random.uniform([], 0, tf.maximum(1, w - cut_w + 1), dtype=tf.int32)

        box = tf.ones([cut_h, cut_w, c], dtype=x.dtype)
        box = tf.image.pad_to_bounding_box(box, top, left, h, w)

        fill_tensor = tf.cast(fill, x.dtype)
        return x * (1.0 - box) + fill_tensor * box

    return tf.cond(apply_cutout, _do_cutout, lambda: x)

def _maybe_random_rot90(x: tf.Tensor, p: float = 0.1) -> tf.Tensor:
    apply_rot = tf.random.uniform([]) < p

    def _do_rot() -> tf.Tensor:
        k = tf.random.uniform([], minval=1, maxval=4, dtype=tf.int32)
        return tf.image.rot90(x, k=k)

    return tf.cond(apply_rot, _do_rot, lambda: x)


def augment_view(
    x: tf.Tensor,
    bright: float,
    contrast: float,
    random_rotation: bool,
    rotation_prob: float,
) -> tf.Tensor:
    """SimCLR-style augmentations for grayscale images in [0,1]."""
    x = tf.image.random_flip_left_right(x)

    if random_rotation and rotation_prob > 0.0:
        x = _maybe_random_rot90(x, p=rotation_prob)

    # stronger random crop + resize back (key)
    h = tf.shape(x)[0]
    w = tf.shape(x)[1]
    c = tf.shape(x)[2]
    scale = tf.random.uniform([], 0.6, 1.0)  # was 0.8..1.0
    crop_h = tf.maximum(1, tf.cast(tf.cast(h, tf.float32) * scale, tf.int32))
    crop_w = tf.maximum(1, tf.cast(tf.cast(w, tf.float32) * scale, tf.int32))
    x = tf.image.random_crop(x, [crop_h, crop_w, c])
    x = tf.image.resize(x, [h, w], method=tf.image.ResizeMethod.BILINEAR)

    # photometric jitter
    bright_delta = float(bright)
    contrast_lower = max(0.01, 1.0 - float(contrast))
    contrast_upper = 1.0 + float(contrast)
    x = tf.image.random_brightness(x, max_delta=bright_delta)
    x = tf.image.random_contrast(x, lower=contrast_lower, upper=contrast_upper)

    # kill texture shortcuts
    x = _maybe_gaussian_blur(x, p=0.5, ksize=9, sigma_min=0.1, sigma_max=1.5)
    x = _maybe_add_noise(x, p=0.2, std_min=0.01, std_max=0.05)

    # encourage part-based features
    x = _maybe_cutout(x, p=0.25, frac_min=0.08, frac_max=0.25, fill=0.0)

    return tf.clip_by_value(x, 0.0, 1.0)


def make_ds(
    paths: List[str],
    batch_size: int,
    training: bool,
    img_size: int,
    bright: float,
    contrast: float,
    random_rotation: bool,
    rotation_prob: float,
    seed: int
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(paths)
    if training:
        ds = ds.shuffle(buffer_size=min(len(paths), 4096), seed=seed, reshuffle_each_iteration=True)

    def _map(path):
        x = decode_gray(path, img_size)
        if training:
            v1 = augment_view(
                x,
                bright=bright,
                contrast=contrast,
                random_rotation=random_rotation,
                rotation_prob=rotation_prob,
            )
            v2 = augment_view(
                x,
                bright=bright,
                contrast=contrast,
                random_rotation=random_rotation,
                rotation_prob=rotation_prob,
            )
        else:
            v1 = x
            v2 = x
        return v1, v2

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=training)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def nt_xent_loss(z1: tf.Tensor, z2: tf.Tensor, temperature: float) -> tf.Tensor:
    n = tf.shape(z1)[0]
    z1 = tf.math.l2_normalize(z1, axis=1)
    z2 = tf.math.l2_normalize(z2, axis=1)
    z = tf.concat([z1, z2], axis=0)  # [2N, D]

    logits = tf.matmul(z, z, transpose_b=True) / temperature
    mask = tf.eye(2 * n, dtype=logits.dtype)
    logits = logits * (1.0 - mask) + (-1e9) * mask

    labels = tf.concat([tf.range(n, 2 * n), tf.range(0, n)], axis=0)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    return tf.reduce_mean(loss)


def main() -> int:
    args = parse_args()

    if not 0.0 <= args.rotation_prob <= 1.0:
        raise ValueError(f"--rotation-prob must be in [0, 1], got {args.rotation_prob}")

    project_root = Path(args.project_root).resolve()
    images_dir = (project_root / args.images_dir).resolve()
    manifest_path = (project_root / args.manifest).resolve()
    build_manifest_script = (project_root / args.build_manifest_script).resolve()

    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")

    runtime_info = runtime_device_report(args.require_gpu)
    set_seed(args.seed)

    maybe_build_manifest(
        project_root=project_root,
        images_dir=images_dir,
        manifest_path=manifest_path,
        script_path=build_manifest_script,
        force=args.rebuild_manifest,
    )

    paths = read_paths_from_manifest(manifest_path, images_dir, project_root, args.max_images)
    if len(paths) < 64:
        raise RuntimeError(f"Too few usable images ({len(paths)}).")

    train_paths, val_paths = split_train_val(paths, args.val_fraction, args.seed)
    print(f"Loaded {len(paths)} images (train={len(train_paths)}, val={len(val_paths)}) from {images_dir}")

    run_name = args.run_name or datetime.now().strftime("smoke_%Y%m%d_%H%M%S")
    run_dir = (project_root / args.weights_dir / run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    with (run_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                **vars(args),
                "resolved_images_dir": str(images_dir),
                "resolved_manifest": str(manifest_path),
                "train_count": len(train_paths),
                "val_count": len(val_paths),
                "runtime": runtime_info,
            },
            f,
            indent=2,
        )

    train_ds = make_ds(
        train_paths, args.batch_size, True, args.img_size,
        args.jitter_brightness, args.jitter_contrast,
        args.random_rotation, args.rotation_prob, args.seed
    )
    val_ds = make_ds(
        val_paths, args.batch_size, False, args.img_size,
        args.jitter_brightness, args.jitter_contrast,
        args.random_rotation, args.rotation_prob, args.seed
    )

    cfg = MicahNetConfig(
        input_shape=(args.img_size, args.img_size, 1),
        width_mult=args.width_mult,
        embedding_dim=args.embedding_dim,
        proj_dim=args.proj_dim,
        proj_hidden_dim=args.proj_hidden_dim,
    )
    model = build_model(cfg)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    train_loss = tf.keras.metrics.Mean()
    val_loss = tf.keras.metrics.Mean()

    @tf.function
    def train_step(x1, x2):
        with tf.GradientTape() as tape:
            _, z1 = model(x1, training=True)
            _, z2 = model(x2, training=True)
            loss = nt_xent_loss(z1, z2, args.temperature)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss.update_state(loss)

    @tf.function
    def eval_step(x1, x2):
        _, z1 = model(x1, training=False)
        _, z2 = model(x2, training=False)
        loss = nt_xent_loss(z1, z2, args.temperature)
        val_loss.update_state(loss)

    def save_ckpt(tag: str) -> None:
        ckpt_path = run_dir / f"{tag}.weights.h5"
        model.save_weights(str(ckpt_path))
        print(f"[checkpoint] saved -> {ckpt_path}")

    history = []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        epoch_t0 = time.perf_counter()
        train_loss.reset_state()
        val_loss.reset_state()
        epoch_train_loss = tf.keras.metrics.Mean()

        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        for step_idx, (x1, x2) in enumerate(train_ds, start=1):
            train_step(x1, x2)
            batch_loss = float(train_loss.result().numpy())
            epoch_train_loss.update_state(batch_loss)
            global_step += 1

            if args.log_every > 0 and (step_idx % args.log_every == 0):
                print(
                    f"[train] epoch={epoch} step={step_idx} global_step={global_step} "
                    f"running_loss={float(epoch_train_loss.result().numpy()):.4f}"
                )

            if args.ckpt_every > 0 and (global_step % args.ckpt_every == 0):
                save_ckpt(f"step_{global_step:07d}")

            if args.max_train_steps > 0 and step_idx >= args.max_train_steps:
                print(f"[train] reached max_train_steps={args.max_train_steps} for epoch {epoch}")
                break

        val_batches = 0
        val_examples = 0
        for x1, x2 in val_ds:
            eval_step(x1, x2)
            val_batches += 1
            val_examples += int(x1.shape[0]) if x1.shape[0] is not None else int(tf.shape(x1)[0].numpy())

        epoch_seconds = time.perf_counter() - epoch_t0
        train_examples = len(train_paths)
        train_ex_per_sec = (train_examples / epoch_seconds) if epoch_seconds > 0 else 0.0

        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": float(train_loss.result().numpy()),
            "val_loss": float(val_loss.result().numpy()),
            "val_batches": val_batches,
            "epoch_seconds": float(epoch_seconds),
            "train_examples": int(train_examples),
            "val_examples": int(val_examples),
            "train_examples_per_sec": float(train_ex_per_sec),
        }
        history.append(row)

        print(
            f"[epoch-summary] epoch={epoch} global_step={global_step} "
            f"train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} "
            f"val_batches={val_batches} val_examples={val_examples} "
            f"epoch_seconds={row['epoch_seconds']:.2f} train_ex_per_sec={row['train_examples_per_sec']:.1f}"
        )

        save_ckpt(f"epoch_{epoch:03d}")

    model.save_weights(str(run_dir / "final.weights.h5"))
    print(f"[checkpoint] saved -> {run_dir / 'final.weights.h5'}")

    with (run_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    model.encoder.summary()
    model.projector.summary()
    print(f"Saved run artifacts to: {run_dir}")
    print("Training run completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
