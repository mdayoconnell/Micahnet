#!/usr/bin/env python3
"""Feature visualization for MicahNet (SimCLR).

Goal: reveal semantic meaning of an embedding direction by synthesizing an input
image that maximizes that direction in the encoder embedding space.

This is *not* image generation in the artistic sense; it's gradient ascent on the
input pixels with regularization.

Typical usage:
  python evaluation/featurevis.py \
    --weights weights/coco10k_smoke_clean_v1/final.weights.h5 \
    --outdir weights/coco10k_smoke_clean_v1/featurevis \
    --img-size 100 \
    --embedding-dim 256 \
    --dim 17 \
    --steps 800 \
    --lr 0.05

Notes:
- Visualize encoder space (h) by default (recommended). You can switch to
  projector space (z) with --use-projector.
- For interpretability, regularization matters (TV + L2) and so does jitter.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.simclr_model import (
    MicahNetConfig,
    build_model,
    SimCLRModel,
    MicahNetBackbone,
    ProjectionHead,
    ConvBNAct,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MicahNet feature visualization (gradient ascent on input pixels)."
    )

    p.add_argument("--weights", required=True, help="Path to weights file (.weights.h5 or full .h5 model).")
    p.add_argument("--outdir", required=True, help="Directory to write PNG snapshots.")

    # Architecture is stable; keep these defaults hardcoded.
    p.add_argument("--img-size", type=int, default=128, help="Input image size (H=W).")
    p.add_argument("--width-mult", type=float, default=1.0)
    p.add_argument("--embedding-dim", type=int, default=384)
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--proj-hidden-dim", type=int, default=512)

    # Objective
    p.add_argument("--dim", type=int, default=0, help="Embedding dimension index to maximize.")
    p.add_argument("--use-projector", action="store_true", help="Maximize projector output z instead of encoder h.")

    # Scan mode (find responsive dimensions)
    p.add_argument("--scan-dims", action="store_true", help="If set, scan multiple dimensions and rank responsiveness.")
    p.add_argument("--dim-start", type=int, default=0, help="First dimension to scan (inclusive).")
    p.add_argument("--dim-end", type=int, default=63, help="Last dimension to scan (inclusive).")
    p.add_argument("--scan-steps", type=int, default=200, help="Gradient-ascent steps per dimension during scan.")
    p.add_argument("--topk", type=int, default=12, help="How many top responsive dims to print.")

    # Optimization
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-every", type=int, default=50)

    # Regularization / stabilization
    p.add_argument("--tv-weight", type=float, default=1e-4, help="Total variation weight.")
    p.add_argument("--l2-weight", type=float, default=1e-6, help="L2 weight on pixels.")
    p.add_argument("--jitter", type=int, default=8, help="Max pixel shift for random jitter per step.")
    p.add_argument("--blur-every", type=int, default=10, help="Apply small blur every N steps (0 disables).")

    return p.parse_args()

    # Objective
    p.add_argument("--dim", type=int, default=0, help="Embedding dimension index to maximize.")
    p.add_argument("--use-projector", action="store_true", help="Maximize projector output z instead of encoder h.")

    # Scan mode (find responsive dimensions)
    p.add_argument("--scan-dims", action="store_true", help="If set, scan multiple dimensions and rank responsiveness.")
    p.add_argument("--dim-start", type=int, default=0, help="First dimension to scan (inclusive).")
    p.add_argument("--dim-end", type=int, default=63, help="Last dimension to scan (inclusive).")
    p.add_argument("--scan-steps", type=int, default=200, help="Gradient-ascent steps per dimension during scan.")
    p.add_argument("--topk", type=int, default=12, help="How many top responsive dims to print.")

    # Optimization
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-every", type=int, default=50)

    # Regularization / stabilization
    p.add_argument("--tv-weight", type=float, default=1e-4, help="Total variation weight.")
    p.add_argument("--l2-weight", type=float, default=1e-6, help="L2 weight on pixels.")
    p.add_argument("--jitter", type=int, default=8, help="Max pixel shift for random jitter per step.")
    p.add_argument("--blur-every", type=int, default=10, help="Apply small blur every N steps (0 disables).")

    return p.parse_args()


def _gaussian_kernel2d(ksize: int = 7, sigma: float = 1.0) -> tf.Tensor:
    radius = ksize // 2
    xs = tf.cast(tf.range(-radius, radius + 1), tf.float32)
    s = tf.cast(sigma, tf.float32)
    s = tf.maximum(s, 1e-3)
    g = tf.exp(-0.5 * tf.square(xs / s))
    g = g / tf.reduce_sum(g)
    k2 = tf.tensordot(g, g, axes=0)  # [ksize, ksize]
    return tf.reshape(k2, [ksize, ksize, 1, 1])


def _blur(x: tf.Tensor, k: tf.Tensor) -> tf.Tensor:
    # x: [1,H,W,1]
    return tf.nn.depthwise_conv2d(x, k, strides=[1, 1, 1, 1], padding="SAME")



def _save_png(x01: tf.Tensor, path: Path) -> None:
    """Save [1,H,W,1] float32 in [0,1] to grayscale PNG."""
    x01 = tf.clip_by_value(x01, 0.0, 1.0)
    u8 = tf.cast(tf.round(x01 * 255.0), tf.uint8)
    png = tf.io.encode_png(tf.squeeze(u8, axis=0))
    tf.io.write_file(str(path), png)


# Helper to robustly load weights or model file
def _load_model_or_weights(model: tf.keras.Model, weights_path: str) -> tf.keras.Model:
    """Try loading weights into an already-built model; if that fails, try loading a full model file.

    Returns a model ready for inference (built + weights loaded).
    """
    try:
        model.load_weights(weights_path)
        return model
    except Exception as e1:
        # Try legacy name-based loading (useful if layer ordering/names drifted).
        try:
            model.load_weights(weights_path, by_name=True, skip_mismatch=True)
            print("[featurevis] warning: loaded weights with by_name=True, skip_mismatch=True (partial load)")
            return model
        except Exception:
            pass

        # Try loading as a full saved Keras model (.h5). This can happen if the file was saved with model.save(...).
        try:
            loaded = tf.keras.models.load_model(
                weights_path,
                compile=False,
                custom_objects={
                    "SimCLRModel": SimCLRModel,
                    "MicahNetBackbone": MicahNetBackbone,
                    "ProjectionHead": ProjectionHead,
                    "ConvBNAct": ConvBNAct,
                },
            )
            # Ensure variables are built.
            dummy_shape = model.inputs[0].shape if model.inputs else None
            if dummy_shape is not None:
                dummy = tf.zeros([1, int(dummy_shape[1]), int(dummy_shape[2]), int(dummy_shape[3])], dtype=tf.float32)
                _ = loaded(dummy, training=False)
            return loaded
        except Exception as e2:
            raise ValueError(
                "Failed to load weights/model from the provided path. "
                "This usually means the file was created with a different saving method (weights-only vs full model) "
                "or the architecture/layer names changed.\n"
                f"Path: {weights_path}\n"
                f"load_weights error: {type(e1).__name__}: {e1}\n"
                f"load_model error: {type(e2).__name__}: {e2}"
            )


def _objective_value(model: tf.keras.Model, x: tf.Tensor, dim: int, use_projector: bool) -> tf.Tensor:
    h, z = model(x, training=False)
    e = z if use_projector else h
    e = tf.math.l2_normalize(e, axis=1)
    return e[:, dim]  # [B]


def _run_single_dim_scan(
    model: tf.keras.Model,
    img_size: int,
    dim: int,
    use_projector: bool,
    steps: int,
    lr: float,
    tv_weight: float,
    l2_weight: float,
    jitter: int,
    blur_every: int,
    blur_k: tf.Tensor,
    seed: int,
) -> tuple[float, float, float]:
    """Returns (start_target, end_target, gain) for one dim."""
    tf.random.set_seed(seed)
    np.random.seed(seed)

    x = tf.Variable(tf.random.uniform([1, img_size, img_size, 1], 0.45, 0.55, dtype=tf.float32))
    opt = tf.keras.optimizers.Adam(learning_rate=lr)

    start_val = float(_objective_value(model, x, dim, use_projector).numpy()[0])

    for _ in range(int(steps)):
        if jitter > 0:
            jy = tf.random.uniform([], -jitter, jitter + 1, dtype=tf.int32)
            jx = tf.random.uniform([], -jitter, jitter + 1, dtype=tf.int32)
            x_in = tf.roll(x, shift=[jy, jx], axis=[1, 2])
        else:
            x_in = x

        with tf.GradientTape() as tape:
            tape.watch(x)
            target = _objective_value(model, x_in, dim, use_projector)
            loss = -tf.reduce_mean(target)
            loss += tv_weight * tf.reduce_mean(tf.image.total_variation(x))
            loss += l2_weight * tf.reduce_mean(tf.square(x - 0.5))

        grads = tape.gradient(loss, x)
        opt.apply_gradients([(grads, x)])
        x.assign(tf.clip_by_value(x, 0.0, 1.0))

        if blur_every and ((_ + 1) % int(blur_every) == 0):
            x.assign(_blur(x, blur_k))
            x.assign(tf.clip_by_value(x, 0.0, 1.0))

    end_val = float(_objective_value(model, x, dim, use_projector).numpy()[0])
    return start_val, end_val, end_val - start_val


def _scan_dims(args: argparse.Namespace, model: tf.keras.Model, outdir: Path) -> int:
    out_dim = args.proj_dim if args.use_projector else args.embedding_dim
    d0 = int(max(0, args.dim_start))
    d1 = int(min(out_dim - 1, args.dim_end))
    if d1 < d0:
        raise SystemExit(f"Invalid scan range: [{args.dim_start}, {args.dim_end}] for out_dim={out_dim}")

    blur_k = _gaussian_kernel2d(ksize=7, sigma=1.0)
    rows: list[tuple[int, float, float, float]] = []

    print(f"[featurevis-scan] scanning dims {d0}..{d1} ({d1-d0+1} dims), steps={args.scan_steps}")
    for d in range(d0, d1 + 1):
        s, e, g = _run_single_dim_scan(
            model=model,
            img_size=args.img_size,
            dim=d,
            use_projector=args.use_projector,
            steps=args.scan_steps,
            lr=args.lr,
            tv_weight=args.tv_weight,
            l2_weight=args.l2_weight,
            jitter=args.jitter,
            blur_every=args.blur_every,
            blur_k=blur_k,
            seed=args.seed + d,
        )
        rows.append((d, s, e, g))
        print(f"[featurevis-scan] dim={d:03d} start={s:+.5f} end={e:+.5f} gain={g:+.5f}")

    rows.sort(key=lambda t: t[3], reverse=True)

    topk = max(1, int(args.topk))
    print(f"\n[featurevis-scan] top {min(topk, len(rows))} by gain")
    for i, (d, s, e, g) in enumerate(rows[:topk], start=1):
        print(f" {i:2d}. dim={d:03d} gain={g:+.5f} (start={s:+.5f} -> end={e:+.5f})")

    csv_path = outdir / "responsive_dims.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        f.write("dim,start,end,gain\n")
        for d, s, e, g in rows:
            f.write(f"{d},{s:.8f},{e:.8f},{g:.8f}\n")

    print(f"[featurevis-scan] wrote ranking CSV: {csv_path}")
    return 0


def main() -> int:
    args = parse_args()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = MicahNetConfig(
        input_shape=(args.img_size, args.img_size, 1),
        width_mult=args.width_mult,
        embedding_dim=args.embedding_dim,
        proj_dim=args.proj_dim,
        proj_hidden_dim=args.proj_hidden_dim,
    )
    model = build_model(cfg)

    # Build variables
    dummy = tf.zeros([1, args.img_size, args.img_size, 1], dtype=tf.float32)
    _ = model(dummy, training=False)

    model = _load_model_or_weights(model, args.weights)

    dim = int(args.dim)
    if args.use_projector:
        out_dim = args.proj_dim
        space = "z(projector)"
    else:
        out_dim = args.embedding_dim
        space = "h(encoder)"

    if dim < 0 or dim >= out_dim:
        raise SystemExit(f"--dim must be in [0, {out_dim-1}] for space {space}")

    if args.scan_dims:
        return _scan_dims(args, model, outdir)

    print(f"[featurevis] weights={args.weights}")
    print(f"[featurevis] outdir={outdir}")
    print(f"[featurevis] target space={space} dim={dim}")

    # Initialize image with small noise around mid-gray
    x = tf.Variable(tf.random.uniform([1, args.img_size, args.img_size, 1], 0.45, 0.55, dtype=tf.float32))

    opt = tf.keras.optimizers.Adam(learning_rate=args.lr)
    blur_k = _gaussian_kernel2d(ksize=7, sigma=1.0)

    # Save initial
    _save_png(x, outdir / "step_0000.png")

    for step in range(1, args.steps + 1):
        # Random jitter (translation) to reduce grid overfitting
        if args.jitter > 0:
            jy = tf.random.uniform([], -args.jitter, args.jitter + 1, dtype=tf.int32)
            jx = tf.random.uniform([], -args.jitter, args.jitter + 1, dtype=tf.int32)
            x_in = tf.roll(x, shift=[jy, jx], axis=[1, 2])
        else:
            x_in = x

        with tf.GradientTape() as tape:
            tape.watch(x)
            h, z = model(x_in, training=False)
            e = z if args.use_projector else h

            # Normalize to make dimensions comparable and avoid runaway scaling
            e = tf.math.l2_normalize(e, axis=1)
            target = e[:, dim]  # [1]

            # We do gradient *ascent* on target, so loss is negative target.
            loss = -tf.reduce_mean(target)

            # Regularizers: smoothness (TV) and small pixel magnitude (L2)
            loss += args.tv_weight * tf.reduce_mean(tf.image.total_variation(x))
            loss += args.l2_weight * tf.reduce_mean(tf.square(x - 0.5))

        grads = tape.gradient(loss, x)
        opt.apply_gradients([(grads, x)])

        # Clamp to valid image range
        x.assign(tf.clip_by_value(x, 0.0, 1.0))

        # Occasional mild blur to kill high-frequency adversarial noise
        if args.blur_every and (step % int(args.blur_every) == 0):
            x.assign(_blur(x, blur_k))
            x.assign(tf.clip_by_value(x, 0.0, 1.0))

        if args.save_every and (step % int(args.save_every) == 0 or step == args.steps):
            _save_png(x, outdir / f"step_{step:04d}.png")
            # Report current objective value
            h, z = model(x, training=False)
            e = z if args.use_projector else h
            e = tf.math.l2_normalize(e, axis=1)
            cur = float(e[:, dim].numpy()[0])
            print(f"[featurevis] step={step:04d} target={cur:.4f}")

    print(f"[featurevis] done. Wrote snapshots to: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
