#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf

# Ensure project-root imports work when running this file directly.
# Prefer the directory that contains this repo (../.. from this file).
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Fallback to ~/MicahNet if the computed root doesn't look right.
_FALLBACK_ROOT = Path("~/MicahNet").expanduser().resolve()

if (_PROJECT_ROOT / "models").exists() and str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
elif (_FALLBACK_ROOT / "models").exists() and str(_FALLBACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_FALLBACK_ROOT))

from models.simclr_model import MicahNetConfig, build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="kNN classifier on SimCLR embeddings.")
    p.add_argument(
        "--run-dir",
        default=None,
        help="Run directory from main() (contains run_config.json + weights). Required unless --embeddings-npy is provided.",
    )
    p.add_argument("--weights", default=None, help="Weights file name or path (default: final.weights.h5 in run-dir).")
    p.add_argument(
        "--embeddings-npy",
        default=None,
        help="Optional .npy file of precomputed embeddings (shape [N,D]). If provided, the script will NOT run the model to embed images.",
    )
    p.add_argument(
        "--embeddings-paths",
        default=None,
        help="Text file with N lines of image paths corresponding 1:1 with rows in --embeddings-npy.",
    )
    p.add_argument(
        "--manifest",
        default=None,
        help="Alias for --embeddings-paths. A manifest file with one image path per line aligned 1:1 with rows in --embeddings-npy.",
    )
    p.add_argument(
        "--normalize",
        action="store_true",
        help="L2-normalize embeddings before kNN. Recommended for cosine-sim kNN.",
    )

    p.add_argument(
        "--unlabeled",
        action="store_true",
        help="Run an unlabeled neighbor report using only --embeddings-npy + --manifest/--embeddings-paths (no COCO required).",
    )
    p.add_argument(
        "--report-neighbors",
        type=int,
        default=20,
        help="Number of query images to report neighbors for in --unlabeled mode (default: 20).",
    )
    p.add_argument(
        "--report-k",
        type=int,
        default=10,
        help="Number of neighbors to show per query in --unlabeled mode (default: 10).",
    )
    p.add_argument(
        "--dump-neighbors-dir",
        default=None,
        help="If set in --unlabeled mode, write neighbors.csv and (optionally) symlink images into subfolders.",
    )
    p.add_argument(
        "--dump-mode",
        choices=["none", "symlink"],
        default="none",
        help="How to materialize images into --dump-neighbors-dir (default: none).",
    )

    p.add_argument(
        "--annotations",
        default="data/raw/coco/coco2017/annotations/instances_val2017.json",
        help="COCO instances annotations JSON (for labels).",
    )
    p.add_argument(
        "--images-dir",
        default="data/raw/coco/coco2017/val2017",
        help="Directory containing COCO images referenced by annotations.",
    )
    p.add_argument("--max-images", type=int, default=5000, help="Max labeled images to use (0 = all).")
    p.add_argument("--val-fraction", type=float, default=0.2, help="Validation fraction for kNN eval.")
    p.add_argument("--k", type=int, default=5, help="Number of neighbors.")
    p.add_argument("--embed-batch", type=int, default=128, help="Batch size for embedding extraction.")
    p.add_argument("--knn-batch", type=int, default=1024, help="Batch size for kNN similarity blocks.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling/split.")
    p.add_argument("--use-projection", action="store_true", help="Use projection head embeddings (z) instead of h.")
    return p.parse_args()


def load_run_config(run_dir: Path) -> dict:
    cfg_path = run_dir / "run_config.json"
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


# --- Embedding table helpers ---

def load_embeddings_table(npy_path: Path, paths_path: Path, normalize: bool) -> Tuple[np.ndarray, List[str]]:
    emb = np.load(str(npy_path)).astype(np.float32)
    with paths_path.open("r", encoding="utf-8") as f:
        paths = [ln.strip() for ln in f if ln.strip()]

    if emb.ndim != 2:
        raise ValueError(f"Embeddings must be 2D [N,D], got shape {emb.shape}")
    if len(paths) != emb.shape[0]:
        raise ValueError(
            f"Path count ({len(paths)}) must match embeddings rows ({emb.shape[0]})."
        )

    if normalize:
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / (norms + 1e-12)

    return emb, paths


def gather_embeddings_for_paths(
    emb: np.ndarray,
    emb_paths: List[str],
    query_paths: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (gathered_embeddings, keep_mask) aligned to query_paths.

    keep_mask[i] = True if query_paths[i] was found in emb_paths.
    """
    index = {p: i for i, p in enumerate(emb_paths)}
    keep = np.zeros(len(query_paths), dtype=bool)
    rows: List[int] = []
    for i, p in enumerate(query_paths):
        j = index.get(p)
        if j is None:
            continue
        keep[i] = True
        rows.append(j)

    if not rows:
        return np.zeros((0, emb.shape[1]), dtype=np.float32), keep

    return emb[np.asarray(rows, dtype=np.int64)], keep


def build_model_from_run(run_dir: Path, weights_path: Optional[str]) -> Tuple[tf.keras.Model, int]:
    cfg_data = load_run_config(run_dir)
    img_size = int(cfg_data.get("img_size", 100))

    cfg = MicahNetConfig(
        input_shape=(img_size, img_size, 1),
        width_mult=float(cfg_data.get("width_mult", 1.0)),
        embedding_dim=int(cfg_data.get("embedding_dim", 256)),
        proj_dim=int(cfg_data.get("proj_dim", 128)),
        proj_hidden_dim=int(cfg_data.get("proj_hidden_dim", 512)),
    )

    model = build_model(cfg)
    w_path = Path(weights_path) if weights_path else (run_dir / "final.weights.h5")
    if w_path.is_dir():
        w_path = w_path / "final.weights.h5"
    model.load_weights(str(w_path))
    return model, img_size


def decode_gray(path: tf.Tensor, img_size: int) -> tf.Tensor:
    b = tf.io.read_file(path)
    img = tf.io.decode_image(b, channels=1, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [img_size, img_size], method=tf.image.ResizeMethod.BILINEAR)
    img.set_shape([img_size, img_size, 1])
    return img


def embed_paths(
    model: tf.keras.Model,
    paths: List[str],
    img_size: int,
    batch_size: int,
    use_projection: bool,
) -> np.ndarray:
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(lambda p: decode_gray(p, img_size), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    @tf.function
    def embed_step(batch: tf.Tensor) -> tf.Tensor:
        h, z = model(batch, training=False)
        return z if use_projection else h

    chunks = []
    for batch in ds:
        chunks.append(embed_step(batch).numpy())

    if not chunks:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def load_coco_labels(
    annotations_path: Path,
    images_dir: Path,
    max_images: int,
    seed: int,
) -> Tuple[List[str], np.ndarray, List[str]]:
    with annotations_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    cat_id_to_name = {c["id"]: c.get("name", str(c["id"])) for c in categories if "id" in c}

    best_cat: dict[int, int] = {}
    best_area: dict[int, float] = {}
    for ann in annotations:
        img_id = ann.get("image_id")
        cat_id = ann.get("category_id")
        if img_id is None or cat_id is None:
            continue
        area = float(ann.get("area", 0.0))
        if img_id not in best_area or area > best_area[img_id]:
            best_area[img_id] = area
            best_cat[img_id] = cat_id

    items: List[Tuple[str, int]] = []
    for img in images:
        img_id = img.get("id")
        file_name = img.get("file_name")
        if img_id is None or not file_name:
            continue
        cat_id = best_cat.get(img_id)
        if cat_id is None:
            continue
        path = images_dir / file_name
        if path.exists():
            items.append((str(path), cat_id))

    if not items:
        raise RuntimeError("No labeled images found. Check annotations/images-dir paths.")

    if max_images > 0 and len(items) > max_images:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(items), size=max_images, replace=False)
        items = [items[i] for i in idx]

    cat_ids = sorted({cat_id for _, cat_id in items})
    cat_to_idx = {cid: i for i, cid in enumerate(cat_ids)}

    paths = [p for p, _ in items]
    labels = np.fromiter((cat_to_idx[cid] for _, cid in items), dtype=np.int64, count=len(items))
    class_names = [cat_id_to_name.get(cid, str(cid)) for cid in cat_ids]

    return paths, labels, class_names


def split_train_val(n: int, val_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if n == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(n * val_fraction)
    if n > 1:
        n_val = max(1, n_val)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_idx, val_idx


def knn_predict(
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    query_emb: np.ndarray,
    k: int,
    batch_size: int,
) -> np.ndarray:
    if train_emb.shape[0] == 0:
        raise RuntimeError("Empty training embeddings.")

    k = max(1, min(k, train_emb.shape[0]))

    train_emb = np.ascontiguousarray(train_emb, dtype=np.float32)
    query_emb = np.ascontiguousarray(query_emb, dtype=np.float32)
    train_labels = np.asarray(train_labels, dtype=np.int64)

    num_classes = int(train_labels.max()) + 1
    train_emb_t = train_emb.T

    preds = np.empty(query_emb.shape[0], dtype=np.int64)
    for start in range(0, query_emb.shape[0], batch_size):
        end = min(start + batch_size, query_emb.shape[0])
        q = query_emb[start:end]
        sim = q @ train_emb_t

        idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
        neigh_labels = train_labels[idx]
        neigh_sims = np.take_along_axis(sim, idx, axis=1)

        scores = np.zeros((q.shape[0], num_classes), dtype=np.float32)
        row = np.arange(q.shape[0])[:, None]
        np.add.at(scores, (row, neigh_labels), neigh_sims)
        preds[start:end] = scores.argmax(axis=1)

    return preds


# --- Unlabeled neighbor report (OpenImages-friendly) ---
def unlabeled_neighbor_report(
    emb: np.ndarray,
    paths: List[str],
    n_report: int,
    k: int,
    seed: int,
    dump_dir: Optional[Path],
    dump_mode: str,
) -> None:
    """Cosine neighbor report for unlabeled embeddings (OpenImages-friendly)."""
    if emb.ndim != 2:
        raise ValueError(f"Embeddings must be 2D [N,D], got shape {emb.shape}")
    if len(paths) != emb.shape[0]:
        raise ValueError(f"Path count ({len(paths)}) must match embeddings rows ({emb.shape[0]})")
    n = emb.shape[0]
    if n == 0:
        raise RuntimeError("Empty embeddings table")

    k = max(1, min(k, n - 1))
    n_report = max(1, min(n_report, n))

    rng = np.random.default_rng(seed)
    q_idx = rng.choice(n, size=n_report, replace=False if n_report <= n else True)

    emb_t = emb.T

    rows = []
    if dump_dir is not None:
        dump_dir.mkdir(parents=True, exist_ok=True)

    for qi, i0 in enumerate(q_idx):
        q = emb[i0:i0 + 1]
        sim = (q @ emb_t).ravel()
        sim[i0] = -1e9  # exclude self

        nn = np.argpartition(-sim, kth=k - 1)[:k]
        nn = nn[np.argsort(sim[nn])[::-1]]

        print("\n=== QUERY ===")
        print(paths[i0])
        for rank, j in enumerate(nn, 1):
            print(f" {rank:2d}  sim={float(sim[j]):.4f}  {paths[j]}")
            rows.append((qi, paths[i0], rank, float(sim[j]), paths[j]))

        if dump_dir is not None and dump_mode == "symlink":
            q_out = dump_dir / f"q{qi:04d}"
            q_out.mkdir(parents=True, exist_ok=True)

            def _symlink(src: str, dst: Path) -> None:
                try:
                    if dst.exists() or dst.is_symlink():
                        dst.unlink()
                    os.symlink(src, dst)
                except Exception as e:
                    print(f"[warn] symlink failed {src} -> {dst}: {e}")

            _symlink(paths[i0], q_out / f"0_query__{Path(paths[i0]).name}")
            for rank, j in enumerate(nn, 1):
                _symlink(paths[j], q_out / f"{rank:02d}_nn__{Path(paths[j]).name}")

    if dump_dir is not None:
        import csv
        out_csv = dump_dir / "neighbors.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["query_index", "query_path", "neighbor_rank", "cosine_sim", "neighbor_path"])
            w.writerows(rows)
        print(f"\nWrote neighbor report to {out_csv}")

    print(f"\nUnlabeled neighbor report complete: queries={n_report} k={k} N={n}")


def main() -> int:
    args = parse_args()

    annotations_path = Path(args.annotations).resolve()
    images_dir = Path(args.images_dir).resolve()

    if not args.unlabeled:
        if not annotations_path.exists():
            raise FileNotFoundError(f"Annotations not found: {annotations_path}")
        if not images_dir.exists():
            raise FileNotFoundError(f"Images dir not found: {images_dir}")

    use_precomputed = args.embeddings_npy is not None
    run_dir: Optional[Path] = None
    if not use_precomputed:
        if args.run_dir is None:
            raise ValueError("--run-dir is required unless --embeddings-npy is provided")
        run_dir = Path(args.run_dir).resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"Run dir not found: {run_dir}")

    # Allow --manifest as an alias for --embeddings-paths
    if args.embeddings_paths is None and args.manifest is not None:
        args.embeddings_paths = args.manifest
    model = None
    img_size = None

    if use_precomputed:
        if args.embeddings_paths is None:
            raise ValueError("--embeddings-paths (or --manifest) is required when using --embeddings-npy")
        emb_table, emb_paths = load_embeddings_table(
            Path(args.embeddings_npy).resolve(),
            Path(args.embeddings_paths).resolve(),
            normalize=args.normalize,
        )
        if args.unlabeled:
            dump_dir = Path(args.dump_neighbors_dir).expanduser().resolve() if args.dump_neighbors_dir else None
            unlabeled_neighbor_report(
                emb=emb_table,
                paths=emb_paths,
                n_report=args.report_neighbors,
                k=args.report_k,
                seed=args.seed,
                dump_dir=dump_dir,
                dump_mode=args.dump_mode,
            )
            return 0
    else:
        assert run_dir is not None
        model, img_size = build_model_from_run(run_dir, args.weights)

    paths, labels, class_names = load_coco_labels(
        annotations_path=annotations_path,
        images_dir=images_dir,
        max_images=args.max_images,
        seed=args.seed,
    )

    if use_precomputed:
        # Filter to labeled COCO paths that exist in the embeddings table
        gathered, keep_mask = gather_embeddings_for_paths(emb_table, emb_paths, paths)
        kept_paths = [p for p, keep in zip(paths, keep_mask) if keep]
        kept_labels = labels[keep_mask]

        if gathered.shape[0] == 0:
            raise RuntimeError(
                "No labeled image paths matched the embeddings manifest. "
                "This is expected if your embeddings/manifest come from a different dataset (e.g., OpenImages). "
                "For OpenImages qualitative checks, rerun with --unlabeled."
            )

        if len(kept_paths) != gathered.shape[0]:
            raise RuntimeError("Internal mismatch: kept_paths and gathered embeddings differ in length.")

        paths = kept_paths
        labels = kept_labels
        all_emb = gathered

        train_idx, val_idx = split_train_val(len(paths), args.val_fraction, args.seed)
        if train_idx.size == 0 or val_idx.size == 0:
            raise RuntimeError("Need at least 1 train and 1 val sample for kNN evaluation.")

        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        train_emb = all_emb[train_idx]
        val_emb = all_emb[val_idx]
    else:
        train_idx, val_idx = split_train_val(len(paths), args.val_fraction, args.seed)
        if train_idx.size == 0 or val_idx.size == 0:
            raise RuntimeError("Need at least 1 train and 1 val sample for kNN evaluation.")

        train_paths = [paths[i] for i in train_idx]
        val_paths = [paths[i] for i in val_idx]
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]

        train_emb = embed_paths(model, train_paths, img_size, args.embed_batch, args.use_projection)
        val_emb = embed_paths(model, val_paths, img_size, args.embed_batch, args.use_projection)

        if args.normalize:
            train_emb = train_emb / (np.linalg.norm(train_emb, axis=1, keepdims=True) + 1e-12)
            val_emb = val_emb / (np.linalg.norm(val_emb, axis=1, keepdims=True) + 1e-12)

    preds = knn_predict(train_emb, train_labels, val_emb, args.k, args.knn_batch)
    acc = float(np.mean(preds == val_labels))

    src = "precomputed" if use_precomputed else "model"
    print(f"kNN accuracy: {acc:.4f} (source={src})")
    print(
        f"samples: train={train_emb.shape[0]} val={val_emb.shape[0]} "
        f"classes={len(class_names)} k={min(args.k, train_emb.shape[0])}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
