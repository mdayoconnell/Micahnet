#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Analyze exported embeddings and print top dimensions by activation "
            "plus top linear combinations."
        )
    )
    p.add_argument(
        "embeddings",
        nargs="?",
        help="Path to exported embeddings .npy file (or use --embeddings).",
    )
    p.add_argument(
        "-e",
        "--embeddings",
        dest="embeddings_flag",
        help="Path to exported embeddings .npy file.",
    )
    p.add_argument("--top-k", type=int, default=5, help="How many dimensions/combinations to print.")
    p.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional row cap for faster analysis (<=0 uses all rows).",
    )
    p.add_argument("--seed", type=int, default=0, help="Seed used when --max-samples is active.")
    p.add_argument(
        "--combo-terms",
        type=int,
        default=4,
        help="How many strongest coefficients to show for each linear combination.",
    )
    args = p.parse_args()
    args.embeddings_path = args.embeddings_flag or args.embeddings
    if not args.embeddings_path:
        p.error("embeddings path is required (positional or --embeddings).")
    return args


def _top_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(int(k), scores.shape[0]))
    idx = np.argpartition(scores, -k)[-k:]
    return idx[np.argsort(scores[idx])[::-1]]


def _load_embeddings(path: Path, max_samples: int, seed: int) -> tuple[np.ndarray, int]:
    emb = np.load(path, mmap_mode="r")
    if not isinstance(emb, np.ndarray) or emb.ndim != 2:
        raise RuntimeError(f"Expected a 2D embeddings array, got shape={getattr(emb, 'shape', None)}")

    total_rows = int(emb.shape[0])
    if max_samples > 0 and max_samples < total_rows:
        rng = np.random.default_rng(seed)
        row_idx = np.sort(rng.choice(total_rows, size=max_samples, replace=False))
        return np.asarray(emb[row_idx], dtype=np.float32), total_rows

    return np.asarray(emb, dtype=np.float32), total_rows


def _format_combo(vec: np.ndarray, n_terms: int) -> str:
    idx = _top_indices(np.abs(vec), n_terms)
    parts: list[str] = []
    for d in idx:
        coeff = float(vec[d])
        sign = "+" if coeff >= 0.0 else "-"
        parts.append(f"{sign}{abs(coeff):.4f}*d{int(d)}")
    return " ".join(parts).lstrip("+").strip()


def main() -> int:
    args = parse_args()
    npy_path = Path(args.embeddings_path).expanduser().resolve()
    if not npy_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {npy_path}")

    x, total_rows = _load_embeddings(npy_path, args.max_samples, args.seed)
    n_rows, n_dims = map(int, x.shape)
    if n_rows < 1 or n_dims < 1:
        raise RuntimeError(f"Embeddings are empty: shape={x.shape}")

    top_k = max(1, min(int(args.top_k), n_dims))
    combo_terms = max(1, min(int(args.combo_terms), n_dims))

    meta_path = npy_path.with_suffix(".meta.json")
    meta = {}
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

    mean_act = np.mean(x, axis=0, dtype=np.float64)
    mean_abs_act = np.mean(np.abs(x), axis=0, dtype=np.float64)
    top_dims = _top_indices(mean_abs_act, top_k)

    x_centered = x - mean_act.astype(np.float32, copy=False)
    denom = float(max(n_rows - 1, 1))
    cov = (x_centered.T @ x_centered) / denom
    eigvals, eigvecs = np.linalg.eigh(cov.astype(np.float64, copy=False))
    eigvals = np.maximum(eigvals, 0.0)

    top_combo_idx = _top_indices(eigvals, top_k)
    total_var = float(np.sum(eigvals))

    print(f"[dimension-response] embeddings={npy_path}")
    if meta:
        print(
            f"[dimension-response] space={meta.get('space', 'unknown')} "
            f"embed_dim={meta.get('embed_dim', n_dims)} "
            f"count={meta.get('count', total_rows)}"
        )
    sampled_txt = f"sampled={n_rows}/{total_rows}" if n_rows != total_rows else f"rows={n_rows}"
    print(f"[dimension-response] dims={n_dims} {sampled_txt}")

    print(f"\nTop {top_k} dimensions by activation (mean |value|):")
    for rank, d in enumerate(top_dims, start=1):
        print(
            f" {rank:2d}. d{int(d):03d} "
            f"mean|act|={mean_abs_act[d]:.6f} "
            f"mean={mean_act[d]:+.6f}"
        )

    print(f"\nTop {top_k} linear combinations (principal directions):")
    for rank, pc_idx in enumerate(top_combo_idx, start=1):
        vec = eigvecs[:, pc_idx]
        proj = x_centered @ vec.astype(np.float32, copy=False)
        proj_mean_abs = float(np.mean(np.abs(proj), dtype=np.float64))
        proj_std = float(np.sqrt(eigvals[pc_idx]))
        explained = (float(eigvals[pc_idx]) / total_var) if total_var > 0.0 else 0.0
        combo = _format_combo(vec, combo_terms)
        print(
            f" {rank:2d}. pc{rank} "
            f"mean|act|={proj_mean_abs:.6f} std={proj_std:.6f} "
            f"explained={explained:.2%} combo={combo}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
