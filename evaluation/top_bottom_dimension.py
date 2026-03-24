#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Print top/bottom image paths for a chosen embedding dimension.")
    p.add_argument("--npy", required=True, help="Embeddings .npy file (shape [N, D]).")
    p.add_argument("--paths", default=None, help="Matching .paths.txt (one path per row). If omitted, tries <npy>.paths.txt")
    p.add_argument("--dim", type=int, required=True, help="Dimension index d to inspect (0-based).")
    p.add_argument("--n", type=int, default=20, help="How many top/bottom items to print.")
    p.add_argument("--max-samples", type=int, default=0, help="Optional cap (0 = all). Uses deterministic sampling.")
    p.add_argument("--seed", type=int, default=0, help="Seed for sampling when --max-samples is used.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    npy_path = Path(args.npy).expanduser().resolve()
    if not npy_path.exists():
        raise FileNotFoundError(f"npy not found: {npy_path}")

    paths_path = Path(args.paths).expanduser().resolve() if args.paths else npy_path.with_suffix(".paths.txt")
    if not paths_path.exists():
        raise FileNotFoundError(f"paths not found: {paths_path}")

    X = np.load(str(npy_path), mmap_mode="r")
    N, D = X.shape

    lines = [ln.strip() for ln in paths_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) != N:
        raise RuntimeError(f"paths count mismatch: {len(lines)} paths vs {N} embedding rows")

    d = int(args.dim)
    if d < 0 or d >= D:
        raise ValueError(f"--dim {d} out of range (D={D})")

    # Optional deterministic sampling
    if args.max_samples and 0 < args.max_samples < N:
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(N, size=args.max_samples, replace=False))
        Xd = np.asarray(X[idx, d], dtype=np.float32)
        paths = [lines[i] for i in idx.tolist()]
    else:
        Xd = np.asarray(X[:, d], dtype=np.float32)
        paths = lines

    n = max(1, min(int(args.n), len(paths)))

    top_idx = np.argpartition(Xd, -n)[-n:]
    bot_idx = np.argpartition(Xd, n - 1)[:n]
    top_idx = top_idx[np.argsort(Xd[top_idx])[::-1]]
    bot_idx = bot_idx[np.argsort(Xd[bot_idx])]

    print(f"[top_bottom_dim] npy={npy_path}")
    print(f"[top_bottom_dim] paths={paths_path}")
    print(f"[top_bottom_dim] rows={len(paths)} dim={d} n={n}")

    print(f"\nTop {n} (highest) for dim d{d:03d}:")
    for rank, i in enumerate(top_idx, start=1):
        print(f" {rank:2d}. score={float(Xd[i]):+.6f} path={paths[int(i)]}")

    print(f"\nBottom {n} (lowest) for dim d{d:03d}:")
    for rank, i in enumerate(bot_idx, start=1):
        print(f" {rank:2d}. score={float(Xd[i]):+.6f} path={paths[int(i)]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())