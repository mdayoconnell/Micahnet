#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class EmbeddingSample:
    x: np.ndarray
    total_rows: int
    total_dims: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compute embedding dimensional-utilization diagnostics "
            "(effective rank, participation ratio, variance@k, pairwise metrics, hubness)."
        )
    )
    p.add_argument(
        "embeddings",
        nargs="?",
        help="Path to embeddings .npy file (or use --embeddings).",
    )
    p.add_argument(
        "-e",
        "--embeddings",
        dest="embeddings_flag",
        help="Path to embeddings .npy file.",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=20000,
        help="Max rows to use for stats (<=0 uses all).",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for row/pair sampling.")
    p.add_argument(
        "--random-pairs",
        type=int,
        default=50000,
        help="Number of random pairs for pairwise cosine/uniformity stats.",
    )
    p.add_argument(
        "--hubness-sample",
        type=int,
        default=4000,
        help="Subset size for top-1 hubness analysis (<=1 disables).",
    )
    p.add_argument(
        "--hubness-block",
        type=int,
        default=512,
        help="Block size for hubness similarity computation.",
    )
    p.add_argument(
        "--thresholds",
        default="0.8,0.9,0.95,0.99,0.995,0.999",
        help="Comma-separated cumulative variance thresholds.",
    )
    p.add_argument("--tag", default=None, help="Optional experiment tag to include in output.")
    p.add_argument("--space", default=None, help="Optional space label (e.g., h or z).")
    p.add_argument(
        "--out",
        default=None,
        help="Output JSON path. Default: <embeddings>.dim_usage.json",
    )
    p.add_argument(
        "--append-jsonl",
        default=None,
        help="If set, append one compact JSON record per run to this .jsonl path.",
    )
    p.add_argument("--quiet", action="store_true", help="Suppress console summary.")
    args = p.parse_args()
    args.embeddings_path = args.embeddings_flag or args.embeddings
    if not args.embeddings_path:
        p.error("embeddings path is required (positional or --embeddings).")
    return args


def parse_thresholds(raw: str) -> list[float]:
    vals: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        v = float(part)
        if not (0.0 < v <= 1.0):
            raise ValueError(f"threshold must be in (0,1], got {v}")
        vals.append(v)
    if not vals:
        raise ValueError("no valid thresholds provided")
    vals = sorted(set(vals))
    return vals


def load_sample(path: Path, max_samples: int, seed: int) -> EmbeddingSample:
    emb = np.load(path, mmap_mode="r")
    if not isinstance(emb, np.ndarray) or emb.ndim != 2:
        raise RuntimeError(f"Expected 2D embeddings, got shape={getattr(emb, 'shape', None)}")

    total_rows, total_dims = int(emb.shape[0]), int(emb.shape[1])
    if total_rows < 2:
        raise RuntimeError(f"Need at least 2 rows, got {total_rows}")

    use_rows = total_rows if max_samples <= 0 else min(int(max_samples), total_rows)
    if use_rows == total_rows:
        x = np.asarray(emb, dtype=np.float32)
    else:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(total_rows, size=use_rows, replace=False))
        x = np.asarray(emb[idx], dtype=np.float32)

    return EmbeddingSample(x=x, total_rows=total_rows, total_dims=total_dims)


def normalize_rows(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(x, axis=1)
    x_unit = x / (norms[:, None] + 1e-12)
    return x_unit, norms


def covariance_eigenspectrum(x: np.ndarray) -> np.ndarray:
    x64 = x.astype(np.float64, copy=False)
    mu = x64.mean(axis=0, dtype=np.float64)
    xc = x64 - mu
    denom = float(max(x64.shape[0] - 1, 1))
    cov = (xc.T @ xc) / denom
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 0.0)[::-1]
    return eigvals


def spectral_metrics(eigvals: np.ndarray, thresholds: list[float]) -> dict[str, Any]:
    total = float(np.sum(eigvals))
    sumsq = float(np.sum(np.square(eigvals)))
    out: dict[str, Any] = {
        "total_variance": total,
        "top_eigenvalue": float(eigvals[0]) if eigvals.size else 0.0,
        "eigenvalue_count": int(eigvals.size),
    }
    if total <= 0.0 or eigvals.size == 0:
        out.update(
            {
                "effective_rank": 0.0,
                "participation_ratio": 0.0,
                "variance_at_k": {},
                "k_at_threshold": {},
            }
        )
        return out

    p = eigvals / total
    nz = p[p > 0.0]
    entropy = float(-(nz * np.log(nz)).sum())
    r_eff = float(np.exp(entropy))
    r_pr = float((total * total) / sumsq) if sumsq > 0 else 0.0

    cum = np.cumsum(p)
    k_at: dict[str, int] = {}
    for t in thresholds:
        k = int(np.searchsorted(cum, t) + 1)
        k_at[f"{t:.3f}"] = k

    var_at_k: dict[str, float] = {}
    for k in (1, 2, 4, 8, 16, 32, 64, 128, 256, eigvals.size):
        if k <= eigvals.size:
            var_at_k[str(k)] = float(cum[k - 1])

    out.update(
        {
            "effective_rank": r_eff,
            "participation_ratio": r_pr,
            "variance_at_k": var_at_k,
            "k_at_threshold": k_at,
        }
    )
    return out


def random_pair_metrics(x_unit: np.ndarray, n_pairs: int, seed: int) -> dict[str, Any]:
    n = int(x_unit.shape[0])
    if n < 2 or n_pairs <= 0:
        return {
            "pairs": 0,
            "cosine_mean": 0.0,
            "cosine_std": 0.0,
            "cosine_p05": 0.0,
            "cosine_p50": 0.0,
            "cosine_p95": 0.0,
            "sqdist_mean": 0.0,
            "uniformity_u": 0.0,
        }

    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    mask = i != j
    i = i[mask]
    j = j[mask]
    if i.size == 0:
        return {
            "pairs": 0,
            "cosine_mean": 0.0,
            "cosine_std": 0.0,
            "cosine_p05": 0.0,
            "cosine_p50": 0.0,
            "cosine_p95": 0.0,
            "sqdist_mean": 0.0,
            "uniformity_u": 0.0,
        }

    cos = np.sum(x_unit[i] * x_unit[j], axis=1)
    sqdist = 2.0 - 2.0 * cos
    uniformity_u = float(np.log(np.mean(np.exp(-2.0 * sqdist))))
    return {
        "pairs": int(cos.size),
        "cosine_mean": float(np.mean(cos)),
        "cosine_std": float(np.std(cos)),
        "cosine_p05": float(np.quantile(cos, 0.05)),
        "cosine_p50": float(np.quantile(cos, 0.50)),
        "cosine_p95": float(np.quantile(cos, 0.95)),
        "sqdist_mean": float(np.mean(sqdist)),
        "uniformity_u": uniformity_u,
    }


def hubness_metrics(x_unit: np.ndarray, sample_size: int, block: int, seed: int) -> dict[str, Any]:
    n = int(x_unit.shape[0])
    if sample_size <= 1 or n <= 1:
        return {
            "enabled": False,
            "subset_size": 0,
        }

    m = min(int(sample_size), n)
    rng = np.random.default_rng(seed)
    if m == n:
        sub = x_unit
    else:
        idx = rng.choice(n, size=m, replace=False)
        sub = x_unit[idx]

    nn = np.empty(m, dtype=np.int64)
    block = max(1, int(block))
    for s in range(0, m, block):
        e = min(s + block, m)
        q = sub[s:e]
        sim = q @ sub.T
        rows = np.arange(s, e)
        sim[np.arange(e - s), rows] = -1e9
        nn[s:e] = np.argmax(sim, axis=1)

    freq = np.bincount(nn, minlength=m)
    nz = freq[freq > 0]
    return {
        "enabled": True,
        "subset_size": int(m),
        "unique_hubs": int(nz.size),
        "max_freq": int(freq.max()) if freq.size else 0,
        "freq_p95": float(np.quantile(freq, 0.95)),
        "freq_p99": float(np.quantile(freq, 0.99)),
        "share_freq_ge_5": float(np.mean(freq >= 5)),
        "share_freq_eq_1": float(np.mean(freq == 1)),
    }


def load_sidecar_meta(emb_path: Path) -> dict[str, Any]:
    meta_path = emb_path.with_suffix(".meta.json")
    if not meta_path.exists():
        return {}
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        if isinstance(meta, dict):
            return meta
    except Exception:
        return {}
    return {}


def main() -> int:
    args = parse_args()
    emb_path = Path(args.embeddings_path).expanduser().resolve()
    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {emb_path}")

    thresholds = parse_thresholds(args.thresholds)
    sample = load_sample(emb_path, args.max_samples, args.seed)
    x_unit, norms = normalize_rows(sample.x)

    eigvals = covariance_eigenspectrum(sample.x)
    spectral = spectral_metrics(eigvals, thresholds)
    pairwise = random_pair_metrics(x_unit, args.random_pairs, args.seed + 1)
    hubness = hubness_metrics(x_unit, args.hubness_sample, args.hubness_block, args.seed + 2)
    sidecar_meta = load_sidecar_meta(emb_path)

    result: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tag": args.tag,
        "space": args.space or sidecar_meta.get("space"),
        "embeddings_path": str(emb_path),
        "shape_total": [sample.total_rows, sample.total_dims],
        "shape_used": [int(sample.x.shape[0]), int(sample.x.shape[1])],
        "dtype": str(sample.x.dtype),
        "norm": {
            "mean": float(np.mean(norms)),
            "std": float(np.std(norms)),
            "min": float(np.min(norms)),
            "max": float(np.max(norms)),
        },
        "spectral": spectral,
        "pairwise": pairwise,
        "hubness": hubness,
        "config": {
            "max_samples": int(args.max_samples),
            "random_pairs": int(args.random_pairs),
            "hubness_sample": int(args.hubness_sample),
            "hubness_block": int(args.hubness_block),
            "thresholds": thresholds,
            "seed": int(args.seed),
        },
    }
    if sidecar_meta:
        result["source_meta"] = sidecar_meta

    out_path = (
        Path(args.out).expanduser().resolve()
        if args.out
        else emb_path.with_suffix(".dim_usage.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    if args.append_jsonl:
        jsonl_path = Path(args.append_jsonl).expanduser().resolve()
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, separators=(",", ":")) + "\n")

    if not args.quiet:
        r_eff = result["spectral"]["effective_rank"]
        r_pr = result["spectral"]["participation_ratio"]
        k95 = result["spectral"]["k_at_threshold"].get("0.950")
        cos_m = result["pairwise"]["cosine_mean"]
        uni = result["pairwise"]["uniformity_u"]
        print(f"[dim-usage] embeddings={emb_path}")
        print(
            f"[dim-usage] used_rows={result['shape_used'][0]}/{result['shape_total'][0]} "
            f"dims={result['shape_used'][1]} space={result.get('space') or 'unknown'}"
        )
        print(
            f"[dim-usage] r_eff={r_eff:.4f} r_pr={r_pr:.4f} "
            f"k95={k95} cos_mean={cos_m:.5f} uniformity_u={uni:.5f}"
        )
        if result["hubness"].get("enabled"):
            print(
                f"[dim-usage] hubness subset={result['hubness']['subset_size']} "
                f"max_freq={result['hubness']['max_freq']} "
                f"share_ge_5={result['hubness']['share_freq_ge_5']:.5f}"
            )
        print(f"[dim-usage] wrote {out_path}")
        if args.append_jsonl:
            print(f"[dim-usage] appended {Path(args.append_jsonl).expanduser().resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
