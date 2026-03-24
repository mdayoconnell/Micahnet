# MicahNet — Working Notes

## A) High-Level Intent

- Core ambition is self-supervised visual learning from unlabeled data.
- “AlexNet style” means convolution-first spirit, not exact historical architecture.
- Primary visible win = emergent face-related features via feature visualization.

---

## B) Clarified Success Ordering

1. Featureviz evidence of face-sensitive representations
2. Transferable embeddings
3. Optional face-probability video demo

Interpretation:
- If we achieve (1), project is already a strong success.
- (2) and (3) extend from (1), but are not mandatory for initial milestone.

---

## C) Transferable Embeddings (Practical Definition)

Embeddings are “transferable” if frozen features from self-supervised pretraining can support downstream tasks with small labeled data and modest heads.

Evidence examples:
- Good linear probe results with few labels
- Good nearest-neighbor semantic retrieval
- Robustness across modest domain shifts

What this unlocks:
- Fast adaptation to new tasks
- Few-shot downstream fine-tuning
- Retrieval and clustering demos

---

## D) Why Objective Choice Matters

Important reminder:
- Unlabeled training alone does not guarantee semantic feature emergence.
- SSL objective and augment policy strongly determine outcome quality.

Branch summary:
- BYOL: strong practical baseline, often stable
- VICReg: robust anti-collapse behavior under constrained settings
- SimCLR: strong but more batch-sensitive; can be costly if tuned badly

---

## E) Input/Data Notes

Current preference:
- grayscale, around 100x100

Tradeoff:
- Grayscale simplifies and reduces cost
- RGB generally improves representational ceiling

Data caveats:
- Quantity likely not bottleneck
- Quality/diversity/dedup are critical
- Kaggle sourcing is fine if dedup/sanity checks are included

Suggested checks:
- duplicate/near-duplicate filtering
- class/source skew checks
- watermark/text prevalence checks
- fixed validation/eval slice for consistent comparison

---

## F) Budget and Compute Notes

Budget target:
- hard cap around $40–50

Likely split:
- Local smoke test first
- Most spend on one main cloud run
- Keep reserve for one recovery/ablation run

Operational principles:
- prioritize throughput per dollar, not “best GPU class”
- use spot pricing with robust checkpointing
- test resume flow before long run

---

## G) M1 Local Feasibility Notes (2020, 8GB)

Feasible locally:
- architecture smoke tests
- short SSL proof-of-concept runs
- embedding extraction
- basic feature visualization
- downstream tiny probes

Not ideal locally:
- large-scale SSL convergence
- heavy long-run contrastive training with large batch

Conclusion:
- local for correctness + profiling
- cloud for main training

---

## H) Current Architecture State

A minimal TensorFlow SimCLR-style architecture has been drafted:

Components:
- `MicahNetConfig` dataclass
- `MicahNetBackbone` (5 conv stages, BN/ReLU, maxpool, GAP, dense embedding)
- `ProjectionHead` (MLP)
- `SimCLRModel` returning normalized `(h, z)`

Current default draft settings:
- input `(100,100,1)`
- width multiplier `1.0`
- embedding dim `256`
- projection hidden `512`
- projection dim `128`

Local-smoke smaller preset candidate:
- width `0.75`
- embedding `192`
- projection hidden `384`
- projection dim `96`

---

## I) Current Plan Status

Completed:
- project framing and branch mapping
- bottlenecks/failure modes identified
- minimal architecture draft completed

In progress:
- formal project docs (this file + PLAN.md)
- defining data wrangling + augment pipeline spec

Next:
1. implement tf.data loader + two-view augment pipeline
2. implement minimal SimCLR train loop + NT-Xent
3. add checkpoint/resume and forced resume test
4. run local C-smoketest (1–4h)
5. evaluate go/no-go for cloud C vs A/B

---

## J) Cloud Readiness Checklist (pre-RunPod)

Must be true before spending:
- [ ] train step runs stably for >N iterations locally
- [ ] checkpoints save/load without shape/state errors
- [ ] resume preserves optimizer + global step
- [ ] logging and metrics are readable
- [ ] tiny embedding sanity checks show non-random structure
- [ ] estimated images/sec known from local and short cloud pilot

---

## K) Open Questions

1. Keep grayscale at v0 or run small RGB ablation early?
2. Which objective for first full cloud run if C is only “okay” locally?
3. How much dataset balancing toward human/faces vs general scenes?
4. What exact stop criteria triggers fallback to BYOL or VICReg?

---

## L) Suggested Repo Layout (Codex-friendly)

micahnet/
  configs/
    simclr_local.yaml
    simclr_cloud.yaml
    byol_cloud.yaml
    vicreg_cloud.yaml
  src/
    models/
      micahnet_simclr_model.py
    data/
      dataset_loader.py
      augmentations.py
    training/
      losses.py
      train_ssl.py
      checkpoints.py
    eval/
      knn_probe.py
      linear_probe.py
      retrieval.py
    viz/
      featureviz.ipynb
  scripts/
    run_local.sh
    run_cloud.sh
  PLAN.md
  NOTES.md
  RUNBOOK.md

---

## M) Reference Orientation (for implementation context)

- AlexNet historical anchor: architecture scale and supervised baseline precedent.
- SimCLR/BYOL/VICReg family: modern self-supervised objectives for unlabeled representation learning.
- TensorFlow mixed precision and tf.data performance: key levers for cost/perf efficiency.
- RunPod pricing/availability should be checked at run time for actual cost assumptions.

(Implementation should treat pricing/availability as dynamic and re-query before launch.)

---

## N) Patch Log

### 2026-02-13
- Updated `utils/build_manifest.py` to use deterministic directory and filename sorting during recursive scan.
- Determinism detail:
  - `os.walk(...)` now captures `dirnames` and `filenames`.
  - Applies `dirnames.sort()` and `filenames.sort()` before yielding paths.
- Why this matters:
  - Reproducible manifest order when `--shuffle` is **not** used.
  - More stable diffs and easier debugging across machines/runs.

### Next suggested patch
- Replace global RNG usage in `write_manifest` with a local RNG instance:
  - From: `random.seed(seed); random.shuffle(all_paths)`
  - To: `rng = random.Random(seed); rng.shuffle(all_paths)`
- Rationale: avoids mutating global random state.