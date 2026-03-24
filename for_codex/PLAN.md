# MicahNet — Project Plan (Self-Supervised CNN, Unlabeled Images)

## 1) Project Goal

Build **MicahNet**, a self-supervised image representation model (CNN core, AlexNet-inspired spirit but modernized) that learns from **unlabeled images** and develops meaningful visual features (especially face-sensitive ones) without explicit face labels.

Primary success criteria (in order):
1. **Feature visualization evidence** of emergent face-related units/patterns.
2. **Transferable embeddings** (good downstream performance with few labels).
3. Optional demo: **video frame face-probability party trick**.

---

## 2) Constraints and Preferences

- Framework: **TensorFlow**
- Data: Unlabeled, likely assembled from Kaggle/open datasets
- Preferred input: grayscale, around **100x100** (negotiable)
- Training strategy: local smoke tests + cloud (RunPod) for main run
- Budget cap: approximately **$40–50 total**
- Failure tolerance: willing to learn through failed runs, but minimize wasted spend
- Bottlenecks:
  - Cost
  - Spot interruption handling / resume robustness
  - Architecture/objective choice
  - GPU choice efficiency

---

## 3) Chosen Direction

### Architecture philosophy
- Not strict AlexNet replication
- CNN encoder remains core
- Use modern SSL components and training practices

### Candidate branches
- **A**: BYOL (quick visual emergence, practical stability/cost)
- **B**: VICReg (strong robustness, good transfer under constrained batch)
- **C**: SimCLR-style contrastive (higher upside, batch-sensitive, higher risk)

Current position:
- Leaning A/B for main run
- Want local proof-of-concept for C to map risk before cloud spend

---

## 4) MicahNet v0 (Current Draft for Branch C POC)

- Input: `100x100x1`
- Backbone: 5-stage CNN (AlexNet-inspired kernels early, modern BN/ReLU/GAP)
- Head:
  - Backbone embedding `h` (e.g. 256d)
  - Projection head `z` (e.g. 128d) for contrastive loss
- Output:
  - `h`: normalized embedding for downstream transfer/probing
  - `z`: normalized projection for SimCLR loss
- No giant AlexNet FC stack (cost/efficiency reasons)

---

## 5) Execution Phases

## Phase 0 — Local Smoke Test (no cloud)
Goal: prove pipeline correctness and training dynamics.

Tasks:
1. Finalize v0 architecture code.
2. Build tf.data ingest for unlabeled images.
3. Implement two-view augment pipeline (SimCLR-compatible).
4. Implement contrastive loss (NT-Xent) + train step.
5. Add checkpoint save/load + resume test.
6. Run short training on 10k–30k images for 1–4 hours.
7. Inspect:
   - loss curve behavior
   - embedding norm stats
   - nearest-neighbor retrieval sanity
   - no-collapse indicators

Gate to pass:
- Stable non-collapsing training
- Resume works after restart
- Embeddings show meaningful local structure vs random baseline

---

## Phase 1 — Cloud Main Run (RunPod spot)
Goal: generate useful embeddings and featureviz results under budget.

Tasks:
1. Choose best GPU by **throughput/$** from short pilots.
2. Launch with aggressive checkpoint cadence.
3. Track metrics and snapshots.
4. Stop early on bad dynamics; restart with known fallback branch.

Budget plan:
- ~70–80% of budget for main run
- reserve remainder for re-runs/ablations

---

## Phase 2 — Evaluation and Feature Visualization
Goal: validate emergence and transferability.

Tasks:
1. Feature visualization on intermediate/high-level units.
2. kNN retrieval and qualitative embedding checks.
3. Linear probe on small labeled subset (face/non-face or similar).
4. Report transferability and representation quality.

---

## Phase 3 — Optional Video Demo
Goal: frame-level face probability demo.

Tasks:
1. Freeze encoder.
2. Train tiny downstream head on small labeled frame set.
3. Run per-frame probabilities + smoothing for stable demo output.

---

## 6) Branch Decision Rules

### If local C (SimCLR) is stable and promising:
- Keep C as an active candidate for cloud run.
- Compare to A/B using same backbone size and data slice.

### If C underperforms locally due to small batch constraints:
- Treat that as expected hardware limit, not automatic rejection.
- Prefer A (BYOL) or B (VICReg) for budget-critical main run.

### If objective/optimization instability appears:
- Fallback priority: **BYOL -> VICReg -> SimCLR**

---

## 7) Risk Register + Mitigations

1. **Spot interruption data loss**
   - Mitigation: frequent persistent checkpoints; tested resume path

2. **Wrong GPU selection**
   - Mitigation: short throughput pilot benchmark on 2–3 candidate GPUs

3. **Dataset quality issues (duplicates/skew/watermarks)**
   - Mitigation: dedup + sampling balance + fixed eval slice

4. **False confidence from pretty featureviz only**
   - Mitigation: pair visuals with probe metrics (kNN/linear probe)

5. **Budget overrun**
   - Mitigation: hard stop criteria per run; staged ramp-up only

---

## 8) Deliverables

- `micahnet_simclr_model.py` (model architecture)
- `data_pipeline.py` (dataset + augment views)
- `train_ssl.py` (training loop + loss + logging)
- `eval_embeddings.py` (kNN/linear probe/NN retrieval)
- `featureviz.ipynb` (interpretability analysis)
- `RUNBOOK.md` (cloud launch + resume + failover instructions)

---

## 9) Immediate Next Tasks (Current Sprint)

1. Lock final v0 architecture (done in draft form).
2. Implement augment pipeline for two-view SSL training.
3. Create minimal training script with checkpoints.
4. Run local 1–4h C-smoketest.
5. Decide cloud branch using measured evidence.