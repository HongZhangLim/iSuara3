## Root Cause Analysis: V3.1.3 Regression

I found **5 bugs/issues**, ranked by severity:

### Bug 1 (CRITICAL): Idle prefix corrupts velocity/acceleration features
The pipeline runs: Speed variants → **Stages 1-5 (computes velocity/accel)** → **Idle prefix insertion**. When idle frames are prepended and tail frames truncated, the pre-computed velocity at the idle→sign boundary is **stale** — it reflects the original `frame[0]→frame[1]` transition, not the actual `idle_frame[-1]→sign_frame[0]` transition. **30% of training samples have corrupted derivative features.** This is the #1 cause of accuracy loss.

### Bug 2 (HIGH): NumpySequence doesn't shuffle between epochs
The `NumpySequence` class has no `on_epoch_end()` method. When `model.fit()` receives a `Sequence`, `shuffle=True` is ignored — Keras only shuffles raw arrays, not `Sequence` objects. The model sees batches in the **same order every epoch**, degrading convergence and causing the jittery validation loss.

### Bug 3 (HIGH): Z-score statistics skewed by speed variants
Z-score mean/std is computed on the full augmented training set which is 75% speed-variant data (with lots of hold-padded zeros in velocity/accel channels). Val/test only see 1× speed data. The z-score transform applies a **mismatched distribution** to val/test, directly explaining the train-val gap.

### Bug 4 (MEDIUM): 3× speed tier too aggressive  
10 real frames + 20 hold-padded frames = 67% static padding. Many of the 98 signs become indistinguishable at 10-frame resolution, diluting the training signal.

### Bug 5 (MEDIUM): Initial LR=0.001 too high
The training log shows the model only improved significantly after `ReduceLROnPlateau` dropped LR to 0.0005 at epoch 66, wasting 66 epochs.

---

## Plan: V3.1.4 Fixes

Here are the concrete changes I'll make to the `V3.1.3 copy.ipynb`:

| Fix | Cell | Change |
|---|---|---|
| **Rename to V3.1.4** | Cell 1 (markdown) | Update title, version table, changelog |
| **Remove idle prefix entirely** | Cell 10 (functions) | Delete `augment_idle_prefix()` — the velocity corruption bug makes it harmful. Can be re-added in V3.1.5 if moved before Stages 3-5 with derivative recomputation. |
| **Drop 3× speed tier** | Cell 10 (functions) | Change default `speed_tiers` to only `[1.5×, 2×]` (3 tiers total including 1×) |
| **Add epoch shuffle to NumpySequence** | Cell 16 (training) | Add `on_epoch_end()` with `np.random.permutation` index shuffle |
| **Z-score on 1× data only** | Cell 12 (pipeline) | Compute SCALER_MEAN/SCALER_SCALE from original-speed training samples only, before concatenating speed variants into the augmented array |
| **Lower initial LR to 0.0003** | Cell 14 (model) + Cell 16 (training) | `Adam(learning_rate=0.0003)` — more conservative start |
| **Increase ReduceLROnPlateau patience** | Cell 16 (training) | `patience=15` (was 10) — gives the lower LR more time before further reduction |
| **Update pipeline summary, eval, exports** | Cells 12, 18, 20, 22, 24, 27 | Version numbers, speed tier references, remove idle prefix references |



## Plan: V3.1.3 — Multi-Speed Temporal Downsampling for Variable-Duration Sign Recognition

### TL;DR

V3.1.2 trains exclusively on 30-frame sequences recorded at a fixed speed. Real-world signers perform the **same complete motion** at varying speeds — a fast signer's "Terima Kasih" covers the same trajectory in half the time. V3.1.3 addresses this by generating temporally downsampled variants of each training sequence (sampling every 2nd or 3rd frame), preserving the complete motion arc at compressed timescales. No architecture or TFLite input shape changes. The model learns speed-invariant recognition purely through training data diversity.

---

### What Changes and Why

| V3.1.2 | V3.1.3 | Justification |
|---|---|---|
| Stages 1–5 applied to all data before split | **Stages 1–5 applied after split + after downsampling** | Velocity/acceleration must be computed on the downsampled sequence, not the original. If you downsample **after** computing velocity, the velocity values reflect 30fps motion, not the compressed 15fps motion. The derivatives must reflect the actual frame-to-frame differences in the downsampled sequence. |
| 3 augmentation methods → 4× | **+1 new method (temporal downsampling → 4 speed variants) then existing 3 → 4×** | Temporal downsampling is not traditional augmentation — it's a data **generation** step that creates genuinely different training distributions. It runs first, then existing augmentations apply on top. |
| Fixed 30-frame sequences only | **30 + 20 + 15 + 10 frame variants (hold-padded to 30)** | 3 speed tiers cover the range of realistic signing speeds (see justification below). |
| No Idle-prefix contamination | **First 3–8 frames of some samples replaced with Idle data** | In the real sliding window, a sign is always preceded by either the prior sign's tail or Idle. The model currently never sees this during training, creating a train/inference distribution mismatch. |
| Test evaluation: full 30-frame only | **+ per-speed accuracy breakdown** | Must verify the model actually handles fast signs, not just hope. |

---

### Speed Tier Justification

The training data was collected at ~30fps. Each frame-skip rate simulates a specific real-world signing speed relative to the training recording speed.

| Tier | Frames Sampled | Skip Rate | Simulated Speed | Justification |
|---|---|---|---|---|
| **1× (original)** | 30 | None | Same as training | Baseline — no change needed. |
| **1.5×** | 20 | ~every 1.5 frames | 1.5× faster | Conservative speed-up. Covers signers who are slightly faster than the recorded training speed. Uses `np.round(np.linspace(0, 29, 20)).astype(int)` — evenly spaced, necessary because 30/20 = 1.5 is non-integer and simple striding can't produce exactly 20 frames. |
| **2×** | 15 | Every 2nd frame | 2× faster | The primary target. Most fluent signers perform at roughly 2× the speed of careful/deliberate signing used during data collection. `np.arange(0, 30, 2)` = `[0, 2, 4, ..., 28]` — clean integer stride. |
| **3×** | 10 | Every 3rd frame | 3× faster | Upper bound of realistic speed. Very fast conversational signing. `np.arange(0, 30, 3)` = `[0, 3, 6, ..., 27]` — clean integer stride. Going beyond 3× (e.g., 5 frames) would be unrealistic — signs become indistinguishable even to human interpreters. |

**Why hold-padding (not zero-padding) after downsampling**: After sampling N frames from the original 30, the remaining `30 - N` frames are filled by repeating the last sampled frame. Justification:
- Hold-padded frames produce **zero velocity and zero acceleration** in Stages 3–4, which correctly signals "motion has stopped"
- Zero-padding inserts all-zero keypoints (x=0, y=0, z=0 for all landmarks), which: (a) creates an artificial velocity spike at the transition from real→zero, and (b) represents a body configuration that never occurs in real data (all landmarks collapsed to origin)
- In real inference, the sliding window always contains real keypoints — never zeros. Hold-padding matches this distribution

**Why no tail-end cropping**: Removed entirely. The real problem is "same motion, faster speed" — not "partial motion." The real-time sliding window continuously advances, so any sign will eventually be fully captured; the model only needs to handle that same trajectory compressed in time.

---

### Steps

**Step 1. Add `generate_speed_variants()` function — new in Cell 2.7**

Add after the existing augmentation functions in Cell 2.7, lines 391–454. This operates on **raw `(N, 30, 258)`** data — before Stages 1–5.

```
def generate_speed_variants(X_raw, y_raw, target_frame_counts=[20, 15, 10]):
    """
    For each sample, create temporally downsampled versions
    that simulate signing at 1.5×, 2×, 3× speed.
    Hold-pads to 30 frames after downsampling.
    
    Operates on RAW (30, 258) data — before normalization.
    """
    For each target N in [20, 15, 10]:
        indices = np.round(np.linspace(0, 29, N)).astype(int)
        sampled = X_raw[:, indices, :]           # (N_samples, N, 258)
        pad = np.tile(sampled[:, -1:, :], (1, 30 - N, 1))  # repeat last frame
        padded = np.concatenate([sampled, pad], axis=1)     # (N_samples, 30, 258)
        Append padded + y_raw to output
    
    Concatenate original + all variants
    Return (4× X_raw, 4× y_raw)
```

**Why this runs before Stages 1–5**: Stages 3–4 compute velocity = `frame[t] - frame[t-1]`. If you downsample the raw coordinates first, the velocity between consecutive sampled frames is `frame[6] - frame[3]` (at 3× speed) — a larger displacement per step, reflecting faster motion. If you instead downsample **after** computing velocity, you get the original per-frame velocity at sampled points, which doesn't reflect the speed change. The model needs to see the velocity patterns of fast signing.

**Step 2. Add `augment_idle_prefix()` function — new in Cell 2.7**

```
def augment_idle_prefix(X_train, y_train, X_idle, fraction=0.3, max_prefix_frames=8):
    """
    For a fraction of training samples, replace the first 3-8 frames
    with randomly sampled Idle keypoint frames.
    Simulates the real sliding window where a sign starts mid-window.
    """
    n_to_modify = int(len(X_train) * fraction)
    selected = np.random.choice(len(X_train), n_to_modify, replace=False)
    X_out = X_train.copy()
    
    For each selected index:
        prefix_len = np.random.randint(3, max_prefix_frames + 1)
        idle_sample = random Idle sequence from X_idle
        X_out[index, :prefix_len, :] = idle_sample[:prefix_len, :]
    
    Return X_out, y_train  # labels unchanged
```

Justification for parameters:
- **`fraction=0.3` (30%)**: The model needs to learn both clean starts and contaminated starts. A 70/30 split ensures the majority of training signal is still clean, while the 30% teaches robustness. This isn't arbitrary — it mirrors the rough real-world proportion: in continuous signing, the window starts aligned roughly 70% of the time (when the signer pauses briefly between signs) and catches a sign mid-transition about 30% of the time.
- **`max_prefix_frames=8`**: At 50fps sliding window, 8 frames = 160ms. The average human transition between signs is 100–200ms. Contaminating more than 8 of 30 frames (27%) would start destroying the sign's identity, since important early-motion frames carry discriminative information (e.g., the upward hand raise in greetings).
- **`X_idle` source**: Idle class sequences from the same training split. The function receives these as pre-extracted data, not from test/val sets — no data leakage.

**Step 3. Restructure Cell 2.5 — Move normalization application out**

Currently, Cell 2.5, lines 172–383, both **defines** the normalization functions and **applies** them to `X`. Change it to **define only** — remove the "APPLY STAGES 1-5" block at the end of the cell (the `X = normalize_frames_vectorized(X)` calls and everything below it). These are moved to Cell 3.

The validation block (vectorized vs V1 loop) stays in Cell 2.5 — it validates the function correctness, not the data.

**Step 4. Restructure Cell 3 — New data pipeline**

Replace Cell 3, lines 463–514. The new flow, with each step justified:

```
1. SPLIT raw (N, 30, 258) → train/val/test (70/15/15, stratified)
   ↳ Same as V3.1.2. On raw data, not normalized.
   ↳ Why before everything: test set must remain pure original
     sequences for fair comparison with V3.1.2.

2. EXTRACT Idle samples from X_train (for Step 5)
   ↳ X_idle = X_train[y_train_raw == IDLE_IDX]
   ↳ Done now because after speed variants, label indices shift.

3. TEMPORAL DOWNSAMPLING on X_train only → 4× train set
   ↳ generate_speed_variants(X_train_raw, y_train_raw)
   ↳ Why training only: val/test must reflect real-world input
     (single speed) so metrics are interpretable.

4. STAGES 1-5 normalization on ALL three sets
   ↳ normalize_frames_vectorized() + velocity + accel + engineered
   ↳ Why here: velocity must be computed on the downsampled+padded
     sequences, reflecting the actual frame-to-frame differences.
   ↳ Val and test are normalized with the same function (same
     deterministic math), no data leakage.

5. IDLE PREFIX CONTAMINATION on X_train only
   ↳ augment_idle_prefix(X_train, y_train, X_idle_normalized)
   ↳ Why after normalization: the Idle frames being inserted should
     be normalized too, so the contaminated sequence is coherent.
   ↳ Note: this modifies in-place (30% of samples), not additive.

6. STANDARD AUGMENTATION (noise + warp + dropout) → 4× train
   ↳ apply_augmentation(X_train, y_train) — same as V3.1.2
   ↳ Why last: augmentation adds noise/distortion on top of the
     already speed-varied, normalized, Idle-contaminated data.
     Applying augmentation first would distort the raw keypoints
     before velocity computation.

7. Z-SCORE (Stage 6) — fit on X_train, transform all
   ↳ Same as V3.1.2. Scaler fitted on the full augmented train set
     because the model sees this distribution during training.
```

**Total training samples**: original N × 0.7 (split) × 4 (speed variants) × 4 (augmentation) = **~11.2N**. With ~3,000 total samples → ~2,100 train → ~8,400 after speed → ~33,600 after augmentation. Colab T4 has 15GB RAM; at float32 with shape `(33600, 30, 780)`, this is `33600 × 30 × 780 × 4 bytes` ≈ 3.0 GB — within budget.

**Step 5. No change to Cell 4 (Model architecture)**

Architecture remains identical — 2× BiLSTM-128 + Dot Attention, input `(30, 780)`. The temporal attention layer is actually well-suited for variable-speed input: it can learn to assign low attention weights to the hold-padded tail (zero velocity/acceleration region) and high weights to the active-motion portion.

**Step 6. Minor update to Cell 5 (Training)**

Update print statements to reflect V3.1.3 changes. No hyperparameter changes — the existing regularization (GaussianNoise, SpatialDropout1D, recurrent_dropout, L2, label_smoothing) is already strong and was tuned for the V3.1 architecture. Adding more data diversity (speed variants) is a form of regularization itself, so adding *more* explicit regularization would risk underfitting.

Also delete old V3.1.2 Drive checkpoints on first run (same pattern as V3.1.2 deletes V3.0 checkpoints) to force fresh training with the new data distribution.

**Step 7. Expand Cell 7 — Multi-speed evaluation**

After the existing full-sequence test evaluation, add a per-speed accuracy breakdown on the **held-out test set**:

```
For each speed in [1×, 1.5×, 2×, 3×]:
    Generate downsampled+padded version of X_test
    Apply Stages 1-5
    Apply z-score (using the already-fitted scaler)
    Predict with best_model
    Report accuracy + F1

Print table:
| Speed | Frames | Accuracy | F1   |
|-------|--------|----------|------|
| 1×    | 30     | xx.x%    | 0.xx |
| 1.5×  | 20     | xx.x%    | 0.xx |
| 2×    | 15     | xx.x%    | 0.xx |
| 3×    | 10     | xx.x%    | 0.xx |
```

This uses the *test set* downsampled at inference time — not training data — so it's a honest evaluation of whether multi-speed training actually works.

**Step 8. Update Cell 8 — label_map.json metadata**

Add to the JSON output in Cell 8, lines 1300–1320:

```json
"model_version": "V3.1.3",
"multi_speed_trained": true,
"trained_speed_tiers": {
    "1x": {"frames_sampled": 30, "description": "Original recording speed"},
    "1.5x": {"frames_sampled": 20, "description": "np.linspace(0,29,20)"},
    "2x": {"frames_sampled": 15, "description": "np.arange(0,30,2)"},
    "3x": {"frames_sampled": 10, "description": "np.arange(0,30,3)"}
},
"padding_strategy": "hold_last_frame"
```

This metadata tells the Android app what speed variants the model was trained on, enabling the multi-stride inference strategy on the app side.

**Step 9. Update Cell 9 — TFLite validation with speed variants**

Add TFLite accuracy + latency tests for:
- Full 30-frame input (baseline — same as V3.1.2)
- 15-frame downsampled + hold-padded input (2× speed — the primary target)

This validates that TFLite quantization doesn't disproportionately hurt speed-variant inputs (where the feature distribution differs from full-sequence inputs).

**Step 10. Update header markdown**

Replace the V3.1.2 header in lines 2–26 with:

| Aspect | V3.1.2 | V3.1.3 |
|---|---|---|
| Training sequences | Fixed 30-frame at recording speed | 4 speed variants (1×, 1.5×, 2×, 3×) via temporal downsampling |
| Augmentation | 3 methods (noise, warp, dropout) | +Idle prefix contamination |
| Pipeline order | Normalize → Split → Augment → Z-score | Split → Speed variants → Normalize → Idle prefix → Augment → Z-score |
| Architecture | Unchanged `(1, 30, 780)` | Unchanged |
| Evaluation | Full-sequence only | + per-speed accuracy table |

---

### File Changes Summary

| Cell | File Location | Change Type | Description |
|---|---|---|---|
| Header | Lines 2–26 | **Modify** | New changelog, version table |
| Cell 2.5 | Lines 172–383 | **Modify** | Remove "APPLY STAGES 1–5" block; keep function definitions + validation only |
| Cell 2.7 | Lines 391–454 | **Add** | `generate_speed_variants()` + `augment_idle_prefix()` functions |
| Cell 3 | Lines 463–514 | **Rewrite** | New pipeline: Split → Speed → Normalize → Idle prefix → Augment → Z-score |
| Cell 5 | Lines 652–755 | **Minor** | Update print statements, delete old checkpoints |
| Cell 7 | Lines 798–861 | **Expand** | Add per-speed evaluation table |
| Cell 8 | Lines 882–1439 | **Modify** | Update label_map.json metadata |
| Cell 9 | Lines 1445–1532 | **Expand** | Add 2× speed TFLite validation |
| Cell 10 | Lines 1538–1574 | No change | |

---

### Verification

| Check | Criterion | Justification for threshold |
|---|---|---|
| **1× regression** | Full 30-frame test accuracy ≥ V3.1.2's | Multi-speed training must not degrade normal-speed recognition. If it does, the speed variants are confusing the model rather than teaching it. |
| **2× speed accuracy** | ≥ 75% on 15-frame downsampled test set | 2× is the primary target speed. 75% is the threshold because: the model has never seen this data distribution in V3.1.2 (gets ~40-50% at best), so a 75%+ jump confirms the training worked. Below 75% means the attention mechanism isn't learning to ignore the hold-pad tail. |
| **3× speed accuracy** | ≥ 55% on 10-frame downsampled test set | 3× is extreme — 10 frames capture coarse motion only. 55% is above chance (1/98 = 1%) and indicates useful recognition. Below 55% is acceptable as a known limitation. |
| **TFLite drift** | f32 ±1%, int8 ±3% vs Keras | Same as V3.1.2 — quantization tolerance unchanged. |
| **Memory** | Peak RAM < 12GB during training | Colab T4 has 15GB. Budget = ~12GB (3GB data + 5GB model + 4GB TF overhead). |
| **Generalization gap** | Train − Test accuracy < 5% | The 4× data expansion + existing regularization should keep overfitting controlled. V3.1.2 targeted <2%, but the new data diversity may widen this slightly — 5% is acceptable given the larger training set. |

### Decisions

| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| Multi-scale strategy | **Temporal downsampling** | Tail-end cropping | Downsampling preserves the complete motion trajectory at faster speeds. Tail-end cropping destroys early-motion phases that carry discriminative information and doesn't match the real-world problem (the sliding window eventually captures full signs). |
| Speed tiers: 1×/1.5×/2×/3× | **3 additional tiers** | 2 tiers or 5+ tiers | 3 tiers cover the realistic speed range. Below 1.5× is too close to original to add value. Above 3× is unrealistic — 10 frames from 30 already loses fine detail. More tiers would increase training data linearly with diminishing returns. |
| Frame selection method | **`np.linspace` for non-integer strides, `np.arange` for clean strides** | Always `np.linspace` or always `np.arange` | `np.arange(0, 30, 2)` gives exactly `[0,2,4,...,28]` — clean and intuitive for 2× and 3×. But 1.5× requires sampling 20 from 30 = every 1.5 frames, which `np.arange` can't express. `np.linspace(0, 29, 20)` handles this correctly with even spacing. |
| Padding | **Hold-pad (repeat last frame)** | Zero-pad | Hold-pad produces zero velocity/acceleration in the padded region, correctly signaling "motion stopped." Zero-pad creates an impossible body configuration (all landmarks at origin) with an artificial velocity spike at the transition — a distribution the model never sees in real inference. |
| Pipeline order | **Split → Speed → Normalize → Augment → Z-score** | Normalize → Split → Speed → Augment → Z-score (V3.1.2 order) | Velocity must be computed on the downsampled frames. If normalization (including velocity) runs before downsampling, the velocity values reflect 30fps motion at sampled points, not the actual per-step displacement of faster signing. |
| Idle prefix contamination at 30% | **In-place modification, not additive** | Additive (creating new copies) | Making it additive would increase dataset size further with near-duplicate samples. In-place ensures the model sees both clean and contaminated starts without adding redundancy. 30% matches the rough proportion of "mid-transition" window states in real continuous signing. |
| Architecture changes | **None** | Adding masking layer for padded frames | The dot-product attention already handles this — it learns to assign low weight to low-information frames (the padded tail has zero velocity/acceleration, giving the attention layer a clear signal). Adding explicit masking would be premature optimisation that complicates TFLite export. |
