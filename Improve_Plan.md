## Plan: High-Speed Sign Detection Pipeline

### TL;DR

The current pipeline hits ~30fps because **MediaPipe runs both Pose and Hand on CPU sequentially**, there's a **double Bitmap allocation per frame**, and the **30-frame fixed window + cooldown (5-10 frames)** causes missed fast signs. The fix has two prongs: **(A) speed up the pipeline** to 45-50fps via GPU delegate, hardware parallelism, hand crop strategy, and bitmap pooling; **(B) replace fixed cooldown with continuous sliding-window prediction + confidence peak detection** so both 5-frame static signs and 15-frame dynamic signs are caught without per-class configuration. Retraining with variable-length sequences is a recommended Phase 2 enhancement.

---

### Current Bottleneck Breakdown

| Bottleneck | Where | Cost | Impact |
|---|---|---|---|
| MediaPipe Pose on CPU | LandmarkExtractor.kt `Delegate.CPU` | ~15-20ms | **Biggest bottleneck** |
| MediaPipe Hand on CPU | LandmarkExtractor.kt `Delegate.CPU` | ~15-20ms | Runs in parallel with pose, but both compete for CPU cores |
| Double Bitmap allocation | CameraScreen.kt `toBitmap()` + `createBitmap()` | ~3-5ms + GC | Per-frame GC pressure at 30fps |
| Cooldown blocks 5-10 frames | SignPredictor.kt `cooldownCounter.set(10)` | 167-333ms blind | Fast signs missed entirely |
| Fixed 30-frame window | SignPredictor.kt `frameBuffer.size == 30` | 1s cold start | No adaptivity to sign duration |
| Camera not requesting 60fps | CameraScreen.kt | Capped at 30fps | Hardware supports 60fps but CameraX defaults to 30 |

---

### Steps

#### Phase 1: Speed — Reach 45-50fps (App-Side Only)

**Step 1. Enable 60fps camera output**
In CameraScreen.kt, use Camera2Interop to request 60fps from the sensor. Add `CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE` set to `Range(45, 60)` on the `ImageAnalysis.Builder`. Also add the `camera-camera2` interop dependency if not already present (it is present as `camera-camera2` in build.gradle.kts). Reduce analysis resolution from `640×480` to `480×360` — MediaPipe doesn't need higher resolution for landmark detection.

**Step 2. MediaPipe — GPU for Pose, CPU for Hand (hardware parallelism)**
In LandmarkExtractor.kt:
- Change `PoseLandmarker` delegate from `Delegate.CPU` → `Delegate.GPU`
- Keep `HandLandmarker` on `Delegate.CPU`
- Both already use `LIVE_STREAM` mode with async callbacks, so they naturally run in parallel
- **Why this split**: Pose on GPU (~10ms) runs on the Mali-G720. Hand on CPU (~12-15ms) runs on A725 cores. True hardware parallelism means `max(10ms, 12ms) = 12ms` instead of the current `~25-30ms` (both competing for CPU)
- Add a GPU fallback: if `Delegate.GPU` init fails, fall back to CPU with a log warning

**Step 3. Eliminate double Bitmap allocation**
In CameraScreen.kt:
- Pre-allocate a reusable `Bitmap` at the analysis resolution (480×360) using `Bitmap.createBitmap(width, height, ARGB_8888)`
- Use `Canvas` to draw the rotated/mirrored frame onto the pre-allocated bitmap instead of `Bitmap.createBitmap(..., matrix, true)` which allocates a new bitmap every frame
- This eliminates ~60-100 allocations/sec and associated GC pauses

**Step 4. Hand crop strategy using previous frame's wrist positions**
In LandmarkExtractor.kt:
- Cache the last known wrist positions (pose landmarks 15 and 16) from the most recent `onPoseResult` callback
- For each new frame, **before** sending to `HandLandmarker`, crop a 160×160 region centered on each cached wrist position
- Run `HandLandmarker` on these small crops instead of the full frame
- The 1-frame latency in wrist positions is negligible at 50fps (~20ms, wrists barely move)
- This cuts hand detection time from ~12-15ms to ~6-8ms on CPU
- Handle edge cases: if no previous wrist data exists (first frame), fall back to full-frame hand detection; if wrist is near frame border, clamp the crop region
- **Remap hand landmark coordinates** back to full-frame space after detection (scale by crop dimensions + add crop offset)

**Step 5. Pipeline TFLite inference with MediaPipe**
In SignPredictor.kt:
- Currently inference runs on `Dispatchers.Default` but blocks the next prediction via `isPredicting` — this is fine, they use different hardware (NPU vs GPU/CPU)
- The real change: remove the requirement that inference must complete before the **next frame enters the buffer**. Currently `onLandmarksExtracted` holds `synchronized(frameBuffer)` while checking conditions and launching inference
- Restructure so buffer insertion and inference launching are decoupled: insert the frame immediately, then check inference eligibility outside the synchronized block
- This prevents the buffer lock from blocking incoming MediaPipe callbacks

**Expected performance after Phase 1:**

| Operation | Hardware | Time |
|---|---|---|
| Camera frame capture | ISP | <1ms |
| Bitmap transform (reuse) | CPU | ~1ms |
| MediaPipe Pose (GPU) | Mali-G720 | ~10ms |
| MediaPipe Hand (CPU, cropped) | A725 cores | ~7ms |
| *Pose + Hand parallel* | | *~10ms total* |
| Normalize (Stages 1-2) | CPU | <1ms |
| TFLite BiLSTM (NPU, pipelined) | APU 880 | ~3ms (overlapped) |
| **Per-frame total** | | **~12-14ms → 45-50fps** |

---

#### Phase 2: Dynamic Sign Duration — Catch Fast and Slow Signs

**Step 6. Replace cooldown with continuous sliding-window prediction**
In SignPredictor.kt:
- **Remove** `cooldownCounter` entirely
- **Remove** the `isPredicting` single-inference gate (or change it to allow prediction every N frames using a simple frame counter)
- Instead, predict **every 2 frames** (configurable stride). At 50fps, this is 25 inferences/sec. Each inference costs ~3ms on NPU = 75ms/sec of NPU time — well within budget
- The sliding window continues to be 30 frames, but it now produces predictions at 25Hz instead of the current ~3-5Hz

**Step 7. Confidence-based sign emission (peak detection)**
Add a new class `SignSegmenter` in ml that replaces the current `updatePrediction()` logic. The segmenter tracks a rolling window of recent predictions and emits signs based on **confidence pattern** rather than fixed cooldown:

- **State machine** with 3 states:
  - `IDLE` — model is predicting "Idle" or confidence < threshold (0.5)
  - `TRACKING` — a non-Idle class is being predicted with confidence ≥ threshold. Track: the class, entry timestamp, peak confidence
  - `EMITTING` — the tracked class has been confirmed; emit and transition
- **Transition rules**:
  - `IDLE → TRACKING`: when a non-Idle class appears with confidence ≥ 0.5 for `MIN_ONSET` consecutive predictions (e.g., 2 predictions = ~80ms at 25Hz). This filters noise
  - `TRACKING → EMITTING`: when EITHER (a) the class has persisted for `MIN_HOLD` predictions (e.g., 3 = ~120ms), OR (b) confidence exceeds a high-confidence threshold (0.85) for even 2 predictions (fast commit for obvious signs)
  - `EMITTING → IDLE`: after emission, enter a short `DEBOUNCE` period (3 predictions = ~120ms) before tracking a new sign. This prevents double-triggers but is much shorter than the current 333ms cooldown
  - `TRACKING → IDLE`: if the class drops below threshold or changes to a different class before `MIN_HOLD` is reached → discard as noise

- **Why this handles variable duration naturally**:
  - **Fast static sign (5 frames)**: appears in the window, confidence spikes quickly. `MIN_ONSET=2` + `MIN_HOLD=3` = 5 predictions ≈ 200ms total response time. Since the sign is static, the 30-frame window will contain many frames of the same pose → strong confidence
  - **Slow dynamic sign (15+ frames)**: confidence builds gradually as the motion pattern fills the window. The segmenter tracks it throughout and emits when confirmed. No timeout kills it early
  - **No per-class configuration needed**: the model's own confidence patterns adapt. Static signs saturate confidence quickly; complex signs take longer

**Step 8. Exponential moving average (EMA) smoothing on predictions**
Before feeding predictions to the segmenter, smooth them with per-class EMA:
```
smoothed_conf[class] = α × raw_conf[class] + (1-α) × prev_smoothed_conf[class]
```
Use α = 0.4 (responsive but filters single-frame noise). This prevents the segmenter from triggering on a single anomalous frame. Apply to all 98 classes per prediction.

---

#### Phase 3: Model Enhancement (Retraining — Optional)

**Step 9. Retrain with multi-scale window (if Phase 2 accuracy is insufficient)**
If the fixed 30-frame window misses very fast signs even with the segmenter:
- Retrain the model with **variable-length sequences** padded to 30 frames
- During training, for each sign video, sample sub-sequences of length 10, 15, 20, 25, 30 frames, pad shorter sequences with the last frame (hold padding) or zeros, and train with all sub-sequence lengths
- This teaches the model to recognize signs even when they only occupy part of the window
- The TFLite input shape remains (1, 30, 780) — only the training data changes
- Add a masking layer or a sequence-length feature so the model knows how many frames are "real"

**Step 10. Multi-stride inference (alternative to retraining)**
If you don't want to retrain, run inference at **two strides simultaneously**:
- Primary window: last 30 frames (current approach, stride 2)
- Fast window: last 15 frames padded to 30 by repeating each frame twice (effectively 2× time-stretch)
- Take the prediction with higher confidence from either window
- This lets the model "see" fast signs at a slower playback speed, improving recognition
- Cost: doubles TFLite work (6ms instead of 3ms on NPU) — still well within budget

---

### File Change Summary

| File | Changes |
|---|---|
| CameraScreen.kt | Camera2Interop 60fps, lower resolution to 480×360, reusable Bitmap pool |
| LandmarkExtractor.kt | Pose → GPU delegate, hand crop strategy with cached wrist positions, coordinate remapping |
| SignPredictor.kt | Remove cooldown, predict every 2 frames, decouple buffer lock from inference launch |
| **New: `SignSegmenter.kt`** | Confidence peak detection state machine, EMA smoothing, sign emission logic |
| build.gradle.kts | Possibly add `camera-camera2` interop if needed for Camera2Interop |

---

### Verification

- **FPS test**: measure displayed FPS counter — target 45-50fps sustained
- **Latency test**: perform a fast sign (e.g., "Satu" — a quick number sign) and measure time from gesture start to UI display. Target: <300ms
- **Accuracy test**: compare prediction accuracy with the Python `test_webcam_v2.py` — fast signs should now be caught that were previously missed
- **GPU delegate stability**: run for 5+ minutes continuously — verify no crashes, OOM, or frame drops
- **Variable duration test**: perform a mix of fast static signs ("Satu", "Dua") and slow dynamic signs ("Assalamualaikum", "Terima Kasih") back-to-back — verify both types are detected
- **Double-trigger test**: hold a static sign — verify it emits only once, not repeatedly

### Decisions

- **GPU for Pose + CPU for Hand** over both-on-GPU: true hardware parallelism (Mali-G720 + A725 cores) is faster than sharing the GPU
- **Hand crop with cached wrist positions** over full-frame hand detection: ~50% hand detection speedup at the cost of 1-frame wrist position latency (negligible at 50fps)
- **Confidence peak segmenter** over per-class frame counts: no need to encode 98 frame-count rules; the model's confidence pattern naturally differentiates sign durations
- **Predict every 2 frames** over every frame: 25 inferences/sec is sufficient to catch all signs; every frame would be 50/sec with marginal benefit
- **EMA smoothing (α=0.4)** over raw predictions: filters noise without adding perceptible latency (~80ms effective lag)
- **Retraining with multi-scale windows** kept as Phase 3: app-side optimizations should solve most cases; retraining is the nuclear option if accuracy remains insufficient
