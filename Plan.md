## Plan: iSuara Android App — Framework & Architecture

### TL;DR

**Use native Kotlin + Jetpack Compose.** Flutter adds a Dart↔native bridge penalty on every camera frame (30+ fps), and Google AI Studio is a prototyping tool, not an app framework. Native Kotlin gives you zero-copy camera frames, direct GPU/NPU delegate access for TFLite, and the official MediaPipe Tasks Android SDK — all without bridge overhead. For a single-page app, the "complexity" of native Android is effectively the same as Flutter.

---

### Framework Comparison

| Criteria | **Native Kotlin (Compose)** | **Flutter** | **Google AI Studio** |
|---|---|---|---|
| TFLite NPU/GPU delegate | Direct `NnApiDelegate` + `GpuDelegate` — first-class | Via `tflite_flutter` plugin — wraps native, adds FFI bridge overhead per frame | N/A (not an app framework) |
| MediaPipe Tasks | Official Android SDK (`com.google.mediapipe:tasks-vision`) — actively maintained | `google_mlkit_pose_detection` — incomplete (no hand landmarks), or custom platform channel to Java MediaPipe | N/A |
| Camera → ML pipeline | CameraX `ImageAnalysis` callback — zero-copy `ImageProxy` on a dedicated thread | `camera` plugin → platform channel → Dart isolate → platform channel → TFLite. 2 bridge crossings per frame | N/A |
| NPU acceleration | `NnApiDelegate()` — hardware vendor maps ops to NPU automatically. Dimensity 8400's APU 880 is fully supported via NNAPI | Same delegate available, but must go through FFI. `IsolateInterpreter` adds overhead | N/A |
| Multithreading | Kotlin coroutines + `ImageAnalysis` executor — camera, ML, UI on separate threads natively | Dart is single-threaded; heavy work needs `Isolate` + platform channels. Camera plugin manages its own thread but you can't control it | N/A |
| TTS | `android.speech.tts.TextToSpeech` — 3 lines of code, supports Bahasa Melayu | `flutter_tts` plugin — wraps the same native API, works fine | N/A |
| App size | ~8 MB APK (Compose + TFLite + MediaPipe) | ~15-20 MB (Flutter engine + plugins + TFLite + MediaPipe) | N/A |
| One-page app complexity | Equally simple — one `@Composable` function | Equally simple — one `StatefulWidget` | N/A |

**Google AI Studio** is for prompt-testing Gemini models in a browser — it doesn't build mobile apps. Dismissed.

**Flutter** works but adds unnecessary overhead for a real-time ML pipeline. Every camera frame crosses the Dart↔native bridge twice, and you'd need custom platform channels for MediaPipe hand landmarks anyway (the Flutter ML Kit plugin doesn't expose hand landmarks the way you need them).

**Native Kotlin** is the clear winner for this use case: real-time camera → ML inference → UI, with NPU acceleration.

---

### Pipeline Simplification (Python → Kotlin)

The current Python pipeline has 6 stages. Here's what simplifies on mobile:

| Stage | Python (test_webcam_v2.py) | Android (Kotlin) | Simplification |
|---|---|---|---|
| Keypoint extraction | MediaPipe Holistic (deprecated API) | MediaPipe Tasks: `PoseLandmarker` + `HandLandmarker` — merge outputs to 258-vector | Use newer, supported API |
| Stage 1-2 (anchor/scale) | `normalize_single_frame()` — pure NumPy | Pure Kotlin `FloatArray` math — no library needed | Same logic, ~40 lines |
| Stage 3-5 (vel/accel/eng) | `build_sequence_features()` — NumPy | Pure Kotlin `FloatArray` — no library needed | Same logic, ~50 lines |
| Stage 6 (z-score) | Baked in model | Baked in model | No work needed |
| TFLite inference | `ai-edge-litert` interpreter | `org.tensorflow:tensorflow-lite` + `NnApiDelegate` | Native, faster |
| Gemini translation | `google-genai` Python SDK | Gemini REST API or `com.google.ai.client.generativeai` SDK | Same prompt |
| UI | OpenCV overlays | Jetpack Compose `CameraPreview` + `Text` | Much cleaner |
| TTS | None | `android.speech.tts.TextToSpeech` | 3 lines |

---

### Recommended Architecture (Single-Page App)

```
┌─────────────────────────────────────────────┐
│              UI Thread (Compose)            │
│  ┌─────────────────────────────────────┐    │
│  │  CameraX Preview (full screen)      │    │
│  │  ┌───────────────────────────────┐  │    │
│  │  │  Overlay: detected word +     │  │    │
│  │  │  confidence bar               │  │    │
│  │  └───────────────────────────────┘  │    │
│  │  ┌───────────────────────────────┐  │    │
│  │  │  Bottom: sentence text +      │  │    │
│  │  │  [Translate] button           │  │    │
│  │  └───────────────────────────────┘  │    │
│  └─────────────────────────────────────┘    │
└──────────────────┬──────────────────────────┘
                   │ StateFlow
┌──────────────────┴──────────────────────────┐
│         ML Thread (CameraX Executor)        │
│  Frame → MediaPipe → extract 258 kp        │
│  → normalize (stages 1-2)                   │
│  → sliding buffer (30 frames)               │
│  → build features (stages 3-5) → 780       │
│  → TFLite invoke (NnApiDelegate/GPU)        │
│  → argmax → emit prediction via StateFlow   │
└─────────────────────────────────────────────┘
                   │ coroutine
┌──────────────────┴──────────────────────────┐
│         IO Thread (Dispatchers.IO)          │
│  Gemini API call (on [Translate] press)     │
│  TTS playback                               │
└─────────────────────────────────────────────┘
```

**3 threads, zero bridge overhead:**
1. **UI thread** — Compose rendering only
2. **ML thread** — CameraX `ImageAnalysis` executor runs MediaPipe + normalization + TFLite. All native, no copies
3. **IO thread** — Gemini network call + TTS (triggered by button)

---

### NPU Utilization (Dimensity 8400 / APU 880)

```kotlin
// In your TFLite setup:
val nnApiOptions = NnApiDelegate.Options().apply {
    setUseNnapiCpu(false)           // force hardware accelerator (NPU)
    setAllowFp16(true)              // allow FP16 on NPU — faster
    setExecutionPreference(         // prefer sustained speed
        NnApiDelegate.Options.EXECUTION_PREFERENCE_SUSTAINED_SPEED
    )
}
val nnApiDelegate = NnApiDelegate(nnApiOptions)

val interpreterOptions = Interpreter.Options().apply {
    addDelegate(nnApiDelegate)
    setNumThreads(4)                // fallback CPU threads for unsupported ops
}
val interpreter = Interpreter(modelFile, interpreterOptions)
```

The NNAPI delegate automatically maps supported ops (Dense, Conv, etc.) to the NPU. BiLSTM ops that the NPU doesn't support fall back to the 4 CPU threads. On Dimensity 8400 with 8× A725 cores, set `numThreads = 4` (uses big cores).

---

### Steps

1. **Create Android project** — Android Studio, Kotlin, Jetpack Compose, min SDK 26, target SDK 34
2. **Add dependencies** — `tensorflow-lite`, `tensorflow-lite-gpu`, `tensorflow-lite-support`, `mediapipe-tasks-vision`, `generativeai`, CameraX
3. **Copy model assets** — Put `bim_lstm_v3_int8.tflite`, `bim_lstm_v3_f32.tflite`, `label_map.json` into `app/src/main/assets/`
4. **Port normalization** — Translate `normalize_single_frame()` and `build_sequence_features()` to Kotlin `FloatArray` operations (~90 lines)
5. **Build MediaPipe landmarker** — Use `PoseLandmarker` + `HandLandmarker` (Tasks API), merge into 258-float vector matching the training format
6. **Build TFLite interpreter** — Load int8 model with `NnApiDelegate`, fallback to f32 + `GpuDelegate` if NNAPI fails
7. **Implement sliding window** — 30-frame `ArrayDeque<FloatArray>`, predict when full, cooldown logic
8. **Build Compose UI** — Single `Activity`, `CameraPreview` + overlay `Text` for prediction + sentence + translate button
9. **Add Gemini translation** — HTTP POST to Gemini API with the same prompt, on `Dispatchers.IO`
10. **Add TTS** — `TextToSpeech` engine, set locale to `ms_MY` (Bahasa Melayu), speak the translated sentence

### Verification
- Test inference latency: should be <5ms on NPU, <10ms on CPU for the 1.4M-param model
- Test end-to-end FPS: target 15+ fps (MediaPipe is the bottleneck, not TFLite)
- Verify the 258-feature vector from MediaPipe Tasks matches the training format (same landmark order, same x/y/z/visibility layout)
- Compare predictions with `test_webcam_v2.py` on the same signs to verify port correctness

### Decisions
- **Native Kotlin over Flutter**: eliminates Dart↔native bridge overhead on every frame; direct NPU delegate access; official MediaPipe Tasks SDK
- **NNAPI delegate over GPU delegate**: NPU is faster and more power-efficient than GPU for this model size; GPU delegate is fallback
- **MediaPipe Tasks API over legacy Holistic**: Holistic is deprecated; Tasks API is the supported path on Android
- **Single Activity + Compose**: minimal architecture for a one-page app


# IMPORTANT CLARIFICATION
### 1. FP16 vs Int8 on Dimensity 8400 Ultra (APU 880)

**Int8 is faster.** The `setAllowFp16(true)` line I suggested is a **fallback option** — it tells the NPU "if you can't do int8 for a specific op, try fp16 instead of falling back to CPU." It doesn't mean running the whole model in fp16. The priority chain is:

```
Int8 (fastest) → FP16 (if int8 op unsupported) → CPU float32 (last resort)
```

MediaTek's APU 880 has dedicated int8 MAC units. Your int8 TFLite model will run on those. The `allowFp16` just prevents unnecessary CPU fallback for any ops the int8 path doesn't cover (like certain activation functions).

**However** — your "int8" model is actually **dynamic-range quantized** (weights int8, activations float32), not full int8. The NPU will still accelerate the weight-heavy ops (Dense, LSTM kernels) using int8 weights, but activations flow as float32. This is still significantly faster than pure float32.

### 2. Separated PoseLandmarker + HandLandmarker vs Legacy Holistic

**It depends on how you use them:**

| Approach | What happens | Speed |
|---|---|---|
| **Legacy Holistic** | Runs pose + face + hands in a single optimized pipeline. One model, one pass. | ~25-35ms per frame on mobile |
| **Tasks API: Pose + 2× Hand separately** | 3 separate model invocations per frame. Each is lighter, but combined = more total work. | ~40-60ms per frame (slower!) |
| **Tasks API: Pose only → crop hands → Hand** | Run pose first, use wrist coordinates to crop hand regions, then run hand landmarker on crops only. | ~30-40ms (comparable to Holistic) |

The Tasks API is **not inherently faster** — it's just the actively maintained path. Holistic was actually well-optimized as a single pipeline. The advantage of Tasks API is that it's **supported** and you can run pose on GPU while hands run on CPU in parallel.

### 3. How to Hit 30+ FPS on Dimensity 8400 Ultra

You're getting 25-30 fps on a **laptop CPU**. On Dimensity 8400 Ultra with NPU + 8× A725, you should target **30-45 fps**. Here's how:

**A. Pipeline the MediaPipe + TFLite work (biggest win)**

Don't run them sequentially. Pipeline them:

```
Frame 1: [MediaPipe landmarks]
Frame 2: [MediaPipe landmarks] + [TFLite predict on Frame 1's features]
Frame 3: [MediaPipe landmarks] + [TFLite predict on Frame 2's features]
```

MediaPipe and TFLite can run on **different hardware**: MediaPipe on GPU, TFLite on NPU. They never block each other.

**B. Run MediaPipe on GPU delegate**

```kotlin
val poseOptions = PoseLandmarker.PoseLandmarkerOptions.builder()
    .setBaseOptions(BaseOptions.builder()
        .setDelegate(Delegate.GPU)      // ← GPU, not CPU
        .build())
    .setRunningMode(RunningMode.LIVE_STREAM)  // ← async callback, doesn't block
    .setNumPoses(1)
    .build()
```

`LIVE_STREAM` mode is critical — it returns results via callback instead of blocking the camera thread.

**C. Skip frames intelligently**

You don't need to run MediaPipe on every camera frame. At 30 fps camera, run MediaPipe every **2nd frame** and interpolate landmarks:

```kotlin
// In ImageAnalysis callback:
frameCounter++
if (frameCounter % 2 == 0) {
    // Run MediaPipe on this frame
} else {
    // Reuse last landmarks (or interpolate)
}
```

This halves MediaPipe load while maintaining smooth predictions.

**D. Run TFLite on NPU with int8 model**

Already covered — your model is only 1.5 MB and 1.4M params. TFLite inference alone should be **<3ms** on the NPU.

**E. Use Pose-only + hand crop strategy**

Instead of running full HandLandmarker on the entire frame:
1. Run `PoseLandmarker` (fast, ~15ms on GPU)
2. Extract wrist positions from pose landmarks
3. Crop a small ROI around each wrist (~150×150 pixels)
4. Run `HandLandmarker` on the small crops only (~8ms each)

This is ~30% faster than running HandLandmarker on the full frame.

**F. Reduce camera resolution**

```kotlin
val imageAnalysis = ImageAnalysis.Builder()
    .setTargetResolution(Size(480, 360))  // ← smaller than preview
    .setBackpressureStrategy(STRATEGY_KEEP_ONLY_LATEST)
    .build()
```

MediaPipe doesn't need 1080p — 480×360 is enough for landmark detection and dramatically reduces processing time.

### Expected Performance Budget (Dimensity 8400 Ultra)

| Operation | Hardware | Estimated Time |
|---|---|---|
| Camera frame capture | ISP | <1ms |
| MediaPipe Pose (GPU, 480p) | Mali-G720 | ~12ms |
| MediaPipe Hand ×2 (GPU, crop) | Mali-G720 | ~10ms |
| Normalize stages 1-5 (Kotlin) | CPU (1 core) | <1ms |
| TFLite BiLSTM (NPU, int8) | APU 880 | ~2-3ms |
| UI update (Compose) | CPU | <2ms |
| **Total (pipelined)** | | **~15-18ms → 55-65 fps** |
| **Total (sequential)** | | **~28ms → 35 fps** |

With pipelining and frame skipping, **40+ fps** is realistic. Without any optimization, **30 fps** is the floor on this chip.
