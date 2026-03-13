## Plan: Dynamic Hand Crop Strategy (Hybrid)

**TL;DR** — Replace full-frame hand detection with a dynamic-crop pipeline using **previous-frame pose caching** to maintain parallel execution. Each frame: pose and two hand detectors fire simultaneously — hand detectors use wrist coordinates cached from the prior frame's pose result to crop small regions. Crop size = `shoulderWidthPx × 1.5`, adapting to user distance. Two separate `HandLandmarker` instances (`numHands=1` each) eliminate hand-entanglement bugs. Cropped hand landmarks are remapped back to full-frame `[0..1]` space before merging into the 258-float feature array. No changes to `FrameNormalizer`, `SignPredictor`, `SignInterpreter`, or `CameraScreen`. Reduces hand detection pixel count by **~85-95%** while preserving parallel latency profile.

**Steps**

1. **Add cached pose state variables** in LandmarkExtractor.kt
   - Add instance variables to cache wrist positions and crop size from the previous frame's pose result:
     - `lastLeftWristNormX/Y: Float` — normalized `[0..1]` left wrist (pose landmark 15), init to `-1f`
     - `lastRightWristNormX/Y: Float` — normalized `[0..1]` right wrist (pose landmark 16), init to `-1f`
     - `lastCropSize: Int` — dynamic crop dimension in pixels, init to `200`
   - These are updated at the end of each `onPoseResult` and consumed at the start of the next `extractAsync`.

2. **Update `FrameResult` class** in LandmarkExtractor.kt
   - Replace `handDone: Boolean` with `leftHandDone: Boolean` and `rightHandDone: Boolean` (both default `false`).
   - Add per-hand crop metadata for coordinate remapping:
     - `leftCropStartX/Y: Int`, `leftCropSize: Int`
     - `rightCropStartX/Y: Int`, `rightCropSize: Int`
     - `bitmapWidth: Int`, `bitmapHeight: Int`
   - Keep existing fields: `poseDone`, `features`, `hasData`, `isFrontCamera`.

3. **Replace single `HandLandmarker` with two instances** in LandmarkExtractor.kt
   - Remove the single `handLandmarker` field.
   - Create `leftHandLandmarker` and `rightHandLandmarker`, both configured identically: model `hand_landmarker.task`, `Delegate.CPU`, `RunningMode.LIVE_STREAM`, **`numHands = 1`**.
   - Each gets its own callback: `onLeftHandResult()` and `onRightHandResult()`.
   - Update `close()` to close both instances.

4. **Rewrite `extractAsync()` to crop using cached pose data** in LandmarkExtractor.kt
   - Still fires `poseLandmarker.detectAsync(fullImage, ts)` on the full bitmap (unchanged).
   - **Hand detection now uses cached wrist coords from previous frame:**
     - If `lastLeftWristNormX == -1f` (cold start / no prior pose): mark both `leftHandDone = true` and `rightHandDone = true` immediately (zero-fill, skip hand detection). This only happens on the very first frame.
     - Otherwise, for each wrist that has valid cached coords:
       - Convert normalized coords to pixel coords: `wristPx = (normX * bitmap.width, normY * bitmap.height)`.
       - Compute crop box: `startX = (wristPxX - cropSize/2).coerceIn(0, bitmap.width - cropSize)`, same for Y. If `cropSize > bitmap.width` or `bitmap.height`, clamp `cropSize` to `min(bitmap.width, bitmap.height)`.
       - `Bitmap.createBitmap(bitmap, startX, startY, cropSize, cropSize)` → wrap in `BitmapImageBuilder` → `MPImage`.
       - Store `startX`, `startY`, `cropSize` on `FrameResult` for that hand.
       - Fire `leftHandLandmarker.detectAsync(leftCropImage, ts)` / `rightHandLandmarker.detectAsync(rightCropImage, ts)`.
     - If only one wrist has valid cached coords, only crop and detect that hand; mark the other as done (zero-fill).
   - Store `bitmap.width` and `bitmap.height` on `FrameResult` for coordinate remapping.
   - **Key**: Pose and both hand detectors now run in **parallel** — no sequential wait. Frame latency stays at `max(pose, hand)`, not `pose + hand`.

5. **Update `onPoseResult()` to cache wrist coords for next frame** in LandmarkExtractor.kt
   - After writing pose landmarks to `frame.features` (existing logic — unchanged):
   - Extract raw (pre-mirror) coordinates for:
     - Pose landmark 15 (left wrist): `x = poseLms[15].x()`, `y = poseLms[15].y()`, `vis = poseLms[15].visibility()`
     - Pose landmark 16 (right wrist): similarly
     - Pose landmarks 11/12 (shoulders): for dynamic crop sizing
   - **Visibility guard**: Only cache a wrist's coords if `visibility > 0.5f`. If below threshold, set that wrist's cached norm to `-1f` (will cause skip on next frame).
   - **Dynamic crop size**: `shoulderWidthPx = euclidean distance between landmark 11 and 12 in pixel space`. `lastCropSize = (shoulderWidthPx * 1.5f).toInt().coerceIn(120, 300)`.
   - **Front/rear camera handling**: Cache the **raw** normalized coords (before any mirroring). The crop in `extractAsync()` operates on the original bitmap which hasn't been coordinate-flipped — it's the same bitmap for front and rear. Mirroring is only applied when writing to the feature array.
   - Call `checkCompletion(ts, frame)` as before.

6. **Implement `onLeftHandResult()` and `onRightHandResult()` with coordinate remapping** in LandmarkExtractor.kt
   - Each callback follows the same pattern (parameterized by offset: left → 132, right → 195):
     1. Retrieve `FrameResult` from `pendingFrames[ts]`.
     2. If `result.landmarks().isNotEmpty()`, set `frame.hasData = true`.
     3. For each of the 21 hand landmarks (index `j`):
        - Get crop-local normalized coords: `rawCropX = result.landmarks()[0][j].x()`, `rawCropY = ...y()`, `rawZ = ...z()`.
        - **Remap to full-frame normalized coords:**
          ```
          fullFrameX = (cropStartX + rawCropX * cropSize) / bitmapWidth
          fullFrameY = (cropStartY + rawCropY * cropSize) / bitmapHeight
          ```
        - **Z coordinate**: Keep as-is — MediaPipe Z is depth relative to wrist, not spatial position in the image. No remapping needed.
        - **Rear camera mirroring**: Apply `if (!frame.isFrontCamera) fullFrameX = 1f - fullFrameX` — same logic as the current `onHandResult`.
        - **Hand label**: No label-based left/right assignment needed — each callback already knows which hand it is (left landmarker → offset 132, right landmarker → offset 195). For rear camera, swap offsets: left landmarker → 195, right landmarker → 132 (mirrors the existing `isLeft = !isLeft` logic).
     4. Write `fullFrameX`, `fullFrameY`, `rawZ` to `frame.features!![offset + j*3 .. offset + j*3 + 2]`.
     5. Mark `frame.leftHandDone = true` (or `rightHandDone`).
     6. Call `checkCompletion(ts, frame)`.

7. **Update `checkCompletion()`** in LandmarkExtractor.kt
   - Change condition from `poseDone && handDone` to `poseDone && leftHandDone && rightHandDone`.
   - No bitmap reference to clean up (bitmap is not stored on `FrameResult`).

8. **No changes needed** to:
   - FrameNormalizer.kt — Anchor subtraction and shoulder-width scaling operate on the feature array values, which are full-frame normalized coordinates after remapping. Identical to current behavior.
   - SignPredictor.kt — Same 258-float callback, same EMA, same buffer, same prediction trigger.
   - SignInterpreter.kt — TFLite model input `(1, 30, 780)` and output `(1, 98)` unchanged.
   - CameraScreen.kt — Camera config (`480×360` analysis resolution), frame delivery, and overlay drawing unchanged.

**Verification**
- **Crop tracking**: Log `lastCropSize` and wrist pixel coords per frame — expect cropSize ~120-200px at typical signing distance on 480×360 frames
- **Feature parity**: On a recorded test sequence, compare the 258-float arrays before and after the change — values should be nearly identical (minor differences from crop-edge interpolation)
- **Latency**: Measure `extractAsync()` → `onResult()` time before and after — expect **40-60% reduction** in hand detection latency while pose latency stays the same, overall frame time improves
- **Accuracy**: Run a full 30-frame prediction cycle on known signs and verify predictions match pre-change results
- **Edge cases to test**:
  - One hand off-screen (one wrist cached, one `-1f`) → only detected hand has landmarks
  - Hands overlapping / crossed → two separate landmarkers avoid entanglement
  - Hands near frame edges → crop clamping prevents `createBitmap` crash
  - Rear camera → X-flip and offset-swap produce correct features
  - Cold start (frame 1) → zero-filled hands, pose still detected, frame 2 onwards has crops
  - User moves closer/farther → crop size tracks shoulder width dynamically

**Decisions**
- **Previous-frame caching** (from Gemini): Preserves parallel `max(pose, hand)` latency instead of sequential `pose + hand`. 1-frame lag (~33ms at 30 FPS) causes negligible crop offset (~3-5px of wrist movement).
- **Dynamic crop**: `shoulderWidthPx × 1.5`, clamped `[120, 300]` — adapts to user distance, prevents degenerate sizes.
- **Two HandLandmarkers** (`numHands=1` each): Eliminates hand-entanglement on overlapping hands; ~10-15 MB extra memory is acceptable.
- **Cold-start fallback**: Skip hand detection on first frame (zero-fill) — simplest, and a single zero-filled frame has no impact on the 30-frame prediction buffer.
- **Visibility threshold** (from Gemini): Only cache wrists with `visibility > 0.5f` — prevents tracking ghost/occluded limbs.
