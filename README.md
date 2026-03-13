# iSuara Android App

**Real-time BIM (Bahasa Isyarat Malaysia) Sign Language Detection**

Single-page app: live camera → detect 98 BIM signs → build sentence → translate (Gemini AI) → text-to-speech.

## Architecture

```
CameraX (front camera, 640×480)
  ↓ ImageProxy → Bitmap
MediaPipe (PoseLandmarker + HandLandmarker, GPU delegate)
  ↓ 258 keypoints (33×4 pose + 21×3 left hand + 21×3 right hand)
FrameNormalizer (Stage 1-5)
  ↓ 780 features per frame × 30-frame sliding window
TFLite (V3.1 BiLSTM, dynamic-range int8, GPU delegate)
  ↓ 98-class softmax
SignPredictor (cooldown, sentence buffer)
  ↓ detected words
GeminiTranslator → natural Bahasa Melayu sentence
TtsService → spoken output
```

## Setup

### 1. Prerequisites
- Android Studio Ladybug (2024.2) or newer
- JDK 17+
- Android device with API 26+ (Android 8.0)

### 2. Gemini API Key
Create `android/local.properties` and add:
```properties
GEMINI_API_KEY=your_api_key_here
```
Get a key from [Google AI Studio](https://aistudio.google.com/apikey).
Translation works without this, but will show an error message.

### 3. Build & Run
```bash
cd android
./gradlew assembleDebug
# Or open in Android Studio and run
```

### 4. First Run
- Grant camera permission when prompted
- Point camera at a person signing BIM
- Detected words appear at the top
- Word buffer builds at the bottom
- Tap **Translate** to form a sentence with Gemini AI
- Tap **Speak** to hear it via TTS
- Tap **Reset** to clear everything

## Project Structure

```
android/app/src/main/
├── assets/
│   ├── bim_lstm_v3_int8.tflite    # Sign language model (dynamic-range quantized)
│   ├── bim_lstm_v3_f32.tflite     # Float32 fallback
│   ├── label_map.json             # 98 BIM sign labels
│   ├── pose_landmarker_lite.task  # MediaPipe pose model
│   └── hand_landmarker.task       # MediaPipe hand model
├── java/com/isuara/app/
│   ├── MainActivity.kt            # Entry point, permissions, init
│   ├── ml/
│   │   ├── FrameNormalizer.kt     # Stages 1-5 normalization (port of normalize_frame.py)
│   │   ├── SignInterpreter.kt     # TFLite model wrapper
│   │   ├── LandmarkExtractor.kt   # MediaPipe pose + hand landmark extraction
│   │   └── SignPredictor.kt       # Full pipeline orchestrator
│   ├── service/
│   │   ├── GeminiTranslator.kt    # BIM keywords → Bahasa Melayu sentence
│   │   └── TtsService.kt          # Text-to-speech (Malay/Indonesian/English)
│   └── ui/
│       └── CameraScreen.kt        # Single-page Compose UI
└── res/
    └── values/
        ├── strings.xml
        └── themes.xml
```

## Model Details

- **V3.1**: 2× BiLSTM-128 + Dot Attention, ~1.37M params
- **Input**: (1, 30, 780) float32O — 30-frame window, 780 features/frame
- **Output**: (1, 98) softmax — 97 BIM signs + Idle
- **Scaler**: z-score normalization baked into model (no external scaler needed)
- **Quantization**: Dynamic-range (weights int8, activations float32)

## Performance Targets (Dimensity 8400 Ultra)

| Component | Target |
|-----------|--------|
| MediaPipe (GPU) | ~15ms |
| Normalization | ~1ms |
| TFLite inference | ~4ms |
| Total pipeline | ~20ms (~50 FPS) |
