package com.isuara.app.ml

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import org.json.JSONObject
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

class SignPredictor(context: Context) {

    data class PredictionState(
        val currentWord: String = "",
        val confidence: Float = 0f,
        val isConfident: Boolean = false,
        val bufferProgress: Float = 0f,
        val sentence: List<String> = emptyList(),
        val keypoints: FloatArray? = null,
        val imageWidth: Int = 480,  // PINPOINT: Fixes "No parameter found"
        val imageHeight: Int = 640  // PINPOINT: Fixes "No parameter found"
    )

    private val landmarkExtractor = LandmarkExtractor(context, this::onLandmarksExtracted)
    private val signInterpreter = SignInterpreter(context)
    private val labels: List<String>
    private val inferenceScope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    private val isPredicting = AtomicBoolean(false)
    private val cooldownCounter = AtomicInteger(0)
    private val frameBuffer = ArrayDeque<FloatArray>(31)
    private val sentenceWords = mutableListOf<String>()
    private var lastWord = ""
    private var previousFrame: FloatArray? = null // Holds the EMA state

    private val _state = MutableStateFlow(PredictionState())
    val state: StateFlow<PredictionState> = _state.asStateFlow()

    init {
        val jsonStr = context.assets.open("label_map.json").bufferedReader().use { it.readText() }
        labels = JSONObject(jsonStr).getJSONArray("actions_ordered").let { arr ->
            (0 until arr.length()).map { arr.getString(it) }
        }
    }

    // We added the isFrontCamera parameter here
    fun processFrame(bitmap: Bitmap, timestampMs: Long, isFrontCamera: Boolean = true) {
        // Update dimensions immediately for the UI mapping
        _state.update { it.copy(imageWidth = bitmap.width, imageHeight = bitmap.height) }

        // Pass the flag down to the extractor
        landmarkExtractor.extractAsync(bitmap, timestampMs, isFrontCamera)
    }

    private fun onLandmarksExtracted(rawKeypoints: FloatArray?, timestampMs: Long) {
        _state.update { it.copy(keypoints = rawKeypoints) }

        // If no body/hands are detected, clear the previous frame so it doesn't
        // awkwardly "morph" when hands reappear.
        if (rawKeypoints == null) {
            previousFrame = null
            return
        }

        val rawNormalized = FrameNormalizer.normalizeSingleFrame(rawKeypoints)
        val smoothedNormalized = FloatArray(rawNormalized.size)
        val prev = previousFrame

        // --- APPLY EMA SMOOTHING FILTER ---
        if (prev == null) {
            // First frame: Nothing to smooth against, just use the raw values
            System.arraycopy(rawNormalized, 0, smoothedNormalized, 0, rawNormalized.size)
        } else {
            // EMA Math: alpha defines how much we trust the new frame vs the old frame.
            // 0.4f means 40% new frame, 60% old frame (Heavy smoothing)
            val alpha = 0.4f
            for (i in rawNormalized.indices) {
                smoothedNormalized[i] = (rawNormalized[i] * alpha) + (prev[i] * (1f - alpha))
            }
        }

        // Save this smoothed frame to be used as the "previous" frame next time
        previousFrame = smoothedNormalized.clone()
        // ----------------------------------

        var readyToPredict = false
        var snapshot: Array<FloatArray>? = null

        synchronized(frameBuffer) {
            // Add the smoothed data to the buffer instead of the raw data
            frameBuffer.addLast(smoothedNormalized)
            if (frameBuffer.size > 30) frameBuffer.removeFirst()

            // Check if ready, but don't launch coroutine inside the lock
            if (frameBuffer.size == 30 && cooldownCounter.get() <= 0 && !isPredicting.get()) {
                isPredicting.set(true) // Set immediately so we don't trigger twice
                readyToPredict = true
                snapshot = frameBuffer.toTypedArray()
            } else if (cooldownCounter.get() > 0) {
                cooldownCounter.decrementAndGet()
            }
            updateProgress()
        }

        // Launch inference OUTSIDE the lock so MediaPipe isn't blocked waiting for AI
        if (readyToPredict && snapshot != null) {
            inferenceScope.launch {
                try {
                    val features = FrameNormalizer.buildSequenceFeatures(snapshot!!)
                    val (idx, conf) = signInterpreter.predictTopClass(features)
                    updatePrediction(labels[idx], conf)
                } finally {
                    isPredicting.set(false)
                }
            }
        }
    }

    private fun updatePrediction(word: String, confidence: Float) {
        val isConfident = confidence >= 0.6f
        if (isConfident && word != lastWord && word != "Idle") {
            sentenceWords.add(word)
            lastWord = word
            if (sentenceWords.size > 8) sentenceWords.removeAt(0)
        }
        _state.update { it.copy(
            currentWord = if (isConfident) word else "$word?",
            confidence = confidence,
            isConfident = isConfident,
            sentence = sentenceWords.toList(),
            bufferProgress = 1f
        ) }
        cooldownCounter.set(if (isConfident) 10 else 5)
    }

    private fun updateProgress() {
        val progress = synchronized(frameBuffer) { frameBuffer.size.toFloat() / 30 }
        _state.update { it.copy(bufferProgress = progress) }
    }

    fun getSentenceWords() = sentenceWords.toList()
    fun resetAll() {
        synchronized(frameBuffer) { frameBuffer.clear() }
        sentenceWords.clear()
        lastWord = ""
        previousFrame = null // Reset the smoothing state!
        _state.update { PredictionState() }
    }
    fun close() {
        landmarkExtractor.close()
        signInterpreter.close()
    }
}