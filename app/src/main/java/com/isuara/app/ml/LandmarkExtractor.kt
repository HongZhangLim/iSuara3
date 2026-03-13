package com.isuara.app.ml

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import java.util.concurrent.ConcurrentHashMap

class LandmarkExtractor(
    context: Context,
    private val onResult: (FloatArray?, Long) -> Unit // PINPOINT: Fixes "Too many arguments"
) {

    companion object {
        private const val TAG = "LandmarkExtractor"
    }

    private var poseLandmarker: PoseLandmarker? = null
    private var handLandmarker: HandLandmarker? = null

    // -----------------------------------------------------
    // ADDED: Cached Pose State Variables for Hand Cropping
    // -----------------------------------------------------
    private var lastLeftWristNormX = -1f
    private var lastLeftWristNormY = -1f
    private var lastRightWristNormX = -1f
    private var lastRightWristNormY = -1f
    private var lastShoulderWidthPx = 150f

    private class FrameResult {
        var poseDone = false
        var handDone = false
        var features: FloatArray? = null
        var hasData = false
        var isFrontCamera = true // Added this flag

        // -----------------------------------------------------
        // ADDED: Unified Crop metadata for coordinate remapping
        // -----------------------------------------------------
        var bitmapWidth = 0
        var bitmapHeight = 0
        var cropStartX = 0
        var cropStartY = 0
        var cropWidth = 0
        var cropHeight = 0
    }
    private val pendingFrames = ConcurrentHashMap<Long, FrameResult>()

    init {
        val handOptions = HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(BaseOptions.builder().setModelAssetPath("hand_landmarker.task").setDelegate(Delegate.GPU).build())
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setNumHands(2)
            .setResultListener(this::onHandResult)
            .build()
        handLandmarker = HandLandmarker.createFromOptions(context, handOptions)

        val poseOptionsGPU = PoseLandmarker.PoseLandmarkerOptions.builder()
            .setBaseOptions(BaseOptions.builder().setModelAssetPath("pose_landmarker_lite.task").setDelegate(Delegate.GPU).build())
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setResultListener(this::onPoseResult)
            .build()
        poseLandmarker = PoseLandmarker.createFromOptions(context, poseOptionsGPU)
        Log.d(TAG, "PoseLandmarker and HandLandmarker loaded on GPU")
    }

    // Added isFrontCamera parameter
    fun extractAsync(bitmap: Bitmap, timestampMs: Long, isFrontCamera: Boolean) {
        val frame = FrameResult()
        frame.isFrontCamera = isFrontCamera // Save the flag for this specific frame
        frame.bitmapWidth = bitmap.width
        frame.bitmapHeight = bitmap.height
        pendingFrames[timestampMs] = frame

        // 1. Pose always runs on full resolution FIRST
        poseLandmarker?.detectAsync(BitmapImageBuilder(bitmap).build(), timestampMs)

        // ---------------------------------------------------------
        // ADDED: Unified Hand Crop Logic
        // ---------------------------------------------------------
        var minX = Float.MAX_VALUE
        var maxX = Float.MIN_VALUE
        var minY = Float.MAX_VALUE
        var maxY = Float.MIN_VALUE
        var activeWrists = 0

        if (lastLeftWristNormX >= 0f) {
            minX = minOf(minX, lastLeftWristNormX)
            maxX = maxOf(maxX, lastLeftWristNormX)
            minY = minOf(minY, lastLeftWristNormY)
            maxY = maxOf(maxY, lastLeftWristNormY)
            activeWrists++
        }
        if (lastRightWristNormX >= 0f) {
            minX = minOf(minX, lastRightWristNormX)
            maxX = maxOf(maxX, lastRightWristNormX)
            minY = minOf(minY, lastRightWristNormY)
            maxY = maxOf(maxY, lastRightWristNormY)
            activeWrists++
        }

        if (activeWrists > 0) {
            // Padding based on user's shoulder width
            val paddingXNorm = (lastShoulderWidthPx * 1.5f) / bitmap.width
            val paddingYNorm = (lastShoulderWidthPx * 1.5f) / bitmap.height

            // Shift crop slightly UPWARDS (lower Y value) because fingers are above the wrist
            val yOffsetNorm = paddingYNorm * 0.4f

            val startXNorm = (minX - paddingXNorm).coerceIn(0f, 1f)
            val endXNorm = (maxX + paddingXNorm).coerceIn(0f, 1f)
            val startYNorm = (minY - paddingYNorm - yOffsetNorm).coerceIn(0f, 1f)
            val endYNorm = (maxY + paddingYNorm - yOffsetNorm).coerceIn(0f, 1f)

            frame.cropStartX = (startXNorm * bitmap.width).toInt()
            frame.cropStartY = (startYNorm * bitmap.height).toInt()
            frame.cropWidth = ((endXNorm - startXNorm) * bitmap.width).toInt()
            frame.cropHeight = ((endYNorm - startYNorm) * bitmap.height).toInt()

            // Safety check for valid crop dimensions
            if (frame.cropWidth > 10 && frame.cropHeight > 10) {
                val crop = Bitmap.createBitmap(bitmap, frame.cropStartX, frame.cropStartY, frame.cropWidth, frame.cropHeight)
                handLandmarker?.detectAsync(BitmapImageBuilder(crop).build(), timestampMs)
            } else {
                runFullFrameHand(bitmap, timestampMs, frame)
            }
        } else {
            // Cold start / wrists hidden -> run full frame
            runFullFrameHand(bitmap, timestampMs, frame)
        }
    }

    // ADDED: Fallback helper function for full-frame processing
    private fun runFullFrameHand(bitmap: Bitmap, timestampMs: Long, frame: FrameResult) {
        frame.cropStartX = 0
        frame.cropStartY = 0
        frame.cropWidth = bitmap.width
        frame.cropHeight = bitmap.height
        handLandmarker?.detectAsync(BitmapImageBuilder(bitmap).build(), timestampMs)
    }

    private fun onPoseResult(result: PoseLandmarkerResult, image: MPImage) {
        val ts = result.timestampMs()
        val frame = pendingFrames[ts] ?: return

        // THE FIX: A map to swap Left and Right body landmarks (eyes, ears, shoulders, wrists, etc.)
        val POSE_SWAP_MAP = intArrayOf(
            0,  // 0: nose
            4, 5, 6, // 1,2,3 -> 4,5,6 (eyes)
            1, 2, 3, // 4,5,6 -> 1,2,3 (eyes)
            8, 7,    // 7,8 -> 8,7 (ears)
            10, 9,   // 9,10 -> 10,9 (mouth)
            12, 11,  // 11,12 -> 12,11 (shoulders)
            14, 13,  // 13,14 -> 14,13 (elbows)
            16, 15,  // 15,16 -> 16,15 (wrists)
            18, 17,  // 17,18 -> 18,17 (pinkies)
            20, 19,  // 19,20 -> 20,19 (indexes)
            22, 21,  // 21,22 -> 22,21 (thumbs)
            24, 23,  // 23,24 -> 24,23 (hips)
            26, 25,  // 25,26 -> 26,25 (knees)
            28, 27,  // 27,28 -> 28,27 (ankles)
            30, 29,  // 29,30 -> 30,29 (heels)
            32, 31   // 31,32 -> 32,31 (foot indexes)
        )

        synchronized(frame) {
            if (frame.features == null) frame.features = FloatArray(258)
            if (result.landmarks().isNotEmpty()) {
                frame.hasData = true
                val poseLms = result.landmarks()[0]
                for (i in 0 until 33) {

                    // THE FIX: Use the swapped index for the rear camera!
                    val mappedIndex = if (frame.isFrontCamera) i else POSE_SWAP_MAP[i]
                    val idx = mappedIndex * 4
                    val rawX = poseLms[i].x()

                    frame.features!![idx] = if (frame.isFrontCamera) rawX else (1f - rawX)
                    frame.features!![idx + 1] = poseLms[i].y()
                    frame.features!![idx + 2] = poseLms[i].z()
                    frame.features!![idx + 3] = poseLms[i].visibility().orElse(0f)
                }

                // ---------------------------------------------------------
                // ADDED: Cache RAW coords for next frame's unified crop
                // ---------------------------------------------------------
                lastLeftWristNormX = if (poseLms[15].visibility().orElse(0f) > 0.5f) poseLms[15].x() else -1f
                lastLeftWristNormY = if (poseLms[15].visibility().orElse(0f) > 0.5f) poseLms[15].y() else -1f
                lastRightWristNormX = if (poseLms[16].visibility().orElse(0f) > 0.5f) poseLms[16].x() else -1f
                lastRightWristNormY = if (poseLms[16].visibility().orElse(0f) > 0.5f) poseLms[16].y() else -1f

                // Dynamic padding based on shoulder width
                val dx = (poseLms[11].x() - poseLms[12].x()) * frame.bitmapWidth
                val dy = (poseLms[11].y() - poseLms[12].y()) * frame.bitmapHeight
                lastShoulderWidthPx = Math.sqrt((dx * dx + dy * dy).toDouble()).toFloat().coerceAtLeast(100f)

            } else {
                // If nobody is on screen, reset the crop trackers
                lastLeftWristNormX = -1f
                lastRightWristNormX = -1f
            }
            frame.poseDone = true
            checkCompletion(ts, frame)
        }
    }

    private fun onHandResult(result: HandLandmarkerResult, image: MPImage) {
        val ts = result.timestampMs()
        val frame = pendingFrames[ts] ?: return
        synchronized(frame) {
            if (frame.features == null) frame.features = FloatArray(258)
            if (result.landmarks().isNotEmpty()) {
                frame.hasData = true
                for (i in result.landmarks().indices) {
                    var isLeft = result.handednesses()[i][0].categoryName() == "Left"

                    // THE FIX: Swap hand labels for the rear camera
                    if (!frame.isFrontCamera) {
                        isLeft = !isLeft
                    }

                    val offset = if (isLeft) 132 else 195
                    for (j in 0 until 21) {
                        val idx = offset + (j * 3)

                        // ---------------------------------------------------------
                        // ADDED: Remap from Crop space back to Full Frame space
                        // ---------------------------------------------------------
                        val cropX = result.landmarks()[i][j].x()
                        val cropY = result.landmarks()[i][j].y()
                        val fullFrameX = (frame.cropStartX + cropX * frame.cropWidth) / frame.bitmapWidth
                        val fullFrameY = (frame.cropStartY + cropY * frame.cropHeight) / frame.bitmapHeight

                        // THE FIX: Mathematically mirror X for the rear camera
                        frame.features!![idx] = if (frame.isFrontCamera) fullFrameX else (1f - fullFrameX)
                        frame.features!![idx+1] = fullFrameY
                        frame.features!![idx+2] = result.landmarks()[i][j].z()
                    }
                }
            }
            frame.handDone = true
            checkCompletion(ts, frame)
        }
    }

    private fun checkCompletion(ts: Long, frame: FrameResult) {
        if (frame.poseDone && frame.handDone) {
            pendingFrames.remove(ts)
            onResult(if (frame.hasData) frame.features else null, ts)
        }
    }

    fun close() {
        poseLandmarker?.close()
        handLandmarker?.close()
    }
}