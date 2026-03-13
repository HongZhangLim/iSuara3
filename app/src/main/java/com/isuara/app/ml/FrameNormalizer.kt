package com.isuara.app.ml

/**
 * FrameNormalizer — Kotlin port of normalize_frame.py (V3)
 *
 * Stages 1-5 only. Stage 6 (z-score) is BAKED INTO the TFLite model.
 *
 * Keypoint layout (258 floats):
 *   [0..131]   Pose: 33 landmarks × 4 (x, y, z, visibility)
 *   [132..194]  Left Hand: 21 landmarks × 3 (x, y, z)
 *   [195..257]  Right Hand: 21 landmarks × 3 (x, y, z)
 */
object FrameNormalizer {

    const val RAW_FEATURES = 258
    const val FINAL_FEATURES = 780
    const val SEQUENCE_LENGTH = 30

    /**
     * Stage 1 (Anchor Subtraction) + Stage 2 (Scale by Shoulder Width)
     * for a single frame.
     *
     * @param frame FloatArray of size 258 — raw keypoints
     * @return FloatArray of size 258 — anchor + scale normalized
     */
    fun normalizeSingleFrame(frame: FloatArray): FloatArray {
        require(frame.size == RAW_FEATURES) { "Expected $RAW_FEATURES features, got ${frame.size}" }
        val f = frame.copyOf()

        // ── Stage 1: Anchor Subtraction ──

        // Shoulder landmarks: left=[44..46](xyz), right=[48..50](xyz)
        val shoulderLx = f[44]; val shoulderLy = f[45]; val shoulderLz = f[46]
        val shoulderRx = f[48]; val shoulderRy = f[49]; val shoulderRz = f[50]

        // Pose anchor = shoulder midpoint
        val anchorX = (shoulderLx + shoulderRx) / 2f
        val anchorY = (shoulderLy + shoulderRy) / 2f
        val anchorZ = (shoulderLz + shoulderRz) / 2f

        // Subtract anchor from pose xyz (33 landmarks × 4, only xyz not visibility)
        if (anchorX != 0f || anchorY != 0f || anchorZ != 0f) {
            for (i in 0 until 33) {
                val base = i * 4
                f[base] -= anchorX      // x
                f[base + 1] -= anchorY  // y
                f[base + 2] -= anchorZ  // z
                // f[base + 3] = visibility — unchanged
            }
        }

        // Left hand: anchor to left wrist
        val lhWristX = f[132]; val lhWristY = f[133]; val lhWristZ = f[134]
        if (lhWristX != 0f || lhWristY != 0f || lhWristZ != 0f) {
            for (i in 0 until 21) {
                val base = 132 + i * 3
                f[base] -= lhWristX
                f[base + 1] -= lhWristY
                f[base + 2] -= lhWristZ
            }
        }

        // Right hand: anchor to right wrist
        val rhWristX = f[195]; val rhWristY = f[196]; val rhWristZ = f[197]
        if (rhWristX != 0f || rhWristY != 0f || rhWristZ != 0f) {
            for (i in 0 until 21) {
                val base = 195 + i * 3
                f[base] -= rhWristX
                f[base + 1] -= rhWristY
                f[base + 2] -= rhWristZ
            }
        }

        // ── Stage 2: Scale by Shoulder Width ──
        val dx = shoulderLx - shoulderRx
        val dy = shoulderLy - shoulderRy
        val dz = shoulderLz - shoulderRz
        val shoulderWidth = Math.sqrt((dx * dx + dy * dy + dz * dz).toDouble()).toFloat()

        if (shoulderWidth > 0.01f) {
            val invWidth = 1f / shoulderWidth

            // Scale pose xyz (skip visibility at every 4th element)
            for (i in 0 until 33) {
                val base = i * 4
                f[base] *= invWidth
                f[base + 1] *= invWidth
                f[base + 2] *= invWidth
            }

            // Scale left hand
            for (i in 132 until 195) {
                f[i] *= invWidth
            }

            // Scale right hand
            for (i in 195 until 258) {
                f[i] *= invWidth
            }
        }

        return f
    }

    /**
     * Stages 3 (Velocity) + 4 (Acceleration) + 5 (Engineered Features)
     * for a full 30-frame sequence.
     *
     * @param sequence Array of 30 FloatArrays, each size 258
     * @return FloatArray of size 30 × 780 = 23400, row-major [frame][feature]
     */
    fun buildSequenceFeatures(sequence: Array<FloatArray>): FloatArray {
        require(sequence.size == SEQUENCE_LENGTH) {
            "Expected $SEQUENCE_LENGTH frames, got ${sequence.size}"
        }
        val t = sequence.size
        val n = RAW_FEATURES  // 258

        // Stage 3: Velocity (first derivative)
        val velocity = Array(t) { FloatArray(n) }
        for (i in 1 until t) {
            for (j in 0 until n) {
                velocity[i][j] = sequence[i][j] - sequence[i - 1][j]
            }
        }

        // Stage 4: Acceleration (second derivative)
        val acceleration = Array(t) { FloatArray(n) }
        for (i in 1 until t) {
            for (j in 0 until n) {
                acceleration[i][j] = velocity[i][j] - velocity[i - 1][j]
            }
        }

        // Stage 5: Engineered features (6 per frame)
        // Indices in normalized pose:
        //   left wrist xyz  = [60, 61, 62]  (pose landmark 15: 15*4=60)
        //   right wrist xyz = [64, 65, 66]  (pose landmark 16: 16*4=64)
        //   nose xyz        = [0, 1, 2]     (pose landmark 0)
        val engineered = Array(t) { FloatArray(6) }
        for (i in 0 until t) {
            val s = sequence[i]
            val lwX = s[60]; val lwY = s[61]; val lwZ = s[62]
            val rwX = s[64]; val rwY = s[65]; val rwZ = s[66]
            val noseX = s[0]; val noseY = s[1]; val noseZ = s[2]

            // [0] Distance between wrists
            val dwX = lwX - rwX; val dwY = lwY - rwY; val dwZ = lwZ - rwZ
            engineered[i][0] = Math.sqrt((dwX * dwX + dwY * dwY + dwZ * dwZ).toDouble()).toFloat()

            // [1:4] Wrist difference vector
            engineered[i][1] = dwX
            engineered[i][2] = dwY
            engineered[i][3] = dwZ

            // [4] Left wrist to nose distance
            val lnX = lwX - noseX; val lnY = lwY - noseY; val lnZ = lwZ - noseZ
            engineered[i][4] = Math.sqrt((lnX * lnX + lnY * lnY + lnZ * lnZ).toDouble()).toFloat()

            // [5] Right wrist to nose distance
            val rnX = rwX - noseX; val rnY = rwY - noseY; val rnZ = rwZ - noseZ
            engineered[i][5] = Math.sqrt((rnX * rnX + rnY * rnY + rnZ * rnZ).toDouble()).toFloat()
        }

        // Concatenate: [position(258) | velocity(258) | acceleration(258) | engineered(6)] = 780
        val result = FloatArray(t * FINAL_FEATURES)
        for (i in 0 until t) {
            val offset = i * FINAL_FEATURES
            System.arraycopy(sequence[i], 0, result, offset, n)
            System.arraycopy(velocity[i], 0, result, offset + n, n)
            System.arraycopy(acceleration[i], 0, result, offset + n * 2, n)
            System.arraycopy(engineered[i], 0, result, offset + n * 3, 6)
        }

        return result
    }
}
