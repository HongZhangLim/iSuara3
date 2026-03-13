package com.isuara.app

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.core.content.ContextCompat
import com.isuara.app.ml.SignPredictor
import com.isuara.app.service.GeminiTranslator
import com.isuara.app.service.TtsService
import com.isuara.app.ui.CameraScreen

class MainActivity : ComponentActivity() {

    companion object {
        private const val TAG = "MainActivity"
    }

    private var signPredictor: SignPredictor? = null
    private var geminiTranslator: GeminiTranslator? = null
    private var ttsService: TtsService? = null

    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            initAndShow()
        } else {
            Toast.makeText(this, "Camera permission required", Toast.LENGTH_LONG).show()
            finish()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            initAndShow()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun initAndShow() {
        // ── Initialize ML & services ──
        try {
            signPredictor = SignPredictor(this)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to init SignPredictor", e)
            Toast.makeText(this, "Model load failed: ${e.message}", Toast.LENGTH_LONG).show()
        }

        // Gemini translator (optional — needs API key)
        val apiKey = try {
            BuildConfig.GEMINI_API_KEY
        } catch (_: Exception) {
            ""
        }
        geminiTranslator = if (apiKey.isNotBlank() && apiKey != "\"\"") {
            try {
                GeminiTranslator(apiKey)
            } catch (e: Exception) {
                Log.w(TAG, "Gemini init failed: ${e.message}")
                null
            }
        } else {
            Log.w(TAG, "No Gemini API key — translate feature disabled")
            null
        }

        ttsService = TtsService(this)

        // ── Compose UI ──
        setContent {
            MaterialTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = Color.Black
                ) {
                    val predictor = signPredictor
                    val tts = ttsService

                    if (predictor != null && tts != null) {
                        CameraScreen(
                            signPredictor = predictor,
                            geminiTranslator = geminiTranslator,
                            ttsService = tts
                        )
                    } else {
                        Box(
                            modifier = Modifier.fillMaxSize(),
                            contentAlignment = Alignment.Center
                        ) {
                            Text(
                                text = "Failed to initialize ML models",
                                color = Color.White
                            )
                        }
                    }
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        signPredictor?.close()
        ttsService?.close()
    }
}
