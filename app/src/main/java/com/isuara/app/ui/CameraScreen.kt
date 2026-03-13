package com.isuara.app.ui

import android.graphics.Bitmap
import android.util.Log
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Sync
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.isuara.app.R
import com.isuara.app.ml.SignPredictor
import com.isuara.app.service.GeminiTranslator
import com.isuara.app.service.TtsService
import kotlinx.coroutines.launch
import java.util.concurrent.Executors

private const val TAG = "CameraScreen"

val googleSansFlex = FontFamily(Font(R.font.google_sans_flex))

@androidx.annotation.OptIn(androidx.camera.camera2.interop.ExperimentalCamera2Interop::class)
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CameraScreen(
    signPredictor: SignPredictor,
    geminiTranslator: GeminiTranslator?,
    ttsService: TtsService
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val scope = rememberCoroutineScope()

    val predictionState by signPredictor.state.collectAsState()

    var showLandmarks by remember { mutableStateOf(false) }
    var translatedText by remember { mutableStateOf("") }
    var isTranslating by remember { mutableStateOf(false) }
    var fpsCounter by remember { mutableIntStateOf(0) }
    var displayFps by remember { mutableIntStateOf(0) }
    var lastFpsTime by remember { mutableLongStateOf(System.currentTimeMillis()) }
    var lensFacing by remember { mutableIntStateOf(CameraSelector.LENS_FACING_FRONT) }

    val mlExecutor = remember { Executors.newSingleThreadExecutor() }

    // =========================================================================
    // 1. AUTO-TRANSLATE & TTS FEATURE
    // =========================================================================
    LaunchedEffect(predictionState.currentWord) {
        // Notice the new condition: !predictionState.isWaitingForNewSentence
        // This prevents the TTS from looping twice if the user stays idle!
        if (predictionState.currentWord == "Idle" &&
            predictionState.sentence.isNotEmpty() &&
            !isTranslating &&
            !predictionState.isWaitingForNewSentence) {

            kotlinx.coroutines.delay(2000)

            val words = signPredictor.getSentenceWords()
            if (words.isNotEmpty()) {
                isTranslating = true
                translatedText = ""

                try {
                    val result = geminiTranslator?.translate(words) ?: "Error: Gemini unavailable"
                    translatedText = result

                    val textToSpeak = result.ifEmpty { words.joinToString(" ") }
                    if (textToSpeak.isNotEmpty()) {
                        ttsService.speak(textToSpeak)
                    }
                } catch (e: Exception) {
                    translatedText = "Translation failed"
                    Log.e(TAG, "Auto-translate error", e)
                } finally {
                    isTranslating = false
                    // INSTEAD OF resetAll(), tell the predictor to hold the text on
                    // screen until the exact moment the next sign is made!
                    signPredictor.prepareForNewSentence()
                }
            }
        }
    }

    // =========================================================================
    // 2. UI POLISH (OPTIONAL BUT RECOMMENDED)
    // Clear the previous translation text from the screen when a NEW sentence starts
    // =========================================================================
    LaunchedEffect(predictionState.sentence.size) {
        if (predictionState.sentence.size == 1 && !isTranslating) {
            translatedText = ""
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black)
    ) {
        // 1. EXACT 4:3 CAMERA PREVIEW
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .aspectRatio(3f / 4f)
                .background(Color.DarkGray)
        ) {
            val previewView = remember {
                androidx.camera.view.PreviewView(context).apply {
                    scaleType = androidx.camera.view.PreviewView.ScaleType.FILL_CENTER
                    implementationMode = androidx.camera.view.PreviewView.ImplementationMode.PERFORMANCE
                }
            }

            LaunchedEffect(lensFacing) {
                val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
                cameraProviderFuture.addListener({
                    val cameraProvider = cameraProviderFuture.get()

                    val previewBuilder = Preview.Builder()
                        .setTargetResolution(android.util.Size(640, 480))

                    val previewExt = androidx.camera.camera2.interop.Camera2Interop.Extender(previewBuilder)
                    previewExt.setCaptureRequestOption(
                        android.hardware.camera2.CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE,
                        android.util.Range(30, 60)
                    )

                    val preview = previewBuilder.build().also {
                        it.surfaceProvider = previewView.surfaceProvider
                    }

                    val analysisBuilder = ImageAnalysis.Builder()
                        .setTargetResolution(android.util.Size(480, 360))
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)

                    val imageAnalysis = analysisBuilder.build()
                    val isFront = lensFacing == CameraSelector.LENS_FACING_FRONT
                    var reusedBitmap: Bitmap? = null
                    val canvas = android.graphics.Canvas()
                    val matrix = android.graphics.Matrix()

                    imageAnalysis.setAnalyzer(mlExecutor) { imageProxy ->
                        try {
                            val rawBitmap = imageProxy.toBitmap()
                            val rotation = imageProxy.imageInfo.rotationDegrees
                            val isPortrait = rotation == 90 || rotation == 270
                            val targetWidth = if (isPortrait) rawBitmap.height else rawBitmap.width
                            val targetHeight = if (isPortrait) rawBitmap.width else rawBitmap.height

                            if (reusedBitmap == null || reusedBitmap!!.width != targetWidth) {
                                reusedBitmap = Bitmap.createBitmap(targetWidth, targetHeight, Bitmap.Config.ARGB_8888)
                            }

                            matrix.reset()
                            matrix.postTranslate(-rawBitmap.width / 2f, -rawBitmap.height / 2f)
                            matrix.postRotate(rotation.toFloat())
                            if (isFront) matrix.postScale(-1f, 1f)
                            matrix.postTranslate(targetWidth / 2f, targetHeight / 2f)

                            canvas.setBitmap(reusedBitmap)
                            canvas.drawColor(android.graphics.Color.BLACK, android.graphics.PorterDuff.Mode.CLEAR)
                            canvas.drawBitmap(rawBitmap, matrix, null)

                            signPredictor.processFrame(reusedBitmap!!, imageProxy.imageInfo.timestamp / 1_000_000, isFront)
                        } finally {
                            imageProxy.close()
                        }

                        fpsCounter++
                        val now = System.currentTimeMillis()
                        if (now - lastFpsTime >= 1000) {
                            displayFps = fpsCounter
                            fpsCounter = 0
                            lastFpsTime = now
                        }
                    }

                    val cameraSelector = CameraSelector.Builder().requireLensFacing(lensFacing).build()
                    try {
                        cameraProvider.unbindAll()
                        cameraProvider.bindToLifecycle(lifecycleOwner, cameraSelector, preview, imageAnalysis)
                    } catch (e: Exception) {
                        Log.e(TAG, "Camera bind failed", e)
                    }
                }, ContextCompat.getMainExecutor(context))
            }

            AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())

            if (showLandmarks && predictionState.keypoints != null) {
                Canvas(modifier = Modifier.fillMaxSize()) {
                    val keypoints = predictionState.keypoints!!
                    val radius = 6f
                    val scale = java.lang.Math.max(size.width / predictionState.imageWidth, size.height / predictionState.imageHeight)
                    val scaledWidth = predictionState.imageWidth * scale
                    val scaledHeight = predictionState.imageHeight * scale
                    val offsetX = (size.width - scaledWidth) / 2f
                    val offsetY = (size.height - scaledHeight) / 2f

                    fun mapX(xNorm: Float) = if (lensFacing == CameraSelector.LENS_FACING_FRONT) (xNorm * scaledWidth) + offsetX else ((1f - xNorm) * scaledWidth) + offsetX
                    fun mapY(yNorm: Float) = (yNorm * scaledHeight) + offsetY

                    for (i in 0 until 33) {
                        val x = mapX(keypoints[i * 4]); val y = mapY(keypoints[i * 4 + 1])
                        if (keypoints[i * 4 + 3] > 0.5f) drawCircle(Color.Green, radius, androidx.compose.ui.geometry.Offset(x, y))
                    }
                    for (i in 0 until 21) {
                        val idx = 132 + (i * 3)
                        if (keypoints[idx] > 0f) drawCircle(Color.Magenta, radius, androidx.compose.ui.geometry.Offset(mapX(keypoints[idx]), mapY(keypoints[idx + 1])))
                    }
                    for (i in 0 until 21) {
                        val idx = 195 + (i * 3)
                        if (keypoints[idx] > 0f) drawCircle(Color.Cyan, radius, androidx.compose.ui.geometry.Offset(mapX(keypoints[idx]), mapY(keypoints[idx + 1])))
                    }
                }
            }

            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .statusBarsPadding()
                    .padding(16.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Surface(
                    color = Color.Black.copy(alpha = 0.5f),
                    shape = RoundedCornerShape(16.dp)
                ) {
                    Text(
                        text = "$displayFps FPS",
                        fontFamily = googleSansFlex,
                        color = Color.White,
                        modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold
                    )
                }

                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    IconButton(
                        onClick = { showLandmarks = !showLandmarks },
                        colors = IconButtonDefaults.iconButtonColors(containerColor = if (showLandmarks) Color(0xFF2196F3) else Color.Black.copy(alpha = 0.5f))
                    ) {
                        Icon(Icons.Default.Settings, contentDescription = "Toggle Landmarks", tint = Color.White)
                    }
                    IconButton(
                        onClick = { lensFacing = if (lensFacing == CameraSelector.LENS_FACING_FRONT) CameraSelector.LENS_FACING_BACK else CameraSelector.LENS_FACING_FRONT },
                        colors = IconButtonDefaults.iconButtonColors(containerColor = Color.Black.copy(alpha = 0.5f))
                    ) {
                        Icon(Icons.Default.Sync, contentDescription = "Switch Camera", tint = Color.White)
                    }
                }
            }

            LinearProgressIndicator(
                progress = { predictionState.bufferProgress },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(4.dp)
                    .align(Alignment.BottomCenter),
                color = Color(0xFF4CAF50),
                trackColor = Color.Transparent,
            )
        }

        // 2. CENTERED UI PANEL
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f)
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Box(modifier = Modifier.height(40.dp), contentAlignment = Alignment.Center) {
                androidx.compose.animation.AnimatedVisibility(
                    visible = predictionState.currentWord.isNotEmpty(),
                    enter = fadeIn(),
                    exit = fadeOut()
                ) {
                    val textColor by animateColorAsState(if (predictionState.isConfident) Color(0xFF4CAF50) else Color.White.copy(alpha = 0.7f), label = "color")
                    Text(
                        text = predictionState.currentWord.uppercase(),
                        fontFamily = googleSansFlex,
                        color = textColor,
                        fontSize = 24.sp,
                        fontWeight = FontWeight.Black,
                        letterSpacing = 2.sp
                    )
                }
            }

            val animatedConfidence by animateFloatAsState(targetValue = predictionState.confidence, label = "confidence")
            LinearProgressIndicator(
                progress = { animatedConfidence },
                modifier = Modifier.width(100.dp).height(2.dp).clip(RoundedCornerShape(1.dp)),
                color = if (predictionState.isConfident) Color(0xFF4CAF50) else Color(0xFFFF9800),
                trackColor = Color.DarkGray
            )

            Spacer(modifier = Modifier.height(16.dp))

            val displaySentence = predictionState.sentence.joinToString(" ")

            Surface(
                color = Color.White.copy(alpha = 0.1f),
                shape = RoundedCornerShape(12.dp),
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f)
            ) {
                Column(
                    modifier = Modifier.padding(16.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Center
                ) {
                    Text(
                        text = displaySentence.ifEmpty { "Waiting for signs..." },
                        fontFamily = googleSansFlex,
                        color = if (displaySentence.isEmpty()) Color.White.copy(alpha = 0.3f) else Color.White,
                        fontSize = 18.sp,
                        lineHeight = 24.sp,
                        textAlign = TextAlign.Center
                    )

                    AnimatedVisibility(visible = isTranslating || translatedText.isNotEmpty()) {
                        Column(
                            modifier = Modifier.padding(top = 8.dp),
                            horizontalAlignment = Alignment.CenterHorizontally
                        ) {
                            HorizontalDivider(color = Color.White.copy(alpha = 0.2f), modifier = Modifier.padding(vertical = 8.dp))
                            if (isTranslating) {
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    CircularProgressIndicator(color = Color(0xFF2196F3), modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
                                    Spacer(modifier = Modifier.width(8.dp))
                                    Text("Refining grammar...", fontFamily = googleSansFlex, color = Color(0xFF2196F3), fontSize = 14.sp)
                                }
                            } else {
                                Text(
                                    text = translatedText,
                                    fontFamily = googleSansFlex,
                                    color = Color(0xFF64B5F6),
                                    fontSize = 18.sp,
                                    fontWeight = FontWeight.Medium,
                                    textAlign = TextAlign.Center
                                )
                            }
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            val uniformButtonHeight = 56.dp

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                FilledIconButton(
                    onClick = {
                        signPredictor.resetAll()
                        translatedText = ""
                        isTranslating = false
                        ttsService.stop()
                    },
                    modifier = Modifier.size(uniformButtonHeight),
                    colors = IconButtonDefaults.filledIconButtonColors(containerColor = Color.White.copy(alpha = 0.1f))
                ) {
                    Icon(Icons.Default.Delete, contentDescription = "Clear", tint = Color.White)
                }

                Button(
                    onClick = {
                        val words = signPredictor.getSentenceWords()
                        if (words.isNotEmpty() && !isTranslating) {
                            isTranslating = true
                            translatedText = ""
                            scope.launch {
                                translatedText = geminiTranslator?.translate(words) ?: "Error: Gemini unavailable"
                                isTranslating = false
                            }
                        }
                    },
                    modifier = Modifier
                        .weight(1f)
                        .height(uniformButtonHeight),
                    enabled = predictionState.sentence.isNotEmpty() && !isTranslating,
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF2196F3))
                ) {
                    Text("Translate", fontFamily = googleSansFlex, fontSize = 16.sp, fontWeight = FontWeight.Bold)
                }

                FloatingActionButton(
                    onClick = {
                        val textToSpeak = translatedText.ifEmpty { predictionState.sentence.joinToString(" ") }
                        if (textToSpeak.isNotEmpty()) ttsService.speak(textToSpeak)
                    },
                    modifier = Modifier.size(uniformButtonHeight),
                    containerColor = Color(0xFF4CAF50),
                    contentColor = Color.White,
                    elevation = FloatingActionButtonDefaults.elevation(0.dp)
                ) {
                    Icon(Icons.Default.PlayArrow, contentDescription = "Speak")
                }
            }
        }
    }
}