package com.isuara.app.service

import android.content.Context
import android.speech.tts.TextToSpeech
import android.util.Log
import java.util.Locale

/**
 * TtsService — Android TextToSpeech wrapper for reading out
 * translated Bahasa Melayu sentences.
 */
class TtsService(context: Context) {

    companion object {
        private const val TAG = "TtsService"
        private val LOCALE_MS = Locale("ms", "MY")
    }

    private var tts: TextToSpeech? = null
    private var isReady = false

    init {
        tts = TextToSpeech(context) { status ->
            if (status == TextToSpeech.SUCCESS) {
                // Try Malay first, fall back to Indonesian (very similar), then English
                val result = tts?.setLanguage(LOCALE_MS)
                if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                    val idResult = tts?.setLanguage(Locale("id", "ID"))
                    if (idResult == TextToSpeech.LANG_MISSING_DATA || idResult == TextToSpeech.LANG_NOT_SUPPORTED) {
                        tts?.setLanguage(Locale.US)
                        Log.w(TAG, "Malay & Indonesian TTS not available, using English")
                    } else {
                        Log.i(TAG, "Using Indonesian TTS (close to Malay)")
                    }
                } else {
                    Log.i(TAG, "Using Malay TTS")
                }

                tts?.setSpeechRate(0.9f)
                isReady = true
            } else {
                Log.e(TAG, "TTS initialization failed: $status")
            }
        }
    }

    /**
     * Speak the given text. Interrupts any ongoing speech.
     */
    fun speak(text: String) {
        if (!isReady || text.isBlank()) return
        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "isuara_tts")
    }

    /**
     * Stop any ongoing speech.
     */
    fun stop() {
        tts?.stop()
    }

    fun isSpeaking(): Boolean = tts?.isSpeaking == true

    fun close() {
        tts?.stop()
        tts?.shutdown()
        tts = null
        isReady = false
    }
}
