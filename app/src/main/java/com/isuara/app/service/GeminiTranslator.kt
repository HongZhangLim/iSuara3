package com.isuara.app.service

import android.util.Log
import com.google.ai.client.generativeai.GenerativeModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import com.google.ai.client.generativeai.type.generationConfig
import com.google.ai.client.generativeai.type.content
/**
 * GeminiTranslator — translates detected BIM sign keywords into
 * a natural Bahasa Melayu sentence using Gemini 2.5 Flash Lite.
 */
class GeminiTranslator(apiKey: String) {

    companion object {
        private const val TAG = "GeminiTranslator"
    }

    private val model = GenerativeModel(
        modelName = "gemini-3.1-flash-lite-preview",
        apiKey = apiKey,
        generationConfig = generationConfig {
            temperature = 0.2f  // Controls randomness (0.0 to 2.0)
            topP = 0.8f         // Controls vocabulary diversity (0.0 to 1.0)
        },
                systemInstruction = content {
            text("""
            You are a professional Bahasa Isyarat Malaysia (BIM) sign language interpreter.
            
            Rules:
            1. Rearrange and expand the BIM keywords (glosses) into a natural Bahasa Melayu sentence (Subject + Verb + Object).
            2. Infer context and add implied verbs (e.g., "rasa," "mahu"), emotions(e.g.,"Gembira","Sedih","Kecewa","Maaf" and grammatical particles.
            3. If a phrase is ambiguous, default to the most direct, standard interpretation.
            4. Return ONLY one single sentence. Do NOT explain your process.
            
            Examples:
            Input: [Polis, Siapa, Salah]
            Output: Siapa yang polis salahkan tadi
            
            Input: [Saya, Makan, Sudah]
            Output: Saya sudah makan.
            
            Input: [Awak, Nama, Apa]
            Output: Siapakah nama awak?
            
            Input: [Hari ini, Tengok, Cantik]
            Output: Saya rasa gembira hari ini kerana melihat pemandangan yang cantik.
            
        """.trimIndent())
        }
    )

    /**
     * Convert a list of BIM sign keywords into a natural sentence.
     * Runs on IO dispatcher. Returns the translated sentence or error string.
     */
    suspend fun translate(words: List<String>): String = withContext(Dispatchers.IO) {
        if (words.isEmpty()) return@withContext ""

        // Keep it clean. The model already knows the rules from systemInstruction!
        val prompt = "Input: $words\nOutput:"

        try {
            val response = model.generateContent(prompt)
            val text = response.text?.trim() ?: ""
            Log.i(TAG, "Translation: $words → $text")
            text
        } catch (e: Exception) {
            Log.e(TAG, "Translation failed", e)
            "[Translation error: ${e.message}]"
        }
    }
}