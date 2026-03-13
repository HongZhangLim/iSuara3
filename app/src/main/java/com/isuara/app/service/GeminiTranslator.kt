package com.isuara.app.service

import android.util.Log
import com.google.ai.client.generativeai.GenerativeModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * GeminiTranslator — translates detected BIM sign keywords into
 * a natural Bahasa Melayu sentence using Gemini 2.5 Flash Lite.
 */
class GeminiTranslator(apiKey: String) {

    companion object {
        private const val TAG = "GeminiTranslator"
    }

    private val model = GenerativeModel(
        modelName = "gemini-2.5-flash-lite",
        apiKey = apiKey
    )

    /**
     * Convert a list of BIM sign keywords into a natural sentence.
     * Runs on IO dispatcher. Returns the translated sentence or error string.
     */
    suspend fun translate(words: List<String>): String = withContext(Dispatchers.IO) {
        if (words.isEmpty()) return@withContext ""

        val prompt = """
You are a professional Bahasa Isyarat Malaysia (BIM) sign language interpreter.

Rules:

Input is a sequence of BIM keywords (glosses).

Rearrange and expand the keywords into a natural, conversational Bahasa Melayu sentence following the Subject + Verb + Object + Time/Location format.

Crucial: BIM often omits connecting verbs, transitions, and expressions of intent. You must infer the context and add implied verbs of action, feeling, or motion (e.g., adding "rasa," "ingin," "mahu," or "pergi ke") to capture the signer's true meaning, rather than just doing a literal grammatical fix.

Add any missing pronouns, prepositions, or grammatical particles needed for natural spoken Malay.

Keep the core meaning accurate to the signer's overall intent.

Do NOT explain your reasoning or translation process.

Return ONLY one single sentence.

Input: $words
Output:
""".trimIndent()

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
