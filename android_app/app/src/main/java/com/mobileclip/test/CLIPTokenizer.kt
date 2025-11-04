package com.mobileclip.test

import android.content.Context
import org.json.JSONObject
import java.util.regex.Pattern

/**
 * CLIP tokenizer implementing BPE (Byte Pair Encoding) for text tokenization.
 * Based on OpenAI's CLIP tokenizer.
 *
 * Context length: 77 tokens
 * Special tokens:
 * - SOT (Start of Text): 49406
 * - EOT (End of Text): 49407
 * - Padding: 0
 */
class CLIPTokenizer(context: Context) {
    private val contextLength = 77
    private val bpeRanks: Map<Pair<String, String>, Int>
    private val encoder: Map<String, Int>
    private val decoder: Map<Int, String>

    // GPT-2 byte encoder for UTF-8 encoding
    private val byteEncoder: Map<Int, String> = createByteEncoder()
    private val byteDecoder: Map<String, Int> = byteEncoder.entries.associate { it.value to it.key }

    init {
        // Load BPE merges
        val mergesText = context.assets.open("clip-merges.txt").bufferedReader().use { it.readText() }
        val mergeLines = mergesText.split("\n")
        bpeRanks = mergeLines.drop(1) // Skip header
            .filter { it.isNotBlank() }
            .mapIndexed { index, line ->
                val parts = line.split(" ")
                if (parts.size >= 2) {
                    Pair(parts[0], parts[1]) to index
                } else null
            }
            .filterNotNull()
            .toMap()

        // Load vocabulary
        val vocabText = context.assets.open("clip-vocab.json").bufferedReader().use { it.readText() }
        val vocabJson = JSONObject(vocabText)
        encoder = vocabJson.keys().asSequence().associateWith { vocabJson.getInt(it) }
        decoder = encoder.entries.associate { it.value to it.key }
    }

    /**
     * Creates the byte encoder mapping for GPT-2 BPE.
     * Maps UTF-8 bytes to unicode characters.
     */
    private fun createByteEncoder(): Map<Int, String> {
        val map = mutableMapOf<Int, String>()

        // Printable ASCII characters (33-126)
        for (i in 33..126) {
            map[i] = i.toChar().toString()
        }

        // Extended characters (161-255, excluding 173)
        for (i in 161..255) {
            if (i != 173) {
                map[i] = i.toChar().toString()
            }
        }

        // Non-printable characters mapped to unicode private use area
        var n = 0
        for (b in 0..255) {
            if (!map.containsKey(b)) {
                map[b] = (256 + n).toChar().toString()
                n++
            }
        }

        return map
    }

    /**
     * Byte-encodes text tokens using the GPT-2 encoding scheme.
     */
    private fun byteEncode(text: String): List<String> {
        // Regex pattern for tokenization
        val pattern = Pattern.compile(
            """<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            Pattern.CASE_INSENSITIVE
        )

        val matcher = pattern.matcher(text)
        val tokens = mutableListOf<String>()

        while (matcher.find()) {
            val token = matcher.group()
            // Convert each byte to its encoded character
            val encoded = token.toByteArray(Charsets.UTF_8)
                .map { byteEncoder[it.toInt() and 0xFF]!! }
                .joinToString("")
            tokens.add(encoded)
        }

        return tokens
    }

    /**
     * Get all adjacent pairs in a word.
     */
    private fun getPairs(word: List<String>): Set<Pair<String, String>> {
        val pairs = mutableSetOf<Pair<String, String>>()
        for (i in 0 until word.size - 1) {
            pairs.add(Pair(word[i], word[i + 1]))
        }
        return pairs
    }

    /**
     * Apply BPE merging algorithm to a token.
     */
    private fun bpe(token: String): String {
        if (token.length <= 1) {
            return "$token</w>"
        }

        var word = token.map { it.toString() }.toMutableList()
        word[word.size - 1] = word.last() + "</w>"

        var pairs = getPairs(word).toList()
        if (pairs.isEmpty()) {
            return "$token</w>"
        }

        while (true) {
            // Find bigrams that exist in our BPE ranks
            val bigrams = pairs.filter { bpeRanks.containsKey(it) }
            if (bigrams.isEmpty()) {
                break
            }

            // Get the bigram with minimum rank (most common)
            val bigram = bigrams.minByOrNull { bpeRanks[it]!! }!!

            val first = bigram.first
            val second = bigram.second
            val newWord = mutableListOf<String>()
            var i = 0

            while (i < word.size) {
                val j = word.subList(i, word.size).indexOf(first)
                if (j == -1) {
                    newWord.addAll(word.subList(i, word.size))
                    break
                }

                newWord.addAll(word.subList(i, i + j))
                i += j

                if (word[i] == first && i < word.size - 1 && word[i + 1] == second) {
                    newWord.add(first + second)
                    i += 2
                } else {
                    newWord.add(word[i])
                    i += 1
                }
            }

            word = newWord
            if (word.size == 1) {
                break
            }
            pairs = getPairs(word).toList()
        }

        return word.joinToString(" ")
    }

    /**
     * Tokenize text into BPE tokens.
     */
    private fun tokenize(text: String): List<String> {
        val tokens = mutableListOf<String>()
        val lowercased = text.lowercase()

        for (token in byteEncode(lowercased)) {
            val bpeTokens = bpe(token).split(" ")
            tokens.addAll(bpeTokens)
        }

        return tokens
    }

    /**
     * Encode text to token IDs.
     */
    private fun encode(text: String): List<Int> {
        return tokenize(text).mapNotNull { encoder[it] }
    }

    /**
     * Full encoding with SOT, EOT, and padding to context length.
     * This is the main method to use for model input.
     */
    fun encodeFull(text: String): LongArray {
        val tokens = encode(text)
        val fullTokens = LongArray(contextLength) { 0 } // Initialize with padding

        // Add SOT token
        fullTokens[0] = encoder["<|startoftext|>"]!!.toLong()

        // Add text tokens (truncate if necessary)
        val maxTokens = minOf(tokens.size, contextLength - 2)
        for (i in 0 until maxTokens) {
            fullTokens[i + 1] = tokens[i].toLong()
        }

        // Add EOT token
        fullTokens[maxTokens + 1] = encoder["<|endoftext|>"]!!.toLong()

        return fullTokens
    }

    /**
     * Decode token IDs back to text.
     */
    fun decode(tokens: List<Int>): String {
        val text = tokens.mapNotNull { decoder[it] }.joinToString("")
        val utfBytes = text.map { byteDecoder[it.toString()]!!.toByte() }.toByteArray()
        return String(utfBytes, Charsets.UTF_8)
    }
}
