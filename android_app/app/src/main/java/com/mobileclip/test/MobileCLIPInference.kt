package com.mobileclip.test

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import java.io.File
import kotlin.math.exp
import kotlin.math.sqrt

/**
 * MobileCLIP inference engine for running image and text encoders.
 * Uses ExecuTorch .pte models for efficient inference.
 */
class MobileCLIPInference(private val context: Context) {
    private var imageModule: Module? = null
    private var textModule: Module? = null
    private val tokenizer: CLIPTokenizer = CLIPTokenizer(context)

    companion object {
        private const val IMAGE_MODEL_NAME = "mobileclip-s2_visual.pte"
        private const val TEXT_MODEL_NAME = "mobileclip-s2_text.pte"
        private const val EMBEDDING_DIM = 512
    }

    /**
     * Load the .pte model files from assets.
     */
    fun loadModels(): Boolean {
        return try {
            // Copy models from assets to internal storage if needed
            val imageModelPath = copyAssetToFile(IMAGE_MODEL_NAME)
            val textModelPath = copyAssetToFile(TEXT_MODEL_NAME)

            // Load ExecuTorch modules
            imageModule = Module.load(imageModelPath)
            textModule = Module.load(textModelPath)

            true
        } catch (e: Exception) {
            e.printStackTrace()
            false
        }
    }

    /**
     * Copy asset file to internal storage for ExecuTorch loading.
     */
    private fun copyAssetToFile(assetName: String): String {
        val assetPath = "models/$assetName"
        val outputFile = File(context.filesDir, assetName)

        if (!outputFile.exists()) {
            context.assets.open(assetPath).use { input ->
                outputFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
        }

        return outputFile.absolutePath
    }

    /**
     * Run inference on multiple images and text descriptions.
     * Returns a map of image names to their results.
     */
    fun runTest(imageAssetPaths: List<String>, texts: List<String>): TestResults {
        val results = mutableListOf<ImageResult>()

        for (imagePath in imageAssetPaths) {
            val result = processImage(imagePath, texts)
            results.add(result)
        }

        return TestResults(results)
    }

    /**
     * Process a single image against multiple text descriptions.
     */
    private fun processImage(imageAssetPath: String, texts: List<String>): ImageResult {
        val imageName = imageAssetPath.substringAfterLast('/')

        // Load and preprocess image
        val bitmap = loadBitmapFromAssets(imageAssetPath)
        val imageFeatures = encodeImage(bitmap)

        // Encode all text descriptions
        val textFeatures = texts.map { encodeText(it) }

        // Normalize features
        val normalizedImageFeatures = normalize(imageFeatures)
        val normalizedTextFeatures = textFeatures.map { normalize(it) }

        // Compute similarities
        val similarities = normalizedTextFeatures.map { textFeat ->
            100.0f * dotProduct(normalizedImageFeatures, textFeat)
        }

        // Apply softmax to get probabilities
        val probabilities = softmax(similarities)

        // Find top match
        val topIndex = probabilities.indices.maxByOrNull { probabilities[it] } ?: 0
        val topMatch = texts[topIndex]
        val topProb = probabilities[topIndex]

        return ImageResult(
            imageName = imageName,
            texts = texts,
            probabilities = probabilities,
            topMatch = topMatch,
            topProbability = topProb
        )
    }

    /**
     * Encode image to feature vector.
     */
    private fun encodeImage(bitmap: Bitmap): FloatArray {
        val preprocessed = ImagePreprocessor.preprocess(bitmap)

        // Create input tensor shape: [1, 3, 256, 256]
        val inputTensor = org.pytorch.executorch.Tensor.fromBlob(
            preprocessed,
            longArrayOf(1, 3, 256, 256)
        )

        // Run inference
        val output = imageModule!!.forward(EValue.from(inputTensor))
        val outputTensor = output[0]

        // Extract features (shape: [1, 512])
        return outputTensor.toTensor().dataAsFloatArray
    }

    /**
     * Encode text to feature vector.
     */
    private fun encodeText(text: String): FloatArray {
        val tokens = tokenizer.encodeFull(text)

        // Create input tensor shape: [1, 77]
        val inputTensor = org.pytorch.executorch.Tensor.fromBlob(
            tokens,
            longArrayOf(1, 77)
        )

        // Run inference
        val output = textModule!!.forward(EValue.from(inputTensor))
        val outputTensor = output[0]

        // Extract features (shape: [1, 512])
        return outputTensor.toTensor().dataAsFloatArray
    }

    /**
     * Load bitmap from assets.
     */
    private fun loadBitmapFromAssets(path: String): Bitmap {
        return context.assets.open(path).use { inputStream ->
            BitmapFactory.decodeStream(inputStream)
        }
    }

    /**
     * Normalize feature vector (L2 normalization).
     */
    private fun normalize(features: FloatArray): FloatArray {
        val norm = sqrt(features.map { it * it }.sum())
        return features.map { it / norm }.toFloatArray()
    }

    /**
     * Compute dot product of two vectors.
     */
    private fun dotProduct(a: FloatArray, b: FloatArray): Float {
        return a.indices.sumOf { (a[it] * b[it]).toDouble() }.toFloat()
    }

    /**
     * Apply softmax to get probabilities.
     */
    private fun softmax(values: List<Float>): List<Float> {
        val maxVal = values.maxOrNull() ?: 0f
        val exps = values.map { exp((it - maxVal).toDouble()).toFloat() }
        val sumExps = exps.sum()
        return exps.map { it / sumExps }
    }

    /**
     * Release resources.
     */
    fun release() {
        imageModule?.destroy()
        textModule?.destroy()
    }
}

/**
 * Results for a single image.
 */
data class ImageResult(
    val imageName: String,
    val texts: List<String>,
    val probabilities: List<Float>,
    val topMatch: String,
    val topProbability: Float
)

/**
 * Overall test results.
 */
data class TestResults(
    val imageResults: List<ImageResult>
) {
    /**
     * Format results as readable text.
     */
    fun formatResults(): String {
        val sb = StringBuilder()

        for ((index, result) in imageResults.withIndex()) {
            if (index > 0) sb.append("\n\n")

            sb.append("=== ${result.imageName} ===\n\n")
            sb.append("Probabilities:\n")
            for ((text, prob) in result.texts.zip(result.probabilities)) {
                sb.append(String.format("  %s: %.4f\n", text, prob))
            }

            sb.append(String.format(
                "\nTop Match: %s (%.4f)\n",
                result.topMatch,
                result.topProbability
            ))
        }

        return sb.toString()
    }
}
