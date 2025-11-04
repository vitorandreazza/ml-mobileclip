package com.mobileclip.test

import android.graphics.Bitmap
import android.graphics.Matrix

/**
 * Preprocesses images for MobileCLIP-S2 model inference.
 *
 * Pipeline: Resize -> Center Crop -> Normalize
 * - Target size: 256x256
 * - Mean: [0.48145466, 0.4578275, 0.40821073]
 * - Std: [0.26862954, 0.26130258, 0.27577711]
 */
class ImagePreprocessor {
    companion object {
        private const val IMAGE_SIZE = 256

        // ImageNet normalization values used by MobileCLIP
        private val MEAN = floatArrayOf(0.48145466f, 0.4578275f, 0.40821073f)
        private val STD = floatArrayOf(0.26862954f, 0.26130258f, 0.27577711f)

        /**
         * Preprocesses a bitmap for model inference.
         * Returns a FloatArray in CHW format (channels × height × width).
         */
        fun preprocess(bitmap: Bitmap): FloatArray {
            // Step 1: Resize to 256x256
            val resized = Bitmap.createScaledBitmap(
                bitmap,
                IMAGE_SIZE,
                IMAGE_SIZE,
                true // Use bilinear filtering
            )

            // Step 2: Center crop (already 256x256, so this is a no-op)
            val cropped = centerCrop(resized, IMAGE_SIZE)

            // Step 3: Convert to float array and normalize
            return bitmapToNormalizedFloatArray(cropped)
        }

        /**
         * Center crops the image to the target size.
         */
        private fun centerCrop(bitmap: Bitmap, size: Int): Bitmap {
            val width = bitmap.width
            val height = bitmap.height

            if (width == size && height == size) {
                return bitmap
            }

            val x = (width - size) / 2
            val y = (height - size) / 2

            return Bitmap.createBitmap(bitmap, x, y, size, size)
        }

        /**
         * Converts bitmap to normalized float array in CHW format.
         * Shape: [3, 256, 256]
         */
        private fun bitmapToNormalizedFloatArray(bitmap: Bitmap): FloatArray {
            val width = bitmap.width
            val height = bitmap.height
            val pixels = IntArray(width * height)

            bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

            // Output array: 3 channels × height × width
            val output = FloatArray(3 * height * width)

            for (i in pixels.indices) {
                val pixel = pixels[i]

                // Extract RGB values (0-255)
                val r = ((pixel shr 16) and 0xFF) / 255.0f
                val g = ((pixel shr 8) and 0xFF) / 255.0f
                val b = (pixel and 0xFF) / 255.0f

                // Normalize using mean and std, store in CHW format
                val idx = i % width + (i / width) * width
                output[idx] = (r - MEAN[0]) / STD[0]                          // Red channel
                output[height * width + idx] = (g - MEAN[1]) / STD[1]         // Green channel
                output[2 * height * width + idx] = (b - MEAN[2]) / STD[2]     // Blue channel
            }

            return output
        }
    }
}
